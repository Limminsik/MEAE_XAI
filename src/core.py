"""공용 원시 함수 — 번호 붙은 단계 스크립트(01~08)가 공유한다.

여기에는 **여러 단계가 함께 쓰는 것만** 둔다. 한 단계에서만 쓰는 계산·표·그림은
그 단계 스크립트 안에 둔다.

  · 체크포인트 로드          load_ckpt · load_run (확정 모델 / 딥러닝 비교선)
  · 딥러닝 비교선            DeepFilter · DeScoD · baseline_restore
  · 성분·재구성 추출         component_bank · reconstruct
  · 분절 내 표준화와 지표    znorm · pearson · aggregate · rmse_norm_matrix · mad_matrix
  · 표 렌더링                top_idx · mark · render

모든 계산은 **crop 후 중앙 3600 구간**에서 한다.
"""
import os

import numpy as np
import torch

from .data.dataset import REF_KEYS
from .model import meae
from .model.meae import enc_label

NOISE_REFS = ("bw", "ma", "em")


# ---------------------------------------------------------------- 모델·데이터
def load_ckpt(cfg, run):
    """run 은 실행 이름이거나 체크포인트 파일 경로다.
    후자를 허용하는 이유: pool/ 에 보관한 후보 에폭을 재학습 없이 그대로 불러 쓰기 위해서다."""
    if run.endswith(".pt") and os.path.exists(run):
        path = run
    else:
        name = os.path.basename(run)
        cand = [os.path.join("results", "02_model", run, f"{name}.pt"),
                os.path.join("_work", "archive", "runs", run, f"{name}.pt")]
        path = next((c for c in cand if os.path.exists(c)), cand[0])
    ck = torch.load(path, map_location="cpu", weights_only=False)
    # 체크포인트에 저장된 config를 우선한다 — hidden 등 구조 오버라이드가 반영돼 있다
    model = meae.build(ck.get("cfg", cfg), ck["n_encoders"])
    model.load_state_dict(ck["model"])
    return model.eval(), ck


# ════════════════════════════════════════════════════════════════════════
# 딥러닝 비교선 — **기법만 바꾼 대조군**
# ════════════════════════════════════════════════════════════════════════
# 데이터·기록 분할·잡음 주입·평가 지표·집계를 그대로 두고 디노이징 방법만 공개 기법으로
# 갈아 끼운다. 확정 모델과 **같은 파이프라인 위에 별개의 run** 으로 올린다 —
# `results/02_model/<run>/` 에 학습 산출물이 서고, 04·05·06 이 `--run <run>` 으로
# 그 run 의 결과 폴더를 따로 낸다. 03(성분 ↔ 참조 대응)은 돌리지 않는다 —
# 인코더가 하나뿐이라 대응표가 성립하지 않는다.
#
# 비교선은 두 가지만 약속한다 (`BaselineDenoiser`):
#   loss(x_noisy, x_clean) -> 스칼라      02 가 학습에 쓴다
#   denoise(x_noisy)       -> (B,1,L)     04·05·06 이 복원에 쓴다
# 손실·옵티마이저·콜백·체크포인트 선정은 **각 원 논문 코드 방식 그대로** 둔다.
# 그것도 기법의 일부이므로 우리 학습 설정으로 바꾸지 않는다.


class BaselineDenoiser(torch.nn.Module):
    """비교선 공통 인터페이스. 학습 손실 하나, 추론 하나."""

    def loss(self, x_noisy, x_clean):
        raise NotImplementedError

    @torch.no_grad()
    def denoise(self, x_noisy):
        raise NotImplementedError


# ──────────────────────────────────────────────────────────────────────
# ① DeepFilter — Multibranch LANLD
# ──────────────────────────────────────────────────────────────────────
#   원 구현  Perdigón-Romero et al. (2021) "DeepFilter: an ECG baseline wander
#            removal filter using deep learning techniques", Biomed Signal Process
#            Control 70:102992. github.com/fperdigon/DeepFilter — deepFilter/dl_models.py
#            `deep_filter_model_I_LANL_dilated` (논문 표기 Multibranch LANLD)
#   이식     Keras → PyTorch. 커널 크기·채널 수·드롭아웃·배치정규화 배치·활성함수를
#            원본과 같게 옮겼다. 학습 파라미터 68,719개로 원본과 같다.
#            **완전합성곱**(stride 1, padding same, 풀링 없음)이라 길이에 무관하다 —
#            원 논문 512 샘플 대신 우리 3,600 샘플을 그대로 받는다.
#   입출력   x_noisy (B,1,3600) → x_clean 추정. mV 원단위, 표준화하지 않는다.
#            고전 비교선과 달리 DC 복원이 필요 없다 — 오프셋까지 학습해 낸다.

class _LANL(torch.nn.Module):
    """LANLFilter_module — 선형 4갈래 + ReLU 4갈래. 커널 3·5·9·15, 각 layers//8 채널."""

    KS = (3, 5, 9, 15)

    def __init__(self, cin, layers):
        super().__init__()
        c = int(layers / 8)
        mk = lambda k: torch.nn.Conv1d(cin, c, k, padding="same")
        self.lin = torch.nn.ModuleList([mk(k) for k in self.KS])
        self.nlin = torch.nn.ModuleList([mk(k) for k in self.KS])
        self.out_channels = c * 8

    def forward(self, x):
        return torch.cat([m(x) for m in self.lin]
                         + [torch.relu(m(x)) for m in self.nlin], dim=1)


class _LANLD(torch.nn.Module):
    """LANLFilter_module_dilated — 같은 구성에 dilation 3. 커널 5·9·15, 각 layers//6 채널."""

    KS = (5, 9, 15)
    DILATION = 3

    def __init__(self, cin, layers):
        super().__init__()
        c = int(layers / 6)
        mk = lambda k: torch.nn.Conv1d(cin, c, k, padding="same", dilation=self.DILATION)
        self.lin = torch.nn.ModuleList([mk(k) for k in self.KS])
        self.nlin = torch.nn.ModuleList([mk(k) for k in self.KS])
        self.out_channels = c * 6

    def forward(self, x):
        return torch.cat([m(x) for m in self.lin]
                         + [torch.relu(m(x)) for m in self.nlin], dim=1)


def ssd_mad_loss(pred, true):
    """저자 combined_ssd_mad_loss — max(d²)·50 + sum(d²). 분절마다 재고 배치 평균."""
    d = (pred - true) ** 2
    return (d.amax(-1) * 50.0 + d.sum(-1)).mean()


DEEPFILTER_LOSSES = {"ssd_mad": ssd_mad_loss,
                     "mse": lambda p, t: ((p - t) ** 2).mean()}


class DeepFilter(BaselineDenoiser):
    """Multibranch LANLD. LANL·LANLD 를 번갈아 6블록, 블록마다 Dropout → BatchNorm.

    채널 폭은 64 → 64 → 32 → 32 → 16 → 16 이고 실제 출력 채널은
    64 → 60 → 32 → 30 → 16 → 12 다(정수 나눗셈 때문에 LANLD 가 하나씩 적다).
    마지막은 커널 9 선형 합성곱 1채널 — 원본과 같다.
    """

    SPECS = ((_LANL, 64), (_LANLD, 64), (_LANL, 32),
             (_LANLD, 32), (_LANL, 16), (_LANLD, 16))

    def __init__(self, dropout=0.4, loss="ssd_mad"):
        super().__init__()
        blocks, cin = [], 1
        for cls, layers in self.SPECS:
            m = cls(cin, layers)
            # Keras BatchNormalization 기본값 — eps 1e-3, momentum 0.99 (PyTorch 표기 0.01)
            blocks += [m, torch.nn.Dropout(dropout),
                       torch.nn.BatchNorm1d(m.out_channels, eps=1e-3, momentum=0.01)]
            cin = m.out_channels
        self.body = torch.nn.Sequential(*blocks)
        self.head = torch.nn.Conv1d(cin, 1, 9, padding="same")
        self.crit = DEEPFILTER_LOSSES[loss]
        # Keras Conv1D 기본 초기화 — glorot_uniform 가중치, 0 편향. PyTorch 기본(kaiming
        # uniform)과 다르면 학습 궤적이 달라진다. "저자 코드 그대로" 를 위해 맞춘다.
        for m in self.modules():
            if isinstance(m, torch.nn.Conv1d):
                torch.nn.init.xavier_uniform_(m.weight)
                torch.nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.head(self.body(x))

    def loss(self, x_noisy, x_clean):
        return self.crit(self(x_noisy), x_clean)

    @torch.no_grad()
    def denoise(self, x_noisy):
        return self(x_noisy)


# ──────────────────────────────────────────────────────────────────────
# ② DeScoD-ECG — 조건부 확산 모델
# ──────────────────────────────────────────────────────────────────────
#   원 구현  Li H, Ditzler G, Roveda J, Li A. (2024) "DeScoD-ECG: Deep Score-Based
#            Diffusion Model for ECG Baseline Wander and Noise Removal",
#            IEEE J Biomed Health Inform 28(11):5081–5091.
#            github.com/huayuLiArizona/Score-based-ECG-Denoising
#            — denoising_model_small.py `ConditionalModel` · main_model.py `DDPM`
#   이식     구조·스케줄·손실·다중 표본 평균을 원본 그대로 옮겼다. 원본이 이미
#            PyTorch 라 코드가 아니라 **경로만** 우리 것으로 바꾼 셈이다.
#            쓰이지 않던 `leaf_audio_pytorch` 임포트만 뺐다.
#   방식     x_noisy 를 조건으로 두고 x_clean 을 확산 복원한다. 추론이 확률적이라
#            같은 입력을 `shots` 번 복원해 평균한다(저자 multi-shot averaging).
#            복원 1회가 num_steps 번의 역방향 통과이므로 shots × num_steps 만큼
#            무겁다 — 고전·DeepFilter 와 비용이 다르다는 것은 기법의 성질이다.
#   길이     완전합성곱(reflect 패딩, 풀링 없음)이라 3,600 샘플을 그대로 받는다.
#            단 HNF 블록의 InstanceNorm1d 가 **전체 길이의 통계**로 정규화하므로 수용영역은
#            창 전체다(기울기 탐침으로 확인). DeepFilter 처럼 짧은 수용영역 문제는 없지만,
#            창 길이가 달라지면 정규화 통계가 달라진다 — 그래서 저자 설계 길이(512, beat
#            창)로 맞추는 것이 여기서도 의미가 있다.
#   눈금     mV 원단위 그대로 넣는다. 확산 모델은 데이터 눈금에 민감하지만 원 논문도
#            mV 신호를 그대로 썼다. 표준화를 새로 넣으면 그것은 기법의 변경이다.

class _DSConv1d(torch.nn.Conv1d):
    """저자 Conv1d — kaiming_normal 가중치, 0 편향."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        torch.nn.init.kaiming_normal_(self.weight)
        torch.nn.init.zeros_(self.bias)


class _PositionalEncoding(torch.nn.Module):
    """잡음 수준(연속 √ᾱ)을 sin/cos 로 인코딩한다."""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, noise_level):
        import math
        noise_level = noise_level.view(-1)
        count = self.dim // 2
        step = torch.arange(count, dtype=noise_level.dtype,
                            device=noise_level.device) / count
        enc = noise_level.unsqueeze(1) * torch.exp(-math.log(1e4) * step.unsqueeze(0))
        return torch.cat([torch.sin(enc), torch.cos(enc)], dim=-1)


class _FeatureWiseAffine(torch.nn.Module):
    def __init__(self, cin, cout, use_affine_level=False):
        super().__init__()
        self.use_affine_level = use_affine_level
        self.noise_func = torch.nn.Sequential(
            torch.nn.Linear(cin, cout * (1 + int(self.use_affine_level))))

    def forward(self, x, noise_embed):
        b = x.shape[0]
        if self.use_affine_level:
            gamma, beta = self.noise_func(noise_embed).view(b, -1, 1).chunk(2, dim=1)
            return (1 + gamma) * x + beta
        return x + self.noise_func(noise_embed).view(b, -1, 1)


class _HNFBlock(torch.nn.Module):
    """다중 커널(3·5·9·15) 잔차 블록. 절반만 InstanceNorm 을 거친다."""

    def __init__(self, cin, hidden, dilation):
        super().__init__()
        h4 = hidden // 4
        self.filters = torch.nn.ModuleList([
            _DSConv1d(cin, h4, 3, dilation=dilation, padding=1 * dilation,
                      padding_mode="reflect"),
            _DSConv1d(hidden, h4, 5, dilation=dilation, padding=2 * dilation,
                      padding_mode="reflect"),
            _DSConv1d(hidden, h4, 9, dilation=dilation, padding=4 * dilation,
                      padding_mode="reflect"),
            _DSConv1d(hidden, h4, 15, dilation=dilation, padding=7 * dilation,
                      padding_mode="reflect"),
        ])
        self.conv_1 = _DSConv1d(hidden, hidden, 9, padding=4, padding_mode="reflect")
        self.norm = torch.nn.InstanceNorm1d(hidden // 2)
        self.conv_2 = _DSConv1d(hidden, hidden, 9, padding=4, padding_mode="reflect")

    def forward(self, x):
        residual = x
        filts = torch.cat([layer(x) for layer in self.filters], dim=1)
        nfilts, filts = self.conv_1(filts).chunk(2, dim=1)
        filts = torch.nn.functional.leaky_relu(
            torch.cat([self.norm(nfilts), filts], dim=1), 0.2)
        filts = torch.nn.functional.leaky_relu(self.conv_2(filts), 0.2)
        return filts + residual


class _Bridge(torch.nn.Module):
    def __init__(self, cin, hidden):
        super().__init__()
        self.encoding = _FeatureWiseAffine(cin, hidden, use_affine_level=1)
        self.input_conv = _DSConv1d(cin, cin, 3, padding=1, padding_mode="reflect")
        self.output_conv = _DSConv1d(cin, hidden, 3, padding=1, padding_mode="reflect")

    def forward(self, x, noise_embed):
        x = self.input_conv(x)
        x = self.encoding(x, noise_embed)
        return self.output_conv(x)


class _ConditionalModel(torch.nn.Module):
    """두 흐름(잡음 섞인 x, 조건 신호)을 다리로 잇는 잡음 예측망."""

    def __init__(self, feats=80):
        super().__init__()
        mk_stream = lambda: torch.nn.ModuleList([
            torch.nn.Sequential(
                _DSConv1d(1, feats, 9, padding=4, padding_mode="reflect"),
                torch.nn.LeakyReLU(0.2)),
            _HNFBlock(feats, feats, 1), _HNFBlock(feats, feats, 2),
            _HNFBlock(feats, feats, 4), _HNFBlock(feats, feats, 2),
            _HNFBlock(feats, feats, 1)])
        self.stream_x = mk_stream()
        self.stream_cond = mk_stream()
        self.embed = _PositionalEncoding(feats)
        self.bridge = torch.nn.ModuleList([_Bridge(feats, feats) for _ in range(5)])
        self.conv_out = _DSConv1d(feats, 1, 9, padding=4, padding_mode="reflect")

    def forward(self, x, cond, noise_scale):
        noise_embed = self.embed(noise_scale)
        xs = []
        for layer, br in zip(self.stream_x, self.bridge):
            x = layer(x)
            xs.append(br(x, noise_embed))
        for xi, layer in zip(xs, self.stream_cond):
            cond = layer(cond) + xi
        return self.conv_out(cond)


class DeScoD(BaselineDenoiser):
    """DDPM 래퍼 — 저자 main_model.DDPM 과 같은 스케줄·손실·역방향 표집."""

    def __init__(self, feats=80, num_steps=50, beta_start=1e-4, beta_end=0.5,
                 schedule="quad", shots=10, sample_seed=42):
        super().__init__()
        self.model = _ConditionalModel(feats)
        self.num_steps = int(num_steps)
        self.shots = int(shots)
        self.sample_seed = int(sample_seed)
        self.loss_func = torch.nn.L1Loss(reduction="sum")
        self._set_schedule(schedule, beta_start, beta_end)

    def _set_schedule(self, schedule, start, end):
        n = self.num_steps
        if schedule == "linear":
            betas = torch.linspace(start, end, n)
        elif schedule == "quad":
            betas = torch.linspace(start ** 0.5, end ** 0.5, n) ** 2
        elif schedule == "sigmoid":
            betas = torch.sigmoid(torch.linspace(-6, 6, n)) * (end - start) + start
        else:
            raise ValueError(f"알 수 없는 스케줄: {schedule}")
        betas = betas.numpy()          # 저자와 같은 float32 경로로 둔다
        alphas = 1.0 - betas
        ac = np.cumprod(alphas, axis=0)
        ac_prev = np.append(1.0, ac[:-1])
        self.sqrt_alphas_cumprod_prev = np.sqrt(np.append(1.0, ac))
        t = lambda v: torch.tensor(v, dtype=torch.float32)
        self.register_buffer("betas", t(betas))
        self.register_buffer("sqrt_recip_alphas_cumprod", t(np.sqrt(1.0 / ac)))
        self.register_buffer("sqrt_recipm1_alphas_cumprod", t(np.sqrt(1.0 / ac - 1)))
        pv = betas * (1.0 - ac_prev) / (1.0 - ac)
        self.register_buffer("posterior_log_variance_clipped",
                             t(np.log(np.maximum(pv, 1e-20))))
        self.register_buffer("posterior_mean_coef1",
                             t(betas * np.sqrt(ac_prev) / (1.0 - ac)))
        self.register_buffer("posterior_mean_coef2",
                             t((1.0 - ac_prev) * np.sqrt(alphas) / (1.0 - ac)))

    # ---- 학습 (저자 p_losses)
    #   확산 시점 t 와 잡음을 매 호출 무작위로 뽑는다 — 학습에서는 저자 그대로다.
    #   **평가(eval 모드)에서는 고정 시드로 뽑는다.** 저자 코드는 val 손실도 무작위
    #   t 로 재서 에폭마다 2~3배씩 출렁이고, 그 최소를 체크포인트로 고른다 — 우리 실행에서
    #   val 이 23k~69k 를 오가며 "운 좋은 draw" 를 골랐다. 기법이 아니라 **채점의 잡음**이므로
    #   평가 draw 만 고정해 에폭 간 비교가 같은 눈금에 놓이게 한다.
    def loss(self, x_noisy, x_clean):
        b = x_clean.shape[0]
        if self.training:
            rng, gen = np.random, None
        else:
            rng = np.random.RandomState(self.sample_seed + b)
            gen = torch.Generator(device=x_clean.device).manual_seed(self.sample_seed)
        step = rng.randint(1, self.num_steps + 1)
        cs = torch.FloatTensor(rng.uniform(
            self.sqrt_alphas_cumprod_prev[step - 1],
            self.sqrt_alphas_cumprod_prev[step], size=b)).to(x_clean.device).view(b, -1)
        noise = (torch.randn_like(x_clean) if gen is None
                 else torch.randn(x_clean.shape, generator=gen, device=x_clean.device,
                                  dtype=x_clean.dtype))
        x_t = cs.view(-1, 1, 1) * x_clean + (1 - cs.view(-1, 1, 1) ** 2).sqrt() * noise
        return self.loss_func(noise, self.model(x_t, x_noisy, cs))

    # ---- 추론 (저자 p_sample_loop + multi-shot averaging)
    @torch.no_grad()
    def _one_shot(self, cond):
        x = torch.randn_like(cond)
        for i in reversed(range(self.num_steps)):
            b = x.shape[0]
            nl = torch.FloatTensor([self.sqrt_alphas_cumprod_prev[i + 1]]
                                   ).repeat(b, 1).to(x.device)
            eps = self.model(x, cond, nl)
            x0 = self.sqrt_recip_alphas_cumprod[i] * x - \
                self.sqrt_recipm1_alphas_cumprod[i] * eps
            mean = self.posterior_mean_coef1[i] * x0 + self.posterior_mean_coef2[i] * x
            logvar = self.posterior_log_variance_clipped[i]
            noise = torch.randn_like(x) if i > 0 else torch.zeros_like(x)
            x = mean + noise * (0.5 * logvar).exp()
        return x

    @torch.no_grad()
    def denoise(self, x_noisy):
        # 표집이 확률적이므로 시드를 고정해 재현 가능하게 둔다
        g = torch.random.get_rng_state()
        torch.manual_seed(self.sample_seed)
        out = sum(self._one_shot(x_noisy) for _ in range(self.shots)) / self.shots
        torch.random.set_rng_state(g)
        return out


# ──────────────────────────────────────────────────────────────────────
# 비교선 등록 · run 갈래 처리
# ──────────────────────────────────────────────────────────────────────
BASELINE_KIND = "deepfilter"          # 기본값 (하위 호환)
BASELINE_NAMES = ("deepfilter", "descod")


def baseline_cfg(cfg, name=BASELINE_KIND):
    """configs 의 baselines.<name> 절. 코드에 값을 두지 않는다."""
    b = (cfg.get("baselines") or {}).get(name)
    if b is None:
        raise KeyError(f"configs 에 baselines.{name} 절이 없다")
    return b


def baseline_run_name(cfg, name=BASELINE_KIND):
    return baseline_cfg(cfg, name)["run"]


def build_baseline(cfg, name=BASELINE_KIND):
    """이름으로 비교선을 만든다. 하이퍼파라미터는 전부 config 에서 온다."""
    b = baseline_cfg(cfg, name)
    if name == "deepfilter":
        return DeepFilter(dropout=float(b["dropout"]), loss=b["loss"])
    if name == "descod":
        return DeScoD(feats=int(b["feats"]), num_steps=int(b["num_steps"]),
                      beta_start=float(b["beta_start"]), beta_end=float(b["beta_end"]),
                      schedule=b["schedule"], shots=int(b["shots"]),
                      sample_seed=int(b["seed"]))
    raise ValueError(f"알 수 없는 비교선: {name} (가능: {BASELINE_NAMES})")


# ──────────────────────────────────────────────────────────────────────
# 저자가 설계한 입력 길이로 잘라 넣기 — **데이터셋이 아니라 창만 바꾼다**
# ──────────────────────────────────────────────────────────────────────
# DeepFilter·DeScoD 는 둘 다 **512 표본** 입력으로 설계·보고된 기법이다. 우리 분절은
# 3,600 표본(10초)이라 그대로 넣으면 기법을 설계 범위 밖에서 쓰는 셈이 된다 —
# 수용영역이 177 표본(0.49초)뿐이라 2~20초 주기의 기저선 변동을 창 안에서 볼 수 없다.
#
# 그래서 **잡음 규약은 우리 것 그대로 두고**(bw·ma·em, SNR 0–12 dB) 입력 창만 저자
# 설계대로 잘라 넣는다. 잘린 창은 이미 우리 규약으로 잡음이 주입된 신호의 일부다.
#
#   full  분절 3,600 표본을 그대로 (지금까지의 판)
#   tile  512 표본 비중첩 타일링. 전 구간을 덮고 이어 붙인다
#   beat  R-피크 중심 512 표본 창. 저자의 박동 단위에 대응한다.
#         저자는 QT Database 주석으로 박동을 자르지만 우리 npz 에는 R-피크가 있으므로
#         **R-피크 중심 창**이 가장 가까운 대응이다 — 이것이 유일한 불가피한 각색이고
#         원고에 그대로 적는다. 저자와 같이 창의 양끝 평균을 빼 기준선을 맞추고
#         복원 뒤 되돌린다. 겹치는 구간은 평균, 어느 창에도 안 닿는 구간은 입력 그대로.


def window_spec(cfg, name):
    w = dict((baseline_cfg(cfg, name).get("window") or {}))
    w.setdefault("mode", "full")
    w.setdefault("length", 512)
    return w


def cut_windows(x, spec, peaks=None, base=None):
    """(B,1,L) → (창, 되돌릴 정보). mode 가 full 이면 그대로 돌려준다."""
    mode, n = spec["mode"], int(spec["length"])
    if mode == "full":
        return x, None
    B, _, L = x.shape
    starts, owner = [], []
    if mode == "tile":
        # 비중첩 타일 + 마지막 창은 끝에 맞춘다(3600 = 7×512 + 16 이라 반사 패딩으로 가짜
        # 표본을 만들지 않는다). 겹치는 16 표본은 평균.
        st0 = list(range(0, L - n + 1, n))
        if st0[-1] != L - n:
            st0.append(L - n)
        for b in range(B):
            starts += st0
            owner += [b] * len(st0)
    elif mode == "beat":
        if peaks is None:
            raise ValueError("beat 창에는 R-피크가 필요하다")
        for b in range(B):
            pk = np.asarray(peaks[b], dtype=np.int64)
            if len(pk) == 0:                              # 검출 실패 분절은 타일로 덮는다
                pk = np.arange(n // 2, L, n)
            st = sorted({int(np.clip(p - n // 2, 0, max(L - n, 0))) for p in pk}
                        | {0, max(L - n, 0)})             # 양끝 창을 항상 넣어 전 구간을 덮는다
            starts += st
            owner += [b] * len(st)
    else:
        raise ValueError(f"알 수 없는 창 모드: {mode}")
    idx = torch.arange(n, device=x.device)
    st = torch.tensor(starts, device=x.device, dtype=torch.long)
    ow = torch.tensor(owner, device=x.device, dtype=torch.long)
    gather = st.unsqueeze(1) + idx.unsqueeze(0)       # (W, n)
    win = x[ow, 0][torch.arange(len(st), device=x.device).unsqueeze(1), gather]
    if mode == "beat":
        # 저자 기준선 정렬 — 창 양끝 평균을 뺀다. **오프셋은 입력(x_noisy)에서 한 번만
        # 재고, 학습 목표(x_clean)에도 같은 값을 쓴다** (base 인자로 전달). 각자 자기
        # 양끝으로 빼면 입력과 목표가 다른 오프셋을 잃어 목표가 어긋난다. 추론에는
        # 참값이 없으므로 입력 오프셋만 쓰는 것이 유일하게 일관된 선택이다.
        b0 = base if base is not None else (win[:, :1] + win[:, -1:]) / 2.0
        win = win - b0
    else:
        b0 = torch.zeros(win.shape[0], 1, device=x.device, dtype=x.dtype)
    return win.unsqueeze(1), {"mode": mode, "B": B, "L": L, "n": n, "st": st,
                              "ow": ow, "gather": gather, "base": b0}


def join_windows(y, meta, x_in):
    """복원한 창을 원래 길이로 되돌린다. meta 가 None 이면 그대로."""
    if meta is None:
        return y
    B, L = meta["B"], meta["L"]
    out = x_in.clone()                                    # 창이 안 닿는 구간은 입력 그대로
    acc = torch.zeros(B, L, device=y.device, dtype=y.dtype)
    cnt = torch.zeros(B, L, device=y.device, dtype=y.dtype)
    vals = (y.squeeze(1) + meta["base"])                  # 뺐던 기준선을 되돌린다
    acc.index_put_((meta["ow"].unsqueeze(1).expand_as(meta["gather"]), meta["gather"]),
                   vals, accumulate=True)
    cnt.index_put_((meta["ow"].unsqueeze(1).expand_as(meta["gather"]), meta["gather"]),
                   torch.ones_like(vals), accumulate=True)
    hit = cnt > 0
    out[:, 0][hit] = (acc[hit] / cnt[hit])
    return out


def _ckpt_path(run):
    """load_ckpt 와 같은 탐색 규칙 — results/02_model/<run>/<run>.pt"""
    if run.endswith(".pt") and os.path.exists(run):
        return run
    name = os.path.basename(run)
    cand = [os.path.join("results", "02_model", run, f"{name}.pt"),
            os.path.join("_work", "archive", "runs", run, f"{name}.pt")]
    return next((c for c in cand if os.path.exists(c)), cand[0])


def run_kind(run):
    """run 이 확정 모델(meae)인지 어느 비교선인지. 체크포인트의 kind 로 가른다."""
    p = _ckpt_path(run)
    if not os.path.exists(p):
        return "meae"
    return torch.load(p, map_location="cpu", weights_only=False).get("kind", "meae")


def load_run(cfg, run):
    """(kind, model, ck). kind 가 'meae' 면 load_ckpt 와 같고, 아니면 비교선을 올린다.

    04·05·06 이 run 이름 하나로 모든 갈래를 받게 하는 진입점이다.
    """
    kind = run_kind(run)
    if kind == "meae":
        model, ck = load_ckpt(cfg, run)
        return kind, model, ck
    ck = torch.load(_ckpt_path(run), map_location="cpu", weights_only=False)
    model = build_baseline(cfg, kind)
    model.load_state_dict(ck["model"])
    return kind, model.eval(), ck


@torch.no_grad()
def baseline_restore(model, ds, device, idx, batch=100, progress=None, spec=None):
    """비교선의 복원 신호 (n, 3600). 마스킹이 없다 — 출력이 하나뿐이다.

    `spec` 을 주면 저자가 설계한 입력 창으로 잘라 넣고 원래 길이로 되돌린다.
    """
    spec = spec or {"mode": "full", "length": 512}
    out = np.zeros((len(idx), ds.x_noisy.shape[1]))
    for s in range(0, len(idx), batch):
        j = idx[s:s + batch]
        x = ds.tensor(j).to(device)
        pk = [ds.rpeaks[int(g)] for g in j] if spec["mode"] == "beat" else None
        win, meta = cut_windows(x, spec, pk)
        y = join_windows(model.denoise(win), meta, x)
        out[s:s + len(j)] = y.squeeze(1).cpu().numpy().astype(np.float64)
        if progress and s % progress == 0:
            print(f"  복원 {s}/{len(idx)}", flush=True)
    return out


@torch.no_grad()
def component_bank(model, ds, device, idx, batch=100):
    """(n, K, 3600) 성분과 (n, R, 3600) 참조."""
    K, pad = model.n_encoders, model.pad_each
    comps, refs = [], []
    for s in range(0, len(idx), batch):
        j = idx[s:s + batch]
        x = meae.pad(ds.tensor(j).to(device), pad)
        c = torch.stack([meae.crop(model.component(x, k), pad).squeeze(1)
                         for k in range(K)], 1)
        r = torch.stack([ds.ref_tensor(k, j).to(device).squeeze(1)
                         for k in REF_KEYS], 1)
        comps.append(c.cpu().numpy().astype(np.float64))
        refs.append(r.cpu().numpy().astype(np.float64))
    return np.concatenate(comps), np.concatenate(refs)


@torch.no_grad()
def reconstruct(model, ds, device, idx, batch=100):
    """(n, 3600) 입력과 마스킹 없는 재구성."""
    pad = model.pad_each
    rec = []
    for s in range(0, len(idx), batch):
        j = idx[s:s + batch]
        x = meae.pad(ds.tensor(j).to(device), pad)
        rec.append(meae.crop(model(x)[0], pad).squeeze(1).cpu().numpy().astype(np.float64))
    return ds.x_noisy[idx].astype(np.float64), np.concatenate(rec)


# ---------------------------------------------------------------- 표준화와 지표
def _center(a):
    return a - a.mean(-1, keepdims=True)


def znorm(a):
    """분절 내 표준화 — 마지막 축 기준 평균 0, 표준편차 1. 상수 신호는 0으로 둔다."""
    s = a.std(-1, keepdims=True)
    return np.where(s > 0, _center(a) / np.maximum(s, 1e-12), 0.0)


def pearson(comps, refs):
    """[S4-01] 1단계 — **분절 내** Pearson 상관. (n, K, R) 부호 있는 ρ.

    평균·표준편차는 해당 분절 안에서만 구한다. 분절을 이어붙여 일괄 계산하지 않는다 —
    분절마다 다른 잡음 구간이 주입되었고, 진폭 큰 분절이 결과를 지배하기 때문이다.
    상관 관련 코드는 전부 이 함수를 거친다.
    """
    zc, zr = znorm(comps), znorm(refs)
    return np.einsum("nkt,nrt->nkr", zc, zr) / zc.shape[-1]


def aggregate(rho):
    """[S4-01] 2단계 — 분절 간 집계. (ρ̄, σ, 양수비율) 각 (K, R).

      ρ̄ = mean_s |ρ|,  σ = std_s |ρ| (ddof=1)

    절댓값을 쓰는 이유: 인코딩과 디코더 가중치가 동시에 부호 반전되어도 재구성이 불변하므로
    성분이 참조와 반대 위상으로 수렴할 수 있다. 부호 반전은 무관이 아니라 반대 위상의 일치다.
    원 부호의 분포는 양수 비율로 따로 기록한다.
    """
    a = np.abs(rho)
    return a.mean(0), a.std(0, ddof=1), (rho > 0).mean(0)


def rmse_norm_matrix(comps, refs):
    """[S4-02] 정규화 RMSE. (n, K, R) 분절별 `sqrt(mean_i (ã−r̃)²)`.

    표준화하는 이유: 성분 진폭은 비선형 디코더의 임의 출력이고 참조 4종의 RMS도 서로 다르다
    (clean 0.204, 잡음 0.113~0.119 mV). 표준화하지 않으면 크기 차이가 값을 지배한다.
    부호 정렬을 하지 않으므로 반대 위상은 값이 커진다.
    """
    zc, zr = znorm(comps), znorm(refs)
    out = np.empty((zc.shape[0], zc.shape[1], zr.shape[1]))
    for k in range(zc.shape[1]):          # (n,K,R,T) 를 한 번에 만들면 수백 MB가 된다
        d = zc[:, k, None, :] - zr
        out[:, k, :] = np.sqrt((d ** 2).mean(-1))
    return out


def mad_matrix(comps, refs, with_argmax=False):
    """[S4-03] 국소 최대 편차. (n, K, R) 분절별 `max_i |ã−r̃|`. 단위는 표준편차.

    with_argmax=True 이면 최대 편차가 **어느 표본에서** 났는지도 함께 준다.
    """
    zc, zr = znorm(comps), znorm(refs)
    out = np.empty((zc.shape[0], zc.shape[1], zr.shape[1]))
    pos = np.empty(out.shape, dtype=np.int64) if with_argmax else None
    for k in range(zc.shape[1]):
        d = np.abs(zc[:, k, None, :] - zr)
        out[:, k, :] = d.max(-1)
        if with_argmax:
            pos[:, k, :] = d.argmax(-1)
    return (out, pos) if with_argmax else out


# ---------------------------------------------------------------- 표 렌더링
def top_idx(m, n=2, largest=True):
    """(K, R) 에서 **열별** 상·하위 n개 인덱스. 반환 (n, R), 1위가 첫 행.

    표시는 열 기준이다 — 한 참조를 어느 인코더가 가장 잘 잡는지를 열 안에서 비교한다.
    """
    return np.argsort(-m if largest else m, axis=0)[:n]


def mark(m, n=2, largest=True):
    """(K, R) → 열별 상·하위 n개 순위 배열. 0=미표시, 1=1위, 2=2위."""
    out = np.zeros(m.shape, dtype=int)
    for rank, ks in enumerate(top_idx(m, n, largest)):
        out[ks, np.arange(m.shape[1])] = rank + 1
    return out


def render(m, sd, flag, fmt="{:.3f}"):
    """CSV·콘솔용 표기. [1] = 열 1위, [2] = 열 2위 (굵은 글씨를 못 쓰므로)."""
    tag = {0: "", 1: " [1]", 2: " [2]"}
    return [[f"{fmt.format(m[k, r])}±{fmt.format(sd[k, r])}{tag[flag[k, r]]}"
             for r in range(m.shape[1])] for k in range(m.shape[0])]


def enc_names(K):
    return [enc_label(k) for k in range(K)]
