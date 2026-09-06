"""손실 함수.

**이 갈래가 쓰는 것은 아래쪽 `SupervisedLoss` 하나다.** 정의와 근거는 그 앞의
주석 블록에 있다. 인코더마다 참조를 하나씩 배정하고, 성분을 그 참조에 직접 맞춘다.

이 파일 위쪽의 `MEAELoss` 는 자기지도 갈래(version1~3)에서 쓰던 4항 손실이다.
version5 에서는 **호출되지 않는다** — 참조를 직접 주므로 "어떻게 나눌지 모를 때 돕는"
정칙화가 필요 없고, 배정 목표와 충돌할 수 있다. 두 갈래의 손실을 나란히 두어
무엇을 껐는지 코드로 확인할 수 있게 남긴다. `configs/default.yaml` 의 λ_m·λ_o·λ_z 가
전부 0.0 인 것이 그 기록이다.
"""
from typing import Dict, List

import torch
from torch import Tensor, nn

from ._vendor_separation_loss import WeightSeparationLoss, WeightSeparationLossAlternative

SEP_IMPLS = {"alternative": WeightSeparationLossAlternative, "blockwise": WeightSeparationLoss}


# ================================================================
# [version5 에서 사용하지 않음] 자기지도 4항 손실 — version1~3 갈래의 손실이다.
#
# 선행 4항을 차용하되 **재구성만 BCE → MSE 로 교체**했다. 선행은 신호를 [0,1] 로
# min-max 정규화한 뒤 BCEWithLogits 를 썼지만, 우리는 정규화하지 않고 ECG 는 음수를 갖는다.
#
#   재구성          MSE(x̂, x) + γ·MSE(Δx̂, Δx)      선행 BCE 를 교체 + 차분항 추가
#   sparse mixing   디코더 가중치 비대각 L1          선행 sep_lr = 1e-3 (mesa_ecg_bss)
#   zero recon      전영 인코딩 → 출력 0 (MSE)      선행 zero_lr = 1e-2
#   인코딩 L2       각 z 의 평균 제곱               선행 코드 하드코딩 1e-2
#
# sparse mixing 구현은 `WeightSeparationLossAlternative` 를 쓴다 — 선행의 ECG 실험
# 설정(`mesa_ecg_bss`)이 쓴 구현이기 때문이다.
#
# version5 는 참조를 직접 주므로 이 세 정칙화 항을 모두 0.0 으로 끈다.
# ================================================================
class MEAELoss(nn.Module):
    def __init__(self, n_encoders: int, lambda_mixing: float, lambda_zero_recon: float,
                 lambda_z_l2: float, sep_impl: str = "alternative", sep_norm: str = "L1",
                 gamma_diff: float = 0.0):
        super().__init__()
        self.gamma_diff = gamma_diff
        if sep_impl not in SEP_IMPLS:
            raise ValueError(f"sep_impl={sep_impl!r} 는 {sorted(SEP_IMPLS)} 중 하나여야 한다.")
        self.recon = nn.MSELoss()
        self.separation = SEP_IMPLS[sep_impl](n_encoders, sep_norm)
        self.lambda_mixing = lambda_mixing
        self.lambda_zero_recon = lambda_zero_recon
        self.lambda_z_l2 = lambda_z_l2

    def forward(self, model, x: Tensor, x_pred: Tensor, zs: List[Tensor]) -> Dict[str, Tensor]:
        """항별 값을 그대로 돌려준다 — 에폭 로그에 4항을 각각 남기기 위함."""
        recon = self.recon(x_pred, x)
        if self.gamma_diff:                      # 1차 차분 = 고역강조. 저역통과 감쇠를 막는다
            recon = recon + self.gamma_diff * self.recon(torch.diff(x_pred, dim=-1),
                                                         torch.diff(x, dim=-1))
        z_l2 = sum(torch.mean(z ** 2) for z in zs)
        # 출력층(model.net.output)은 제외하고 디코더 가중치만 대상 — 선행과 동일
        mixing = self.separation(model.net.decoder)

        zeros = model.zero_encoding(1, zs[0].shape[-1], x.device)
        x_zero = model.decode(zeros, zeros_train=True)
        zero_recon = self.recon(x_zero, torch.zeros_like(x_zero))
        if self.gamma_diff:                      # 재구성과 같은 정의를 쓴다
            zero_recon = zero_recon + self.gamma_diff * self.recon(
                torch.diff(x_zero, dim=-1), torch.zeros_like(torch.diff(x_zero, dim=-1)))

        total = (recon
                 + self.lambda_z_l2 * z_l2
                 + self.lambda_mixing * mixing
                 + self.lambda_zero_recon * zero_recon)
        return {"total": total, "recon": recon, "mixing": mixing,
                "zero_recon": zero_recon, "z_l2": z_l2}


def build(cfg, n_encoders: int) -> MEAELoss:
    lo = cfg["loss"]
    if lo["recon"] != "mse":
        raise ValueError("재구성 손실은 MSE 로 고정된다 (loss.recon: mse).")
    return MEAELoss(n_encoders=n_encoders, lambda_mixing=lo["lambda_mixing"],
                    lambda_zero_recon=lo["lambda_zero_recon"],
                    lambda_z_l2=lo["lambda_z_l2"], sep_impl=lo["sep_impl"],
                    sep_norm=lo["sep_norm"], gamma_diff=lo.get("gamma_diff", 0.0))


# ================================================================
# 지도 손실
#
#   L      = (1/L)‖x̂ − x_noisy‖²  +  λ_sup · L_sup
#
#   L_sup  = (1/K) Σ_k (1/σ_k²) [ (1/L)‖ŝ_k − r_k‖²
#                      + γ₁ · (1/(L−1))‖Δŝ_k  − Δr_k‖²
#                      + γ₂ · (1/(L−2))‖Δ²ŝ_k − Δ²r_k‖²
#                      + β  · (1/B)‖ |F ŝ_k| − |F r_k| ‖² ]      B = L//2+1
#
# **소스별 정규화** σ_k — 훈련셋 전체에서 잰 참조 r_k 의 표준편차(소스당 스칼라 하나).
# `sup_normalize: source_std` 일 때만 나눈다. 끄면 σ_k = 1 이라 기존과 완전히 같다.
#
# 참조들의 진폭이 다르다 — 훈련셋에서 σ²(clean) 이 잡음 3종의 약 4.2배다. MSE 는 절대
# 오차를 보므로 잡음을 틀려도 손실이 거의 늘지 않고, 자원을 clean 에 몰아주는 것이
# 최적해가 된다. σ_k² 로 나누면 "bw 를 20% 틀리는 것"과 "clean 을 20% 틀리는 것"이 같은
# 벌점이 된다. 항 전체(파형·차분·주파수)를 같은 σ_k² 로 나눈다 — 한 소스 안에서 γ·β 의
# 상대 비중은 설계 그대로 유지된다.
#
# F 는 실수 FFT(`torch.fft.rfft`)이고 **norm="ortho"** 다. 그래야 Parseval 로
# Σ|X|² = Σ|x|² 가 성립해 빈 평균이 표본 평균과 같은 눈금에 놓인다 — β 를 γ 와 같은
# 방식으로 읽을 수 있다. 크기만 비교하므로 위상은 보지 않는다 (파형 항이 이미 본다).
# FFT 는 미분 가능해 기울기가 인코더까지 그대로 흐른다.
#
#   (Δx)[t]  = x[t] − x[t−1]                         길이 L−1
#   (Δ²x)[t] = (Δx)[t] − (Δx)[t−1] = x[t] − 2x[t−1] + x[t−2]    길이 L−2
#
# 배정: r1 = x_clean · r2 = bw · r3 = ma · r4 = em (`loss.supervise`).
#
# ŝ_k 는 기존 `model.component(x, k)` 경로 — 다른 인코딩을 0으로 두고 디코드하는,
# 04에서 쓰는 것과 **같은** 마스킹 디코드다. 학습과 평가가 같은 경로를 쓰게 하려는
# 것이다. 크롭도 기존과 같이 중앙 3600.
#
# **차분항을 두는 이유**: 파형 일치만으로는 참조의 국소 고주파(자글자글한 변화)가
# 학습되지 않는다. 그 성분은 제곱오차에 거의 기여하지 않아 평탄한 해가 이득이기
# 때문이다. 변화량을 따로 맞추면 그 대역이 손실에 잡힌다.
#
# 노름은 표기상 ‖·‖² 이지만 구현은 **각 항의 길이로 나눈 평균**이다 (파형은 L,
# 차분은 L−1). 길이가 다른 둘을 같은 눈금에 놓아야 γ가 상대 비중 그대로가 된다.
# k 에 대해서도 합이 아니라 평균(1/K)이라 K 를 바꿔도 λ_sup 의 뜻이 유지된다.
# ================================================================
def _fft_mag(x: Tensor) -> Tensor:
    """실수 FFT 크기. (..., L) -> (..., L//2+1). norm='ortho' 로 눈금을 맞춘다."""
    return torch.fft.rfft(x, norm="ortho").abs()


def _diff(x: Tensor, n: int = 1) -> Tensor:
    """n차 차분. 마지막 축 기준, 길이는 L−n 이 된다.

    n=2 면 x[t] − 2x[t−1] + x[t−2] 이고 `torch.diff(x, n=2, dim=-1)` 과 같다.
    """
    return torch.diff(x, n=n, dim=-1)


class SupervisedLoss(nn.Module):
    """[s4_상한 사본에서만 다름] `active` 로 **일부 인코더만** 지도할 수 있다.

    본 트리(`version5_supervised/src/model/losses.py`)에는 이 인자가 없다. 상한 측정
    실험에만 필요해 사본에서만 더했다 — 확정 모델의 코드 경로는 건드리지 않는다.

    `active=None` 이면 전 인코더를 지도한다 → **본 트리와 완전히 같은 계산**이다.
    일부만 주면 나머지 인코더에는 지도 항이 걸리지 않고 재구성에만 기여한다.

    **평균은 지도하는 인코더 수로 나눈다.** λ_sup 이 "지도 항 하나의 비중"이라는 뜻을
    유지하기 위해서다. 그래서 한 개만 지도하면 그 항의 실효 가중치가 전체 지도(1/K)보다
    K 배 커진다 — 상한을 재는 것이 목적이라 의도한 바이지만, 결과를 읽을 때
    **"경쟁 제거"와 "가중치 증가"가 함께 작용했다**는 점을 같이 적어야 한다.
    """

    def __init__(self, n_encoders: int, lambda_sup: float, pad_each: int,
                 gamma_sup: float = 0.0, gamma2_sup: float = 0.0,
                 beta_sup: float = 0.0, sigmas=None, active=None):
        super().__init__()
        self.recon = nn.MSELoss()
        self.n_encoders = n_encoders
        self.lambda_sup = lambda_sup
        self.gamma_sup = gamma_sup        # γ₁ — 1차 차분
        self.gamma2_sup = gamma2_sup      # γ₂ — 2차 차분
        self.beta_sup = beta_sup          # β  — |FFT| 크기
        self.pad_each = pad_each
        self.active = list(range(n_encoders)) if active is None else sorted(active)
        if not self.active or any(not 0 <= k < n_encoders for k in self.active):
            raise ValueError(f"active={active} 는 0..{n_encoders - 1} 안의 "
                             "비어 있지 않은 목록이어야 한다.")
        # σ_k² 의 역수를 미리 담아 둔다. None 이면 전부 1 (정규화 없음)
        w = [1.0] * n_encoders if sigmas is None else [1.0 / (s ** 2) for s in sigmas]
        self.register_buffer("inv_var", torch.tensor(w, dtype=torch.float32))

    def forward(self, model, x: Tensor, x_pred: Tensor, zs: List[Tensor],
                refs: Tensor) -> Dict[str, Tensor]:
        """refs: (B, K, L) — 배정 순서대로 쌓은 참조. L은 크롭 뒤 길이와 같다."""
        from . import meae
        recon = self.recon(x_pred, x)
        z = recon.new_zeros(())
        per = {k: z for k in range(self.n_encoders)}
        acc = {"sup_wave": z, "sup_diff": z, "sup_diff2": z, "sup_freq": z}
        for k in self.active:
            s_k = meae.crop(model.component(x, k), self.pad_each).squeeze(1)
            r_k = refs[:, k]
            wave = self.recon(s_k, r_k)                          # (1/L)‖ŝ−r‖²
            d1 = self.recon(_diff(s_k, 1), _diff(r_k, 1))        # (1/(L−1))‖Δŝ−Δr‖²
            d2 = self.recon(_diff(s_k, 2), _diff(r_k, 2))        # (1/(L−2))‖Δ²ŝ−Δ²r‖²
            fq = self.recon(_fft_mag(s_k), _fft_mag(r_k))        # (1/B)‖|Fŝ|−|Fr|‖²
            acc["sup_wave"] = acc["sup_wave"] + wave
            acc["sup_diff"] = acc["sup_diff"] + d1
            acc["sup_diff2"] = acc["sup_diff2"] + d2
            acc["sup_freq"] = acc["sup_freq"] + fq
            per[k] = self.inv_var[k] * (wave + self.gamma_sup * d1
                                        + self.gamma2_sup * d2 + self.beta_sup * fq)
        n = len(self.active)
        sup = sum(per[k] for k in self.active) / n            # 지도하는 k 에 대해 평균
        out = {"total": recon + self.lambda_sup * sup, "recon": recon, "sup": sup,
               **{key: v / n for key, v in acc.items()}}
        # 지도하지 않은 인코더도 0 으로 남긴다 — history.csv 의 열이 런마다 흔들리지 않게
        for k in range(self.n_encoders):
            out[f"sup_e{k + 1}"] = per[k]
        return out


def source_sigmas(train_set, keys):
    """훈련셋 전체에서 소스별 표준편차. 소스당 스칼라 하나, 학습 시작 전 1회."""
    return [float(train_set.refs[k].std()) + 1e-8 for k in keys]


def build_supervised(cfg, n_encoders: int, pad_each: int, sigmas=None) -> SupervisedLoss:
    lo = cfg["loss"]
    if lo["recon"] != "mse":
        raise ValueError("재구성 손실은 MSE 로 고정된다 (loss.recon: mse).")
    n_ref = len(lo["supervise"])
    if n_ref != n_encoders:
        raise ValueError(f"loss.supervise 가 {n_ref}개인데 인코더는 {n_encoders}개다. "
                         "지도학습은 인코더마다 참조를 하나씩 배정한다.")
    return SupervisedLoss(n_encoders=n_encoders, lambda_sup=lo["lambda_sup"],
                          pad_each=pad_each, gamma_sup=lo.get("gamma_sup", 0.0),
                          gamma2_sup=lo.get("gamma2_sup", 0.0),
                          beta_sup=lo.get("beta_sup", 0.0), sigmas=sigmas,
                          active=active_encoders(cfg, n_encoders))


def active_encoders(cfg, n_encoders: int):
    """지도 항을 거는 인코더 인덱스(0-based). `loss.supervise_only` 가 비면 전부.

    설정에는 **사람이 보는 1-based 번호**로 적는다 (`[2]` = enc2). 표·그림·콘솔이
    전부 1부터 세므로 거기에 맞춘다. 내부 인덱스만 0-based 다.
    """
    only = cfg["loss"].get("supervise_only")
    if not only:
        return None
    idx = sorted(int(v) - 1 for v in only)
    if any(not 0 <= k < n_encoders for k in idx):
        raise ValueError(f"loss.supervise_only={only} 는 1..{n_encoders} 안이어야 한다.")
    return idx
