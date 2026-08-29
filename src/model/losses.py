"""MEAE 손실 4항 (RESEARCH_DESIGN.md §5).

선행 4항을 그대로 차용하되 **재구성만 BCE → MSE로 교체**한다. 선행은 신호를 [0,1]로
min-max 정규화한 뒤 BCEWithLogits를 썼지만, 우리는 정규화를 하지 않고 ECG는 음수를 가진다.
zero reconstruction도 같은 손실 함수를 재사용하므로 함께 MSE가 된다 (설계 §5 요구).

| 항 | 값 | 출처 |
|---|---|---|
| 재구성 | `MSE(x̂, x) + γ·MSE(Δx̂, Δx)` (Δ = 1차 차분) | 선행 BCE를 교체 + 차분항 추가 |
| sparse mixing | 디코더 가중치 비대각 L1 | 선행 sep_lr = 1e-3 (mesa_ecg_bss) |
| zero reconstruction | 전영 인코딩 → 출력 0 (MSE) | 선행 zero_lr = 1e-2 |
| 인코딩 L2 | 각 z의 평균 제곱 | 선행 코드 하드코딩 1e-2 |

**차분항(γ)**: 1차 차분에 대한 MSE를 더한다. 재구성이 저역통과형으로 감쇠하는 것을 막기 위함이다.
차분은 주파수에 비례하는 이득을 갖는 고역강조 연산자이므로, 이 항은 고주파 오차에 더 큰 벌점을
준다. 실측에서 재구성 스펙트럼이 입력보다 1.7–1.8배 가팔랐고 15 Hz부터 단조 하강했다.
γ=0이면 기존 동작과 완전히 같다.

sparse mixing 구현 선택: `separation_loss.py`에는 두 가지가 있고 논문마다 다른 쪽을 쓴다.
우리는 **`WeightSeparationLossAlternative`** 를 쓴다 — ECG 실험 설정인 `mesa_ecg_bss`가
쓴 구현이기 때문이다. 자세한 대조는 results/data_notes.md §8 참조.
"""
from typing import Dict, List

import torch
from torch import Tensor, nn

from ._vendor_separation_loss import WeightSeparationLoss, WeightSeparationLossAlternative

SEP_IMPLS = {"alternative": WeightSeparationLossAlternative, "blockwise": WeightSeparationLoss}


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
        """항별 값을 그대로 돌려준다 — 에폭 로그에 4항을 각각 남기기 위함 (§6)."""
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
        raise ValueError("설계 §5는 재구성 손실을 MSE로 고정한다.")
    return MEAELoss(n_encoders=n_encoders, lambda_mixing=lo["lambda_mixing"],
                    lambda_zero_recon=lo["lambda_zero_recon"],
                    lambda_z_l2=lo["lambda_z_l2"], sep_impl=lo["sep_impl"],
                    sep_norm=lo["sep_norm"], gamma_diff=lo.get("gamma_diff", 0.0))


# ================================================================
# [version4 전용] 지도 손실
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
    def __init__(self, n_encoders: int, lambda_sup: float, pad_each: int,
                 gamma_sup: float = 0.0, gamma2_sup: float = 0.0,
                 beta_sup: float = 0.0, sigmas=None):
        super().__init__()
        self.recon = nn.MSELoss()
        self.n_encoders = n_encoders
        self.lambda_sup = lambda_sup
        self.gamma_sup = gamma_sup        # γ₁ — 1차 차분
        self.gamma2_sup = gamma2_sup      # γ₂ — 2차 차분
        self.beta_sup = beta_sup          # β  — |FFT| 크기
        self.pad_each = pad_each
        # σ_k² 의 역수를 미리 담아 둔다. None 이면 전부 1 (정규화 없음)
        w = [1.0] * n_encoders if sigmas is None else [1.0 / (s ** 2) for s in sigmas]
        self.register_buffer("inv_var", torch.tensor(w, dtype=torch.float32))

    def forward(self, model, x: Tensor, x_pred: Tensor, zs: List[Tensor],
                refs: Tensor) -> Dict[str, Tensor]:
        """refs: (B, K, L) — 배정 순서대로 쌓은 참조. L은 크롭 뒤 길이와 같다."""
        from . import meae
        recon = self.recon(x_pred, x)
        per, per_wave, per_d1, per_d2, per_f = [], [], [], [], []
        for k in range(self.n_encoders):
            s_k = meae.crop(model.component(x, k), self.pad_each).squeeze(1)
            r_k = refs[:, k]
            wave = self.recon(s_k, r_k)                          # (1/L)‖ŝ−r‖²
            d1 = self.recon(_diff(s_k, 1), _diff(r_k, 1))        # (1/(L−1))‖Δŝ−Δr‖²
            d2 = self.recon(_diff(s_k, 2), _diff(r_k, 2))        # (1/(L−2))‖Δ²ŝ−Δ²r‖²
            fq = self.recon(_fft_mag(s_k), _fft_mag(r_k))        # (1/B)‖|Fŝ|−|Fr|‖²
            per_wave.append(wave)
            per_d1.append(d1)
            per_d2.append(d2)
            per_f.append(fq)
            per.append(self.inv_var[k] * (wave + self.gamma_sup * d1
                                         + self.gamma2_sup * d2 + self.beta_sup * fq))
        sup = sum(per) / self.n_encoders                         # k 에 대해 평균
        out = {"total": recon + self.lambda_sup * sup, "recon": recon, "sup": sup,
               "sup_wave": sum(per_wave) / self.n_encoders,
               "sup_diff": sum(per_d1) / self.n_encoders,
               "sup_diff2": sum(per_d2) / self.n_encoders,
               "sup_freq": sum(per_f) / self.n_encoders}
        for k, d in enumerate(per):
            out[f"sup_e{k + 1}"] = d
        return out


def source_sigmas(train_set, keys):
    """훈련셋 전체에서 소스별 표준편차. 소스당 스칼라 하나, 학습 시작 전 1회."""
    return [float(train_set.refs[k].std()) + 1e-8 for k in keys]


def build_supervised(cfg, n_encoders: int, pad_each: int, sigmas=None) -> SupervisedLoss:
    lo = cfg["loss"]
    if lo["recon"] != "mse":
        raise ValueError("설계 §5는 재구성 손실을 MSE로 고정한다.")
    n_ref = len(lo["supervise"])
    if n_ref != n_encoders:
        raise ValueError(f"loss.supervise 가 {n_ref}개인데 인코더는 {n_encoders}개다. "
                         "지도학습은 인코더마다 참조를 하나씩 배정한다.")
    return SupervisedLoss(n_encoders=n_encoders, lambda_sup=lo["lambda_sup"],
                          pad_each=pad_each, gamma_sup=lo.get("gamma_sup", 0.0),
                          gamma2_sup=lo.get("gamma2_sup", 0.0),
                          beta_sup=lo.get("beta_sup", 0.0), sigmas=sigmas)
