"""MEAE 손실 4항 (RESEARCH_DESIGN.md §5).

선행 4항을 그대로 차용하되 **재구성만 BCE → MSE로 교체**한다. 선행은 신호를 [0,1]로
min-max 정규화한 뒤 BCEWithLogits를 썼지만, 우리는 정규화를 하지 않고 ECG는 음수를 가진다.
zero reconstruction도 같은 손실 함수를 재사용하므로 함께 MSE가 된다 (설계 §5 요구).

| 항 | 값 | 출처 |
|---|---|---|
| 재구성 | `MSE(x̂, x_noisy)` | 선행 BCE를 교체 |
| sparse mixing | 디코더 가중치 비대각 L1 | 선행 sep_lr = 1e-3 (mesa_ecg_bss) |
| zero reconstruction | 전영 인코딩 → 출력 0 (MSE) | 선행 zero_lr = 1e-2 |
| 인코딩 L2 | 각 z의 평균 제곱 | 선행 코드 하드코딩 1e-2 |

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
                 lambda_z_l2: float, sep_impl: str = "alternative", sep_norm: str = "L1"):
        super().__init__()
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
        z_l2 = sum(torch.mean(z ** 2) for z in zs)
        # 출력층(model.net.output)은 제외하고 디코더 가중치만 대상 — 선행과 동일
        mixing = self.separation(model.net.decoder)

        zeros = model.zero_encoding(1, zs[0].shape[-1], x.device)
        x_zero = model.decode(zeros, zeros_train=True)
        zero_recon = self.recon(x_zero, torch.zeros_like(x_zero))

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
                    sep_norm=lo["sep_norm"])
