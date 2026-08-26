"""MEAE — 선행 구조를 감싼 우리 인터페이스 (RESEARCH_DESIGN.md §5).

구조 자체는 `_vendor_meae.py`(선행 원본, 무수정)를 그대로 쓴다. 이 파일이 더하는 것은
설계 §5가 요구하는 네 가지뿐이다.

  encode / decode / forward   — 선행 그대로 위임
  component(x, k)             — k번째 인코딩만 남기고 나머지를 0으로 치환한 재구성
  masked_reconstruct(x, idx)  — idx의 인코딩만 0으로 치환한 재구성

§0 원칙 1: 마스킹 대상은 **인코딩(인코더 출력 텐서)** 이다. 인코더 가중치는 건드리지 않고
순전파도 K개 모두 수행하며, 디코더 입력 직전에 해당 인코딩만 영텐서로 바꾼다.

패딩 (T3 확정)
  선행 인코더 깊이가 8이라 MaxPool 2배가 8번 걸린다 → 입력 길이가 256의 배수여야 한다.
  3600 → 3840(=256x15)으로 양쪽 120샘플씩 대칭 제로 패딩한다. 선행 후속 저장소도
  6000 → 6144(=256x24)를 같은 방식으로 처리했다.
  **모든 상관·지표는 crop()으로 중앙 3600을 잘라낸 뒤 계산한다.** 0 구간이 상관을 희석한다.
  R-피크 인덱스는 크롭 좌표(0~3599) 기준을 그대로 쓴다.
"""
from typing import List, Sequence, Union

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ._vendor_meae import ConvolutionalAutoencoder

DOWNSAMPLE_FACTOR = 256          # 선행 인코더 블록 8개 x MaxPool(2)


def enc_label(k: int) -> str:
    """표시용 인코더 이름. **내부 인덱스는 0부터, 사람이 보는 이름은 1부터**다.
    그림·표·콘솔 출력에서 0-based를 쓰면 K와 마지막 번호가 어긋나 혼동이 생긴다."""
    return f"enc{k + 1}"


def pad(x: Tensor, pad_each: int) -> Tensor:
    """(B, C, 3600) → (B, C, 3840). 대칭 제로 패딩."""
    return F.pad(x, (pad_each, pad_each), mode="constant", value=0.0) if pad_each else x


def crop(x: Tensor, pad_each: int) -> Tensor:
    """(B, C, 3840) → (B, C, 3600). 패딩 구간 제거. 지표 계산 전에 반드시 거친다."""
    return x[..., pad_each:x.shape[-1] - pad_each] if pad_each else x


class MEAE(nn.Module):
    def __init__(self, n_encoders: int, input_length: int = 3840,
                 channels: Sequence[int] = (32, 32, 64, 64, 128, 128, 256, 256),
                 hidden: int = 64, norm_type: str = "group_norm",
                 use_weight_norm: bool = True, input_channels: int = 1,
                 pad_each: int = 120):
        super().__init__()
        if input_length % DOWNSAMPLE_FACTOR:
            raise ValueError(
                f"input_length={input_length} 는 {DOWNSAMPLE_FACTOR}의 배수가 아니다. "
                f"선행 인코더 깊이 8 때문에 재구성 길이가 어긋난다.")
        for c in (*channels, hidden):
            if c % n_encoders:
                raise ValueError(
                    f"채널 {c} 가 n_encoders={n_encoders} 로 나눠지지 않는다. "
                    f"디코더 GroupNorm(num_groups=K)이 실패한다. K는 {{4, 8}}만 쓴다.")
        self.n_encoders = n_encoders
        self.pad_each = pad_each
        self.input_length = input_length
        self.net = ConvolutionalAutoencoder(
            input_channels=input_channels, input_length=input_length,
            channels=list(channels), hidden=hidden, num_encoders=n_encoders,
            norm_type=norm_type, use_weight_norm=use_weight_norm)

    # ---- 선행 그대로 위임 -------------------------------------------------
    def encode(self, x: Tensor) -> List[Tensor]:
        """(B,1,L) → 길이 K 리스트, 각 (B, hidden//K, L//256)."""
        return self.net.encode(x)

    def decode(self, zs: List[Tensor], zeros_train: bool = False) -> Tensor:
        """인코딩 K개를 채널 축으로 결합해 단일 공유 디코더에 통과."""
        return self.net.decode(zs, zeros_train)

    def forward(self, x: Tensor):
        return self.net(x)

    # ---- 설계 §5가 요구하는 마스킹 ---------------------------------------
    def _mask(self, zs: List[Tensor], keep: List[int]) -> List[Tensor]:
        """keep에 없는 인코딩을 영텐서로 치환. 인코더 가중치는 불변."""
        return [z if i in keep else torch.zeros_like(z) for i, z in enumerate(zs)]

    def component(self, x: Tensor, k: int) -> Tensor:
        """성분 x̂_k — k번째 인코딩만 남긴 재구성."""
        return self.decode(self._mask(self.encode(x), [k]))

    def masked_reconstruct(self, x: Tensor, mask_idx: Union[int, Sequence[int]]) -> Tensor:
        """mask_idx의 인코딩만 0으로 치환한 재구성 (S5 복원)."""
        idx = {mask_idx} if isinstance(mask_idx, int) else set(mask_idx)
        keep = [i for i in range(self.n_encoders) if i not in idx]
        return self.decode(self._mask(self.encode(x), keep))

    def components(self, x: Tensor) -> List[Tensor]:
        """K개 성분을 한 번의 encode로 모두 얻는다 (S4에서 분절마다 K번 호출을 피함)."""
        zs = self.encode(x)
        return [self.decode(self._mask(zs, [k])) for k in range(self.n_encoders)]

    def zero_encoding(self, batch: int, length: int, device) -> List[Tensor]:
        """zero reconstruction 손실용 전영 인코딩."""
        c = self.net.hidden // self.n_encoders
        return [torch.zeros(batch, c, length, device=device) for _ in range(self.n_encoders)]


def build(cfg, n_encoders: int) -> MEAE:
    """configs/default.yaml 에서 바로 조립. 하드코딩 금지 (§13)."""
    m, d = cfg["model"], cfg["data"]
    return MEAE(n_encoders=n_encoders, input_length=d["pad_to"],
                channels=m["channels"], hidden=m["hidden"],
                norm_type=m["norm_type"], use_weight_norm=m["use_weight_norm"],
                input_channels=m["input_channels"], pad_each=d["pad_each"])
