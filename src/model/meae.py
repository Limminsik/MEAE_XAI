"""MEAE — 선행 구조를 감싼 우리 인터페이스 (RESEARCH_DESIGN.md §5).

구조 자체는 `_vendor_meae.py` 를 쓴다. **version5 에서 그 파일은 수정본이다** —
블록별 dilation 을 받도록 두 곳을 고쳤고, 근거와 변경 지점은 그 파일 머리말에 있다. 이 파일이 더하는 것은
설계 §5가 요구하는 네 가지뿐이다.

  encode / decode / forward   — 선행 그대로 위임
  component(x, k)             — k번째 인코딩만 남기고 나머지를 0으로 치환한 재구성
  masked_reconstruct(x, idx)  — idx의 인코딩만 0으로 치환한 재구성

§0 원칙 1: 마스킹 대상은 **인코딩(인코더 출력 텐서)** 이다. 인코더 가중치는 건드리지 않고
순전파도 K개 모두 수행하며, 디코더 입력 직전에 해당 인코딩만 영텐서로 바꾼다.

패딩 — **압축률에 따라 달라진다**
  인코더 블록 수 D = len(channels) 이고 MaxPool 2배가 D번 걸린다 → 입력 길이가
  2^D 의 배수여야 한다. `fit_pad(3600, D)` 가 필요한 대칭 패딩을 계산한다.

    D=8 (256배)  3600 -> 3840, 양쪽 120   ← 선행 기본값
    D=6 ( 64배)  3600 -> 3648, 양쪽  24
    D=5 ( 32배)  3600 -> 3616, 양쪽   8

  선행 후속 저장소도 6000 -> 6144(=256x24)를 같은 방식으로 처리했다.
  **모든 상관·지표는 crop()으로 중앙 3600을 잘라낸 뒤 계산한다.** 0 구간이 상관을 희석한다.
  R-피크 인덱스는 크롭 좌표(0~3599) 기준을 그대로 쓴다.
"""
from typing import List, Sequence, Union

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ._vendor_meae import ConvolutionalAutoencoder

DOWNSAMPLE_FACTOR = 256          # 선행 기본값 — 인코더 블록 8개 x MaxPool(2)


def fit_pad(length: int, depth: int):
    """(패딩 후 길이, 한쪽 패딩) — 길이를 2^depth 의 배수로 올린다.

    압축률을 바꾸면 이 값이 함께 바뀐다. 양쪽 대칭이어야 하므로 모자란 양이 홀수면
    한 칸 더 올려 짝수로 만든다.
    """
    f = 2 ** depth
    total = -(-length // f) * f - length          # 올림 후 차이
    if total % 2:
        total += f
    return length + total, total // 2


def receptive_field(dilations, fs=360):
    """인코더 수용영역 (입력 표본, 초). 블록 = conv7 - conv7 - maxpool2.

    층을 빼면 수용영역이 줄어든다. bw 의 느린 출렁임을 보려면 그만큼이 필요하므로
    dilation 으로 보전하고, 실제 값을 학습 로그에 남긴다.
    """
    rf, stride = 1, 1
    for d in dilations:
        rf += 2 * (7 - 1) * d * stride + stride
        stride *= 2
    return rf, rf / fs


def enc_label(k: int) -> str:
    """표시용 인코더 이름. **내부 인덱스는 0부터, 사람이 보는 이름은 1부터**다.
    그림·표·콘솔 출력에서 0-based를 쓰면 K와 마지막 번호가 어긋나 혼동이 생긴다."""
    return f"enc{k + 1}"


def pad(x: Tensor, pad_each: int) -> Tensor:
    """(B, C, L) → (B, C, L+2·pad_each). 대칭 제로 패딩."""
    return F.pad(x, (pad_each, pad_each), mode="constant", value=0.0) if pad_each else x


def crop(x: Tensor, pad_each: int) -> Tensor:
    """패딩 구간 제거. 지표 계산 전에 반드시 거친다."""
    return x[..., pad_each:x.shape[-1] - pad_each] if pad_each else x


class MEAE(nn.Module):
    def __init__(self, n_encoders: int, input_length: int = 3840,
                 channels: Sequence[int] = (32, 32, 64, 64, 128, 128, 256, 256),
                 hidden: int = 64, norm_type: str = "group_norm",
                 use_weight_norm: bool = True, input_channels: int = 1,
                 pad_each: int = 120, dilations: Sequence[int] = None,
                 skip_levels: Sequence[int] = None, skip_weight: float = 0.0):
        super().__init__()
        depth = len(channels)
        factor = 2 ** depth
        if input_length % factor:
            raise ValueError(
                f"input_length={input_length} 는 {factor}의 배수가 아니다. "
                f"인코더 깊이 {depth} 때문에 재구성 길이가 어긋난다.")
        if dilations is not None and len(dilations) != depth:
            raise ValueError(f"dilations 는 {depth}개여야 한다 (블록 수).")
        for c in (*channels, hidden):
            if c % n_encoders:
                raise ValueError(
                    f"채널 {c} 가 n_encoders={n_encoders} 로 나눠지지 않는다. "
                    f"디코더 GroupNorm(num_groups=K)이 실패한다. K는 {{4, 8}}만 쓴다.")
        self.n_encoders = n_encoders
        self.pad_each = pad_each
        self.input_length = input_length
        self.depth = depth
        self.downsample = factor
        self.dilations = list(dilations) if dilations else [1] * depth
        self.skip_levels = list(skip_levels) if skip_levels else []
        self.skip_weight = float(skip_weight)
        if any(l >= depth - 1 for l in self.skip_levels):
            raise ValueError(f"skip_levels 는 0..{depth - 2} 범위여야 한다 "
                             f"(가장 깊은 블록은 인코딩 자리라 잔차로 쓰지 않는다).")
        self.net = ConvolutionalAutoencoder(
            input_channels=input_channels, input_length=input_length,
            channels=list(channels), hidden=hidden, num_encoders=n_encoders,
            norm_type=norm_type, use_weight_norm=use_weight_norm,
            dilations=self.dilations, skip_levels=self.skip_levels,
            skip_weight=self.skip_weight)

    # ---- 선행 그대로 위임 -------------------------------------------------
    def encode(self, x: Tensor) -> List[Tensor]:
        """(B,1,L) → 길이 K 리스트, 각 (B, hidden//K, L//256)."""
        return self.net.encode(x)

    def encode_all(self, x: Tensor):
        """(인코딩 K개, 인코더별 잔차). 잔차가 꺼져 있으면 두 번째는 빈 목록들이다."""
        return self.net.encode_all(x)

    def decode(self, zs: List[Tensor], zeros_train: bool = False,
               skips=None) -> Tensor:
        """인코딩 K개를 채널 축으로 결합해 단일 공유 디코더에 통과.

        `skips` 를 주면 잔차도 함께 들어간다. **성분을 뽑을 때는 인코딩과 잔차를 같은
        규칙으로 마스킹해야 한다** — 잔차만 남기면 그 경로로 입력 전체가 새어 들어와
        마스킹이 무의미해진다.
        """
        return self.net.decode(zs, zeros_train, skips=skips)

    def forward(self, x: Tensor):
        return self.net(x)

    # ---- 설계 §5가 요구하는 마스킹 ---------------------------------------
    def _mask(self, zs: List[Tensor], keep: List[int]) -> List[Tensor]:
        """keep에 없는 인코딩을 영텐서로 치환. 인코더 가중치는 불변."""
        return [z if i in keep else torch.zeros_like(z) for i, z in enumerate(zs)]

    def _mask_skips(self, skips, keep: List[int]):
        """잔차도 **같은 규칙으로** 0으로 만든다. 이걸 빠뜨리면 마스킹이 무의미해진다."""
        if not skips:
            return skips
        return [sk if i in keep else [torch.zeros_like(t) for t in sk]
                for i, sk in enumerate(skips)]

    def _decode_masked(self, x: Tensor, keep: List[int], use_skip: bool = True):
        zs, skips = self.encode_all(x)
        if not use_skip:                      # 감시용 — 잔차를 끈 채로 같은 성분을 뽑는다
            skips = self._mask_skips(skips, [])
        return self.decode(self._mask(zs, keep), skips=self._mask_skips(skips, keep))

    def component(self, x: Tensor, k: int, use_skip: bool = True) -> Tensor:
        """성분 x̂_k — k번째 인코딩(과 그 인코더의 잔차)만 남긴 재구성."""
        return self._decode_masked(x, [k], use_skip)

    def masked_reconstruct(self, x: Tensor, mask_idx: Union[int, Sequence[int]],
                           use_skip: bool = True) -> Tensor:
        """mask_idx의 인코딩(과 잔차)만 0으로 치환한 재구성 (S5 복원)."""
        idx = {mask_idx} if isinstance(mask_idx, int) else set(mask_idx)
        keep = [i for i in range(self.n_encoders) if i not in idx]
        return self._decode_masked(x, keep, use_skip)

    def components(self, x: Tensor) -> List[Tensor]:
        """K개 성분을 한 번의 encode로 모두 얻는다 (S4에서 분절마다 K번 호출을 피함)."""
        zs, skips = self.encode_all(x)
        if self.skip_levels:
            return [self.decode(self._mask(zs, [k]),
                                skips=self._mask_skips(skips, [k]))
                    for k in range(self.n_encoders)]
        return [self.decode(self._mask(zs, [k])) for k in range(self.n_encoders)]

    def zero_encoding(self, batch: int, length: int, device) -> List[Tensor]:
        """zero reconstruction 손실용 전영 인코딩."""
        c = self.net.hidden // self.n_encoders
        return [torch.zeros(batch, c, length, device=device) for _ in range(self.n_encoders)]


def build(cfg, n_encoders: int) -> MEAE:
    """configs/default.yaml 에서 바로 조립. 하드코딩 금지 (§13)."""
    m, d = cfg["model"], cfg["data"]
    # 패딩은 채널 깊이에서 유도한다 — 압축률을 바꾸면 자동으로 따라온다.
    length, pad_each = fit_pad(d["fs"] * d["seg_sec"], len(m["channels"]))
    return MEAE(n_encoders=n_encoders, input_length=length,
                channels=m["channels"], hidden=m["hidden"],
                norm_type=m["norm_type"], use_weight_norm=m["use_weight_norm"],
                input_channels=m["input_channels"], pad_each=pad_each,
                dilations=m.get("dilations"), skip_levels=m.get("skip_levels"),
                skip_weight=m.get("skip_weight", 0.0))
