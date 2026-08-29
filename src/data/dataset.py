"""분절 npz를 메모리에 올려 학습에 쓰는 로더 (RESEARCH_DESIGN.md §6).

전체가 float32로 train 5,760 x 3,600 = 83 MB 수준이라 통째로 램에 올린다.
DataLoader 워커를 쓰지 않으므로 워커 시드 문제가 원천적으로 사라지고,
에폭마다 인덱스 셔플만 시드로 통제하면 재현성이 확보된다.

**모델에 들어가는 것은 x_noisy 하나뿐이다** (§0 원칙 3, 자기지도).
clean/bw/ma/em은 val·test에서만 로드하며, 손실이 아니라 채점에만 쓴다 (§0 원칙 4).
"""
import json
import os
from typing import Dict, List, Optional

import numpy as np
import torch

REF_KEYS = ("x_clean", "bw", "ma", "em")


def _paths(proc: str, split: str, records: List[str]) -> List[str]:
    d = os.path.join(proc, "segments", split)
    return sorted(os.path.join(d, f) for f in os.listdir(d)
                  if f.endswith(".npz") and f.split("_")[0] in set(records))


class Segments:
    """한 split 전체를 담는 컨테이너.

    with_refs=False 면 x_noisy만 올린다 (학습셋에는 정답을 아예 올리지 않아
    실수로 손실에 섞이는 사고를 구조적으로 막는다).
    """

    def __init__(self, cfg, split: str, with_refs: bool):
        proc = cfg["paths"]["processed"]
        sp = json.load(open(os.path.join(proc, "split.json"), encoding="utf-8"))
        paths = _paths(proc, split, sp[split])
        self.split = split
        self.paths = paths
        self.n = len(paths)

        seg_len = cfg["data"]["fs"] * cfg["data"]["seg_sec"]
        self.x_noisy = np.empty((self.n, seg_len), dtype=np.float32)
        self.refs: Dict[str, np.ndarray] = (
            {k: np.empty((self.n, seg_len), dtype=np.float32) for k in REF_KEYS}
            if with_refs else {})
        self.rpeaks: List[np.ndarray] = []
        self.meta: List[dict] = []

        for i, p in enumerate(paths):
            z = np.load(p, allow_pickle=False)
            self.x_noisy[i] = z["x_noisy"]
            for k in self.refs:
                self.refs[k][i] = z[k]
            self.rpeaks.append(z["rpeaks"].copy())
            self.meta.append(json.loads(str(z["meta"])))

    def __len__(self):
        return self.n

    def tensor(self, idx: Optional[np.ndarray] = None) -> torch.Tensor:
        """(B, 1, 3600) — 패딩은 학습 루프에서 model.pad_each로 적용한다."""
        a = self.x_noisy if idx is None else self.x_noisy[idx]
        return torch.from_numpy(np.ascontiguousarray(a)).unsqueeze(1)

    def ref_tensor(self, key: str, idx: Optional[np.ndarray] = None) -> torch.Tensor:
        a = self.refs[key] if idx is None else self.refs[key][idx]
        return torch.from_numpy(np.ascontiguousarray(a)).unsqueeze(1)


def load(cfg, split: str) -> Segments:
    """train은 정답 없이, val·test는 정답과 함께 로드한다."""
    return Segments(cfg, split, with_refs=(split != "train"))
