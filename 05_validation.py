"""05 — 외부 데이터 적용 시연 (RESEARCH_DESIGN.md §10). [S6]

**목적은 성능 검증이 아니라 적용 가능성 시연이다.** 외부 데이터에는 clean·bw·ma·em 정답이
없으므로 정량 성능을 주장하지 않는다. **재학습·재판별 금지** — 03이 확정한 대응과 04가
확정한 마스킹 조합을 그대로 적용한다 (§0 원칙 6).

    python 05_validation.py --run K8_seed42 --source mimic_iv --mask 2-6

대상 (연구계획서 기반연구 ①)
  MIMIC-IV Waveform  중환자실  · VitalDB  수술실
  NSRR (shhs·wsc·nfs) 수면      · GalaxyPPG의 Polar H10 ECG  일상 웨어러블

절차: 리샘플·단위 정합 → 성분 분해 → 마스킹 → 전후 제시.

산출물  results/05_validation/<run>/<source>/
  segments.npz            사용한 분절 (리샘플 후 3600 샘플)
  spectra.csv             전후 대역별 전력비
  figures/  case_*.png    대표 사례 [원 분절 → 성분 분해 → 복원 → 스펙트럼 전후]

**주의**: 각 데이터셋의 로더는 원본 규격 확인 후 구현한다 (표본율·단위·유도 이름).
현재는 규격 확인과 리샘플 유틸만 두고, 실제 적용은 04 결과로 마스킹 조합을 확정한 뒤 착수한다.
"""
import argparse
import os

import numpy as np
from scipy.signal import resample_poly

from src.data.build import load_cfg

# 원 표본율 → 360 Hz 리샘플 비. 정수비가 아니면 확인 후 추가한다.
RESAMPLE = {125: (72, 25), 130: (36, 13), 250: (36, 25), 500: (18, 25)}


def to_360(x, fs_src, fs_dst=360):
    """표본율 정합. 정수비가 등록돼 있으면 resample_poly, 아니면 막는다."""
    if fs_src == fs_dst:
        return np.asarray(x, np.float64)
    if fs_src not in RESAMPLE:
        raise ValueError(
            f"표본율 {fs_src} Hz 의 리샘플 비가 등록돼 있지 않다. "
            f"원본 규격을 확인하고 RESAMPLE 에 추가할 것.")
    up, down = RESAMPLE[fs_src]
    return resample_poly(np.asarray(x, np.float64), up, down, axis=-1)


def main(config="configs/default.yaml", run="K8_seed42", source="mimic_iv",
         mask=None, outdir=None):
    cfg = load_cfg(config)
    ext = cfg.get("external", {})
    if source not in ext:
        raise SystemExit(f"[05] configs/default.yaml 의 external 에 '{source}' 가 없다. "
                         f"등록된 것: {list(ext)}")
    outdir = outdir or os.path.join("results", "05_validation", run, source)
    os.makedirs(outdir, exist_ok=True)
    print(f"[05] {source} 설정: {ext[source]}")
    print("[05] 로더 미구현 — 04에서 마스킹 조합을 확정한 뒤 착수한다.")
    print(f"     산출 예정 위치 → {outdir}/")
    return ext[source]


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--run", default="K8_seed42")
    p.add_argument("--source", default="mimic_iv")
    p.add_argument("--mask", default=None, help="예: 2-6 (끌 인코더, 1-based)")
    p.add_argument("--outdir", default=None)
    a = p.parse_args()
    main(a.config, a.run, a.source, a.mask, a.outdir)
