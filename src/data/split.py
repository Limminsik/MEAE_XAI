"""S1 — 기록(환자) 단위 분할 생성 (RESEARCH_DESIGN.md §4-6).

원칙
- 분할 단위는 기록. 분절 단위 분할은 누수다 (§0 원칙 5).
- 층화 없이 완전 무작위로 배정한다.
- 분할 전용 고정 시드(`data.split.split_seed`)만 사용한다. 학습 시드(`seeds`)는
  분할에 관여하지 않으므로, split.json은 1회 생성되고 모든 K·시드 실험이 공유한다.
- 이미 파일이 있으면 재생성하지 않고 동일성만 검사한다.
"""
import json
import os

import numpy as np
import wfdb
import yaml

from .download import MITDB_RECORDS

# 참고 통계용. AAMI N 클래스 = 정상 계열, 나머지(S/V/F/Q)를 부정맥으로 센다.
NORMAL_SYMBOLS = set("NLRej")
BEAT_SYMBOLS = set("NLRVAFjE/aJSeQf")


def load_cfg(path="configs/default.yaml"):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def beat_stats(mitdb, records):
    """기록별 (비트 수, 비정상 비트 수, 비율). 분할에는 쓰지 않고 보고용으로만 쓴다."""
    out = {}
    for r in records:
        ann = wfdb.rdann(os.path.join(mitdb, r), "atr")
        beats = [s for s in ann.symbol if s in BEAT_SYMBOLS]
        abnormal = sum(1 for s in beats if s not in NORMAL_SYMBOLS)
        out[r] = {"n_beats": len(beats), "n_abnormal": abnormal,
                  "ratio": abnormal / len(beats)}
    return out


def build_split(cfg):
    d = cfg["data"]
    sp = d["split"]
    excl = set(d["mitdb_exclude"])
    records = sorted(r for r in MITDB_RECORDS if r not in excl)
    counts = sp["counts"]
    assert sum(counts.values()) == len(records), (counts, len(records))

    rng = np.random.default_rng(sp["split_seed"])
    perm = list(rng.permutation(records))
    n_tr, n_va = counts["train"], counts["val"]
    split = {
        "train": sorted(perm[:n_tr]),
        "val": sorted(perm[n_tr:n_tr + n_va]),
        "test": sorted(perm[n_tr + n_va:]),
    }
    assert not set(split["train"]) & set(split["val"])
    assert not set(split["train"]) & set(split["test"])
    assert not set(split["val"]) & set(split["test"])

    meta = {
        "unit": "record",
        "n_records": len(records),
        "excluded": sorted(excl),
        "stratify": False,
        "split_seed": sp["split_seed"],
        "note": ("분할 전용 고정 시드로 1회 생성. 학습 시드는 분할에 관여하지 않으며 "
                 "모든 K/시드 실험이 이 분할을 공유한다."),
    }
    return split, meta


def main(config="configs/default.yaml", out=None):
    cfg = load_cfg(config)
    out = out or os.path.join(cfg["paths"]["processed"], "split.json")
    split, meta = build_split(cfg)

    if os.path.exists(out):
        prev = json.load(open(out, encoding="utf-8"))
        if {k: prev[k] for k in ("train", "val", "test")} != split:
            raise SystemExit(
                f"[abort] 기존 {out} 과 분할이 다릅니다. 실험 간 분할은 고정돼야 합니다.\n"
                f"        의도한 변경이면 파일을 지우고 다시 실행하세요.")
        print(f"[skip] 기존 분할과 동일: {out}")
    else:
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump({**split, "_meta": meta}, f, ensure_ascii=False, indent=2)
        print(f"[write] {out}  (split_seed={meta['split_seed']}, 층화 없음)")

    stats = beat_stats(cfg["paths"]["mitdb"], sorted(sum(split.values(), [])))
    for name in ("train", "val", "test"):
        rs = split[name]
        ratios = [stats[r]["ratio"] for r in rs]
        beats = sum(stats[r]["n_beats"] for r in rs)
        print(f"{name:>5}: {len(rs):2d}개  분절 {len(rs)*180:,}  비트 {beats:,}  "
              f"부정맥비율 중앙 {np.median(ratios):.3f} "
              f"[{min(ratios):.3f}–{max(ratios):.3f}]")
        print(f"        {' '.join(rs)}")
    return split


if __name__ == "__main__":
    main()
