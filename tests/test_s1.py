"""S1 단위 테스트 (RESEARCH_DESIGN.md §11).

SNR 실측은 §4-4 주입과 동일하게 **신호는 분산, 잡음은 mean(n**2)** 기준으로 잰다.
정의가 어긋나면 테스트가 항상 실패한다.
"""
import json
import os

import numpy as np
import pandas as pd
import pytest

from src.data.build import NOISES, load_cfg, load_noise_pools, measured_snr

CFG = load_cfg()
PROC = CFG["paths"]["processed"]
SEG_LEN = CFG["data"]["fs"] * CFG["data"]["seg_sec"]


@pytest.fixture(scope="module")
def split():
    return json.load(open(os.path.join(PROC, "split.json"), encoding="utf-8"))


@pytest.fixture(scope="module")
def manifest():
    return pd.read_csv(os.path.join(PROC, "manifest.csv"), dtype={"record_id": str})


def _sample_paths(manifest, n=200, seed=0):
    rng = np.random.default_rng(seed)
    rows = manifest.iloc[rng.choice(len(manifest), size=min(n, len(manifest)), replace=False)]
    return [(os.path.join(PROC, "segments", r.split, f"{r.record_id}_{r.seg_idx:04d}.npz"), r)
            for r in rows.itertuples()]


def test_counts(split, manifest):
    """기록 46개 × 180분절 = 8,280. split별 분절 수도 기록 수와 일치."""
    assert len(manifest) == 8280
    for name in ("train", "val", "test"):
        assert (manifest.split == name).sum() == len(split[name]) * 180


def test_split_leakage(split, manifest):
    """train/val/test 기록 교집합 0 (§0 원칙 5)."""
    tr, va, te = (set(split[k]) for k in ("train", "val", "test"))
    assert not (tr & va) and not (tr & te) and not (va & te)
    assert len(tr | va | te) == 46
    for name in ("train", "val", "test"):
        assert set(manifest.loc[manifest.split == name, "record_id"]) == set(split[name])


def test_injection(manifest):
    """실측 SNR = 추첨값 ±0.1 dB, 그리고 x_noisy == x_clean + bw + ma + em."""
    for path, row in _sample_paths(manifest):
        z = np.load(path, allow_pickle=False)
        total = z["x_clean"] + z["bw"] + z["ma"] + z["em"]
        assert np.allclose(z["x_noisy"], total, rtol=0, atol=1e-6), path
        for n in NOISES:
            drawn = getattr(row, f"snr_{n}")
            assert abs(measured_snr(z["x_clean"], z[n]) - drawn) < 0.1, (path, n)


def test_noise_time_split(manifest):
    """train 분절과 val·test 분절이 쓴 잡음 원본 구간이 겹치지 않는다 (§4-3)."""
    cut = int(650000 * CFG["data"]["noise_split_ratio"])
    tr = manifest[manifest.split == "train"]
    ev = manifest[manifest.split != "train"]
    for n in NOISES:
        # 창은 [start, start+3600). train은 cut 이전에서 끝나고 eval은 cut 이후에서 시작해야 한다.
        assert (tr[f"start_{n}"] + SEG_LEN <= cut).all(), n
        assert (ev[f"start_{n}"] >= cut).all(), n
        assert tr[f"start_{n}"].max() + SEG_LEN <= ev[f"start_{n}"].min()


def test_noise_pool_bounds():
    """잡음 풀 경계 자체가 겹치지 않게 잘렸는지."""
    pools = load_noise_pools(CFG)
    for n in NOISES:
        (_, lo_t, hi_t), (_, lo_e, hi_e) = pools[n]["train"], pools[n]["eval"]
        assert lo_t == 0 and hi_t == lo_e and hi_e == 650000
        assert hi_t - lo_t >= SEG_LEN and hi_e - lo_e >= SEG_LEN


def test_shapes_and_rpeaks(manifest):
    """배열 길이 3600, R-피크는 분절 로컬 인덱스 범위 안."""
    for path, row in _sample_paths(manifest, n=100, seed=1):
        z = np.load(path, allow_pickle=False)
        for k in ("x_clean", "x_noisy", *NOISES):
            assert z[k].shape == (SEG_LEN,) and z[k].dtype == np.float32, (path, k)
        r = z["rpeaks"]
        assert r.ndim == 1 and len(r) == row.n_rpeaks
        if len(r):
            assert r.min() >= 0 and r.max() < SEG_LEN
            assert (np.diff(r) > 0).all()


def test_reproducible(manifest):
    """(gen_seed, record, seg_idx) 기반 난수 → 같은 분절은 항상 같은 잡음·SNR."""
    from src.data.build import build_segment, seg_rng
    pools = load_noise_pools(CFG)
    for path, row in _sample_paths(manifest, n=20, seed=2):
        z = np.load(path, allow_pickle=False)
        which = "train" if row.split == "train" else "eval"
        rng = seg_rng(row.gen_seed, row.record_id, row.seg_idx)
        _, x_noisy, comps, info = build_segment(
            z["x_clean"].astype(np.float64), pools, which, rng,
            CFG["data"]["noise_snr_range_db"])
        for n in NOISES:
            assert info[n]["start"] == getattr(row, f"start_{n}"), (path, n)
            assert abs(info[n]["snr"] - getattr(row, f"snr_{n}")) < 1e-9
        assert np.allclose(x_noisy, z["x_noisy"], atol=1e-5)
