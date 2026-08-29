"""T4 단위 테스트 — 지표 (RESEARCH_DESIGN.md §11 test_metrics)."""
import os

import numpy as np
import pandas as pd
import pytest

from src import metrics
from src.data.build import load_cfg

CFG = load_cfg()
FS = CFG["data"]["fs"]
SEG_LEN = FS * CFG["data"]["seg_sec"]
PROC = CFG["paths"]["processed"]


@pytest.fixture(scope="module")
def seg():
    """실제 분절 하나 — 합성 신호로는 R-피크 검출을 검증할 수 없다."""
    man = pd.read_csv(os.path.join(PROC, "manifest.csv"), dtype={"record_id": str})
    # 피크가 넉넉하고 잡음이 약한 분절을 고른다
    r = man[(man.n_rpeaks >= 8) & (man.snr_bw > 8) & (man.snr_ma > 8) & (man.snr_em > 8)].iloc[0]
    z = np.load(os.path.join(PROC, "segments", r.split, f"{r.record_id}_{r.seg_idx:04d}.npz"))
    return {k: z[k] for k in ("x_clean", "x_noisy", "rpeaks")}


def test_identity(seg):
    """clean vs clean → RMSE 0, SNR 무한대 (§11 요구)."""
    c = seg["x_clean"]
    assert metrics.rmse(c, c) == 0.0
    assert metrics.snr_db(c, c) == float("inf")
    s = metrics.score(c, c, seg["rpeaks"], FS, SEG_LEN)
    assert s["rmse"] == 0.0


def test_f1_is_one_for_identical_peak_sets(seg):
    """§11의 'F1 1.0'은 동일한 피크 집합끼리 비교했을 때 성립한다.

    `score(clean, clean, 주석)`의 F1은 1.0이 아니다 — 검출기(neurokit2)와
    사람 주석이 완전히 일치하지 않기 때문이며, 이것이 F1의 실질 상한이다.
    상한 실측치는 results/data_notes.md §13 참조.
    """
    assert metrics.rpeak_prf(seg["rpeaks"], seg["rpeaks"], FS)["f1"] == 1.0
    assert metrics.score(seg["x_clean"], seg["x_clean"], seg["rpeaks"],
                         FS, SEG_LEN)["f1"] > 0.9


def test_snr_definition():
    """§4-4 주입과 동일한 정의인지: 알려진 SNR을 넣으면 그대로 나와야 한다."""
    rng = np.random.default_rng(0)
    clean = rng.normal(size=SEG_LEN) + 5.0          # DC 5는 분산에 안 잡힌다
    noise = rng.normal(size=SEG_LEN)
    for target in (0.0, 6.0, 12.0):
        a = np.sqrt(np.var(clean) / (np.mean(noise ** 2) * 10 ** (target / 10)))
        assert abs(metrics.snr_db(clean, clean + a * noise) - target) < 1e-9


def test_snr_worsens_with_more_noise(seg):
    c, n = seg["x_clean"], seg["x_noisy"]
    assert metrics.snr_db(c, n) < metrics.snr_db(c, c * 0.5 + n * 0.5)
    assert metrics.rmse(c, n) > 0


def test_peak_matching_one_to_one():
    """허용오차 안이어도 한 참조에 두 검출이 붙지 않는다."""
    tol = int(metrics.TOL_MS / 1000 * FS)
    ref = [1000, 2000]
    tp, fp, fn = metrics.match_peaks([1000, 1000 + tol - 1], ref, FS)
    assert (tp, fp, fn) == (1, 1, 1)
    assert metrics.match_peaks([1000 + tol + 5], [1000], FS) == (0, 1, 1)   # 허용오차 밖
    assert metrics.match_peaks([], ref, FS) == (0, 0, 2)
    assert metrics.match_peaks(ref, [], FS) == (0, 2, 0)


def test_prf_edges():
    assert metrics.rpeak_prf([], [], FS)["f1"] == 0.0
    perfect = metrics.rpeak_prf([100, 500], [100, 500], FS)
    assert perfect["f1"] == 1.0 and perfect["precision"] == 1.0


def test_sdnn():
    """RR이 일정하면 SDNN 0, 2개 미만이면 NaN."""
    peaks = np.arange(0, 5) * FS          # RR = 1000 ms 일정
    assert metrics.sdnn_ms(peaks, FS) == 0.0
    assert np.isnan(metrics.sdnn_ms([100], FS))
    assert np.isnan(metrics.sdnn_ms([100, 500], FS))
    rr = np.array([0, 360, 396, 720])     # RR = 1000, 100, 900 ms
    assert abs(metrics.sdnn_ms(rr, FS) - np.std([1000, 100, 900], ddof=1)) < 1e-9


def test_score_rejects_uncropped_length(seg):
    """3840을 그대로 넣으면 막아야 한다 — 패딩 0 구간이 지표를 왜곡한다."""
    pad = CFG["data"]["pad_each"]
    padded = np.pad(seg["x_clean"], (pad, pad))
    with pytest.raises(ValueError, match="crop"):
        metrics.score(padded, padded, seg["rpeaks"], FS, SEG_LEN)


def test_score_pair_direction(seg):
    """마스킹 후가 clean에 가까우면 개선량이 양수로 나온다."""
    c, n = seg["x_clean"], seg["x_noisy"]
    better = c + (n - c) * 0.3            # 잡음을 70% 걷어낸 가상 복원
    d = metrics.score_pair(c, n, better, seg["rpeaks"], FS, SEG_LEN)
    assert d["snr_improvement_db"] > 0
    assert d["rmse_reduction"] > 0
    assert d["before_snr_db"] < d["after_snr_db"]


def test_detect_rpeaks_on_flat_signal():
    """평탄·비유한 입력에서 예외 없이 빈 배열."""
    assert len(metrics.detect_rpeaks(np.zeros(SEG_LEN), FS)) == 0
    assert len(metrics.detect_rpeaks(np.full(SEG_LEN, np.nan), FS)) == 0


def test_detect_rpeaks_on_real_clean(seg):
    """실제 clean 분절에서 주석과 비슷한 수의 피크를 찾는지."""
    det = metrics.detect_rpeaks(seg["x_clean"], FS)
    prf = metrics.rpeak_prf(det, seg["rpeaks"], FS)
    assert prf["f1"] > 0.8, prf
