"""지표 단위 테스트 — src/metrics.py 의 네 묶음을 실제 분절 하나로 검증한다."""
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
    r = man[(man.n_rpeaks >= 8) & (man.snr_bw > 8)
            & (man.snr_ma > 8) & (man.snr_em > 8)].iloc[0]
    z = np.load(os.path.join(PROC, "segments", r.split,
                             f"{r.record_id}_{r.seg_idx:04d}.npz"))
    return {k: z[k] for k in ("x_clean", "x_noisy", "rpeaks")}


# ---------------------------------------------------------------- [S5] 복원 채점
def test_identity(seg):
    """clean vs clean → 거리 0, 유사도 1, SNR 매우 큼."""
    c = seg["x_clean"][None, :].astype(np.float64)
    sc = metrics.s5_score(c, c)
    assert sc["SSD"][0] == 0.0 and sc["MAD"][0] == 0.0 and sc["PRD"][0] == 0.0
    assert sc["CosSim"][0] == pytest.approx(1.0)
    assert sc["SNR"][0] > 100


def test_snr_definition():
    """분산 2배 잡음 → SNR 정확히 10·log10(var/mse)."""
    rng = np.random.default_rng(0)
    c = rng.standard_normal((1, SEG_LEN))
    n = rng.standard_normal((1, SEG_LEN)) * 0.5
    got = metrics.snr_db_vec(c, c + n)[0]
    want = 10 * np.log10(c.var() / (n ** 2).mean())
    assert got == pytest.approx(want, rel=1e-9)


def test_prd_denominator_excludes_dc():
    """PRD 분모는 평균을 뺀 제곱합이다 (PRDN). DC 오프셋에 흔들리지 않는다."""
    rng = np.random.default_rng(5)
    c = rng.standard_normal((1, SEG_LEN))
    e = c + rng.standard_normal((1, SEG_LEN)) * 0.2
    base = metrics.prd(c, e)[0]
    # 두 신호에 같은 상수를 더해도 값이 변하지 않아야 한다
    assert metrics.prd(c + 5.0, e + 5.0)[0] == pytest.approx(base, rel=1e-9)
    want = 100 * np.sqrt(((e - c) ** 2).sum() / ((c - c.mean()) ** 2).sum())
    assert base == pytest.approx(want, rel=1e-9)


def test_snr_and_prd_are_inverse():
    """SNR = −20·log10(PRD/100). 두 지표가 같은 신호 전력 기준을 쓴다는 뜻이다."""
    rng = np.random.default_rng(6)
    c = rng.standard_normal((5, SEG_LEN)) + 3.0        # 평균이 0 이 아닌 신호
    e = c + rng.standard_normal((5, SEG_LEN)) * 0.3
    snr, p = metrics.snr_db_vec(c, e), metrics.prd(c, e)
    assert snr == pytest.approx(-20 * np.log10(p / 100.0), rel=1e-9)


def test_snr_ignores_dc_offset():
    """신호 전력이 분산이므로 상수 오프셋은 SNR 을 바꾸지 않는다.

    주입 정의(data/build.py)와 같은 기준이라는 뜻이다.
    """
    rng = np.random.default_rng(7)
    c = rng.standard_normal((1, SEG_LEN))
    d = rng.standard_normal((1, SEG_LEN)) * 0.2
    assert metrics.snr_db_vec(c + 4.0, c + 4.0 + d)[0] == pytest.approx(
        metrics.snr_db_vec(c, c + d)[0], rel=1e-9)


def test_snr_worsens_with_more_noise(seg):
    c = seg["x_clean"][None, :].astype(np.float64)
    rng = np.random.default_rng(1)
    n = rng.standard_normal(c.shape)
    assert metrics.snr_db_vec(c, c + 0.05 * n)[0] > metrics.snr_db_vec(c, c + 0.2 * n)[0]


# ---------------------------------------------------------------- R-피크 매칭
def test_peak_matching_one_to_one():
    """탐욕 매칭이 배타적인가 — 검출 둘이 참조 하나에 겹치면 TP 는 1."""
    ref = [1000]
    det = [995, 1005]
    tp, fp, fn = metrics.match_peaks(det, ref, FS, tol_ms=50)
    assert (tp, fp, fn) == (1, 1, 0)


def test_prf_edges():
    r = metrics.rpeak_prf([], [100, 200], FS)
    assert r["precision"] == 0.0 and r["recall"] == 0.0 and r["f1"] == 0.0
    r = metrics.rpeak_prf([100, 200], [100, 200], FS)
    assert r["f1"] == 1.0


def test_f1_is_one_for_identical_peak_sets(seg):
    """성분 = 참조면 F1 = 1 (sign_align 포함 경로)."""
    c = seg["x_clean"][None, :].astype(np.float64)
    comps = c[:, None, :]                                  # (1, K=1, L)
    f1 = metrics.f1_vector(comps, c, FS)
    assert f1[0, 0] == pytest.approx(1.0)


def test_detect_rpeaks_on_flat_signal():
    assert len(metrics.detect_rpeaks(np.zeros(SEG_LEN), FS)) == 0


def test_detect_rpeaks_on_real_clean(seg):
    """실제 clean 에서 검출한 피크가 주석과 대체로 일치."""
    det = metrics.detect_rpeaks(seg["x_clean"].astype(np.float64), FS)
    r = metrics.rpeak_prf(det, seg["rpeaks"], FS)
    assert r["f1"] > 0.9


# ---------------------------------------------------------------- [S6] SQI
def test_sqi_directions(seg):
    """잡음이 얹히면 basSQI ↑ · pSQI ↓ · kSQI ↓ — 방향이 정의와 맞는가."""
    c = seg["x_clean"][None, :].astype(np.float64)
    n = seg["x_noisy"][None, :].astype(np.float64)
    qc = metrics.sqi_all(c, FS, with_bsqi=False)
    qn = metrics.sqi_all(n, FS, with_bsqi=False)
    assert qn["basSQI"][0] > qc["basSQI"][0]
    assert qn["pSQI"][0] < qc["pSQI"][0]
    assert qn["kSQI"][0] < qc["kSQI"][0]


def test_bsqi_clean_high(seg):
    """clean 에서는 두 검출기가 대체로 일치한다."""
    v = metrics.bsqi(seg["x_clean"].astype(np.float64), FS)
    assert v > 0.8


def test_ecg_mean_coef(seg):
    """clean 의 박동 일관성은 높고, 강한 백색잡음을 얹으면 떨어진다."""
    c = seg["x_clean"].astype(np.float64)
    pk = metrics.detect_rpeaks(c, FS)
    hi = metrics.ecg_mean_coef(c, FS, peaks=pk)
    rng = np.random.default_rng(2)
    lo = metrics.ecg_mean_coef(c + rng.standard_normal(len(c)) * c.std(), FS, peaks=pk)
    assert hi > 0.9 and lo < hi
    # 박동이 모자라면 NaN
    assert np.isnan(metrics.ecg_mean_coef(c, FS, peaks=pk[:2]))


# ---------------------------------------------------------------- [비교선] 고전 3종
def test_classical_shapes_and_mean(seg):
    """세 방법 모두 모양을 보존하고, 평균 복원이 입력 평균과 일치."""
    n = seg["x_noisy"][None, :].astype(np.float64)
    out = metrics.classical_denoise(n, FS)
    assert set(out) == {"대역통과 0.5-40Hz", "웨이블릿 임계값", "웨이블릿+기저선제거"}
    for v in out.values():
        assert v.shape == n.shape
    # DC 오프셋 복원 — 대역통과·기저선제거는 입력 평균으로 돌아와야 한다
    for k in ("대역통과 0.5-40Hz", "웨이블릿+기저선제거"):
        assert out[k].mean(-1)[0] == pytest.approx(n.mean(-1)[0], abs=5e-2)


def test_wavelet_reduces_hf_noise(seg):
    """백색잡음을 얹으면 웨이블릿 임계값이 SNR 을 올린다 (고주파는 세부계수에 있다)."""
    c = seg["x_clean"][None, :].astype(np.float64)
    rng = np.random.default_rng(3)
    noisy = c + rng.standard_normal(c.shape) * 0.1
    den = metrics.wavelet_denoise(noisy)
    assert metrics.snr_db_vec(c, den)[0] > metrics.snr_db_vec(c, noisy)[0]
