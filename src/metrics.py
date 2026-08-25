"""S5 채점 지표 (RESEARCH_DESIGN.md §8).

전부 **clean 참조 기준**이다 (S4는 잡음 참조, S5는 clean 참조 — 혼동 주의).
비교 대상은 외부 기법이 아니라 **마스킹 전(x_noisy) vs 마스킹 후(복원)** 다.

입력은 **크롭된 원 구간 3600 샘플**이어야 한다. 모델 출력은 3840이므로
`src.model.meae.crop()`을 반드시 먼저 거친다. 패딩 0 구간이 들어오면 지표가 왜곡된다.
`score()`는 길이를 검사해 어긋나면 막는다.
"""
from typing import Dict, Optional, Sequence

import numpy as np

TOL_MS = 150.0          # R-피크 매칭 허용오차 (§8)
_MIN_PEAKS_FOR_SDNN = 3  # RR 2개 이상 있어야 표준편차가 정의된다


def snr_db(clean: np.ndarray, est: np.ndarray) -> float:
    """`10log10(var(clean) / mean((est-clean)**2))`.

    §4-4 주입과 같은 정의 — 신호는 분산(DC 제외), 잡음(잔차)은 mean(·**2).
    """
    clean = np.asarray(clean, dtype=np.float64)
    resid = np.asarray(est, dtype=np.float64) - clean
    p_noise = float(np.mean(resid ** 2))
    if p_noise == 0.0:
        return float("inf")
    return float(10.0 * np.log10(float(np.var(clean)) / p_noise))


def rmse(clean: np.ndarray, est: np.ndarray) -> float:
    d = np.asarray(est, dtype=np.float64) - np.asarray(clean, dtype=np.float64)
    return float(np.sqrt(np.mean(d ** 2)))


def detect_rpeaks(sig: np.ndarray, fs: int) -> np.ndarray:
    """neurokit2로 R-피크 검출. 잡음이 심해 실패하면 빈 배열을 돌려준다."""
    import neurokit2 as nk
    sig = np.asarray(sig, dtype=np.float64)
    if not np.isfinite(sig).all() or np.allclose(sig, sig[0]):
        return np.array([], dtype=int)
    try:
        _, info = nk.ecg_peaks(sig, sampling_rate=fs)
        return np.asarray(info["ECG_R_Peaks"], dtype=int)
    except Exception:
        return np.array([], dtype=int)


def match_peaks(detected: Sequence[int], reference: Sequence[int], fs: int,
                tol_ms: float = TOL_MS):
    """허용오차 안에서 1:1 매칭. 가까운 쌍부터 탐욕적으로 짝짓는다.

    반환: (TP, FP, FN)
    """
    det = np.sort(np.asarray(detected, dtype=int))
    ref = np.sort(np.asarray(reference, dtype=int))
    tol = tol_ms / 1000.0 * fs
    if len(det) == 0 or len(ref) == 0:
        return 0, len(det), len(ref)

    # 거리 행렬에서 허용오차 안의 쌍만 골라 가까운 순으로 배타 매칭
    dist = np.abs(det[:, None] - ref[None, :])
    pairs = [(dist[i, j], i, j) for i, j in zip(*np.where(dist <= tol))]
    pairs.sort()
    used_d, used_r, tp = set(), set(), 0
    for _, i, j in pairs:
        if i not in used_d and j not in used_r:
            used_d.add(i)
            used_r.add(j)
            tp += 1
    return tp, len(det) - tp, len(ref) - tp


def rpeak_prf(detected, reference, fs: int, tol_ms: float = TOL_MS) -> Dict[str, float]:
    tp, fp, fn = match_peaks(detected, reference, fs, tol_ms)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"tp": tp, "fp": fp, "fn": fn,
            "precision": prec, "recall": rec, "f1": f1}


def sdnn_ms(peaks: Sequence[int], fs: int) -> float:
    """RR 간격의 표준편차 (ms). RR이 2개 미만이면 NaN."""
    p = np.sort(np.asarray(peaks, dtype=np.float64))
    if len(p) < _MIN_PEAKS_FOR_SDNN:
        return float("nan")
    rr = np.diff(p) / fs * 1000.0
    return float(np.std(rr, ddof=1))


def score(clean: np.ndarray, est: np.ndarray, rpeaks_ref: Sequence[int], fs: int,
          seg_len: Optional[int] = None, tol_ms: float = TOL_MS) -> Dict[str, float]:
    """단일 채점 진입점 — 신호 하나를 clean 참조로 채점한다.

    마스킹 전은 `est = x_noisy`, 마스킹 후는 `est = 복원 신호`로 같은 함수를 호출한다.
    SDNN은 est에서 검출한 피크 vs **MIT-BIH 주석 피크**를 비교한다.
    """
    clean = np.asarray(clean, dtype=np.float64).ravel()
    est = np.asarray(est, dtype=np.float64).ravel()
    if clean.shape != est.shape:
        raise ValueError(f"길이 불일치: clean {clean.shape} vs est {est.shape}")
    if seg_len is not None and len(clean) != seg_len:
        raise ValueError(
            f"길이 {len(clean)} != {seg_len}. 모델 출력(3840)을 crop()으로 "
            f"중앙 {seg_len}으로 자른 뒤 채점해야 한다 (§8).")

    peaks = detect_rpeaks(est, fs)
    prf = rpeak_prf(peaks, rpeaks_ref, fs, tol_ms)
    sdnn_est, sdnn_ref = sdnn_ms(peaks, fs), sdnn_ms(rpeaks_ref, fs)
    return {"snr_db": snr_db(clean, est), "rmse": rmse(clean, est),
            "n_peaks_det": int(len(peaks)), "n_peaks_ref": int(len(rpeaks_ref)),
            **prf,
            "sdnn_est_ms": sdnn_est, "sdnn_ref_ms": sdnn_ref,
            "sdnn_abs_err_ms": abs(sdnn_est - sdnn_ref)}


def score_pair(clean, x_before, x_after, rpeaks_ref, fs, seg_len=None,
               tol_ms: float = TOL_MS) -> Dict[str, float]:
    """마스킹 전/후를 한 번에 채점하고 개선량까지 계산한다 (§8의 표 한 줄)."""
    b = score(clean, x_before, rpeaks_ref, fs, seg_len, tol_ms)
    a = score(clean, x_after, rpeaks_ref, fs, seg_len, tol_ms)
    out = {f"before_{k}": v for k, v in b.items()}
    out.update({f"after_{k}": v for k, v in a.items()})
    out["snr_improvement_db"] = a["snr_db"] - b["snr_db"]
    out["rmse_reduction"] = b["rmse"] - a["rmse"]
    out["f1_gain"] = a["f1"] - b["f1"]
    out["sdnn_err_reduction_ms"] = b["sdnn_abs_err_ms"] - a["sdnn_abs_err_ms"]
    return out
