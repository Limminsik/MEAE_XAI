"""지표 라이브러리 — 네 묶음이다.

  [S5]   복원 채점 5종 (SSD·MAD·PRD·CosSim·SNR).  clean 참조, **mV 원단위**
  [S4]   성분 정렬 확장 (F1·r_QRS·대역 에너지).  분절 내 표준화 계열과 함께 쓴다
  [S6]   신호 품질 지수 SQI 5종 + ECGMeanCoef.  **참값이 필요 없다**
  [비교선] 고전 디노이징 3종 (대역통과 · 웨이블릿 임계값 · 웨이블릿+기저선제거)

입력은 **크롭된 원 구간 3600 샘플**이어야 한다. 모델 출력은 패딩된 길이이므로
`src.model.meae.crop()`을 반드시 먼저 거친다. 패딩 0 구간이 들어오면 지표가 왜곡된다.
"""
from typing import Dict, Sequence

import numpy as np

TOL_MS = 150.0          # R-피크 매칭 허용오차 (S5 · bSQI)


# ---------------------------------------------------------------- [S5] 지표 5종
# DeepFilter·MECG-E 표준 세트. 전부 clean 참조 기준, **mV 원단위**(표준화하지 않는다).
# 마지막 축 기준 벡터화 — (n, T) 를 넣으면 (n,) 이 나온다.
def ssd(clean, est):
    """Sum of Squared Distance — `Σ_i (est − clean)²`. 낮을수록 유사."""
    d = np.asarray(est, np.float64) - np.asarray(clean, np.float64)
    return (d ** 2).sum(-1)


def mad(clean, est):
    """Maximum Absolute Distance — `max_i |est − clean|`. 낮을수록 유사.
    S4의 MAD와 이름은 같지만 **여기는 표준화하지 않은 mV 원단위**다."""
    d = np.asarray(est, np.float64) - np.asarray(clean, np.float64)
    return np.abs(d).max(-1)


def prd(clean, est):
    """Percentage Root-mean-square Difference — `100·√(Σ(est−clean)² / Σ(clean−mean)²)` [%].

    낮을수록 유사. **분모에서 평균을 뺀다** (문헌의 PRDN / PRD1 형태).

    빼는 이유: 우리 x_clean 은 기저선이 0 이 아니다(test 중앙 −0.295 mV, SD 0.302 mV).
    평균을 그대로 두면 `mean²/var = 1.32` 라 **상수 오프셋 하나가 심전도 파형 전체보다
    큰 전력**으로 잡혀 분모가 2.3배가 되고, PRD 가 1.52배 낮게 나온다. 전극 오프셋은
    정보를 담지 않으므로 신호 전력에서 뺀다.

    `snr_db_vec` 과 같은 기준(신호 전력 = 분산)이라 두 지표는 정확히 역관계다:

        SNR = −20·log10(PRD / 100)
    """
    c = np.asarray(clean, np.float64)
    d = np.asarray(est, np.float64) - c
    den = ((c - c.mean(-1, keepdims=True)) ** 2).sum(-1)
    return 100.0 * np.sqrt((d ** 2).sum(-1) / np.maximum(den, 1e-30))


def cossim(clean, est):
    """Cosine Similarity — 두 벡터 내적 / 노름 곱. 높을수록 유사. 평균 제거 없음."""
    c = np.asarray(clean, np.float64)
    e = np.asarray(est, np.float64)
    den = np.linalg.norm(c, axis=-1) * np.linalg.norm(e, axis=-1)
    return np.where(den > 0, (c * e).sum(-1) / np.maximum(den, 1e-30), 0.0)


def snr_db_vec(clean, est):
    """`10log10(var(clean) / mean((est−clean)²))` 의 벡터판. 높을수록 유사.

    **신호 전력은 분산이다** — 평균을 뺀다. 전극 오프셋은 정보가 아니므로 신호로 세지
    않는다. `data/build.py` 의 주입 정의와 같은 기준이라, x_noisy 를 이 식으로 재면
    주입한 합성 SNR 이 그대로 나온다 (test 중앙 0.305 vs 주입 0.047 dB).

    문헌식 `10log10(Σy²/Σd²)` 을 글자 그대로 쓰면 이 데이터에서는 +3.84 dB 부풀려진다 —
    그 식은 기저선을 먼저 제거한 신호를 전제하기 때문이다. 우리는 기저선을 남기므로
    분자에서 평균을 빼는 것이 같은 식을 올바르게 적용하는 방법이다.

    `prd()` 와 같은 기준이라 두 지표는 정확히 역관계다: SNR = −20·log10(PRD/100).
    """
    c = np.asarray(clean, np.float64)
    d = np.asarray(est, np.float64) - c
    return 10.0 * np.log10(c.var(-1) / np.maximum((d ** 2).mean(-1), 1e-30))


# 지표 이름 → (함수, 높을수록 좋은가)
S5_METRICS = {"SSD": (ssd, False), "MAD": (mad, False), "PRD": (prd, False),
              "CosSim": (cossim, True), "SNR": (snr_db_vec, True)}


def s5_score(clean, est):
    """지표 5종을 한 번에. 반환 {이름: (n,) 배열}."""
    return {k: f(clean, est) for k, (f, _) in S5_METRICS.items()}


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


S4_TOL_MS = 50.0            # F1 매칭 허용오차. S5의 TOL_MS(150)와 목적이 다르다
QRS_BAND = (5.0, 15.0)      # QRS 에너지가 모이는 대역. bw(<1)·T파(<5)·ma 고주파부(>20) 배제
MIN_REF_PEAKS = 3           # 10초에 3박동 미만이면 검출 실패로 보고 그 분절을 뺀다
PSD_BANDS = {"vlf": (0.05, 0.5), "lf": (0.5, 5.0),
             "qrs": (5.0, 15.0), "hf": (15.0, 40.0)}
PSD_NORM = (0.05, 40.0)     # 대역 비율의 분모 구간


def bandpass(x, lo, hi, fs, order=4):
    """Butterworth 대역통과 + filtfilt(영위상). 마지막 축에 적용한다.

    영위상이 필수다 — 위상 지연이 남으면 상관이 그만큼 떨어져 파형 충실도와 섞인다.
    """
    from scipy.signal import butter, sosfiltfilt
    sos = butter(order, [lo, hi], btype="bandpass", fs=fs, output="sos")
    return sosfiltfilt(sos, np.asarray(x, dtype=np.float64), axis=-1)


def _corr_last(a, b, eps=1e-12):
    """마지막 축 기준 Pearson 절댓값. 앞 축은 브로드캐스트한다."""
    a = a - a.mean(-1, keepdims=True)
    b = b - b.mean(-1, keepdims=True)
    d = np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1)
    return np.abs(np.einsum("...t,...t->...", a, b) / np.maximum(d, eps))


def sign_align(comps, ref):
    """성분 부호를 참조에 맞춘다. (S,K,L) x (S,L) -> (S,K,L)

    성분의 R파는 아래로 향할 수 있다. 부호를 맞추지 않으면 검출기가 T파나
    S파를 집어 F1이 실제보다 낮게 나온다.
    """
    s = np.sign(np.einsum("skt,st->sk",
                          comps - comps.mean(-1, keepdims=True),
                          ref - ref.mean(-1, keepdims=True)))
    s[s == 0] = 1.0
    return comps * s[..., None]


def reference_peaks(clean, fs, progress=None):
    """참조 R-피크 — x_clean 에서 **한 번만** 검출해 캐시한다.

    성분·에폭이 바뀌어도 같은 값이어야 비교가 성립한다. 매번 다시 검출하면
    비용만 늘고 값이 흔들린다.
    """
    out = []
    for i, c in enumerate(clean):
        out.append(detect_rpeaks(c, fs))
        if progress and i % progress == 0:
            print(f"  참조 R-피크 {i}/{len(clean)}", flush=True)
    return out


def f1_vector(comps, clean, fs, ref_peaks=None, tol_ms=S4_TOL_MS,
              min_peaks=MIN_REF_PEAKS, progress=None):
    """[S4-04] 성분별 R-피크 검출 F1. (S,K,L) -> (S,K), 참조 피크가 적은 분절은 NaN.

    파형 형태와 무관하게 **박동 정보가 담겼는지**만 본다.
    검출기 설정은 전 성분·전 실험에서 동일해야 한다 (성분마다 바꾸면 비교가 깨진다).
    """
    S, K, _ = comps.shape
    if ref_peaks is None:
        ref_peaks = reference_peaks(clean, fs, progress)
    aligned = sign_align(comps, clean)
    out = np.full((S, K), np.nan)
    for s in range(S):
        if len(ref_peaks[s]) < min_peaks:
            continue
        for k in range(K):
            out[s, k] = rpeak_prf(detect_rpeaks(aligned[s, k], fs),
                                  ref_peaks[s], fs, tol_ms)["f1"]
        if progress and s % progress == 0:
            print(f"  F1 {s}/{S}", flush=True)
    return out


def r_qrs_matrix(comps, refs, fs, band=QRS_BAND):
    """[S4-05] QRS 대역만 통과시킨 뒤의 |r|. (S,K,L) x (S,R,L) -> (S,K,R)

    전대역 |r| 은 저주파 혼입에 희석된다. 대역을 좁혀 **파형 충실도**를 분리해 본다.
    계산은 기존 |r| 과 같고 입력만 필터링된 신호다 (절댓값도 동일).
    """
    c = bandpass(comps, band[0], band[1], fs)
    r = bandpass(refs, band[0], band[1], fs)
    return _corr_last(c[:, :, None, :], r[:, None, :, :])


def band_energy(sig, fs, bands=None, norm=PSD_NORM, nperseg=1024):
    """[S4-06] Welch PSD 의 대역별 전력 **비율**. (..., L) -> (..., n_bands)

    성분마다 진폭 스케일이 다르므로 절대 전력이 아니라 비율로 본다.
    분모는 norm 구간 전체이고 각 행의 합은 1 이하다 (구간 밖은 버린다).
    """
    from scipy.signal import welch
    bands = bands or PSD_BANDS
    f, P = welch(np.asarray(sig, dtype=np.float64), fs=fs, nperseg=nperseg,
                 noverlap=nperseg // 2, window="hann", axis=-1)
    den = P[..., (f >= norm[0]) & (f < norm[1])].sum(-1)
    cols = [P[..., (f >= lo) & (f < hi)].sum(-1) / np.maximum(den, 1e-30)
            for lo, hi in bands.values()]
    return np.stack(cols, axis=-1), list(bands)


# ================================================================
# [S6] 신호 품질 지수 (SQI) — **참값이 필요 없다**
#
# 04 는 x_clean 을 기준으로 재고, 여기는 신호 하나만 보고 잰다. 두 층이 서로를 보완한다 —
# 04 는 표준 지표라 문헌 비교가 되고, SQI 는 참값 의존을 우회해 독립적 근거를 준다.
# MIT-BIH 원본도 완전한 참값이 아니라는 한계에 대한 대비이기도 하다.
#
#   basSQI = P(0-1Hz) / P(0-40Hz)        기저선 대역 비중.  bw 가 빠지면 **낮아진다**
#   pSQI   = P(5-20Hz) / P(0-62.5Hz)     QRS 대역 비중.    잡음이 줄면 **높아진다**
#   kSQI   = 첨도 (4차 모멘트 / 분산²)     잡음 없는 QRS 일수록 높다 (통상 5 이상)
#   bSQI   = |R1 ∩ R2| / |R1 ∪ R2|       서로 다른 두 검출기의 박동 일치 비율
# ================================================================
SQI_BANDS = {"basSQI": ((0.0, 1.0), (0.0, 40.0)),      # (분자 대역, 분모 대역)
             "pSQI": ((5.0, 20.0), (0.0, 62.5))}
BSQI_TOL_MS = 150.0        # 두 검출기 사이 허용오차. S5 의 TOL_MS 와 같은 값이다
BSQI_METHODS = ("neurokit", "pantompkins1985")


def _band_ratio(sig, fs, num, den, nperseg=1024):
    """Welch PSD 의 대역 파워 비. (..., L) -> (...,)"""
    from scipy.signal import welch
    f, P = welch(np.asarray(sig, dtype=np.float64), fs=fs, nperseg=nperseg,
                 noverlap=nperseg // 2, window="hann", axis=-1)
    a = P[..., (f >= num[0]) & (f < num[1])].sum(-1)
    b = P[..., (f >= den[0]) & (f < den[1])].sum(-1)
    return a / np.maximum(b, 1e-30)


def ksqi(sig):
    """첨도. (..., L) -> (...,). 정규분포면 3, 뾰족한 QRS 가 살아 있으면 커진다."""
    x = np.asarray(sig, dtype=np.float64)
    d = x - x.mean(-1, keepdims=True)
    m2 = (d ** 2).mean(-1)
    return (d ** 4).mean(-1) / np.maximum(m2 ** 2, 1e-30)


def _peaks_by(sig, fs, method):
    """검출기를 지정해 R-피크를 찾는다. 실패하면 빈 배열."""
    import neurokit2 as nk
    s = np.asarray(sig, dtype=np.float64)
    if not np.isfinite(s).all() or np.allclose(s, s[0]):
        return np.array([], dtype=int)
    try:
        clean = nk.ecg_clean(s, sampling_rate=fs, method=method)
        info = nk.ecg_findpeaks(clean, sampling_rate=fs, method=method)
        return np.asarray(info["ECG_R_Peaks"], dtype=int)
    except Exception:
        return np.array([], dtype=int)


def bsqi(sig, fs, methods=BSQI_METHODS, tol_ms=BSQI_TOL_MS):
    """두 검출기의 박동 일치 비율 (Jaccard). 잡음이 심하면 검출기끼리 어긋난다."""
    r1 = _peaks_by(sig, fs, methods[0])
    r2 = _peaks_by(sig, fs, methods[1])
    if len(r1) == 0 and len(r2) == 0:
        return np.nan
    tp, _, _ = match_peaks(r1, r2, fs, tol_ms)
    union = len(r1) + len(r2) - tp
    return tp / union if union else np.nan


def sqi_all(sig, fs, with_bsqi=True, progress=None):
    """SQI 4종. (n, L) -> {이름: (n,)}. bSQI 는 분절마다 검출을 두 번 하므로 느리다."""
    x = np.asarray(sig, dtype=np.float64)
    out = {name: _band_ratio(x, fs, num, den)
           for name, (num, den) in SQI_BANDS.items()}
    out["kSQI"] = ksqi(x)
    if with_bsqi:
        vals = np.full(len(x), np.nan)
        for i in range(len(x)):
            vals[i] = bsqi(x[i], fs)
            if progress and i % progress == 0:
                print(f"  bSQI {i}/{len(x)}", flush=True)
        out["bSQI"] = vals
    return out


# ================================================================
# [S6] ECGMeanCoef — 박동 형태의 일관성. **참값이 필요 없다**
#
# R-피크 기준으로 박동을 잘라 정렬하고, 그 평균을 템플릿으로 삼아 각 박동과의 상관을
# 평균한다. 박동 모양이 서로 닮을수록 1 에 가깝다. 잡음이 섞이면 박동마다 모양이
# 흐트러져 값이 내려간다. SQI 4종과 같은 층(참값 불필요)에 놓는다.
# ================================================================
BEAT_WIN_MS = (-200.0, 400.0)      # R 기준 박동 창
MIN_BEATS_FOR_TEMPLATE = 3


def ecg_mean_coef(sig, fs, peaks=None, win_ms=BEAT_WIN_MS,
                  min_beats=MIN_BEATS_FOR_TEMPLATE):
    """평균 템플릿과 각 박동의 상관 평균. 박동이 모자라면 NaN."""
    x = np.asarray(sig, dtype=np.float64)
    pk = detect_rpeaks(x, fs) if peaks is None else np.asarray(peaks, dtype=int)
    a, b = int(round(win_ms[0] / 1000 * fs)), int(round(win_ms[1] / 1000 * fs))
    beats = [x[p + a:p + b] for p in pk if p + a >= 0 and p + b <= len(x)]
    if len(beats) < min_beats:
        return np.nan
    M = np.stack(beats)
    tpl = M.mean(0)
    t0 = tpl - tpl.mean()
    d0 = M - M.mean(-1, keepdims=True)
    den = np.sqrt((d0 ** 2).sum(-1) * (t0 ** 2).sum())
    return float(np.mean((d0 * t0).sum(-1) / np.maximum(den, 1e-30)))


# ================================================================
# [비교선] 고전적 디노이징 두 가지
#
# 학습 없이 도는 표준 방법이다. 같은 test 데이터에 그대로 적용해 04 표에 나란히 싣는다.
# ================================================================
BP_BAND = (0.5, 40.0)              # ECG 판독 통상 대역
WAVELET = "sym8"
WAV_LEVEL = 5                      # 표준 임계값용
WAV_LEVEL_BASE = 7                 # 기저선까지 제거할 때 (근사계수를 버린다)


def _restore_mean(y, src):
    """분절 평균을 되돌린다.

    0.5 Hz 고역통과나 근사계수 제거는 DC 오프셋까지 지운다. x_clean 은 오프셋을 갖고
    있으므로(중앙값 −0.28 mV) 그대로 두면 상수 오차가 얹혀 비교가 불공정해진다.
    **입력(x_noisy)의 평균**을 되돌린다 — 참값을 쓰지 않으므로 누수가 없다.
    """
    return y + np.asarray(src, dtype=np.float64).mean(-1, keepdims=True)


def bandpass_denoise(sig, fs, band=BP_BAND, order=4, restore_mean=True):
    """Butterworth 대역통과 + filtfilt(영위상). (..., L) 그대로."""
    from scipy.signal import butter, sosfiltfilt
    x = np.asarray(sig, dtype=np.float64)
    sos = butter(order, band, btype="bandpass", fs=fs, output="sos")
    y = sosfiltfilt(sos, x, axis=-1)
    return _restore_mean(y, x) if restore_mean else y


def wavelet_denoise(sig, wavelet=WAVELET, level=WAV_LEVEL, drop_approx=False,
                    restore_mean=None):
    """웨이블릿 소프트 임계값. 잡음 표준편차는 최고주파 계수의 MAD 로 추정한다.

    임계값은 universal threshold  σ·sqrt(2 ln N)  (Donoho–Johnstone).

    `drop_approx=True` 면 근사계수도 0 으로 둔다 — 기저선 변동이 거기 실려 있어서다.
    임계값만 걸면 세부계수(고주파)만 건드리므로 bw 는 그대로 남는다.
    """
    import pywt
    x = np.atleast_2d(np.asarray(sig, dtype=np.float64))
    out = np.empty_like(x)
    for i, row in enumerate(x):
        c = pywt.wavedec(row, wavelet, level=level)
        sigma = np.median(np.abs(c[-1])) / 0.6745
        thr = sigma * np.sqrt(2.0 * np.log(len(row)))
        head = np.zeros_like(c[0]) if drop_approx else c[0]
        c = [head] + [pywt.threshold(d, thr, mode="soft") for d in c[1:]]
        r = pywt.waverec(c, wavelet)
        out[i] = r[:len(row)]
    out = out.reshape(np.shape(sig))
    keep = drop_approx if restore_mean is None else restore_mean
    return _restore_mean(out, sig) if keep else out


def classical_denoise(sig, fs):
    """비교선 3종. {이름: 복원 신호}. 학습이 없고 x_clean 도 쓰지 않는다."""
    return {
        "대역통과 0.5-40Hz": bandpass_denoise(sig, fs),
        "웨이블릿 임계값": wavelet_denoise(sig, level=WAV_LEVEL),
        "웨이블릿+기저선제거": wavelet_denoise(sig, level=WAV_LEVEL_BASE,
                                          drop_approx=True),
    }
