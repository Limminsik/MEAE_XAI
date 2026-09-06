"""07 — 외부 데이터 적용.

03 에서 확정한 체크포인트를 **재학습 없이** 외부 ECG 에 적용한다. 학습·마스킹·복원 경로를
하나도 바꾸지 않는다. 배정이 정해져 있으므로 외부에서도 B 를 그대로 만든다.

    python 07_validation.py --run C16_seed42 --source vitaldb

════════════════════════════════════════════════════════════════════════
참값이 없다 — 낼 수 있는 것과 없는 것
════════════════════════════════════════════════════════════════════════
외부 데이터의 잡음은 이미 섞여 들어온 것이라 clean/bw/ma/em 처럼 따로 떼어 낸 파형이
없다. 그래서 **04(SNR·PRD)와 06②(임상 지표 오차)는 낼 수 없다.**

**참값이 필요 없는 층은 그대로 낸다** — 그 층은 test 에서 참값과 대조해 검증했으므로
외부에서도 같은 뜻으로 읽을 수 있다.

**판정에 쓸 수 있는 것은 두 층뿐이다.**
  ① SQI 5종        basSQI · pSQI · kSQI · bSQI · ECGMeanCoef      06① 과 같은 정의
                    참값이 원래 필요 없고, 06 에서 참값과 대조해 방향을 검증해 두었다.
                    **07 에서 좋고 나쁨을 말할 수 있는 유일한 층이다.**
  ② 임상 형태 지표    ST60 · ST80 · R진폭 · QRS면적 의 **값**        06② 와 같은 측정 창
                    참값이 없어 오차가 아니다. 입력과 B 를 나란히 두어 이동만 본다.
                    R-피크 기준점은 **입력에서** 검출해 전 계열에 공유한다.

**나머지는 기술(記述)이다 — 성능 근거가 아니다.**
  ③ 스펙트럼 특성    성분별 중심주파수와 대역 전력비
  ④ 복원 차이       B·A·M0·고전 3종이 입력과 얼마나 다른가

**성분 ↔ 참조 대응(03 의 |r|·RMSE_norm·MAD)은 여기서 낼 수 없다.** 03 은 성분을
주입한 참조 파형(x_clean·bw·ma·em)과 대조하는데 외부 데이터에는 그 파형이 없다.
참조 자리에 입력·M0·B 를 넣으면 이름만 같고 다른 양이 되므로 산출하지 않는다.

고전 비교선 3종(대역통과·웨이블릿 임계값·웨이블릿+기저선제거)은 SQI 표에 함께 넣는다.
다만 **방법 비교 자체는 06 에서 한다** — "어느 쪽이 참값에 더 가까운가"는 참값이 있어야
성립하고, 여기 상자그림은 입력과 B 둘만 둔다.

NSTDB 잡음 조각을 참조로 붙이는 방식은 쓰지 않는다. 외부 데이터에 실제로 섞인 잡음과
무관한 파형이라 상관값이 "대역이 비슷하다"만 반영하고 파형 일치를 뜻하지 않는다.

**정량 성능은 주장하지 않는다. 해석도 붙이지 않는다.**

════════════════════════════════════════════════════════════════════════
데이터 규격 (실측)
════════════════════════════════════════════════════════════════════════
  vitaldb     SNUADC/ECG_II        500 Hz, mV        -> 18/25 로 정확히 360 Hz
  mimic_iv    유도 II              249.89 Hz, mV     -> 36/25 (359.84 Hz, 오차 0.045%)
  galaxyppg   Polar H10 ECG.csv    130 Hz, uV        -> 36/13 로 정확히 360 Hz
                                   단위는 데이터셋 README 가 `ecg (int, uV)` 로 명시 -> /1000

**분절 선정은 사전 규칙**이다 — 지표를 보지 않는다. 다음을 버린다.
  기록 앞 10분 · NaN · 평탄비 > 0.05 · 포화비 > 0.005 · R-피크 < 5 ·
  심박수가 30~200 bpm 밖 · RR 변동계수 > 0.5
  남은 것 중 각 기록의 앞에서부터 순서대로 필요한 개수만 취한다.
  포화비 상한은 VitalDB 40기록 2,400창을 조사해 80.5% 가 남는 값으로 정했다.

**진폭 정합** — 모델은 mV 원값을 그대로 받도록 학습됐고 정규화 단계가 없다. 그 전제는
MIT-BIH 안에서만 성립한다. 외부 데이터는 규모가 다르므로 분절마다 상수배해 학습 입력
SD 중앙값(0.4707 mV)에 맞춘 뒤 통과시킨다. 배율은 `segments.csv` 의 `정합배율` 열에
남는다. `--no-scale` 로 끌 수 있다.

────────────────────────────────────────────────────────────────────────
산출물  results/07_validation/<run>/<source>/
────────────────────────────────────────────────────────────────────────
  [판정] sqi_summary.csv          ① SQI 5종 — 계열별 (입력·B·A·M0·고전 3종)
  [판정] morphology.csv           ② 지표 값·IQR·입력 대비 이동 (오차가 아니다)
  [판정] morphology_beats.csv     ② 박동 x 계열 — 지표 원값
  [판정] morphology_by_record.csv ② 기록별 중앙값 (06 의 error_by_record 와 같은 구성)
  [기술] spectrum.csv             ③ 성분별 중심주파수와 대역 전력비
  [기술] restore.csv              ④ 복원 5종이 입력과 얼마나 다른가
  [기술] segments.csv             사용한 분절 목록과 정합 배율
  note.txt · console.log
  figures/components_top{1..10}.png   입력·재구성·성분 K개
  figures/overlay_top{1..10}.png      입력 위에 B 겹침 + 빠져나간 양
  figures/spectrum.png                성분별 PSD
  figures/sqi_box.png                 ① SQI 가로 상자 (입력 · B 둘만)
  figures/morphology_box.png          ② 임상 형태 지표 가로 상자 (입력 · B 둘만)
      방법 비교(고전 비교선)는 06 에서 한다 — 참값이 있어야 성립하는 비교다.

  --survey 는 분절 품질 조사만 수행한다 (기준을 정하기 전 단계).
"""
import argparse
import os

import numpy as np
import pandas as pd
import torch
from scipy.signal import resample_poly, welch

from src import metrics
from src.core import enc_names, load_ckpt, pearson, znorm
from src.data.build import load_cfg
from src.data.dataset import load as load_split
from src.model import meae
from src.viz import plt

FS = 360
SQI_KEYS = ("basSQI", "pSQI", "kSQI", "bSQI", "ECGMeanCoef")
EDGE_MS = 400.0        # 분절 가장자리 — 측정 창이 잘리는 박동은 뺀다
NL = chr(10)           # 그림 제목 줄바꿈 — 이스케이프 사고를 피한다


def _ms(v, fs=FS):
    return int(round(v / 1000.0 * fs))


def _load_06():
    """06 의 측정 창·함수를 그대로 쓴다 — 정의가 갈라지면 비교가 깨진다."""
    import importlib.util
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "06_ablation.py")
    spec = importlib.util.spec_from_file_location("s06", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod
SEG = 3600
BANDS = [(0.5, 5), (5, 15), (15, 25), (25, 40), (40, 60), (60, 90)]

# 원 표본율 → (up, down). 정확히 떨어지지 않는 것은 근사비이며 note 에 기록한다.
RESAMPLE = {500: (18, 25), 250: (36, 25), 130: (36, 13), 125: (72, 25), 360: (1, 1)}


# ---------------------------------------------------------------- 로더
def _vitaldb(cfg, n_records):
    """VitalDB — SNUADC/ECG_II, 500 Hz, mV."""
    import vitaldb
    root = os.path.join(cfg["external"]["vitaldb"]["root"], "vitaldb_dataset")
    files = sorted(os.listdir(root), key=lambda s: int(s.split(".")[0]))
    got = 0
    for f in files:
        if got >= n_records:
            break
        try:
            vf = vitaldb.VitalFile(os.path.join(root, f),
                                   track_names=["SNUADC/ECG_II"])
            x = vf.to_numpy(["SNUADC/ECG_II"], interval=1 / 500).ravel()
        except Exception as e:                       # 트랙이 없는 파일은 건너뛴다
            print(f"  [건너뜀] {f}: {e}")
            continue
        if x.size < 5 * SEG:
            continue
        got += 1
        yield f.split(".")[0], np.asarray(x, np.float64), 500


def _mimic_iv(cfg, n_records):
    """MIMIC-IV Waveform — 유도 II, 249.89 Hz, mV.

    프레임당 4샘플이므로 `smooth_frames=False` 로 읽어야 네이티브 속도가 나온다.
    """
    import wfdb
    root = cfg["external"]["mimic_iv"]["root"]
    with open(os.path.join(root, "RECORDS"), encoding="utf-8") as fh:
        recs = [l.strip().rstrip("/") for l in fh if l.strip()]
    got = 0
    for rel in recs:
        if got >= n_records:
            break
        d = os.path.join(root, rel)
        if not os.path.isdir(d):
            continue
        subs = [s for s in os.listdir(d) if os.path.isdir(os.path.join(d, s))]
        if not subs:
            continue
        d = os.path.join(d, subs[0])
        segs = sorted(f[:-4] for f in os.listdir(d)
                      if f.endswith(".hea") and "_" in f)
        for sname in segs:
            p = os.path.join(d, sname)
            try:
                h = wfdb.rdheader(p)
                if "II" not in (h.sig_name or []) or h.sig_len < int(h.fs * 60 * 15):
                    continue
                i = h.sig_name.index("II")
                # 앞 10분을 건너뛰므로 그보다 넉넉히 읽는다 (프레임 기준 40분)
                r = wfdb.rdrecord(p, smooth_frames=False,
                                  sampto=min(h.sig_len, int(h.fs * 60 * 40)))
                fs_i = r.fs * r.samps_per_frame[i]
                x = np.asarray(r.e_p_signal[i], np.float64)
            except Exception:
                continue
            if x.size < 5 * SEG:
                continue
            got += 1
            yield f"{rel.split('/')[-1]}_{sname}", x, fs_i
            break


def _galaxyppg(cfg, n_records):
    """GalaxyPPG — Polar H10 ECG.csv, 130 Hz.

    데이터셋 README 가 `ecg (int, μV)` 로 명시한다. **1000으로 나눠 mV 로 바꾼다.**
    변환 후 R파 첨두간이 0.85~1.58 mV 로 MIT-BIH(1.57 mV) 범위에 들어오는 것을 확인했다.
    """
    root = cfg["external"]["galaxyppg"]["root"]
    subs = sorted(d for d in os.listdir(root) if d.startswith("P"))[:n_records]
    for s in subs:
        p = os.path.join(root, s, "PolarH10", "ECG.csv")
        if not os.path.exists(p):
            continue
        t = pd.read_csv(p)
        yield s, t["ecg"].to_numpy(np.float64) / 1000.0, 130


LOADERS = {"vitaldb": _vitaldb, "mimic_iv": _mimic_iv, "galaxyppg": _galaxyppg}


def to_360(x, fs_src):
    if fs_src == FS:
        return x
    key = min(RESAMPLE, key=lambda k: abs(k - fs_src))
    if abs(key - fs_src) / fs_src > 0.02:
        raise ValueError(f"표본율 {fs_src} Hz 의 리샘플 비가 등록돼 있지 않다")
    up, down = RESAMPLE[key]
    return resample_poly(x, up, down)


def training_sd(cfg):
    """학습 입력(x_noisy)의 분절 SD 중앙값. 외부 데이터를 여기에 맞춘다.

    모델은 mV 원값을 그대로 받도록 학습됐고 정규화 단계가 없다(S1 동결).
    그 전제는 MIT-BIH 안에서만 성립한다 — 외부 데이터는 진폭 규모가 다르므로
    분절마다 상수배해 학습 때 본 규모로 맞춘 뒤 통과시킨다.
    """
    tr = load_split(cfg, "train")
    return float(np.median(tr.x_noisy.astype(np.float64).std(-1)))


# 품질 선별 기준 — **결과를 보기 전에 정한다.**
#   기록 앞부분은 전극 부착·장비 안정화 구간이라 건너뛴다.
#   포화비 상한 0.005 는 VitalDB 40기록 2,400창 조사에서 80.5% 가 남는 값이다
#   (40기록 전부에서 창이 남는다).
SKIP_MIN = 10.0        # 기록 앞에서 건너뛸 분
CLIP_MAX = 0.005       # 창의 상·하한 1% 안에 몰린 표본 비율의 상한
FLAT_MAX = 0.05        # 인접 표본 차이가 0에 가까운 비율의 상한
HR_RANGE = (30.0, 200.0)
MIN_PEAKS = 5          # 10초에 5개 = 30 bpm
RR_CV_MAX = 0.5        # RR 변동계수 상한 — 검출이 흐트러진 창을 거른다


def cut(x, fs_src, need, skip_min=SKIP_MIN):
    """품질 선별을 거쳐 10초 분절을 자른다. 지표를 보지 않는다.

    버리는 조건
      NaN 포함 · 평탄(표준편차 0 또는 평탄비 초과) · 포화비 초과 ·
      R-피크가 너무 적음 · 심박수가 범위 밖 · RR 변동계수 초과
    남은 것 중 앞에서부터 need 개를 취한다.
    """
    from src.metrics import detect_rpeaks
    y = to_360(x, fs_src)
    start = int(skip_min * 60 * FS)
    if len(y) - start < SEG:                     # 기록이 짧으면 건너뛰지 않는다
        start = 0
    out = []
    drop = {"nan": 0, "flat": 0, "포화": 0, "R피크": 0, "HR": 0, "RR": 0}
    for s in range(start, len(y) - SEG + 1, SEG):
        w = y[s:s + SEG]
        if np.isnan(w).any():
            drop["nan"] += 1
            continue
        lo, hi = w.min(), w.max()
        span = max(hi - lo, 1e-12)
        if w.std() < 1e-6 or (np.abs(np.diff(w)) < 1e-9 * span).mean() > FLAT_MAX:
            drop["flat"] += 1
            continue
        if ((w > hi - 0.01 * span) | (w < lo + 0.01 * span)).mean() > CLIP_MAX:
            drop["포화"] += 1
            continue
        try:
            pk = detect_rpeaks(w, FS)
        except Exception:
            pk = np.array([], dtype=int)
        if len(pk) < MIN_PEAKS:
            drop["R피크"] += 1
            continue
        rr = np.diff(pk) / FS
        hr = 60 / np.median(rr)
        if not (HR_RANGE[0] <= hr <= HR_RANGE[1]):
            drop["HR"] += 1
            continue
        if np.std(rr) / np.mean(rr) > RR_CV_MAX:
            drop["RR"] += 1
            continue
        out.append((s, w, float(hr)))
        if len(out) >= need:
            break
    return out, drop


# ---------------------------------------------------------------- 표·그림
# ================================================================
# [제거됨] 성분 × (입력, M0, B) 의 |r|·RMSE_norm·MAD 표
#
# 03 은 성분을 **주입한 참조**(x_clean·bw·ma·em)와 대조해 "enc2 가 정말 bw 를
# 담았는가"를 잰다. 외부 데이터에는 그 참조 파형이 없으므로 같은 것을 잴 수 없다.
# 여기서 내던 표는 참조 자리에 입력·M0·B 를 넣은 것이라 이름만 같고 다른 양이었다.
#
# 게다가 B = component(x, enc1) 이라 `enc1 × B복원` 칸이 항등이었다
# (corr 1.0000±0.0000 · MAD 0.0000±0.0000) — 자기 자신과의 상관이다.
#
# 성분의 성격은 `spectrum.csv`(중심주파수·대역 전력비)로, 복원이 입력을 얼마나
# 바꿨는지는 `restore.csv` 로 본다. 둘 다 참조 없이 성립한다.
# ================================================================
def _profile(a):
    f, P = welch(a, fs=FS, nperseg=1024, axis=-1)
    tot = P.sum(-1)
    r = {"중심주파수_Hz": float(np.median((f * P).sum(-1) / tot))}
    for lo, hi in BANDS:
        m = (f >= lo) & (f < hi)
        r[f"{lo}-{hi}Hz"] = float(np.median(P[..., m].sum(-1) / tot))
    return r


def spectrum(comps, inp, rec, ix, outdir):
    """성분별 중심주파수와 대역 전력비. 참조 없이 계산되는 성격 지표다."""
    rows = [{"대상": "입력", **_profile(inp)}, {"대상": "M0 재구성", **_profile(rec)}]
    rows += [{"대상": ix[k], **_profile(comps[:, k])} for k in range(len(ix))]
    t = pd.DataFrame(rows).round(4)
    t.to_csv(f"{outdir}/spectrum.csv", index=False, encoding="utf-8-sig")
    return t


STRIP_MAX = 400          # 상자 위에 찍는 관측점 상한 — 이보다 많으면 균일 추출한다


def fig_box(data, series, keys, out, head, sub, per_beat=False, whis=(1, 99),
            points=False):
    """가로 상자그림 — 06 의 `sqi_box`·`error_box` 와 같은 형식이다.

    관측점은 기본으로 끄다(`points=True` 로 켜면 400개 균일 추출).
    계열이 7개라 점까지 찍으면 상자가 가려 위치 비교가 어려워진다.

    계열이 7개(입력·B·A·M0·고전 3종)라 세로로 세우면 이름이 겹친다. 가로로 두면
    계열 이름이 축에 그대로 들어가고 **분포의 위치 차이**가 한눈에 비교된다.

    참값이 없으므로 여기 그려지는 것은 오차가 아니라 **값**이다. 계열 간 위치 차이가
    곧 처리로 생긴 이동이며, 그 이동이 옳은 방향인지는 이 그림이 말하지 않는다.

    `per_beat=True` 면 `data` 는 박동 표(DataFrame, 열 이름 `<지표>__<계열>`)이고,
    아니면 분절별 SQI dict(`{계열: {지표: 배열}}`)다.
    """
    import seaborn as sns
    sns.set_theme(style="ticks", font=plt.rcParams["font.family"][0])
    rng = np.random.default_rng(0)
    pal = sns.color_palette("vlag", len(series))
    fig, ax = plt.subplots(len(keys), 1, figsize=(9.5, 2.35 * len(keys)))
    ax = np.atleast_1d(ax)
    for a, k in zip(ax, keys):
        cols, vals = [], []
        for s in series:
            v = (data[f"{k}__{s}"].to_numpy(dtype=np.float64) if per_beat
                 else np.asarray(data[s][k], dtype=np.float64))
            v = v[np.isfinite(v)]
            vals.append(v)
            cols.append(np.full(len(v), s))
        df = pd.DataFrame({"값": np.concatenate(vals), "계열": np.concatenate(cols)})
        sns.boxplot(df, x="값", y="계열", hue="계열", order=series, hue_order=series,
                    whis=list(whis), width=.6, palette=pal, legend=False,
                    fliersize=0, ax=a)
        if points:
            keep = []
            for s in series:
                m = np.flatnonzero((df["계열"] == s).to_numpy())
                keep.append(m if len(m) <= STRIP_MAX
                            else rng.choice(m, STRIP_MAX, replace=False))
            sns.stripplot(df.iloc[np.concatenate(keep)], x="값", y="계열",
                          order=series, size=2.2, color=".3", alpha=.35,
                          jitter=.28, ax=a)
        lo = min(np.percentile(v, whis[0]) for v in vals if len(v))
        hi = max(np.percentile(v, whis[1]) for v in vals if len(v))
        pad = (hi - lo) * 0.06 or 1.0
        a.set_xlim(lo - pad, hi + pad)
        # 입력을 세로 점선으로 — 나머지 계열이 어디로 옮겨 갔는지 기준이 된다
        a.axvline(np.nanmedian(vals[0]), color="#1f77b4", ls="--", lw=1.1)
        a.set_title(f"{k}   (점선 = 입력 중앙값 {np.nanmedian(vals[0]):.4g})",
                    fontsize=9.5, loc="left")
        a.xaxis.grid(True, alpha=.35, lw=.4)
        a.set(ylabel="", xlabel="")
        a.tick_params(labelsize=8)
        sns.despine(ax=a, trim=True, left=True)
    fig.suptitle(f"{head}{NL}{sub}{NL}"
                 f"수염과 축은 {whis[0]}-{whis[1]} 백분위", fontsize=11)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, bbox_inches="tight", dpi=130)
    plt.close(fig)
    sns.reset_orig()


def fig_spectrum(comps, inp, rec, ix, out, source):
    f, Pi = welch(inp, fs=FS, nperseg=1024, axis=-1)
    _, Pr = welch(rec, fs=FS, nperseg=1024, axis=-1)
    fig, ax = plt.subplots(figsize=(10, 5.4))
    ax.loglog(f[1:], np.median(Pi, 0)[1:], lw=2, color="#000", label="입력")
    ax.loglog(f[1:], np.median(Pr, 0)[1:], lw=1.6, color="#d62728", label="M0 재구성")
    for k in range(comps.shape[1]):
        _, P = welch(comps[:, k], fs=FS, nperseg=1024, axis=-1)
        ax.loglog(f[1:], np.median(P, 0)[1:], lw=1, alpha=.85, label=ix[k])
    ax.set_xlabel("주파수 (Hz)")
    ax.set_ylabel("PSD (분절 중앙값)")
    ax.grid(alpha=.3, lw=.4, which="both")
    ax.legend(fontsize=8, ncol=2)
    ax.set_title(f"성분별 PSD — {source}", fontsize=11, loc="left")
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def fig_overlay(x, a_sig, out, title, unit, fs=FS):
    """[겹침] 입력 위에 A 성분차감. 참값이 없으므로 **차이를 보는 그림**이다.

    05 와 같은 형식이되 위 칸의 기준선이 x_clean 이 아니라 입력 자체다.
    아래 칸은 빠져나간 양(x - A = 잡음으로 지목된 성분들의 합)이다.
    """
    t = np.arange(len(x)) / fs
    lim = max(np.abs(x).max(), np.abs(a_sig).max()) * 1.08
    fig, ax = plt.subplots(2, 1, figsize=(13, 5.4), sharex=True, sharey=True)
    ax[0].plot(t, x, lw=0.75, color="#000", alpha=.8, label="입력 x")
    ax[0].plot(t, a_sig, lw=0.9, color="#c44e52", label="A 성분차감")
    ax[0].legend(fontsize=8, ncol=2, loc="upper right")
    ax[0].set_title("입력 위에 B 복원 겹침", fontsize=9, loc="left")
    ax[1].plot(t, x - a_sig, lw=0.7, color="#777")
    ax[1].set_title("빠져나간 양  x - B", fontsize=9, loc="left")
    for b in ax:
        b.set_ylim(-lim, lim)
        b.set_ylabel(unit, fontsize=8)
        b.grid(alpha=.28, lw=.4)
        b.tick_params(labelsize=7)
    ax[-1].set_xlabel("시간 (초)", fontsize=8)
    fig.suptitle(title + f"   ·   두 칸 같은 y 범위 (±{lim:.2f} {unit})", fontsize=11)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight", dpi=130)
    plt.close(fig)


def fig_components(comps, inp, rec, i, out, title, unit):
    """입력 · 재구성 · 성분 K개. 참조 행이 없다는 점만 03과 다르다."""
    K = comps.shape[1]
    zc = znorm(comps[i])
    rows = [(f"입력  [{unit}]", inp[i], "#000", False),
            (f"재구성 x_hat  [{unit}]", rec[i], "#d62728", False)]
    rows += [(f"성분 {k+1}", zc[k], "#1f77b4", True) for k in range(K)]
    lim = max(np.abs(v).max() for _, v, _, z in rows if z) * 1.05
    mv = max(np.abs(v).max() for _, v, _, z in rows if not z) * 1.05
    t = np.arange(comps.shape[-1]) / FS
    fig, ax = plt.subplots(len(rows), 1, figsize=(11, 1.0 * len(rows)), sharex=True)
    for a, (lb, v, c, z) in zip(ax, rows):
        a.plot(t, v, lw=.6, color=c)
        a.set_title(lb, fontsize=8, loc="left")
        a.grid(alpha=.25, lw=.4)
        a.tick_params(labelsize=7)
        a.set_ylim(*((-lim, lim) if z else (-mv, mv)))
        a.set_ylabel("z" if z else unit, fontsize=7)
    ax[-1].set_xlabel("시간 (초)")
    fig.suptitle(title + "\n성분은 분절 내 표준화 후 공통 y축. 입력·재구성만 원값",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- main
def main(config="configs/default.yaml", run="C16_seed42", source="vitaldb",
         n_records=10, n_seg=200, scale=True, n_fig=10, outdir=None):
    cfg = load_cfg(config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    outdir = outdir or os.path.join("results", "07_validation", run, source)
    figdir = os.path.join(outdir, "figures")
    os.makedirs(figdir, exist_ok=True)
    if source not in LOADERS:
        raise SystemExit(f"[07] 알 수 없는 source: {source}. {list(LOADERS)}")

    model, ck = load_ckpt(cfg, run)
    model = model.to(device)
    K = model.n_encoders
    ix = enc_names(K)
    unit = "mV"                                  # 세 소스 모두 mV 로 맞춰 읽는다
    print(f"[07] {run} (에폭 {ck['epoch']}) → {source}  재학습 없음")

    tsd = training_sd(cfg) if scale else None
    if scale:
        print(f"  진폭 정합: 분절마다 SD 를 학습 입력 중앙값 {tsd:.4f} mV 에 맞춘다")

    segs, meta, fs_seen = [], [], set()
    per = max(1, n_seg // max(n_records, 1))
    for rid, x, fs_src in LOADERS[source](cfg, n_records):
        got, drop = cut(x, fs_src, per)
        fs_seen.add(round(float(fs_src), 2))
        for s, w, hr in got:
            sd0 = float(w.std())
            g = (tsd / sd0) if (scale and sd0 > 0) else 1.0
            segs.append(w * g)
            meta.append({"기록": rid, "시작표본_360Hz": s, "원_표본율": fs_src,
                         "원_SD": round(sd0, 5), "정합배율": round(g, 5),
                         "HR": round(hr, 1)})
        print(f"  {rid}: {len(got)}분절 채택  (버림 "
              + " · ".join(f"{k} {v}" for k, v in drop.items() if v) + ")", flush=True)
        if len(segs) >= n_seg:
            break
    if not segs:
        raise SystemExit(f"[05] {source} 에서 쓸 분절을 찾지 못했다")
    X = np.stack(segs[:n_seg])
    meta = meta[:len(X)]
    pd.DataFrame(meta).to_csv(f"{outdir}/segments.csv", index=False,
                              encoding="utf-8-sig")
    print(f"  총 {len(X)}분절 · 원 표본율 {sorted(fs_seen)}")

    # ---- 성분·재구성 (추론만)
    pad = model.pad_each
    comps, rec = [], []
    with torch.no_grad():
        for s in range(0, len(X), 100):
            xb = meae.pad(torch.tensor(X[s:s + 100], dtype=torch.float32
                                       ).unsqueeze(1).to(device), pad)
            comps.append(torch.stack(
                [meae.crop(model.component(xb, k), pad).squeeze(1) for k in range(K)],
                1).cpu().numpy().astype(np.float64))
            rec.append(meae.crop(model(xb)[0], pad).squeeze(1).cpu().numpy()
                       .astype(np.float64))
    comps, rec = np.concatenate(comps), np.concatenate(rec)

    # ---- 복원 — 배정이 정해져 있으므로 외부에서도 그대로 만든다.
    # 주 노선은 B(잡음 인코딩을 끈 재구성)다. A 는 비교선, 고전 3종도 함께 낸다.
    sup = list(cfg["loss"]["supervise"])
    k_clean = sup.index("x_clean")
    k_noise = [k for k in range(K) if k != k_clean]
    A = X - comps[:, k_noise].sum(1)
    B = np.zeros_like(X)
    with torch.no_grad():
        for s0 in range(0, len(X), 100):
            xb = meae.pad(torch.tensor(X[s0:s0 + 100], dtype=torch.float32
                                       ).unsqueeze(1).to(device), pad)
            B[s0:s0 + 100] = meae.crop(model.masked_reconstruct(xb, k_noise), pad
                                       ).squeeze(1).cpu().numpy().astype(np.float64)
    classic = metrics.classical_denoise(X, FS)
    cand = {"B 복원": B, "A 성분차감": A, "M0 재구성": rec, **classic}

    rows = []
    for name, v in cand.items():
        c0 = X - X.mean(-1, keepdims=True)
        v0 = v - v.mean(-1, keepdims=True)
        d = np.sqrt((c0 ** 2).sum(-1) * (v0 ** 2).sum(-1))
        r = np.abs((c0 * v0).sum(-1) / np.maximum(d, 1e-30))
        rows.append({"대상": name, "분절수": len(X),
                     "입력과_corr": float(r.mean()), "corr_sd": float(r.std(ddof=1)),
                     "입력과_RMSE_mV": float(np.sqrt(((X - v) ** 2).mean(-1)).mean()),
                     "SD_mV": float(v.std(-1).mean()),
                     "SD비_입력대비": float((v.std(-1) / X.std(-1)).mean())})
    rows.append({"대상": "입력 x", "분절수": len(X), "입력과_corr": 1.0, "corr_sd": 0.0,
                 "입력과_RMSE_mV": 0.0, "SD_mV": float(X.std(-1).mean()),
                 "SD비_입력대비": 1.0})
    restore = pd.DataFrame(rows).round(4)
    restore.to_csv(f"{outdir}/restore.csv", index=False, encoding="utf-8-sig")

    # ---- SQI 5종 — **참값이 없어도 잰다.** 07 의 핵심 지표다.
    print("[07] SQI 5종 — 참값 없이 신호 하나만 보고 잰다 (bSQI 는 검출을 두 번 한다)")
    fid = [metrics.detect_rpeaks(x, FS) for x in X]      # 기준점은 **입력**에서 한 번
    sq, sqi_raw = [], {}
    for name, v in [("입력 x", X)] + list(cand.items()):
        q = metrics.sqi_all(v, FS, progress=200)
        q["ECGMeanCoef"] = np.array(
            [metrics.ecg_mean_coef(v[t], FS, peaks=fid[t]) for t in range(len(X))])
        sqi_raw[name] = q                    # 분절별 원값 — 상자그림이 쓴다
        row = {"계열": name, "분절수": len(X)}
        for k in SQI_KEYS:
            row[k] = float(np.nanmedian(q[k]))
            row[f"{k}_평균"] = float(np.nanmean(q[k]))
        sq.append(row)
    sqi = pd.DataFrame(sq)
    sqi.round(5).to_csv(f"{outdir}/sqi_summary.csv", index=False, encoding="utf-8-sig")

    # ---- 임상 형태 지표 — 참값이 없으므로 **오차가 아니라 값과 이동**이다.
    #
    # 06② 는 x_clean 이 있어 박동별 오차를 낼 수 있지만 여기는 참값이 없다. 대신
    # **같은 기준점·같은 창으로 잰 값**을 계열마다 나란히 두고, 입력에서 B 로 갈 때
    # 값이 어느 쪽으로 얼마나 옮겨 갔는지를 본다. 06 에서는 그 이동의 방향이 참값 쪽인지
    # 확인돼 있으므로, 외부에서 같은 방향의 이동이 나오는지가 볼거리다.
    # **이동을 성능으로 읽지 않는다** — 참값이 없으므로 옳고 그름을 말할 수 없다.
    S06 = _load_06()
    series = [("입력 x", X)] + list(cand.items())
    rec_id = np.array([m["기록"] for m in meta])
    beat_rows = []
    for t in range(len(X)):
        pk = fid[t][(fid[t] >= _ms(EDGE_MS)) & (fid[t] < X.shape[1] - _ms(EDGE_MS))]
        if len(pk) == 0:
            continue
        per = {name: S06.beat_measures(v[t], pk, FS) for name, v in series}
        for bi, r in enumerate(pk):
            row = {"기록": rec_id[t], "분절": t, "박동위치": int(r)}
            for name, _ in series:
                for k in S06.METRICS:
                    row[f"{k}__{name}"] = per[name][bi][k]
            beat_rows.append(row)
    mbeats = pd.DataFrame(beat_rows)
    mbeats.round(5).to_csv(f"{outdir}/morphology_beats.csv", index=False,
                           encoding="utf-8-sig")

    # 계열별 요약 — 중앙값에 산포(IQR)를 더한다. 값 하나로는 분포를 알 수 없다.
    mrows = []
    for name, _ in series:
        row = {"계열": name, "박동수": int(mbeats[f"{S06.METRICS[0]}__{name}"].notna().sum())}
        for k in S06.METRICS:
            v = mbeats[f"{k}__{name}"]
            row[k] = float(v.median())
            row[f"{k}_IQR"] = float(v.quantile(.75) - v.quantile(.25))
            if name != "입력 x":                       # 입력 대비 이동 (오차가 아니다)
                row[f"{k}_입력대비"] = float((v - mbeats[f"{k}__입력 x"]).median())
        mrows.append(row)
    morph = pd.DataFrame(mrows)
    morph.round(5).to_csv(f"{outdir}/morphology.csv", index=False, encoding="utf-8-sig")

    # 기록별 중앙값 — 06 의 error_by_record 와 같은 구성이다 (오차 대신 값).
    mrec = []
    for r in sorted(set(rec_id)):
        m = mbeats["기록"] == r
        row = {"기록": r, "박동수": int(m.sum())}
        for k in S06.METRICS:
            for name, _ in series:
                row[f"{k}_{name}"] = float(mbeats.loc[m, f"{k}__{name}"].median())
        mrec.append(row)
    morph_rec = pd.DataFrame(mrec)
    morph_rec.round(5).to_csv(f"{outdir}/morphology_by_record.csv", index=False,
                              encoding="utf-8-sig")

    # 상자그림은 **입력과 B 둘만** 둔다. 방법 비교(고전 비교선)는 06 에서 한다 —
    # 참값이 있어야 "어느 쪽이 더 가깝다"를 말할 수 있고, 여기에는 참값이 없다.
    # 여기서 볼 것은 우리 처리가 입력을 어느 쪽으로 얼마나 옮겼는가뿐이다.
    pair = ["입력 x", "B 복원"]
    fig_box(mbeats, pair, S06.METRICS,
            f"{figdir}/morphology_box.png",
            f"[07 임상 형태 지표] {source} · {run}",
            f"박동 {len(mbeats):,}개 · 기록 {len(morph_rec)}개 — 참값이 없어 오차가 "
            "아니라 값이다. 두 상자의 위치 차이가 곧 이동", per_beat=True)
    fig_box(sqi_raw, pair, SQI_KEYS,
            f"{figdir}/sqi_box.png",
            f"[07 신호 품질 지수] {source} · {run}",
            f"분절 {len(X):,}개 — 참값이 필요 없는 지표다. basSQI 만 낮을수록 좋다")

    spec = spectrum(comps, X, rec, ix, outdir)
    fig_spectrum(comps, X, rec, ix, f"{figdir}/spectrum.png", source)

    # ---- 그림 분절: 재구성이 입력과 가장 닮은 n_fig 개 (결과 기반임을 제목에 밝힌다)
    score = np.abs(pearson(rec[:, None], X[:, None])[:, 0, 0])
    top = list(np.argsort(-score)[:n_fig])
    for rank, i in enumerate(top, 1):
        m = meta[i]
        head = (f"{source} · 기록 {m['기록']} · 시작 {m['시작표본_360Hz']} "
                f"— 재구성 유사도 상위 {rank}위 (|r| {score[i]:.3f})")
        fig_components(comps, X, rec, i, f"{figdir}/components_top{rank}.png",
                       head, unit)
        fig_overlay(X[i], B[i], f"{figdir}/overlay_top{rank}.png", head, unit)

    with open(f"{outdir}/note.txt", "w", encoding="utf-8") as f:
        f.write(
            f"05 외부 적용 — {source}, {run} (에폭 {ck['epoch']}), 재학습 없음.\n\n"
            f"분절 {len(X)}개, 원 표본율 {sorted(fs_seen)} Hz -> {FS} Hz 리샘플, "
            f"10초(3600표본), 단위 {unit}.\n"
            "분절 선정은 사전 규칙이다 - 기록 앞 %g분 / NaN / 평탄비 > %g /\n"
            "포화비 > %g / R-피크 < %d / 심박수가 %g~%g bpm 밖 / RR 변동계수 > %g\n"
            "인 창을 버리고, 남은 것 중 각 기록의 앞에서부터 순서대로 취했다.\n"
            "지표를 보지 않았다.\n"
            % (SKIP_MIN, FLAT_MAX, CLIP_MAX, MIN_PEAKS,
               HR_RANGE[0], HR_RANGE[1], RR_CV_MAX)
            + (f"진폭 정합: 분절마다 SD 를 학습 입력 중앙값 {tsd:.4f} mV 에 맞췄다.\n"
               "  모델은 mV 원값을 그대로 받도록 학습됐고 정규화 단계가 없다(S1 동결).\n"
               "  그 전제는 MIT-BIH 안에서만 성립한다. 배율은 segments.csv 에 있다.\n\n"
               if scale else "진폭 정합을 하지 않았다 (--no-scale).\n\n")
            + "참조가 없다 - 무엇을 판정할 수 있고 무엇을 못 하는가.\n\n"
            "  [판정 가능]\n"
            "    sqi_summary.csv   SQI 5종. 참값이 원래 필요 없고, 06에서 참값과\n"
            "                      대조해 방향을 검증해 두었다. 좋고 나쁨을 말할 수\n"
            "                      있는 유일한 층이다.\n"
            "    morphology*.csv   06과 같은 측정 창으로 잰 값. 참값이 없으므로\n"
            "                      오차가 아니라 값이고, 입력 대비 이동만 본다.\n\n"
            "  [기술만 - 성능 근거가 아니다]\n"
            "    spectrum.csv      성분별 중심주파수와 대역 전력비. 파형을 맞대지\n"
            "                      않고 계산하므로 참조 없이도 성격을 말할 수 있다.\n"
            "    restore.csv       복원이 입력을 얼마나 바꿨는가.\n\n"
            "  [낼 수 없음]\n"
            "    성분 <-> 참조 대응 (03의 |r| / RMSE_norm / MAD 4열 표).\n"
            "    외부 데이터의 잡음은 이미 섞여 들어온 것이라 clean/bw/ma/em 처럼\n"
            "    따로 떼어 낸 파형이 없다. 참조 자리에 입력이나 M0/B 를 넣으면\n"
            "    이름만 같고 다른 양이 되므로 산출하지 않는다.\n"
            "    (B = component(x, enc1) 이라 enc1 x B 칸은 항등이기도 했다.)\n"
            "    NSTDB 잡음 조각을 참조로 붙이지도 않았다 - 외부 데이터에 실제로\n"
            "    섞인 잡음과 무관한 파형이라 상관값이 파형 일치를 뜻하지 않는다.\n\n"
            f"그림의 분절은 재구성이 입력과 가장 닮은 {len(top)}개다 - "
            "결과를 보고 고른 사례다.\n\n"
            "정량 성능은 주장하지 않는다. 해석도 붙이지 않는다.\n")

    pd.set_option("display.width", 240)
    print("")
    print(f"=== 07 {source} · {len(X)}분절 ===")
    print("[판정 가능] 참값 없이도 좋고 나쁨을 말할 수 있는 층")
    print("")
    print("① SQI 5종 — basSQI 는 낮을수록, 나머지는 높을수록 좋다")
    print(sqi[["계열", "분절수"] + list(SQI_KEYS)].round(4).to_string(index=False))
    print("")
    print("② 임상 형태 지표 — 참값이 없어 오차가 아니라 값이다. 입력 대비 이동만 본다")
    print(morph.round(4).to_string(index=False))
    print("")
    print("[기술] 성능 근거가 아니다 — 무엇이 어떻게 달라졌는지만 적는다")
    print("")
    print("③ 스펙트럼 특성 — 성분별 중심주파수와 대역 전력비")
    print(spec.to_string(index=False))
    print("")
    print("④ 복원 차이 — 각 방식이 입력과 얼마나 다른가")
    print(restore.to_string(index=False))
    print("")
    print("[낼 수 없음] 성분 ↔ 참조 대응 — 외부에는 주입한 참조 파형이 없다")
    print("")
    print(f"산출물 → {outdir}/")
    return sqi, morph, spec


# ================================================================
# 분절 점검 — 어떤 구간이 쓸 만한지 훑어본다. 품질 기준을 정하기 전 단계다.
#   기록 앞부분(전극 부착·장비 안정화)을 건너뛰고, 창마다 품질 지표를 산출한다.
# ================================================================
def quality(w, fs=FS):
    """창 하나의 품질 지표. 판정하지 않고 수치만 낸다."""
    from src.metrics import detect_rpeaks
    z = (w - w.mean()) / max(w.std(), 1e-12)
    lo, hi = w.min(), w.max()
    span = max(hi - lo, 1e-12)
    r = {"SD": float(w.std()),
         "첨두간": float(span),
         "최대_z": float(np.abs(z).max()),
         # 양 끝단 1% 안에 몰린 표본 비율 — 포화·클리핑에서 커진다
         "포화비": float(((w > hi - 0.01 * span) | (w < lo + 0.01 * span)).mean()),
         # 인접 표본 차이가 거의 0인 비율 — 평탄 구간
         "평탄비": float((np.abs(np.diff(w)) < 1e-9 * span).mean())}
    try:
        pk = detect_rpeaks(w, fs)
        rr = np.diff(pk) / fs
        r["R피크수"] = int(len(pk))
        r["HR"] = float(60 / np.median(rr)) if len(rr) else np.nan
        r["RR_변동계수"] = float(np.std(rr) / np.mean(rr)) if len(rr) > 1 else np.nan
    except Exception:
        r.update({"R피크수": 0, "HR": np.nan, "RR_변동계수": np.nan})
    return r


def survey(config="configs/default.yaml", source="vitaldb", n_records=3,
           skip_min=5.0, n_win=40, n_fig=8, outdir=None):
    """기록 앞 skip_min 분을 건너뛰고 창마다 품질 지표를 산출한다.

    선정 규칙을 바꾸지 않는다 — 어떤 구간이 있는지 먼저 보기 위한 점검이다.
    """
    cfg = load_cfg(config)
    outdir = outdir or os.path.join("results", "05_validation", "_check", source)
    os.makedirs(outdir, exist_ok=True)
    rows, keep = [], []
    for rid, x, fs_src in LOADERS[source](cfg, n_records):
        y = to_360(x, fs_src)
        start = int(skip_min * 60 * FS)
        avail = (len(y) - start) // SEG
        if avail < 1:
            start, avail = 0, len(y) // SEG
        step = max(1, avail // n_win)
        for w_i in range(0, avail, step):
            s = start + w_i * SEG
            w = y[s:s + SEG]
            if len(w) < SEG or np.isnan(w).any():
                continue
            q = quality(w)
            rows.append({"기록": rid, "시작표본": s, "시작_분": round(s / FS / 60, 2), **q})
            keep.append((f"{rid}@{s/FS/60:.1f}분", w, q))
            if len(rows) % n_win == 0:
                break
    t = pd.DataFrame(rows).round(4)
    t.to_csv(f"{outdir}/survey.csv", index=False, encoding="utf-8-sig")

    sel = keep[:n_fig]
    fig, ax = plt.subplots(len(sel), 2, figsize=(15, 2.2 * len(sel)))
    ax = np.atleast_2d(ax)
    for i, (lab, w, q) in enumerate(sel):
        from src.metrics import detect_rpeaks
        try:
            pk = detect_rpeaks(w, FS)
        except Exception:
            pk = []
        tt = np.arange(len(w)) / FS
        for c, (a, b) in enumerate([(0, 10), (2, 7)]):
            s_, e_ = int(a * FS), int(b * FS)
            ax[i, c].plot(tt[s_:e_], w[s_:e_], lw=.8, color="#000")
            for p in pk:
                if s_ <= p < e_:
                    ax[i, c].axvline(p / FS, color="#d62728", lw=.6, alpha=.5)
            ax[i, c].set_title(
                f"{lab}  [{a}-{b}초]  HR {q['HR']:.0f} · 포화비 {q['포화비']:.3f} · "
                f"최대z {q['최대_z']:.1f} · RR변동 {q['RR_변동계수']:.2f}",
                fontsize=8, loc="left")
            ax[i, c].grid(alpha=.25, lw=.3)
            ax[i, c].tick_params(labelsize=7)
    for a in ax[-1]:
        a.set_xlabel("시간 (초)")
    fig.suptitle(f"분절 점검 — {source} · 기록 앞 {skip_min:g}분 건너뜀 · "
                 "모델 통과 전, 리샘플만 적용", fontsize=12)
    fig.tight_layout()
    fig.savefig(f"{outdir}/survey.png", dpi=140, bbox_inches="tight")
    plt.close(fig)

    pd.set_option("display.width", 240)
    print(f"=== 분절 점검 · {source} · 창 {len(t)}개 (앞 {skip_min:g}분 건너뜀) ===")
    print(t.to_string(index=False))
    print("\n[분포]")
    print(t[["SD", "첨두간", "최대_z", "포화비", "평탄비", "R피크수", "HR",
             "RR_변동계수"]].describe().round(3).to_string())
    print(f"\n산출물 → {outdir}/")
    return t


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--run", required=True, help="03 에서 확정한 최종 런")
    p.add_argument("--source", default="vitaldb", choices=list(LOADERS))
    p.add_argument("--n-records", dest="n_records", type=int, default=10,
                   help="기록 수 - 보고한 결과는 10기록 x 20분절 = 200분절")
    p.add_argument("--n-seg", dest="n_seg", type=int, default=200)
    p.add_argument("--no-scale", dest="scale", action="store_false",
                   help="진폭 정합 없이 원값 그대로 넣는다")
    p.add_argument("--n-fig", dest="n_fig", type=int, default=10,
                   help="그림으로 낼 분절 수 (재구성 유사도 상위)")
    p.add_argument("--survey", action="store_true",
                   help="분절 점검 — 앞부분을 건너뛰고 창마다 품질 지표를 본다")
    p.add_argument("--skip-min", dest="skip_min", type=float, default=5.0,
                   help="기록 앞에서 건너뛸 분")
    p.add_argument("--outdir", default=None)
    a = p.parse_args()
    if a.survey:
        survey(a.config, a.source, a.n_records, a.skip_min, outdir=a.outdir)
        raise SystemExit
    main(a.config, a.run, a.source, a.n_records, a.n_seg, a.scale, a.n_fig,
         a.outdir)
