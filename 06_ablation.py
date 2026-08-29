"""06 — 임상 형태 지표 대조. x_clean(참값) · x_noisy(처리 전) · A 성분차감(처리 후).

    python 06_ablation.py --run K4_seed42 --split test

디노이징의 효과를 **임상에서 실제로 재는 값**으로 확인한다. 파형 지표(|r|·RMSE·SNR)는
04·05에 있고, 여기서는 그 파형에서 뽑은 진단 지표가 참값과 얼마나 맞는지를 본다.

────────────────────────────────────────────────────────────────────────
대상 셋
────────────────────────────────────────────────────────────────────────
    참값     x_clean
    처리 전   x_noisy
    처리 후   B 재구성 = D(z_clean, 0, 0, 0) — 잡음 인코딩 3개를 0으로 치환

            A(x_noisy - ŝ_bw - ŝ_ma - ŝ_em)는 `--method A` 로 쓸 수 있으나 기본이 아니다.
            A 는 원본 x_noisy 를 그대로 유지한 채 모델 추정치만 빼므로 처리 전과 출발선이
            같지 않다 — 모델이 아무것도 못 뽑아도 x_noisy 만큼은 보장된다.

────────────────────────────────────────────────────────────────────────
기준점을 공유한다 — 이 실험의 핵심 설계
────────────────────────────────────────────────────────────────────────
R-피크는 **x_clean 에서 한 번 검출**해 세 신호에 **똑같이** 적용한다.

신호마다 따로 검출하면 "박동을 찾았는가"(검출 오차)와 "찾은 자리에서 잰 값이 맞는가"
(측정 오차)가 섞인다. 여기서 보려는 것은 뒤쪽이다 — 기준점을 고정하면 박동이 1:1 로
짝지어져 **박동별 오차**를 낼 수 있고, 남는 차이는 온전히 파형 왜곡에서 온다.

검출 성능 자체는 04·05 의 F1 에 있다.

────────────────────────────────────────────────────────────────────────
지표 두 층 — ① SQI(참값 불필요) · ② 임상 형태 지표(참값 대비 오차)
────────────────────────────────────────────────────────────────────────
① ST 분절 편위 — ST60 · ST80 (mV)
    **무엇인가**  QRS 가 끝나는 지점(J 점)에서 60 ms, 80 ms 뒤의 신호 높이를 기저선
    (PQ 구간) 대비로 잰 값이다. 위로 올라가면 ST 상승, 아래로 내려가면 ST 하강이라
    부르고, **심근경색과 허혈 진단의 핵심 지표**다.

    **어떻게 재나**
        기저선  PR 구간 [R-80ms, R-40ms] 의 평균 — 등전위선으로 삼는다
        J 점    R + 40 ms 로 근사한다 (QRS 종료점 검출은 잡음에서 특히 불안정하다)
        측정    ST60 = x[J+60ms] - 기저선,  ST80 = x[J+80ms] - 기저선

    **왜 여기서 보나**  기저선 대비로 재므로 기저선이 흔들리면 그대로 틀린다.
    bw(기저선 변동)가 정확히 그 기저선을 흔든다.

② QT 간격 — **제외한다**
    Q 파 시작부터 T 파 끝까지의 시간이고 길어지면 부정맥 위험이 있는 지표지만, 여기서는
    싣지 않는다. 상용 ECG 판독 프로그램 간 QT 측정 차이가 6-10 ms 인데 우리 절대오차는
    47-64 ms 였다. **지표 자체의 판별력이 없다** — T 종료점 검출이 참값에서도 흔들리기
    때문이다. 분할 코드(`delineate`)는 남겨 두되 집계에서 뺀다.

③ QRS 진폭·면적 — R진폭 (mV) · QRS면적 (mV·ms)
    **무엇인가**  R 파의 높이와 QRS 복합체 아래 면적이다. 둘 다 **심실 비대 판정** 등에
    쓰인다.

    **어떻게 재나**
        R 진폭   max(x[R±50ms]) - 기저선
        QRS 면적  sum |x[R-50ms : R+60ms] - 기저선| / fs * 1000

    **왜 여기서 보나**  진폭이 곧 지표라 잡음이 그대로 오차가 된다. 계산이 단순한 것이
    장점이다.

────────────────────────────────────────────────────────────────────────
집계
────────────────────────────────────────────────────────────────────────
박동별 오차 = (처리 전 또는 처리 후 값) - (참값). 부호를 살린 편향과 절댓값을 함께 본다.
분절 안에서 잰 뒤 **박동 전체**로 모아 중앙값·평균±SD·분위수를 내고, 기록별 표도 낸다.

읽는 법은 단순하다 — **표와 그림의 값은 모두 오차다. 작아졌으면 참값(x_clean)에 더
가까워진 것이다.** 예를 들어

    x_clean 에서 잰 ST60 = 0.05 mV   (참값)
    x_noisy 에서 잰 ST60 = 0.18 mV   -> 오차 |0.18 - 0.05| = 0.13 mV
    A 성분차감에서 잰 ST60 = 0.11 mV  -> 오차 |0.11 - 0.05| = 0.06 mV

`개선된_박동비율` 은 박동 단위로 처리 후 절대오차가 처리 전보다 작아진 비율이다.

값에 대한 해석·판정은 붙이지 않는다.

────────────────────────────────────────────────────────────────────────
산출물  results/06_ablation/<run>/<split>/
────────────────────────────────────────────────────────────────────────
    sqi_summary.csv       ① SQI 4종 — 계열별 중앙값·평균±SD (참값 불필요)
    beats.csv             ② 박동 × 계열 — 지표 원값 (기록·분절·박동 위치 포함)
    metric_summary.csv    지표별 집계 — 참값·처리 전·처리 후의 값과 오차
    error_by_record.csv   기록별 절대오차 중앙값, 처리 전/후 나란히
    breakdown.csv         입력 SNR 구간별·기록별 분해 — 언제 유효한가
    figures/sqi_compare.png     ① SQI 계열별 분포와 분절별 이동
    figures/error_compare.png   ② 지표별 절대오차 분포와 기록별 변화
"""
import argparse
import os

import numpy as np
import pandas as pd
import torch

from src import metrics
from src.core import load_ckpt
from src.data.build import load_cfg
from src.data.dataset import load
from src.model import meae
from src.viz import plt

LABELS = {"B": "B재구성(처리후)", "A": "A성분차감(처리후)"}
SERIES = ("x_clean(참값)", "x_noisy(처리전)", LABELS["B"])   # main() 이 method 로 갱신
BASE_MS = (-80.0, -40.0)      # PR 구간 — 등전위 기저선
J_MS = 40.0                   # R 에서 J 점까지 (근사)
ST_MS = (60.0, 80.0)          # J 점 이후 측정 지점
QRS_MS = (-50.0, 60.0)        # QRS 면적 구간
RAMP_MS = 50.0                # R 진폭 탐색 반경
METRICS = ("ST60_mV", "ST80_mV", "R진폭_mV", "QRS면적_mVms")
# QT·QTc 는 뺀다 — 상용 ECG 판독 프로그램 간 QT 측정 차이가 6-10 ms 인데 우리 절대오차가
# 47-64 ms 다. 지표 자체의 판별력이 없으므로 싣지 않고 그 근거를 note 에 남긴다.
NL = chr(10)          # 그림 제목 줄바꿈 — 이스케이프 사고를 피한다
SQI_KEYS = ("basSQI", "pSQI", "kSQI", "bSQI", "ECGMeanCoef")
# 입력 SNR 구간 — test 는 −5.8 ~ 10.4 dB 에 퍼져 있어 그 범위에 맞춰 나눈다
SNR_BANDS = ((-99.0, -2.0), (-2.0, 0.0), (0.0, 2.0), (2.0, 5.0), (5.0, 99.0))
EDGE_MS = 400.0               # 분절 가장자리 — 창이 잘리는 박동은 뺀다


def _ms(v, fs):
    return int(round(v / 1000.0 * fs))


def beat_measures(sig, peaks, fs, waves=None):
    """박동마다 지표를 잰다. 창이 분절 밖으로 나가면 그 박동은 NaN.

    `waves` 는 참값에서 얻은 파형 분할이다 — QT 는 신호마다 다시 분할해 재므로
    호출부가 그 신호의 분할을 넘긴다.
    """
    L = len(sig)
    b0, b1 = _ms(BASE_MS[0], fs), _ms(BASE_MS[1], fs)
    j = _ms(J_MS, fs)
    s60, s80 = _ms(ST_MS[0], fs), _ms(ST_MS[1], fs)
    q0, q1 = _ms(QRS_MS[0], fs), _ms(QRS_MS[1], fs)
    ra = _ms(RAMP_MS, fs)
    rows = []
    for i, r in enumerate(peaks):
        row = {k: np.nan for k in METRICS}
        lo, hi = r + b0, r + j + s80 + 1
        if lo < 0 or hi > L or r + q0 < 0 or r + q1 > L:
            rows.append(row)
            continue
        base = float(sig[r + b0:r + b1].mean())
        row["ST60_mV"] = float(sig[r + j + s60]) - base
        row["ST80_mV"] = float(sig[r + j + s80]) - base
        seg = sig[max(0, r - ra):min(L, r + ra + 1)]
        row["R진폭_mV"] = float(seg.max()) - base
        row["QRS면적_mVms"] = float(np.abs(sig[r + q0:r + q1] - base).sum()) / fs * 1000.0
        rows.append(row)
    return rows


def delineate(sig, peaks, fs):
    """QRS 시작점·T 종료점. 실패하면 NaN 으로 채운다 (잡음에서 흔한 일이다)."""
    out = {"onset": np.full(len(peaks), np.nan), "offset": np.full(len(peaks), np.nan)}
    if len(peaks) < 3:
        return out
    try:
        import neurokit2 as nk
        _, w = nk.ecg_delineate(np.asarray(sig, dtype=np.float64),
                                rpeaks={"ECG_R_Peaks": np.asarray(peaks)},
                                sampling_rate=fs, method="dwt", show=False)
        on = np.asarray(w.get("ECG_R_Onsets", []), dtype=np.float64)
        off = np.asarray(w.get("ECG_T_Offsets", []), dtype=np.float64)
        n = min(len(peaks), len(on), len(off))
        out["onset"][:n] = on[:n]
        out["offset"][:n] = off[:n]
    except Exception:
        pass
    return out


@torch.no_grad()
def restore(model, ds, device, idx, k_clean, k_noise, batch=100, method="B"):
    """복원 신호. 04 의 A·B 와 **같은 계산**이다.

        B (기본)  잡음 인코딩 3개를 0으로 치환한 재구성 = D(z_clean, 0, 0, 0)
        A         x_noisy - s_bw - s_ma - s_em

    **B 를 기본으로 둔다.** A 는 원본 x_noisy 를 그대로 유지한 채 모델 추정치만 빼므로,
    모델이 아무것도 못 뽑아도 최소한 x_noisy 만큼은 보장된다. 신호 전체를 디코더가 새로
    그려야 하는 B 와 출발선이 다르다. 처리 전(x_noisy)과 나란히 두려면 B 다.
    """
    pad, K = model.pad_each, model.n_encoders
    out = np.zeros((len(idx), ds.x_noisy.shape[1]))
    for s in range(0, len(idx), batch):
        j = idx[s:s + batch]
        x = meae.pad(ds.tensor(j).to(device), pad)
        # [version5] 모델 경로를 쓴다 — 잔차 연결이 있으면 함께 마스킹돼야 한다.
        cut = lambda y: meae.crop(y, pad).squeeze(1).cpu().numpy().astype(np.float64)
        if method == "B":
            out[s:s + len(j)] = cut(model.masked_reconstruct(x, k_noise))
        else:
            noise = np.zeros((len(j), ds.x_noisy.shape[1]))
            for k in k_noise:
                noise += cut(model.component(x, k))
            out[s:s + len(j)] = ds.x_noisy[j].astype(np.float64) - noise
    return out


def fig_sqi(q, out, run, split, n_seg):
    """[SQI] 계열별 분포와 참값 대비 위치. 지표마다 축이 달라 따로 그린다.

    x_clean 을 세로 점선으로 두어 처리 후가 참값 쪽으로 갔는지 바로 보이게 한다.
    basSQI 만 낮을수록 좋으므로 제목에 방향을 적는다.
    """
    keys = [k for k in SQI_KEYS if k in q]
    dirn = {"basSQI": "낮을수록 좋다"}
    fig, ax = plt.subplots(2, len(keys), figsize=(3.1 * len(keys), 7.0))
    ax = np.atleast_2d(ax)
    cols = {SERIES[0]: "#1f77b4", SERIES[1]: "#000", SERIES[2]: "#c44e52"}
    for c, k in enumerate(keys):
        vals = [q[k][s][~np.isnan(q[k][s])] for s in SERIES]
        hi = float(np.nanpercentile(np.concatenate(vals), 98))
        lo = float(np.nanpercentile(np.concatenate(vals), 2))
        a = ax[0, c]
        a.boxplot([np.clip(v, lo, hi) for v in vals], showfliers=False, widths=.55,
                  tick_labels=["참값", "처리 전", "처리 후"])
        a.axhline(np.nanmedian(q[k][SERIES[0]]), color="#1f77b4", ls="--", lw=1)
        a.set_title(f"{k}   {dirn.get(k, '높을수록 좋다')}" + NL
                    + f"전 {np.nanmedian(q[k][SERIES[1]]):.3f} → "
                    f"후 {np.nanmedian(q[k][SERIES[2]]):.3f} "
                    f"(참값 {np.nanmedian(q[k][SERIES[0]]):.3f})",
                    fontsize=8.5, loc="left")
        a.grid(alpha=.3, lw=.4, axis="y")
        a.tick_params(labelsize=7)

        # 아래 줄 — 분절별 처리 전 → 후 이동. 참값 쪽으로 갔으면 파랑
        a = ax[1, c]
        b_, a_ = q[k][SERIES[1]], q[k][SERIES[2]]
        ref = np.nanmedian(q[k][SERIES[0]])
        good = np.abs(a_ - ref) < np.abs(b_ - ref)
        step = max(1, len(b_) // 200)                 # 200개만 그린다 — 과밀 방지
        for i in range(0, len(b_), step):
            if np.isnan(b_[i]) or np.isnan(a_[i]):
                continue
            a.plot([0, 1], [b_[i], a_[i]], lw=.6, alpha=.35,
                   color="#4c72b0" if good[i] else "#c44e52")
        a.axhline(ref, color="#1f77b4", ls="--", lw=1.2, label="참값 중앙값")
        a.set_xticks([0, 1])
        a.set_xticklabels(["처리 전", "처리 후"], fontsize=8)
        a.set_xlim(-.3, 1.3)
        a.set_ylim(lo, hi)
        a.set_title(f"참값에 가까워진 분절 {np.nanmean(good) * 100:.0f}%",
                    fontsize=8.5, loc="left")
        a.legend(fontsize=7)
        a.grid(alpha=.3, lw=.4, axis="y")
        a.tick_params(labelsize=7)

    fig.suptitle(f"[06 ① 신호 품질 지수] {run} · {split} · {n_seg:,}분절 — "
                 "참값이 필요 없는 지표다. 아래 줄에서 파랑은 참값에 가까워진 분절",
                 fontsize=12)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, bbox_inches="tight", dpi=130)
    plt.close(fig)


def fig_error(beats, err_rec, out, run, split, n_beat, n_rec):
    """위: 지표별 절대오차 분포(박동 전체). 아래: 기록별 절대오차 중앙값 변화."""
    keys = [k for k in METRICS if beats[f"{k}_처리전"].notna().any()]
    fig, ax = plt.subplots(2, len(keys), figsize=(3.1 * len(keys), 7.4))
    ax = np.atleast_2d(ax)
    for c, k in zip(range(len(keys)), keys):
        b = beats[f"{k}_처리전"].abs().dropna()
        a2 = beats[f"{k}_처리후"].abs().dropna()
        hi = float(np.nanpercentile(pd.concat([b, a2]), 97)) or 1.0
        a = ax[0, c]
        a.boxplot([b.clip(upper=hi), a2.clip(upper=hi)], showfliers=False,
                  widths=.55, tick_labels=["처리 전", "처리 후"])
        a.set_title(f"|{k} 오차|\n중앙 {b.median():.3f} → {a2.median():.3f}",
                    fontsize=9, loc="left")
        a.grid(alpha=.3, lw=.4, axis="y")
        a.tick_params(labelsize=7)

        a = ax[1, c]
        p, q = err_rec[f"{k}_처리전"], err_rec[f"{k}_처리후"]
        for i in range(len(err_rec)):
            a.plot([0, 1], [p.iloc[i], q.iloc[i]], lw=.9, marker="o", ms=3.5,
                   alpha=.7, color="#4c72b0" if q.iloc[i] < p.iloc[i] else "#c44e52")
        a.set_xticks([0, 1])
        a.set_xticklabels(["처리 전", "처리 후"], fontsize=8)
        a.set_xlim(-.3, 1.3)
        a.set_title(f"기록별 |{k} 오차| 중앙값", fontsize=9, loc="left")
        a.grid(alpha=.3, lw=.4, axis="y")
        a.tick_params(labelsize=7)

    fig.suptitle(f"[06 임상 형태 지표] {run} · {split} · 박동 {n_beat:,}개 · "
                 f"기록 {n_rec}개 — 기준점(R-피크)은 x_clean 에서 뽑아 세 신호에 공유. "
                 "아래 줄에서 파랑은 처리 후 오차가 줄어든 기록", fontsize=12)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, bbox_inches="tight", dpi=130)
    plt.close(fig)


def main(config="configs/default.yaml", run="K4_seed42", split="test", n=None,
         outdir=None, from05=None, method="B"):
    global SERIES
    SERIES = (SERIES[0], SERIES[1], LABELS[method])
    cfg = load_cfg(config)
    fs = cfg["data"]["fs"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    outdir = outdir or os.path.join("results", "06_ablation", run, split)
    os.makedirs(os.path.join(outdir, "figures"), exist_ok=True)

    model, ck = load_ckpt(cfg, run)
    model = model.to(device).eval()
    ds = load(cfg, split)
    idx = np.arange(len(ds) if n is None else min(n, len(ds)))
    picked = None
    if from05:                    # 05 에서 고른 분절만 — 그림과 같은 사례를 수치로 본다
        sel = pd.read_csv(from05, encoding="utf-8-sig")
        want = dict(zip(sel["분절"], sel.get("구간", sel["분절"])))
        pos = {f"{m['record_id']}_{m['seg_idx']:04d}": i for i, m in enumerate(ds.meta)}
        idx = np.array([pos[s] for s in want if s in pos])
        picked = {pos[s]: b for s, b in want.items() if s in pos}
        print(f"[06] 05 선정 분절 {len(idx)}개만 평가한다 ({from05})")

    sup = list(cfg["loss"]["supervise"])
    k_clean = sup.index("x_clean")
    k_noise = [k for k in range(model.n_encoders) if k != k_clean]
    rest = restore(model, ds, device, idx, k_clean, k_noise, method=method)

    print(f"[06] 박동 단위 측정 — {len(idx)}분절 x 3계열")
    fid = []                     # 분절별 기준 R-피크 (x_clean 에서 한 번 검출)
    edge = _ms(EDGE_MS, fs)
    rows = []
    for c, i in enumerate(idx):
        m = ds.meta[int(i)]
        clean = ds.refs["x_clean"][i].astype(np.float64)
        sigs = (clean, ds.x_noisy[i].astype(np.float64), rest[c])
        # 기준점은 참값에서 한 번만. 가장자리 박동은 창이 잘리므로 뺀다.
        pk = np.asarray(metrics.detect_rpeaks(clean, fs), dtype=np.int64)
        pk = pk[(pk >= edge) & (pk < len(clean) - edge)]
        fid.append(pk)
        if len(pk) == 0:
            continue
        per = [beat_measures(sig, pk, fs) for sig in sigs]
        for bi, r in enumerate(pk):
            row = {"기록": m["record_id"], "분절": m["seg_idx"], "박동위치": int(r)}
            if picked is not None:
                row["구간"] = picked.get(int(i), "")
            for s, vals in zip(SERIES, per):
                for k in METRICS:
                    row[f"{k}__{s}"] = vals[bi][k]
            rows.append(row)
        if c % 200 == 0:
            print(f"  {c}/{len(idx)}", flush=True)

    # ---- ① 신호 품질 지수 (SQI) — 참값이 필요 없다
    # x_noisy 와 복원에서 **각각 독립으로** 잰다. x_clean 은 참고로만 함께 싣는다.
    print("[06] SQI 4종 — 참값 없이 신호 하나만 보고 잰다 (bSQI 는 검출을 두 번 한다)")
    sqi_rows, sqi_raw = [], {}
    for label, sig in ((SERIES[0], ds.refs["x_clean"][idx].astype(np.float64)),
                       (SERIES[1], ds.x_noisy[idx].astype(np.float64)),
                       (SERIES[2], rest)):
        q = metrics.sqi_all(sig, fs, progress=300)
        # ECGMeanCoef — 기준점을 공유해 계열 간 비교가 검출 차이에 흔들리지 않게 한다
        q["ECGMeanCoef"] = np.array(
            [metrics.ecg_mean_coef(sig[t], fs, peaks=fid[t]) for t in range(len(idx))])
        sqi_raw[label] = q
        row = {"계열": label, "분절수": len(idx)}
        for k in SQI_KEYS:
            row[k] = float(np.nanmedian(q[k]))
            row[f"{k}_평균"] = float(np.nanmean(q[k]))
            row[f"{k}_SD"] = float(np.nanstd(q[k], ddof=1))
        sqi_rows.append(row)
    sqi = pd.DataFrame(sqi_rows)
    sqi.round(5).to_csv(f"{outdir}/sqi_summary.csv", index=False, encoding="utf-8-sig")
    fig_sqi({k: {s: sqi_raw[s][k] for s in SERIES} for k in SQI_KEYS},
            f"{outdir}/figures/sqi_compare.png", run, split, len(idx))

    beats = pd.DataFrame(rows)
    # ---- 참값 대비 박동별 오차
    for k in METRICS:
        ref = beats[f"{k}__{SERIES[0]}"]
        beats[f"{k}_참값"] = ref
        beats[f"{k}_처리전"] = beats[f"{k}__{SERIES[1]}"] - ref
        beats[f"{k}_처리후"] = beats[f"{k}__{SERIES[2]}"] - ref
    beats.round(5).to_csv(f"{outdir}/beats.csv", index=False, encoding="utf-8-sig")

    # ---- 지표별 집계
    rows = []
    for k in METRICS:
        row = {"지표": k, "박동수": int(beats[f"{k}_참값"].notna().sum()),
               "참값_중앙": beats[f"{k}_참값"].median()}
        for tag in ("처리전", "처리후"):
            e = beats[f"{k}_{tag}"]
            row[f"{tag}_값중앙"] = beats[f"{k}__{SERIES[1 if tag == '처리전' else 2]}"].median()
            row[f"{tag}_편향중앙"] = e.median()
            row[f"{tag}_절대오차중앙"] = e.abs().median()
            row[f"{tag}_절대오차평균"] = e.abs().mean()
            row[f"{tag}_절대오차_p90"] = e.abs().quantile(0.90)
        b, a2 = beats[f"{k}_처리전"].abs(), beats[f"{k}_처리후"].abs()
        row["오차축소_중앙"] = float(b.median() - a2.median())
        row["개선된_박동비율"] = float((a2 < b).mean())
        rows.append(row)
    summ = pd.DataFrame(rows)
    summ.round(5).to_csv(f"{outdir}/metric_summary.csv", index=False,
                         encoding="utf-8-sig")

    # ---- 기록별 절대오차 중앙값
    g = beats.groupby("기록")
    err_rec = pd.DataFrame({"기록": sorted(beats["기록"].unique())}).set_index("기록")
    for k in METRICS:
        for tag in ("처리전", "처리후"):
            err_rec[f"{k}_{tag}"] = g[f"{k}_{tag}"].apply(lambda v: v.abs().median())
    err_rec = err_rec.reset_index()
    err_rec.round(5).to_csv(f"{outdir}/error_by_record.csv", index=False,
                            encoding="utf-8-sig")

    # ---- 분해 — 언제 유효한가. 입력 SNR 구간별, 그리고 기록별 (박동 단위)
    snr_in = metrics.snr_db_vec(ds.refs["x_clean"][idx].astype(np.float64),
                                ds.x_noisy[idx].astype(np.float64))
    seg_snr = {}
    for t, i in enumerate(idx):
        mm = ds.meta[int(i)]
        seg_snr[(mm["record_id"], mm["seg_idx"])] = snr_in[t]
    beats["입력SNR"] = [seg_snr.get((r, g), np.nan)
                       for r, g in zip(beats["기록"], beats["분절"])]
    br = []
    bands = [(f"SNR [{lo:g}, {hi:g})" if lo > -90 else f"SNR < {hi:g}",
              (beats["입력SNR"] >= lo) & (beats["입력SNR"] < hi)) for lo, hi in SNR_BANDS]
    bands += [(f"기록 {r}", beats["기록"] == r) for r in sorted(beats["기록"].unique())]
    for gname, m in bands:
        if not m.any():
            continue
        for k in METRICS:
            b_, a_ = beats.loc[m, f"{k}_처리전"].abs(), beats.loc[m, f"{k}_처리후"].abs()
            br.append({"구간": gname, "박동수": int(m.sum()), "지표": k,
                       "처리전_절대오차중앙": float(b_.median()),
                       "처리후_절대오차중앙": float(a_.median()),
                       "오차축소": float(b_.median() - a_.median()),
                       "개선된_박동비율": float((a_ < b_).mean())})
    pd.DataFrame(br).round(4).to_csv(f"{outdir}/breakdown.csv", index=False,
                                     encoding="utf-8-sig")

    fig_error(beats, err_rec, f"{outdir}/figures/error_compare.png", run, split,
              len(beats), beats["기록"].nunique())

    print("")
    print(f"[06 임상 형태 지표] {run} (에폭 {ck['epoch']}) · {split} "
          f"{len(idx)}분절 · 박동 {len(beats):,}개 · 기록 {beats['기록'].nunique()}개")
    print("기준점(R-피크)은 x_clean 에서 뽑아 세 신호에 공유한다")
    print("")
    print("① 신호 품질 지수 (SQI) — 참값 불필요. basSQI 는 낮을수록, 나머지는 높을수록 좋다")
    print(sqi[["계열", "분절수"] + list(SQI_KEYS)].round(4).to_string(index=False))
    print("")
    print("② 임상 형태 지표 — 참값 대비 오차. 작아졌으면 x_clean 에 가까워진 것이다")
    cols = ["지표", "박동수", "참값_중앙", "처리전_절대오차중앙", "처리후_절대오차중앙",
            "처리전_편향중앙", "처리후_편향중앙", "개선된_박동비율"]
    print(summ[cols].round(4).to_string(index=False))
    print(f"산출물 → {outdir}/")
    return beats, summ, err_rec, sqi


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--run", default="K4_seed42")
    p.add_argument("--split", default="test")
    p.add_argument("--n", type=int, default=None)
    p.add_argument("--from-05", dest="from05", default=None,
                   help="05_figure 의 segments.csv 경로. 그 분절들만 평가한다")
    p.add_argument("--method", default="B", choices=["B", "A"],
                   help="복원 방식. B = 마스킹 재구성(기본), A = 성분차감")
    p.add_argument("--outdir", default=None)
    a = p.parse_args()
    main(a.config, a.run, a.split, a.n, a.outdir, a.from05, a.method)
