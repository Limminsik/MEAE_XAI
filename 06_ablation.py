"""06 — 활용 효과. 두 층으로 본다.

    python 06_ablation.py --run C16_seed42 --split test

파형이 닮았다는 것과 진단값이 맞는다는 것은 다르다. 파형 지표(|r|·RMSE·SNR)는 04·05 에
있고, 여기서는 **신호 품질**과 **임상에서 실제로 재는 값**을 본다.

대상 셋 — x_clean(참값) · x_noisy(처리 전) · **B 재구성**(처리 후)
상자그림은 여기에 고전 비교선 2종(대역통과·웨이블릿 임계값)을 더해 **방법 5종**을 비교한다.
방법 비교는 06 에서만 한다 — 참값이 있어야 "어느 쪽이 더 가깝다"를 말할 수 있고,
07 에는 참값이 없어 입력과 B 둘만 나란히 둔다.
  B = D(z_clean, 0, 0, 0). A(성분차감)는 `--method A` 로 쓸 수 있으나 기본이 아니다 —
  원본 x_noisy 를 유지한 채 모델 추정치만 빼므로 처리 전과 출발선이 같지 않다.

════════════════════════════════════════════════════════════════════════
① 신호 품질 지수 (SQI) — **참값이 필요 없다**
════════════════════════════════════════════════════════════════════════
x_noisy 와 B 에서 **각각 독립으로** 잰다. 이 층이 x_clean 의존을 우회한다 —
MIT-BIH 원본도 완전한 참값이 아니라는 한계에 대한 독립적 근거다.

  basSQI       P(0-1Hz) / P(0-40Hz)          기저선 대역 비중.  **낮을수록** 좋다
  pSQI         P(5-20Hz) / P(0-62.5Hz)       QRS 대역 비중.    높을수록
  kSQI         첨도 = m4 / m2^2               잡음 없는 QRS 일수록 높다 (통상 5 이상)
  bSQI         |R1 & R2| / |R1 | R2|          두 검출기의 박동 일치 (허용 150 ms)
  ECGMeanCoef  mean_b corr(박동_b, 평균 템플릿)  박동 형태의 일관성

bSQI 는 neurokit · pantompkins1985 두 검출기를 쓴다. 10초 분절에 박동이 12~13개뿐이라
값이 이산적이다 — 분별력은 있으나 해상도가 거칠다.

════════════════════════════════════════════════════════════════════════
② 임상 형태 지표 — 참값 기준
════════════════════════════════════════════════════════════════════════
**기준점을 공유한다 — 핵심 설계.** R-피크는 x_clean 에서 한 번 검출해 세 신호에 똑같이
적용한다. 신호마다 따로 검출하면 "박동을 찾았는가"(검출 오차)와 "찾은 자리에서 잰 값이
맞는가"(측정 오차)가 섞인다. 기준점을 고정하면 박동이 1:1 로 짝지어져 **박동별 오차**를
낼 수 있고, 남는 차이는 온전히 파형 왜곡에서 온다. 검출 성능 자체는 03 의 F1 에 있다.
분절 가장자리 400 ms 안쪽 박동은 창이 잘려 뺀다.

    기저선  = mean(x[R-80ms : R-40ms])          PR 구간, 등전위선
    J점     = R + 40ms                           (QRS 종료점 검출은 잡음에 불안정)

    ST60    = x[J+60ms] - 기저선                 [mV]   심근경색·허혈 진단의 핵심
    ST80    = x[J+80ms] - 기저선                 [mV]
    R진폭   = max(x[R+-50ms]) - 기저선            [mV]   심실 비대 판정 등
    QRS면적 = sum|x[R-50ms:R+60ms] - 기저선|/fs*1000   [mV*ms]

**ST 는 기저선 대비로 재므로 기저선이 흔들리면 그대로 틀린다 — bw 가 정확히 그 기저선을
흔든다.** 반면 R진폭·QRS면적은 기저선과 R피크가 ~100 ms 밖에 떨어져 있지 않아 느린 잡음이
**빼면 상쇄된다**(bw 기여 0.0085 mV, 신호 SD 의 1/20). 원래 잡음에 둔감한 지표라 얻을 것이
적고, 대신 재구성 오차가 그대로 얹힌다.

**QT·QTc 는 제외한다.** 상용 ECG 판독 프로그램 간 QT 측정 차이가 6-10 ms 인데 우리
절대오차는 47-64 ms 였다. **지표 자체의 판별력이 없다** — T 종료점 검출이 참값에서도
흔들리기 때문이다. 되살리려면 neurokit2 `ecg_delineate`(dwt)로 R_Onset·T_Offset 을 찾으면 된다.

────────────────────────────────────────────────────────────────────────
집계
────────────────────────────────────────────────────────────────────────
박동별 오차 = (처리 전 또는 처리 후 값) - (참값). 부호를 살린 편향과 절댓값을 함께 본다.
**표와 그림의 값은 모두 오차다. 작아졌으면 참값에 더 가까워진 것이다.**

    x_clean 에서 잰 ST60 = 0.05 mV   (참값)
    x_noisy 에서 잰 ST60 = 0.18 mV   -> 오차 |0.18 - 0.05| = 0.13 mV
    B 재구성에서 잰 ST60  = 0.11 mV   -> 오차 |0.11 - 0.05| = 0.06 mV

`개선된_박동비율` 은 박동 단위로 처리 후 절대오차가 처리 전보다 작아진 비율이다.
입력 SNR 구간별·기록별 분해도 낸다 — "언제 유효한가".

`--from-05 <05의 segments.csv>` 를 주면 05 에서 그림으로 본 그 분절들만 평가한다.

값에 대한 해석·판정은 붙이지 않는다.

────────────────────────────────────────────────────────────────────────
산출물  results/06_ablation/<run>/<split>/
────────────────────────────────────────────────────────────────────────
  sqi_summary.csv       ① SQI 5종 — 계열 5종(참값·처리 전·고전 2종·처리 후)
  beats.csv             ② 박동 x 계열 5종 — 지표 원값 (기록·분절·박동 위치)
  metric_summary.csv    ② 지표별 집계 — 값·편향·절대오차·개선 비율
  error_by_record.csv   ② 기록별 절대오차 중앙값
  breakdown.csv         입력 SNR 구간별·기록별 분해
  sqi_by_record.csv     ① 기록별 SQI 중앙값 (참값·처리 전·처리 후)
  figures/sqi_compare.png     ① 처리 전/후 분포와 **기록별** 이동
  figures/sqi_box.png          ① 가로 상자 — 참값·처리 전·처리 후
  figures/sqi_method_box.png   ① 가로 상자 — **방법 비교** 5종
  figures/error_compare.png   ② 절대오차 분포와 기록별 변화
  figures/error_box.png        ② 가로 상자 — 처리 전/후, 부호 있는 오차
  figures/error_method_box.png ② 가로 상자 — **방법 비교** 5종
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
# 상자그림의 방법 비교에 넣는 고전 비교선. 04 의 `classical_denoise` 와 같은 계산이다.
# **방법 비교는 06 에서만 한다** — 참값이 있어야 "어느 쪽이 더 가까운가"를 말할 수 있고,
# 07 에는 참값이 없어 그 비교가 성립하지 않는다.
CLASSIC = ("대역통과 0.5-40Hz", "웨이블릿 임계값")
# 입력 SNR 구간 — test 는 −5.8 ~ 10.4 dB 에 퍼져 있어 그 범위에 맞춰 나눈다
SNR_BANDS = ((-99.0, -2.0), (-2.0, 0.0), (0.0, 2.0), (2.0, 5.0), (5.0, 99.0))
EDGE_MS = 400.0               # 분절 가장자리 — 창이 잘리는 박동은 뺀다


def _ms(v, fs):
    return int(round(v / 1000.0 * fs))


def beat_measures(sig, peaks, fs):
    """박동마다 지표를 잰다. 창이 분절 밖으로 나가면 그 박동은 NaN."""
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


def fig_sqi(q, sqi_rec, out, run, split, n_seg, n_rec):
    """[SQI] 임상 형태 지표 그림(`fig_error`)과 **같은 구성**이다.

        위  지표별 분포 — 처리 전/후 상자, 참값은 가로 점선(기준선)
        아래 기록별 중앙값 변화 — 파랑이면 그 기록이 참값 쪽으로 갔다

    아래 줄을 분절 단위(1,620개)로 그리면 선이 뭉개져 읽을 수 없다. 기록 단위(9개)로
    묶으면 "어느 환자에서 되고 어느 환자에서 안 되는가"가 그대로 보이고, 임상 형태
    지표 그림과 같은 눈으로 읽을 수 있다.

    basSQI 만 낮을수록 좋으므로 제목에 방향을 적는다.
    """
    keys = [k for k in SQI_KEYS if k in q]
    dirn = {"basSQI": "낮을수록 좋다"}
    fig, ax = plt.subplots(2, len(keys), figsize=(3.1 * len(keys), 7.0))
    ax = np.atleast_2d(ax)
    cols = {SERIES[0]: "#1f77b4", SERIES[1]: "#000", SERIES[2]: "#c44e52"}
    for c, k in enumerate(keys):
        # 상자는 **처리 전·후 둘만** 그린다 — 이 층의 요지는 참값이 필요 없다는 것이다.
        # 참값은 방향을 읽을 기준선으로만 남긴다 (가로 점선).
        vals = [q[k][s][~np.isnan(q[k][s])] for s in SERIES[1:]]
        allv = np.concatenate(vals + [q[k][SERIES[0]][~np.isnan(q[k][SERIES[0]])]])
        hi = float(np.nanpercentile(allv, 98))
        lo = float(np.nanpercentile(allv, 2))
        a = ax[0, c]
        a.boxplot([np.clip(v, lo, hi) for v in vals], showfliers=False, widths=.5,
                  tick_labels=["처리 전", "처리 후"])
        a.axhline(np.nanmedian(q[k][SERIES[0]]), color="#1f77b4", ls="--", lw=1.2,
                  label="참값 중앙값")
        a.legend(fontsize=7, loc="best")
        a.set_title(f"{k}   {dirn.get(k, '높을수록 좋다')}" + NL
                    + f"전 {np.nanmedian(q[k][SERIES[1]]):.3f} → "
                    f"후 {np.nanmedian(q[k][SERIES[2]]):.3f} "
                    f"(참값 {np.nanmedian(q[k][SERIES[0]]):.3f})",
                    fontsize=8.5, loc="left")
        a.grid(alpha=.3, lw=.4, axis="y")
        a.tick_params(labelsize=7)

        # 아래 줄 — **기록별** 중앙값의 처리 전 → 후 이동. 참값 쪽으로 갔으면 파랑
        a = ax[1, c]
        p = sqi_rec[f"{k}_처리전"]
        r2 = sqi_rec[f"{k}_처리후"]
        ref_r = sqi_rec[f"{k}_참값"]
        n_good = 0
        for i in range(len(sqi_rec)):
            good = abs(r2.iloc[i] - ref_r.iloc[i]) < abs(p.iloc[i] - ref_r.iloc[i])
            n_good += bool(good)
            a.plot([0, 1], [p.iloc[i], r2.iloc[i]], lw=.9, marker="o", ms=3.5,
                   alpha=.7, color="#4c72b0" if good else "#c44e52")
        a.axhline(np.nanmedian(q[k][SERIES[0]]), color="#1f77b4", ls="--", lw=1.2,
                  label="참값 중앙값")
        a.set_xticks([0, 1])
        a.set_xticklabels(["처리 전", "처리 후"], fontsize=8)
        a.set_xlim(-.3, 1.3)
        a.set_title(f"기록별 {k} 중앙값" + NL
                    + f"참값에 가까워진 기록 {n_good}/{len(sqi_rec)}",
                    fontsize=8.5, loc="left")
        a.legend(fontsize=7)
        a.grid(alpha=.3, lw=.4, axis="y")
        a.tick_params(labelsize=7)

    fig.suptitle(f"[06 ① 신호 품질 지수] {run} · {split} · {n_seg:,}분절 · "
                 f"기록 {n_rec}개 — 참값이 필요 없는 지표다. "
                 "아래 줄에서 파랑은 참값에 가까워진 기록", fontsize=12)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, bbox_inches="tight", dpi=130)
    plt.close(fig)


STRIP_MAX = 400          # 상자 위에 찍는 관측점 상한 — 이보다 많으면 균일 추출한다


def _hbox(ax, groups, labels, palette, seed=0, strip_max=STRIP_MAX, whis=(0, 100),
          points=False):
    """가로 상자그림 (seaborn boxplot). `points=True` 면 관측점을 겹쳐 찍는다.

    `whis` 는 수염이 덮는 백분위다. 값이 유계인 SQI 는 (0, 100) 으로 전 범위를 보이지만,
    임상 오차처럼 꼬리가 길면 (1, 99) 로 잘라야 한다 — 전 범위로 두면 극단 몇 개가
    축을 늘려 상자가 선으로 눌린다. 자른 경우 부르는 쪽이 제목에 밝힌다.

    관측점은 **기본으로 끈다.** 표본이 1,620개(분절)·18,758개(박동)라 점을 찍으면
    상자와 중앙값선이 가려져 계열 간 위치 비교가 오히려 어려워진다. 켜면 `strip_max`
    개만 균일 추출해 찍는다.
    """
    import seaborn as sns
    vals, cats = [], []
    rng = np.random.default_rng(seed)
    for lab, v in zip(labels, groups):
        v = np.asarray(v, dtype=np.float64)
        v = v[np.isfinite(v)]
        vals.append(v)
        cats.append(np.full(len(v), lab))
    df = pd.DataFrame({"값": np.concatenate(vals), "계열": np.concatenate(cats)})
    sns.boxplot(df, x="값", y="계열", hue="계열", order=labels, hue_order=labels,
                whis=list(whis), width=.6, palette=palette, legend=False,
                fliersize=0, ax=ax)
    if tuple(whis) != (0, 100):        # 잘랐으면 축도 같이 잘라 상자를 보이게 한다
        lo = min(np.percentile(v, whis[0]) for v in vals if len(v))
        hi = max(np.percentile(v, whis[1]) for v in vals if len(v))
        pad = (hi - lo) * 0.06 or 1.0
        ax.set_xlim(lo - pad, hi + pad)
    if points:                    # 관측점은 균일 추출 — 원 분포의 모양을 유지한다
        keep = []
        for lab in labels:
            m = np.flatnonzero((df["계열"] == lab).to_numpy())
            keep.append(m if len(m) <= strip_max
                        else rng.choice(m, strip_max, replace=False))
        sns.stripplot(df.iloc[np.concatenate(keep)], x="값", y="계열", order=labels,
                      size=2.2, color=".3", alpha=.35, jitter=.28, ax=ax)
    ax.xaxis.grid(True, alpha=.35, lw=.4)
    ax.set(ylabel="", xlabel="")
    ax.tick_params(labelsize=8)


def _method_palette(order):
    """참값은 파랑, 처리 전은 회색, 고전 비교선은 붉은 계열, 우리 것은 초록."""
    pal = []
    for s in order:
        if s == SERIES[0]:
            pal.append("#c7d9ec")
        elif s == SERIES[1]:
            pal.append("#d9d9d9")
        elif s in CLASSIC:
            pal.append("#f2c9c4")
        else:
            pal.append("#a8c8a0")
    return pal


def fig_sqi_box(q, order, out, run, split, n_seg, points=False, methods=False):
    """[SQI] 가로 상자그림. 지표마다 눈금이 달라 패널을 따로 둔다.

    `sqi_compare.png` 가 "전 → 후로 얼마나 옮겨 갔나"를 본다면, 이 그림은
    **분포 자체의 모양**을 본다 — 꼬리가 어디까지 뻗는지, 참값 분포와 겹치는지.

    `methods=True` 면 고전 비교선까지 같은 축에 두는 **방법 비교**로 읽힌다. 그 비교는
    06 에서만 뜻이 있다 — 참값이 있어야 "가깝다"를 말할 수 있고, 07 에는 없다.
    """
    import seaborn as sns
    sns.set_theme(style="ticks", font=plt.rcParams["font.family"][0])
    keys = [k for k in SQI_KEYS if k in q]
    dirn = {"basSQI": "낮을수록 좋다"}
    pal = _method_palette(order)
    fig, ax = plt.subplots(len(keys), 1, figsize=(9.6, 2.35 * len(keys)))
    ax = np.atleast_1d(ax)
    for a, k in zip(ax, keys):
        _hbox(a, [q[k][s] for s in order], order, pal, points=points)
        a.axvline(np.nanmedian(q[k][SERIES[0]]), color="#1f77b4", ls="--", lw=1.1)
        a.set_title(f"{k}   {dirn.get(k, '높을수록 좋다')}   ·   "
                    f"전 {np.nanmedian(q[k][SERIES[1]]):.3f} → "
                    f"후 {np.nanmedian(q[k][SERIES[2]]):.3f} "
                    f"(참값 {np.nanmedian(q[k][SERIES[0]]):.3f}, 점선)",
                    fontsize=9.5, loc="left")
        sns.despine(ax=a, trim=True, left=True)
    tag = " — 방법 비교" if methods else ""
    fig.suptitle(f"[06 ① 신호 품질 지수{tag}] {run} · {split} · {n_seg:,}분절{NL}"
                 "가로 상자 · 수염은 전 범위(0-100 백분위) · 점선은 참값 중앙값 · "
                 "참값 쪽에 가까울수록 좋다", fontsize=11)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, bbox_inches="tight", dpi=130)
    plt.close(fig)
    sns.reset_orig()


def fig_error_box(beats, order, out, run, split, n_beat, points=False,
                  methods=False):
    """[임상 형태 지표] 가로 상자그림. 부호를 살린 오차를 그린다.

    절댓값이 아니라 부호를 살리는 이유: 0 선을 기준으로 **치우침(편향)** 과
    **퍼짐(산포)** 이 한 그림에서 같이 읽힌다. 상자가 0 에서 얼마나 벗어났는지가
    편향이고, 상자가 얼마나 퍼졌는지가 산포다.

    `order` 는 참값을 뺀 계열 목록이다 — 참값 자신의 오차는 정의상 0 이라 싣지 않는다.
    오차는 원값 열에서 그때그때 뺀다(`<지표>__<계열>` − `<지표>__참값`).
    """
    import seaborn as sns
    sns.set_theme(style="ticks", font=plt.rcParams["font.family"][0])
    keys = [k for k in METRICS if beats[f"{k}_처리전"].notna().any()]
    pal = _method_palette(order)
    fig, ax = plt.subplots(len(keys), 1, figsize=(9.6, 2.35 * len(keys)))
    ax = np.atleast_1d(ax)
    for a, k in zip(ax, keys):
        ref = beats[f"{k}__{SERIES[0]}"]
        errs = [(beats[f"{k}__{s}"] - ref).dropna() for s in order]
        _hbox(a, [e.to_numpy() for e in errs], order, pal,
              whis=(1, 99), points=points)
        a.axvline(0.0, color="#1f77b4", ls="--", lw=1.1)
        b = beats[f"{k}_처리전"].dropna()
        a2 = beats[f"{k}_처리후"].dropna()
        a.set_title(f"{k} 오차 (측정값 - 참값)   ·   "
                    f"편향 {b.median():+.3f} → {a2.median():+.3f}   ·   "
                    f"|오차| 중앙 {b.abs().median():.3f} → {a2.abs().median():.3f}",
                    fontsize=9.5, loc="left")
        sns.despine(ax=a, trim=True, left=True)
    tag = " — 방법 비교" if methods else ""
    fig.suptitle(f"[06 ② 임상 형태 지표{tag}] {run} · {split} · "
                 f"박동 {n_beat:,}개{NL}"
                 "점선 0 = 참값과 일치. 상자가 0 에서 벗어난 만큼이 편향, "
                 "퍼진 만큼이 산포다. 수염과 축은 1-99 백분위", fontsize=11)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, bbox_inches="tight", dpi=130)
    plt.close(fig)
    sns.reset_orig()


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


def main(config="configs/default.yaml", run="C16_seed42", split="test", n=None,
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

    # 고전 비교선 — **방법 비교는 여기서 한다.** 참값이 있어야 "어느 쪽이 참값에 더
    # 가까운가"를 말할 수 있고, 그 참값은 06 에만 있다(07 에는 없다). 04 와 같은 함수다.
    cls = metrics.classical_denoise(ds.x_noisy[idx].astype(np.float64), fs)
    methods = [(SERIES[0], None), (SERIES[1], None)] + \
              [(nm, cls[nm]) for nm in CLASSIC] + [(SERIES[2], rest)]

    print(f"[06] 박동 단위 측정 — {len(idx)}분절 x {len(methods)}계열")
    fid = []                     # 분절별 기준 R-피크 (x_clean 에서 한 번 검출)
    edge = _ms(EDGE_MS, fs)
    rows = []
    for c, i in enumerate(idx):
        m = ds.meta[int(i)]
        clean = ds.refs["x_clean"][i].astype(np.float64)
        sigs = [clean, ds.x_noisy[i].astype(np.float64)] + \
               [cls[nm][c] for nm in CLASSIC] + [rest[c]]
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
            for (s, _), vals in zip(methods, per):
                for k in METRICS:
                    row[f"{k}__{s}"] = vals[bi][k]
            rows.append(row)
        if c % 200 == 0:
            print(f"  {c}/{len(idx)}", flush=True)

    # ---- ① 신호 품질 지수 (SQI) — 참값이 필요 없다
    # x_noisy 와 복원에서 **각각 독립으로** 잰다. x_clean 은 참고로만 함께 싣는다.
    print(f"[06] SQI 5종 x {len(methods)}계열 — 참값 없이 신호 하나만 보고 잰다 "
          "(bSQI 는 검출을 두 번 한다)")
    sqi_rows, sqi_raw = [], {}
    for label, sig in ((SERIES[0], ds.refs["x_clean"][idx].astype(np.float64)),
                       (SERIES[1], ds.x_noisy[idx].astype(np.float64)),
                       *[(nm, cls[nm]) for nm in CLASSIC],
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

    # ---- 기록별 SQI 중앙값 — 임상 형태 지표의 error_by_record 와 같은 구성이다.
    # 분절 1,620개를 한 그림에 그리면 선이 뭉개진다. 기록으로 묶어야 "어느 환자에서
    # 되고 어느 환자에서 안 되는가"가 보인다.
    rec_id = np.array([ds.meta[int(i)]["record_id"] for i in idx])
    srows = []
    for r in sorted(set(rec_id)):
        m = rec_id == r
        row = {"기록": r, "분절수": int(m.sum())}
        for k in SQI_KEYS:
            for tag, s in (("참값", SERIES[0]), ("처리전", SERIES[1]), ("처리후", SERIES[2])):
                row[f"{k}_{tag}"] = float(np.nanmedian(sqi_raw[s][k][m]))
        srows.append(row)
    sqi_rec = pd.DataFrame(srows)
    sqi_rec.round(5).to_csv(f"{outdir}/sqi_by_record.csv", index=False,
                            encoding="utf-8-sig")

    # 전/후 이동 그림은 처리 전·후 두 계열만 본다 (참값은 기준선).
    fig_sqi({k: {s: sqi_raw[s][k] for s in SERIES} for k in SQI_KEYS}, sqi_rec,
            f"{outdir}/figures/sqi_compare.png", run, split, len(idx), len(sqi_rec))
    # 상자그림 둘 — 기본은 참값·전·후 셋, 방법 비교는 고전 2종을 더해 다섯.
    base = [SERIES[0], SERIES[1], SERIES[2]]
    fig_sqi_box({k: {s: sqi_raw[s][k] for s in base} for k in SQI_KEYS}, base,
                f"{outdir}/figures/sqi_box.png", run, split, len(idx))
    meth = [SERIES[0], SERIES[1], *CLASSIC, SERIES[2]]
    fig_sqi_box({k: {s: sqi_raw[s][k] for s in meth} for k in SQI_KEYS}, meth,
                f"{outdir}/figures/sqi_method_box.png", run, split, len(idx),
                methods=True)

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
    fig_error_box(beats, [SERIES[1], SERIES[2]],
                  f"{outdir}/figures/error_box.png", run, split, len(beats))
    fig_error_box(beats, [SERIES[1], *CLASSIC, SERIES[2]],
                  f"{outdir}/figures/error_method_box.png", run, split,
                  len(beats), methods=True)

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
    p.add_argument("--run", default="C16_seed42")
    p.add_argument("--split", default="test")
    p.add_argument("--n", type=int, default=None)
    p.add_argument("--from-05", dest="from05", default=None,
                   help="05_figure 의 segments.csv 경로. 그 분절들만 평가한다")
    p.add_argument("--method", default="B", choices=["B", "A"],
                   help="복원 방식. B = 마스킹 재구성(기본), A = 성분차감")
    p.add_argument("--outdir", default=None)
    a = p.parse_args()
    main(a.config, a.run, a.split, a.n, a.outdir, a.from05, a.method)
