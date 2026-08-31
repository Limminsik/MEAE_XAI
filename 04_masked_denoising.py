"""04 — 디노이징 성능.

03 에서 확정한 모델의 **인코딩만 마스킹**해 복원하고, 그 결과를 x_clean 으로 채점한다.
가중치를 다시 학습하지 않는 **추론 시점 조작**이다.

    python 04_masked_denoising.py --run C16_seed42 --split test --three-ways

────────────────────────────────────────────────────────────────────────
비교 대상 — 기준은 x_clean, **mV 원단위**(표준화하지 않는다)
────────────────────────────────────────────────────────────────────────
  B  마스킹 재구성   D(z_1, 0, 0, 0)                        **주 결과**
  a  입력           x_noisy                                 기준선
  대역통과 0.5-40Hz  Butterworth 4차 + filtfilt              비교 (고전)
  웨이블릿 임계값     sym8 level 5, universal threshold       비교 (고전)
  웨이블릿+기저선제거  sym8 level 7, 근사계수 제거 + 임계값      비교 (고전)
  A  성분 차감       x_noisy - s_bw - s_ma - s_em            보조 (성분 추정 검증)
  b  M0 재구성       D(z_1, z_2, z_3, z_4)                   참고

**B 를 중심으로 본다.** A 는 원본 x_noisy 를 유지한 채 모델 추정치만 빼므로, 모델이
아무것도 못 뽑아도 x_noisy 만큼은 보장된다. 신호 전체를 디코더가 새로 그려야 하는 B 와
출발선이 같지 않다. C(마스킹 디코드)는 K=4 에서 B 와 같은 연산이라 최대 절대차를 함께 싣는다.

a·b 를 두는 이유: 개선량 중 어디까지가 재구성 자체의 몫이고 어디부터가 마스킹의 몫인지
갈라야 한다.

**고전 비교선 두 가지 주의**
  1. DC 오프셋을 되돌린다. 0.5 Hz 고역통과와 근사계수 제거는 x_clean 이 가진 기저
     오프셋(중앙값 -0.283 mV)까지 지운다. **입력의 평균**을 되돌린다 — 참값을 쓰지
     않으므로 누수가 없다.
  2. 웨이블릿 임계값 단독은 bw 를 못 없앤다. 임계값이 세부계수(고주파)만 건드리는데
     기저선 변동은 근사계수에 있다. 근사계수를 함께 버린 변형을 따로 싣는다.

────────────────────────────────────────────────────────────────────────
지표 — 주 2종 + 보조 5종
────────────────────────────────────────────────────────────────────────
  주   SNR      10log10(var(clean) / mean(잔차^2))  [dB]      높을수록
       dSNR     SNR(추정) - SNR(x_noisy)                      높을수록. **개선량**
       PRD      100 sqrt(sum(d^2) / sum((clean-mean)^2)) [%]  낮을수록

  **두 주 지표는 신호 전력을 똑같이 분산으로 잡는다** (평균을 뺀다). 전극 오프셋은
  정보가 아니므로 신호로 세지 않으며, 주입 SNR 과도 같은 기준이라 x_noisy 를 재면
  주입한 합성 SNR 이 그대로 나온다. 그 결과 SNR = -20log10(PRD/100) 이 정확히 성립해
  두 지표의 순위가 어긋나지 않는다. PRD 는 문헌의 PRDN(정규화 PRD) 형태다.
  보조 corr · RMSE · SSD · MAD(mV 원단위) · CosSim

**분해** — 입력 SNR 구간별·기록별로 같은 지표를 다시 집계한다(`breakdown.csv`).
"언제 유효한가"가 여기서 나온다.

전수 조합(2^K)은 `--three-ways` 없이 부르면 나온다. 최적 조합 선정은 하지 않는다.

값에 대한 해석·명명은 붙이지 않는다.

────────────────────────────────────────────────────────────────────────
산출물  results/04_masked_denoising/<run>/<split>/
────────────────────────────────────────────────────────────────────────
  three_ways.csv        7상태 x 지표 8종 (구분 열에 역할 표시)
  breakdown.csv         입력 SNR 구간별·기록별 재집계
  three_ways_note.txt   정의와 B/C 실측 절대차
  figures/three_ways.png          적층 — 다섯 상태를 세로로
  figures/three_ways_overlay.png  x_clean 위에 처리 전/후를 겹치고 각각의 잔차

  (--three-ways 없이) exhaustive.csv · baseline.csv · best_by_metric.csv ·
  single_mask.csv · cumulative.csv · rpeak_ratio.csv · persegment_top.csv
"""
import argparse
import itertools
import json
import os

import numpy as np
import pandas as pd
import torch

from src import metrics
from src.core import load_ckpt
from src.data.build import load_cfg
from src.data.dataset import load
from src.model import meae
from src.model.meae import enc_label
from src.viz import plt

NOISE_REFS = ("bw", "ma", "em")
# 표에 붙는 역할 표시 — 무엇이 주 결과이고 무엇이 기준선인지 표 안에서 바로 읽히게 한다
ROLE = {"B 심장직접": "주 결과", "a 입력 x_noisy": "기준선",
        "대역통과 0.5-40Hz": "비교 (고전)", "웨이블릿 임계값": "비교 (고전)",
        "웨이블릿+기저선제거": "비교 (고전)",
        "A 성분차감": "보조 (성분 추정 검증)", "C 마스킹디코드": "B 와 동일 연산 확인",
        "b M0 재구성": "참고 (x_noisy 재구성용)"}
# 입력 SNR 구간 — test 는 −5.8 ~ 10.4 dB 에 퍼져 있어 그 범위에 맞춰 나눈다
SNR_BANDS = ((-99.0, -2.0), (-2.0, 0.0), (0.0, 2.0), (2.0, 5.0), (5.0, 99.0))
LOWER_BETTER = {"SSD": True, "MAD": True, "PRD": True, "CosSim": False, "SNR": False}


def _corr_rows(a, b, eps=1e-30):
    """행별 Pearson 절댓값. (n, L) x (n, L) -> (n,)"""
    a = a - a.mean(-1, keepdims=True)
    b = b - b.mean(-1, keepdims=True)
    d = np.sqrt((a ** 2).sum(-1) * (b ** 2).sum(-1))
    return np.abs((a * b).sum(-1) / np.maximum(d, eps))


def combo_label(mask, K):
    """마스킹 조합 이름 — 끈 인코더를 1-based 로 나열한다."""
    return "-".join(enc_label(k) for k in mask) if mask else "M0(없음)"


@torch.no_grad()
def sweep(model, ds, device, idx, combos, batch=100):
    """조합별 지표 5종을 (조합, 분절) 배열로. 인코딩은 배치마다 한 번만 구한다.

    반환 {지표: (n_combo, n_seg)} 와 R-피크 진폭비 (n_combo, n_seg).
    """
    K, pad = model.n_encoders, model.pad_each
    names = list(metrics.S5_METRICS)
    acc = {k: np.zeros((len(combos), len(idx))) for k in names}
    rpk = np.zeros((len(combos), len(idx)))
    for s in range(0, len(idx), batch):
        j = idx[s:s + batch]
        x = meae.pad(ds.tensor(j).to(device), pad)
        clean = ds.refs["x_clean"][j].astype(np.float64)
        peaks = [ds.rpeaks[g] for g in j]
        # [version5] `masked_reconstruct` 를 쓴다 — 잔차 연결이 있으면 그 경로도 같은
        # 규칙으로 마스킹해야 한다. decode() 를 직접 부르면 잔차가 통째로 빠진다.
        for ci, mk in enumerate(combos):
            y = model.masked_reconstruct(x, list(mk))
            y = meae.crop(y, pad).squeeze(1).cpu().numpy().astype(np.float64)
            for name, v in metrics.s5_score(clean, y).items():
                acc[name][ci, s:s + len(j)] = v
            rpk[ci, s:s + len(j)] = _amp_ratio(clean, y, peaks)
    return acc, rpk


def _amp_ratio(clean, est, peaks_list, win=30):
    """R-피크 ±30샘플 첨두간 진폭의 **복원/clean** 비. 형태 훼손 감시용."""
    out = []
    for i, pk in enumerate(peaks_list):
        r = []
        for p in pk:
            a, b = max(0, p - win), min(clean.shape[-1], p + win)
            dc = clean[i, a:b].max() - clean[i, a:b].min()
            de = est[i, a:b].max() - est[i, a:b].min()
            if dc > 0:
                r.append(de / dc)
        out.append(np.median(r) if r else np.nan)
    return np.array(out)


def summarise(acc, combos, K, base_a, base_b, rpk=None):
    """조합별 중앙값·평균±SD와 ⓐ·ⓑ 대비 델타."""
    rows = []
    for ci, mk in enumerate(combos):
        row = {"조합": combo_label(mk, K), "끈_인코더수": len(mk),
               "마스크비트": "".join("1" if k in set(mk) else "0" for k in range(K))}
        for name in metrics.S5_METRICS:
            v = acc[name][ci]
            row[f"{name}_중앙"] = float(np.median(v))
            row[f"{name}_평균"] = float(v.mean())
            row[f"{name}_SD"] = float(v.std(ddof=1))
            row[f"{name}_vs_noisy"] = float(np.median(v) - np.median(base_a[name]))
            row[f"{name}_vs_M0"] = float(np.median(v) - np.median(base_b[name]))
        if rpk is not None:
            row["R피크_진폭비_중앙"] = float(np.nanmedian(rpk[ci]))
        rows.append(row)
    return pd.DataFrame(rows)


def best_by_metric(tab):
    """지표 5종이 각각 지목하는 최적 조합. 선정하지 않고 나열만 한다."""
    rows = []
    for name, lower in LOWER_BETTER.items():
        col = f"{name}_중앙"
        i = tab[col].idxmin() if lower else tab[col].idxmax()
        m0 = tab[tab["끈_인코더수"] == 0].iloc[0]
        rows.append({"지표": name, "방향": "낮을수록 유사" if lower else "높을수록 유사",
                     "지목_조합": tab.loc[i, "조합"],
                     "끈_인코더수": int(tab.loc[i, "끈_인코더수"]),
                     "값": float(tab.loc[i, col]),
                     "vs_ⓐnoisy": float(tab.loc[i, f"{name}_vs_noisy"]),
                     "vs_ⓑM0": float(tab.loc[i, f"{name}_vs_M0"]),
                     "M0값": float(m0[col])})
    out = pd.DataFrame(rows)
    picks = set(out["지목_조합"])
    out["일치"] = "일치" if len(picks) == 1 else f"불일치 ({len(picks)}종)"
    return out


def cumulative_order(run, split):
    """누적 곡선의 추가 순서 — 03의 잡음 유사도 순.

    03 산출 `corr_matrix.csv` 에서 각 인코더의 **잡음 3종 최대 |r|** 을 읽어 내림차순.
    03을 아직 돌리지 않았으면 None 을 돌려주고 호출부가 건너뛴다.
    """
    p = os.path.join("results", "03_bss", run, split, "corr_matrix.csv")
    if not os.path.exists(p):
        return None, None
    t = pd.read_csv(p, index_col=0, encoding="utf-8-sig")
    v = pd.DataFrame({c: [float(str(x).split("±")[0]) for x in t[c]]
                      for c in NOISE_REFS}, index=t.index).max(axis=1)
    order = list(v.sort_values(ascending=False).index)
    return [int(s.replace("enc", "")) - 1 for s in order], v.sort_values(ascending=False)


# ---------------------------------------------------------------- 그림
def fig_exhaustive(tab, out):
    """전수 지도 — 끈 인코더 수 대 지표. 조합 하나가 점 하나."""
    names = list(metrics.S5_METRICS)
    fig, ax = plt.subplots(1, len(names), figsize=(3.4 * len(names), 3.6))
    for a, name in zip(np.atleast_1d(ax), names):
        a.scatter(tab["끈_인코더수"], tab[f"{name}_중앙"], s=14, alpha=.55,
                  color="#1f77b4", edgecolor="none")
        m0 = tab[tab["끈_인코더수"] == 0][f"{name}_중앙"]
        if len(m0):
            a.axhline(float(m0.iloc[0]), color="#d62728", ls="--", lw=1, label="M0")
            a.legend(fontsize=7)
        a.set_xlabel("끈 인코더 수", fontsize=8)
        a.set_title(name + ("  (낮을수록 유사)" if LOWER_BETTER[name] else "  (높을수록 유사)"),
                    fontsize=8.5)
        a.grid(alpha=.3, lw=.4)
        a.tick_params(labelsize=7)
        if name in ("SSD", "PRD"):
            a.set_yscale("log")
    fig.suptitle("전수 2^K 마스킹 지도 — 조합 하나가 점 하나 (분절 중앙값)", fontsize=11)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def fig_single(single, out):
    """단독 마스킹 8개 — M0 대비 변화."""
    names = list(metrics.S5_METRICS)
    fig, ax = plt.subplots(1, len(names), figsize=(3.2 * len(names), 3.4))
    x = np.arange(len(single))
    for a, name in zip(np.atleast_1d(ax), names):
        v = single[f"{name}_vs_M0"].values
        good = (v < 0) if LOWER_BETTER[name] else (v > 0)
        a.bar(x, v, color=np.where(good, "#2ca02c", "#d62728"))
        a.axhline(0, color="k", lw=.8)
        a.set_xticks(x)
        a.set_xticklabels(single["조합"], rotation=90, fontsize=6.5)
        a.set_title(f"{name}  (M0 대비)", fontsize=8.5)
        a.grid(alpha=.3, lw=.4, axis="y")
        a.tick_params(labelsize=7)
    fig.suptitle("단독 마스킹 — 인코더 하나씩 껐을 때 M0 대비 변화 "
                 "(초록 = 유사도 증가 방향)", fontsize=11)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def fig_cumulative(cum, out):
    names = list(metrics.S5_METRICS)
    fig, ax = plt.subplots(1, len(names), figsize=(3.2 * len(names), 3.4))
    for a, name in zip(np.atleast_1d(ax), names):
        a.plot(cum["누적_개수"], cum[f"{name}_중앙"], "o-", lw=1.2, ms=4, color="#1f77b4")
        a.set_xlabel("누적 마스킹 개수", fontsize=8)
        a.set_title(name, fontsize=8.5)
        a.grid(alpha=.3, lw=.4)
        a.tick_params(labelsize=7)
        if name in ("SSD", "PRD"):
            a.set_yscale("log")
    fig.suptitle("누적 마스킹 곡선 — 03의 잡음 유사도 높은 순으로 하나씩 추가", fontsize=11)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def fig_tradeoff(tab, out):
    """지표 간 산포도 — 점 하나가 조합 하나. 색은 마스킹한 인코더 수.

    지표들이 서로 다른 조합을 좋다고 하는지(상충) 눈으로 확인하기 위한 것이다.
    ΔSNR 은 ⓐ x_noisy 대비 개선량이다.
    """
    pairs = [("SNR_vs_noisy", "R피크_진폭비_중앙", "ΔSNR (ⓐ 대비, dB)", "R-피크 진폭비"),
             ("SNR_vs_noisy", "MAD_중앙", "ΔSNR (ⓐ 대비, dB)", "MAD (mV)"),
             ("CosSim_중앙", "MAD_중앙", "CosSim", "MAD (mV)")]
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.8))
    for a, (xc, yc, xl, yl) in zip(np.atleast_1d(ax), pairs):
        sc = a.scatter(tab[xc], tab[yc], c=tab["끈_인코더수"], cmap="viridis",
                       s=26, alpha=.85, edgecolor="none")
        m0 = tab[tab["끈_인코더수"] == 0]
        if len(m0):
            a.scatter(m0[xc], m0[yc], marker="*", s=220, color="#d62728",
                      edgecolor="k", lw=.5, zorder=5, label="M0 (마스킹 없음)")
            a.legend(fontsize=7.5, loc="best")
        a.set_xlabel(xl, fontsize=9)
        a.set_ylabel(yl, fontsize=9)
        a.grid(alpha=.3, lw=.4)
        a.tick_params(labelsize=8)
    cb = fig.colorbar(sc, ax=ax, shrink=.85, pad=.015)
    cb.set_label("마스킹한 인코더 수", fontsize=9)
    fig.suptitle("지표 간 산포 — 점 하나가 마스킹 조합 하나 (전수 2^K, 분절 중앙값)",
                 fontsize=12)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def fig_rpeak(tab, out):
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.scatter(tab["끈_인코더수"], tab["R피크_진폭비_중앙"], s=16, alpha=.6, color="#1f77b4")
    ax.axhline(1.0, color="#d62728", ls="--", lw=1)
    ax.set_xlabel("끈 인코더 수")
    ax.set_ylabel("R-피크 진폭비 (복원/clean)")
    ax.grid(alpha=.3, lw=.4)
    ax.set_title("R-피크 진폭비 — 형태 훼손 감시 (1.0 = clean과 같은 첨두간 진폭)",
                 fontsize=10, loc="left")
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


# ================================================================
# 복원 비교 — 마스킹·차감·고전
#
# NL — 줄바꿈 상수. 이 파일을 스크립트로 고칠 때 이스케이프가 깨지는 사고를 피한다.
#
#   A. 성분 차감    x_noisy − ŝ_bw − ŝ_ma − ŝ_em      비선형 결합을 우회한다
#   B. 심장 직접    ŝ_clean = D(z1,0,0,0)              가장 단순한 안전판
#   C. 마스킹 디코드 잡음 인코딩 3개를 0으로 치환        기존 방식, baseline 과의 비교선
#
# **B와 C는 이 구조에서 같은 계산이다.** 인코딩을 하나만 남기는 것(component)과 나머지
# 셋을 0으로 치환하는 것(masked_reconstruct)이 K=4에서 동일한 연산이기 때문이다.
# 그래도 두 경로를 따로 계산해 싣는다 — 같다는 것 자체가 확인해야 할 사실이다.
#
# 기준선 ⓐ x_noisy · ⓑ M0 재구성(마스킹 없음)을 함께 둔다.
# 지표는 x_clean 대비 |r| · SSD · RMSE · ΔSNR (mV 원단위, 표준화하지 않는다).
# ================================================================
NL = chr(10)


def three_ways(config="configs/default.yaml", run="C16_seed42", split="val",
               n=None, outdir=None):
    cfg = load_cfg(config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    outdir = outdir or os.path.join("results", "04_masked_denoising", run, split)
    os.makedirs(os.path.join(outdir, "figures"), exist_ok=True)

    model, ck = load_ckpt(cfg, run)
    model = model.to(device)
    ds = load(cfg, split)
    idx = np.arange(len(ds) if n is None else min(n, len(ds)))
    sup = list(cfg["loss"]["supervise"])
    k_clean = sup.index("x_clean")
    k_noise = [k for k in range(model.n_encoders) if k != k_clean]

    pad, batch = model.pad_each, 100
    # 고전적 비교선 3종을 같은 test 데이터에 그대로 적용해 나란히 싣는다
    classic = ["대역통과 0.5-40Hz", "웨이블릿 임계값", "웨이블릿+기저선제거"]
    ways = (["B 심장직접", "a 입력 x_noisy"] + classic
            + ["A 성분차감", "C 마스킹디코드", "b M0 재구성"])
    est = {w: np.zeros((len(idx), ds.x_noisy.shape[1])) for w in ways}
    for s in range(0, len(idx), batch):
        j = idx[s:s + batch]
        x = meae.pad(ds.tensor(j).to(device), pad)
        with torch.no_grad():
            # [version5] 모델이 정의한 경로만 쓴다 — 잔차 연결이 있으면 성분·재구성·마스킹
            # 모두 그 경로가 잔차를 같은 규칙으로 처리한다.
            cut = lambda y: meae.crop(y, pad).squeeze(1).cpu().numpy().astype(np.float64)
            comp = {k: cut(model.component(x, k)) for k in range(model.n_encoders)}
            recon = cut(model(x)[0])
            masked = cut(model.masked_reconstruct(x, k_noise))
        raw = ds.x_noisy[j].astype(np.float64)
        est["A 성분차감"][s:s + len(j)] = raw - sum(comp[k] for k in k_noise)
        est["B 심장직접"][s:s + len(j)] = comp[k_clean]
        est["C 마스킹디코드"][s:s + len(j)] = masked
        est["a 입력 x_noisy"][s:s + len(j)] = raw
        est["b M0 재구성"][s:s + len(j)] = recon
        for nm, v in metrics.classical_denoise(raw, cfg["data"]["fs"]).items():
            est[nm][s:s + len(j)] = v

    clean = ds.refs["x_clean"][idx].astype(np.float64)
    snr_in = metrics.snr_db_vec(clean, est["a 입력 x_noisy"])
    rows = []
    for w in ways:
        y = est[w]
        c0 = clean - clean.mean(-1, keepdims=True)
        y0 = y - y.mean(-1, keepdims=True)
        r = np.abs((c0 * y0).sum(-1) /
                   np.maximum(np.sqrt((c0 ** 2).sum(-1) * (y0 ** 2).sum(-1)), 1e-30))
        snr = metrics.snr_db_vec(clean, y)
        sc = metrics.s5_score(clean, y)
        # 주 지표(SNR·ΔSNR·PRD)를 앞에 두고 보조 5종을 뒤에 둔다.
        # ΔSNR 이 곧 x_noisy 대비 개선량이다 — 입력이 기준선이고 x_clean 은 두 SNR 을
        # 계산하기 위해 필요하다.
        rows.append({"방식": w, "구분": ROLE.get(w, ""), "분절수": len(idx),
                     "SNR_dB": np.median(snr),
                     "dSNR_vs_입력": np.median(snr - snr_in),
                     "PRD": np.median(sc["PRD"]),
                     "corr": r.mean(), "corr_sd": r.std(ddof=1),
                     "RMSE": np.median(np.sqrt(((clean - y) ** 2).mean(-1))),
                     "SSD": np.median(sc["SSD"]), "SSD_평균": sc["SSD"].mean(),
                     "MAD": np.median(sc["MAD"]),
                     "CosSim": np.median(sc["CosSim"])})
    tab = pd.DataFrame(rows).round(4)
    tab.to_csv(f"{outdir}/three_ways.csv", index=False, encoding="utf-8-sig")

    # ---- 분해 — 언제 유효한가. 입력 SNR 구간별, 그리고 기록별
    rec = np.array([ds.meta[int(i)]["record_id"] for i in idx])
    groups = ([(f"SNR [{lo:g}, {hi:g})" if lo > -90 else f"SNR < {hi:g}",
                (snr_in >= lo) & (snr_in < hi)) for lo, hi in SNR_BANDS]
              + [(f"기록 {r}", rec == r) for r in sorted(set(rec))])
    br = []
    for gname, m in groups:
        if not m.any():
            continue
        for w in ways:
            y, c = est[w][m], clean[m]
            sn = metrics.snr_db_vec(c, y)
            br.append({"구간": gname, "분절수": int(m.sum()), "방식": w,
                       "SNR_dB": float(np.median(sn)),
                       "dSNR_vs_입력": float(np.median(sn - snr_in[m])),
                       "PRD": float(np.median(metrics.prd(c, y))),
                       "corr": float(_corr_rows(c, y).mean()),
                       "RMSE": float(np.median(np.sqrt(((c - y) ** 2).mean(-1))))})
    pd.DataFrame(br).round(4).to_csv(f"{outdir}/breakdown.csv", index=False,
                                     encoding="utf-8-sig")
    print("")
    print(f"[복원 세 방식] {run} (에폭 {ck['epoch']}) · {split} {len(idx)}분절 · "
          "기준 x_clean, mV 원단위")
    print(tab.to_string(index=False))
    bc = float(np.abs(est["B 심장직접"] - est["C 마스킹디코드"]).max())
    print(f"  B와 C의 최대 절대차 = {bc:.3e}   (같은 계산이면 0에 가깝다)")

    fig_three(est, clean, ds, idx, f"{outdir}/figures/three_ways.png", run, split, tab)
    with open(f"{outdir}/three_ways_note.txt", "w", encoding="utf-8") as f:
        f.write(NL.join([
            f"04 — 복원 세 방식.  {run} (에폭 {ck['epoch']}) · {split} {len(idx)}분절",
            "",
            "A 성분차감      x_noisy - s_bw - s_ma - s_em",
            "B 심장직접      s_clean = D(z1,0,0,0)",
            "C 마스킹디코드   잡음 인코딩 3개를 0으로 치환한 재구성",
            "a 입력          x_noisy (처리 전)",
            "b M0 재구성      마스킹 없이 재구성만 거친 상태",
            "",
            "B와 C는 K=4 에서 같은 연산이다(하나만 남기기 = 나머지 셋 0으로 치환).",
            f"실측 최대 절대차 {bc:.3e}.",
            "",
            "지표는 x_clean 대비. corr 은 분절 간 평균, 나머지는 중앙값이다.",
            "SSD/RMSE/MAD/PRD 는 낮을수록, corr/CosSim/SNR 은 높을수록 유사하다.",
            "mV 원단위이며 표준화하지 않는다.",
            "",
            "값에 대한 해석과 판정은 붙이지 않는다.",
        ]))
    return tab


def fig_three(est, clean, ds, idx, out, run, split, tab):
    """복원 세 방식 + 기준선 2종을 한 분절에 겹쳐 본다. 분절은 corr 중앙값 근처."""
    c0 = clean - clean.mean(-1, keepdims=True)
    y0 = est["B 심장직접"] - est["B 심장직접"].mean(-1, keepdims=True)
    r = np.abs((c0 * y0).sum(-1) /
               np.maximum(np.sqrt((c0 ** 2).sum(-1) * (y0 ** 2).sum(-1)), 1e-30))
    i = int(np.argsort(r)[len(r) // 2])          # 중앙값 분절 — 대표 사례로 고른다
    m = ds.meta[int(idx[i])]
    t = np.arange(clean.shape[1]) / ds.meta[0].get("fs", 360)

    order = ["a 입력 x_noisy", "b M0 재구성", "A 성분차감", "B 심장직접", "C 마스킹디코드"]
    fig, ax = plt.subplots(len(order) + 1, 1, figsize=(13, 1.5 * (len(order) + 1)),
                           sharex=True, sharey=True)
    ax[0].plot(t, clean[i], lw=.8, color="#333")
    ax[0].set_ylabel("x_clean", fontsize=8)
    for a, w in zip(ax[1:], order):
        a.plot(t, est[w][i], lw=.8, color="#c44e52" if w.startswith(("A", "B", "C"))
               else "#4c72b0")
        a.set_ylabel(w, fontsize=8)
    for a in ax:
        a.grid(alpha=.3, lw=.4); a.tick_params(labelsize=7)
    ax[-1].set_xlabel("시간 (초)", fontsize=8)
    head = (f"[복원 세 방식] {run} · {split} · 분절 "
            f"{m['record_id']}_{m['seg_idx']:04d} — B의 corr 중앙값 분절" + NL
            + " | ".join(f"{r0['방식']} corr {r0['corr']:.3f}"
                         for _, r0 in tab.iterrows()))
    fig.suptitle(head, fontsize=11)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight", dpi=130)
    plt.close(fig)
    fig_overlay_clean(est, clean, i, t, out.replace(".png", "_overlay.png"),
                      run, split, m, tab)


def fig_overlay_clean(est, clean, i, t, out, run, split, m, tab):
    """[겹침 그림] **처리 전과 처리 후**를 x_clean 위에 각각 올린다.

    비교 대상은 두 세트다 — (x_clean, x_noisy) 와 (x_clean, **B 재구성**). 같은 분절·같은
    y 범위이므로 잡음이 얼마나 걷혔는지 눈금이 아니라 파형으로 읽힌다. 각 칸 아래에
    잔차를 같은 범위로 두어 무엇이 남았는지 본다.

    **A 가 아니라 B 를 쓴다.** A 는 원본 x_noisy 를 그대로 유지한 채 모델 추정치만 빼므로,
    모델이 아무것도 못 뽑아도 x_noisy 만큼은 보장된다. 처리 전과 출발선이 같지 않다.
    B 는 신호 전체를 디코더가 새로 그린 결과라 처리 전과 나란히 놓을 수 있다.
    A·C 의 수치는 표(three_ways.csv)에 그대로 있다.
    """
    panes = [("처리 전", "a 입력 x_noisy", "x_noisy", "#000"),
             ("처리 후", "B 심장직접", "B 재구성 (잡음 인코딩 마스킹)", "#c44e52")]
    lim = max(np.abs(clean[i]).max(),
              *[np.abs(est[k][i]).max() for _, k, _, _ in panes]) * 1.08
    fig, ax = plt.subplots(len(panes) * 2, 1, figsize=(13, 2.4 * len(panes) * 2),
                           sharex=True, sharey=True)
    for j, (stage, key, label, c) in enumerate(panes):
        a0, a1 = ax[2 * j], ax[2 * j + 1]
        a0.plot(t, est[key][i], lw=0.75, color=c, alpha=.8, label=label)
        a0.plot(t, clean[i], lw=0.95, color="#1f77b4", label="x_clean")
        a0.legend(fontsize=8, ncol=2, loc="upper right")
        row = tab[tab["방식"] == key].iloc[0]
        a0.set_title(f"{stage} — x_clean 과 {label}   corr {row['corr']:.3f} · "
                     f"RMSE {row['RMSE']:.3f} mV · SSD {row['SSD']:.1f} · "
                     f"dSNR {row['dSNR_vs_입력']:+.2f} dB", fontsize=9, loc="left")
        a1.plot(t, clean[i] - est[key][i], lw=0.7, color="#777")
        a1.set_title(f"잔차 x_clean - {label}", fontsize=9, loc="left")
    for a in ax:
        a.set_ylim(-lim, lim)
        a.set_ylabel("mV", fontsize=8)
        a.grid(alpha=.3, lw=.4)
        a.tick_params(labelsize=7)
    ax[-1].set_xlabel("시간 (초)", fontsize=8)
    fig.suptitle(f"[복원 겹침] {run} · {split} · 분절 "
                 f"{m['record_id']}_{m['seg_idx']:04d} — B의 corr 중앙값 분절. "
                 f"전 칸 같은 y 범위 (±{lim:.2f} mV)", fontsize=11)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight", dpi=130)
    plt.close(fig)


def main(config="configs/default.yaml", run="C16_seed42", split="val", n=None, outdir=None):
    cfg = load_cfg(config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    outdir = outdir or os.path.join("results", "04_masked_denoising", run, split)
    figdir = os.path.join(outdir, "figures")
    os.makedirs(figdir, exist_ok=True)

    model, ck = load_ckpt(cfg, run)
    model = model.to(device)
    ds = load(cfg, split)
    idx = np.arange(len(ds) if n is None else min(n, len(ds)))
    K = model.n_encoders
    combos = [tuple(c) for r in range(K + 1) for c in itertools.combinations(range(K), r)]
    print(f"[04] {run} · {split} {len(idx)}분절 · 조합 {len(combos)}개 (2^{K})")

    # ---- ⓐ x_noisy 기준선
    clean = ds.refs["x_clean"][idx].astype(np.float64)
    noisy = ds.x_noisy[idx].astype(np.float64)
    base_a = metrics.s5_score(clean, noisy)

    # ---- ⓒ 전수 (ⓑ M0 = combos[0], 마스킹 없음)
    acc, rpk = sweep(model, ds, device, idx, combos)
    base_b = {k: acc[k][0] for k in acc}

    base = pd.DataFrame([
        {"상태": "ⓐ x_noisy (처리 전)",
         **{f"{k}_중앙": float(np.median(v)) for k, v in base_a.items()},
         **{f"{k}_평균": float(v.mean()) for k, v in base_a.items()}},
        {"상태": "ⓑ M0 복원 (마스킹 없음)",
         **{f"{k}_중앙": float(np.median(v)) for k, v in base_b.items()},
         **{f"{k}_평균": float(v.mean()) for k, v in base_b.items()}}])
    base["R피크_진폭비_중앙"] = [float(np.nanmedian(_amp_ratio(
        clean, noisy, [ds.rpeaks[g] for g in idx]))), float(np.nanmedian(rpk[0]))]
    base.to_csv(f"{outdir}/baseline.csv", index=False, encoding="utf-8-sig")

    tab = summarise(acc, combos, K, base_a, base_b, rpk)
    tab.to_csv(f"{outdir}/exhaustive.csv", index=False, encoding="utf-8-sig")

    best = best_by_metric(tab)
    best.to_csv(f"{outdir}/best_by_metric.csv", index=False, encoding="utf-8-sig")

    # ---- 단독 마스킹 8개
    single = tab[tab["끈_인코더수"] == 1].reset_index(drop=True)
    single.to_csv(f"{outdir}/single_mask.csv", index=False, encoding="utf-8-sig")

    # ---- 누적 곡선 (03의 잡음 유사도 순)
    order, sim = cumulative_order(run, split)
    cum = None
    if order is None:
        print("[04] 03 산출(corr_matrix.csv)이 없어 누적 곡선을 건너뛴다")
    else:
        bit2row = {r["마스크비트"]: r for _, r in tab.iterrows()}
        rows = []
        for m in range(K + 1):
            bits = "".join("1" if k in set(order[:m]) else "0" for k in range(K))
            r = dict(bit2row[bits])
            r["누적_개수"] = m
            r["추가된_인코더"] = enc_label(order[m - 1]) if m else "-"
            rows.append(r)
        cum = pd.DataFrame(rows)
        cum.to_csv(f"{outdir}/cumulative.csv", index=False, encoding="utf-8-sig")
        sim.to_frame("잡음3종_최대_|r|").to_csv(
            f"{outdir}/cumulative_order.csv", encoding="utf-8-sig")

    # ---- R-피크 진폭비
    tab[["조합", "끈_인코더수", "R피크_진폭비_중앙"]].to_csv(
        f"{outdir}/rpeak_ratio.csv", index=False, encoding="utf-8-sig")

    # ---- 주요 조건의 분절별 원값
    keep = [0] + [i for i, c in enumerate(combos) if len(c) == 1]
    per = []
    for ci in keep:
        for t, g in enumerate(idx):
            per.append({"조합": combo_label(combos[ci], K), "분절": int(g),
                        "record_id": ds.meta[int(g)]["record_id"],
                        "seg_idx": ds.meta[int(g)]["seg_idx"],
                        **{k: float(acc[k][ci, t]) for k in acc},
                        "R피크_진폭비": float(rpk[ci, t])})
    pd.DataFrame(per).to_csv(f"{outdir}/persegment_top.csv",
                             index=False, encoding="utf-8-sig")

    with open(f"{outdir}/meta.json", "w", encoding="utf-8") as f:
        json.dump({"run": run, "epoch": ck["epoch"], "split": split,
                   "n_seg": int(len(idx)), "K": int(K), "n_combos": len(combos),
                   "누적_순서": [enc_label(k) for k in order] if order else None,
                   "최적_조합_선정": "하지 않음 — 전수 결과 확인 후 별도 논의"},
                  f, ensure_ascii=False, indent=2)

    # ---- 그림
    fig_exhaustive(tab, f"{figdir}/exhaustive_scatter.png")
    fig_single(single, f"{figdir}/single_mask.png")
    fig_rpeak(tab, f"{figdir}/rpeak_ratio.png")
    fig_tradeoff(tab, f"{figdir}/metric_tradeoff.png")
    if cum is not None:
        fig_cumulative(cum, f"{figdir}/cumulative.png")

    # ---- 콘솔
    pd.set_option("display.width", 260)
    cols = ["상태"] + [f"{k}_중앙" for k in metrics.S5_METRICS] + ["R피크_진폭비_중앙"]
    print("\n[기준선] ⓐ 처리 전 · ⓑ 마스킹 없는 재구성 (clean 대비, mV 원단위)")
    print(base[cols].round(4).to_string(index=False), "\n")
    print("[지표별 지목 조합] 최적 조합 선정은 하지 않는다 — 나열만 한다")
    print(best.round(4).to_string(index=False), "\n")
    show = ["조합"] + [f"{k}_중앙" for k in metrics.S5_METRICS] + ["R피크_진폭비_중앙"]
    print("[단독 마스킹 8개]")
    print(single[show].round(4).to_string(index=False), "\n")
    if cum is not None:
        print("[누적 마스킹] 03의 잡음 3종 최대 |r| 내림차순으로 추가")
        print(cum[["누적_개수", "추가된_인코더"] +
                  [f"{k}_중앙" for k in metrics.S5_METRICS]].round(4).to_string(index=False), "\n")
    print(f"산출물 → {outdir}/")
    return tab, best


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--run", default="C16_seed42")
    p.add_argument("--split", default="val")
    p.add_argument("--n", type=int, default=None)
    p.add_argument("--outdir", default=None)
    p.add_argument("--three-ways", dest="three", action="store_true",
                   help="복원 세 방식(A 성분차감 · B 심장직접 · C 마스킹디코드)만 산출")
    a = p.parse_args()
    if a.three:
        three_ways(a.config, a.run, a.split, a.n, a.outdir)
    else:
        main(a.config, a.run, a.split, a.n, a.outdir)
