"""04 — 마스킹 복원 평가 (RESEARCH_DESIGN.md §9). [S5]

03에서 확정한 모델의 **인코딩만 마스킹**해 복원하고, 그 결과를 clean 참조로 채점한다.
가중치를 다시 학습하지 않는 **추론 시점 조작**이다 (§0 원칙 1).

    python 04_masked_denoising.py --run K8_seed42 --split val

────────────────────────────────────────────────────────────────────────
전수 조합
────────────────────────────────────────────────────────────────────────
2^K = 256개 마스킹 조합 × 해당 split 전체 분절. 최적 조합 선정은 **하지 않는다** —
전수 결과를 먼저 보고, 지표 5종이 각각 어느 조합을 지목하는지 표로 제시한 뒤
일치 여부를 확인하고 선정 기준을 따로 논의해 정한다.

────────────────────────────────────────────────────────────────────────
비교 구조 — 기준은 x_clean, **mV 원단위**(표준화하지 않는다)
────────────────────────────────────────────────────────────────────────
    ⓐ x_noisy          처리 전
    ⓑ M0 복원          마스킹 없이 재구성만 거친 상태
    ⓒ 각 마스킹 조합   최종

    ⓒ − ⓐ = 전체 개선량      ⓒ − ⓑ = 마스킹 순효과

세 상태를 각각 clean과 비교한다. ⓑ를 따로 두는 이유: 개선량 중 어디까지가 재구성 자체의
몫이고 어디부터가 마스킹의 몫인지 갈라야 하기 때문이다.

────────────────────────────────────────────────────────────────────────
지표 5종 (DeepFilter·MECG-E 표준 세트)
────────────────────────────────────────────────────────────────────────
    SSD     Σ(est−clean)²                       낮을수록 유사
    MAD     max|est−clean|                      낮을수록 유사   (S4의 MAD와 달리 mV 원단위)
    PRD     100·√(Σ(est−clean)²/Σclean²)  [%]   낮을수록 유사
    CosSim  코사인 유사도                        높을수록 유사
    ΔSNR    10log10(var(clean)/mean(잔차²))      높을수록 유사

────────────────────────────────────────────────────────────────────────
부수 산출
────────────────────────────────────────────────────────────────────────
  · 단독 마스킹 8개 — 인코더를 하나씩만 껐을 때의 개별 효과
  · 누적 곡선 — 03의 잡음 유사도 순으로 하나씩 추가 마스킹했을 때의 지표 변화
  · R-피크 진폭비(복원/clean) — 형태 훼손 감시

값에 대한 해석·명명은 붙이지 않는다.

────────────────────────────────────────────────────────────────────────
산출물  results/04_masked_denoising/<run>/<split>/
────────────────────────────────────────────────────────────────────────
  exhaustive.csv        256조합 × 지표 5종 (중앙값·평균±SD) + ⓐ·ⓑ 대비 델타
  baseline.csv          ⓐ x_noisy · ⓑ M0 의 지표
  best_by_metric.csv    지표별 최적 조합과 일치 여부
  single_mask.csv       단독 마스킹 8개
  cumulative.csv        누적 마스킹 곡선
  rpeak_ratio.csv       R-피크 진폭비
  persegment_top.csv    주요 조건의 분절별 원값
  figures/  exhaustive_scatter   마스킹 개수 대 지표 5종
            metric_tradeoff     지표 간 산포 (ΔSNR–R피크 / ΔSNR–MAD / CosSim–MAD)
            single_mask · cumulative · rpeak_ratio
"""
import argparse
import itertools
import json
import os

import numpy as np
import pandas as pd
import torch

from src import metrics
from src.core import enc_names, load_ckpt
from src.data.build import load_cfg
from src.data.dataset import load
from src.model import meae
from src.model.meae import enc_label
from src.viz import plt

NOISE_REFS = ("bw", "ma", "em")
LOWER_BETTER = {"SSD": True, "MAD": True, "PRD": True, "CosSim": False, "SNR": False}


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
        zs = model.encode(x)
        zeros = [torch.zeros_like(z) for z in zs]
        for ci, mk in enumerate(combos):
            ms = set(mk)
            y = model.decode([zeros[i] if i in ms else zs[i] for i in range(K)])
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


def main(config="configs/default.yaml", run="K8_seed42", split="val", n=None, outdir=None):
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
    p.add_argument("--run", default="K8_seed42")
    p.add_argument("--split", default="val")
    p.add_argument("--n", type=int, default=None)
    p.add_argument("--outdir", default=None)
    a = p.parse_args()
    main(a.config, a.run, a.split, a.n, a.outdir)
