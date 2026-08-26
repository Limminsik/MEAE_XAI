"""인코더 × 참조 — 상관계수 · RMSE · 기여 몫을 표 하나로.

각 칸에 세 값을 낸다.
  |r|    성분과 참조의 **형태** 일치도 (분절 단위 절대 피어슨 상관, 평균±SD)
  RMSE   성분과 참조의 **절대 차이** (원값, 평균 제거 후, 분절 중앙값)
  기여 몫 참조를 K개 성분에 다중 회귀했을 때 그 성분이 설명하는 분산 비율(%)

산출물
  table_corr_rmse_<run>.csv        표 (사람이 읽는 형식)
  table_corr_rmse_<run>_long.csv   long 형식 (참조 RMS·중앙값 포함, 재분석용)
  figures/table_corr_rmse_<run>.png   표 그림 (원고에 그대로)
  table_corr_rmse.md               모든 모델을 한 파일에 마크다운으로

    python -m experiments.supervised_noise.table_corr_rmse --run SV1_K4_seed42
"""
import argparse
import os

import numpy as np
import pandas as pd
import torch

from src.data.build import load_cfg
from src.data.dataset import REF_KEYS, load
from src.model.meae import enc_label
from src.s4_identify import component_bank, contribution, corr_matrix, load_ckpt
from src.viz import plt

REF_KO = {"x_clean": "clean\n(심장)", "bw": "bw\n(기저선 변동)",
          "ma": "ma\n(근전도)", "em": "em\n(전극 움직임)"}
REF_SHORT = {"x_clean": "clean (심장)", "bw": "bw (기저선)",
             "ma": "ma (근전도)", "em": "em (전극)"}

FOOTNOTE = """· |r|      성분과 참조의 형태 일치도 (분절 단위 절대 피어슨 상관, 평균±SD)
· RMSE     성분과 참조의 절대 차이 (원값, 평균 제거 후, 분절 중앙값).
           성분 진폭은 비선형 디코더의 산출이므로 참조와의 크기 대응이 보장되지 않는다.
· 참조 RMS  그 참조 자체의 크기. RMSE가 이 값에 가까우면 성분이 참조를 거의 설명하지 못한 것이다.
           (long 형식 CSV에 열로 들어 있다)
· 기여 몫   참조를 K개 성분에 다중 회귀했을 때 그 성분이 설명하는 분산 비율(%).
           음수는 다른 성분과 겹쳐 상쇄에 쓰였다는 뜻이다. 합이 그 참조의 R²다.
"""


def rmse_matrix(comps, refs):
    """(n, K, R) 원값 RMSE. 평균만 제거하고 배율 보정은 하지 않는다."""
    c = comps - comps.mean(-1, keepdims=True)
    r = refs - refs.mean(-1, keepdims=True)
    return np.sqrt(((c[:, :, None, :] - r[:, None, :, :]) ** 2).mean(-1))


def build(cfg, run, split="val", n=300, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model, ck = load_ckpt(cfg, run)
    model = model.to(device)
    ds = load(cfg, split)
    idx = np.arange(min(n, len(ds)))
    comps, refs = component_bank(model, ds, device, idx)
    cm = corr_matrix(comps, refs)
    rm = rmse_matrix(comps, refs)
    share, r2 = contribution(comps, refs)
    K = comps.shape[1]

    ref_rms = np.sqrt(((refs - refs.mean(-1, keepdims=True)) ** 2).mean(-1))
    wide = pd.DataFrame(index=[enc_label(k) for k in range(K)])
    for j, c in enumerate(REF_KEYS):
        wide[f"{c}_r"] = [f"{cm[:, k, j].mean():.3f}±{cm[:, k, j].std():.3f}" for k in range(K)]
        wide[f"{c}_RMSE"] = [f"{np.median(rm[:, k, j]):.4f}" for k in range(K)]
        wide[f"{c}_기여%"] = [f"{100 * share[:, k, j].mean():.1f}" for k in range(K)]

    long = []
    for k in range(K):
        for j, c in enumerate(REF_KEYS):
            long.append({"encoder": enc_label(k), "reference": c,
                         "r_mean": cm[:, k, j].mean(), "r_sd": cm[:, k, j].std(),
                         "r_median": np.median(cm[:, k, j]),
                         "rmse_median": np.median(rm[:, k, j]),
                         "rmse_mean": rm[:, k, j].mean(),
                         "ref_rms_median": np.median(ref_rms[:, j]),
                         "share_pct": 100 * share[:, k, j].mean()})
    long = pd.DataFrame(long)
    meta = {"run": run, "epoch": ck["epoch"], "split": split, "n_seg": int(len(idx)),
            "K": K, "R2_%": {c: float(100 * r2[:, j].mean()) for j, c in enumerate(REF_KEYS)}}
    return wide, long, meta


def fig_table(long, meta, out, title=None):
    """표 하나를 그림으로. 칸마다 |r| · RMSE · 기여%, 배경 음영은 |r|."""
    K, R = meta["K"], len(REF_KEYS)
    piv = {c: long[long.reference == c].set_index("encoder") for c in REF_KEYS}
    encs = [enc_label(k) for k in range(K)]
    grid = np.array([[piv[c].loc[e, "r_mean"] for c in REF_KEYS] for e in encs])
    vmax = max(0.8, float(grid.max()))

    fig, ax = plt.subplots(figsize=(2.3 * R + 1.6, 1.05 * K + 2.2))
    ax.imshow(grid, cmap="Blues", vmin=0, vmax=vmax, aspect="auto")
    for i, e in enumerate(encs):
        for j, c in enumerate(REF_KEYS):
            row = piv[c].loc[e]
            col = "white" if row.r_mean > vmax * 0.55 else "#222222"
            ax.text(j, i - 0.22, f"{row.r_mean:.3f}", ha="center", va="center",
                    fontsize=14, fontweight="bold", color=col)
            ax.text(j, i + 0.10, f"RMSE {row.rmse_median:.3f}", ha="center", va="center",
                    fontsize=9, color=col)
            ax.text(j, i + 0.31, f"기여 {row.share_pct:+.1f}%", ha="center", va="center",
                    fontsize=8.5, color=col, alpha=.85)
    ax.set_xticks(range(R))
    ax.set_xticklabels([REF_KO.get(c, c) for c in REF_KEYS], fontsize=10)
    ax.set_yticks(range(K))
    ax.set_yticklabels(encs, fontsize=12)
    ax.set_xticks(np.arange(-.5, R, 1), minor=True)
    ax.set_yticks(np.arange(-.5, K, 1), minor=True)
    ax.grid(which="minor", color="white", lw=2.5)
    ax.tick_params(which="minor", length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    r2 = "    ".join(f"{c} {v:.1f}%" for c, v in meta["R2_%"].items())
    ax.set_title(f"{title or meta['run']}    ·    에폭 {meta['epoch']}"
                 f"    ·    {meta['split']} {meta['n_seg']}분절"
                 f"\n성분 전부로 참조를 설명한 R²:    {r2}", fontsize=11, pad=14)
    fig.text(0.5, -0.015,
             "칸 = |r| (굵게) · RMSE (원값, 분절 중앙값) · 기여 몫 (다중 회귀 설명 분산)"
             "      배경 음영 = |r|",
             ha="center", fontsize=8.5, color="#555555")
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)


def md_table(long, meta):
    """마크다운 한 덩어리 — 보고서에 그대로 붙일 수 있게."""
    piv = {c: long[long.reference == c].set_index("encoder") for c in REF_KEYS}
    encs = [enc_label(k) for k in range(meta["K"])]
    lines = [f"## {meta['run']}",
             "",
             f"에폭 {meta['epoch']} · {meta['split']} {meta['n_seg']}분절",
             "",
             "| 인코더 | " + " | ".join(REF_SHORT.get(c, c) for c in REF_KEYS) + " |",
             "|---|" + "---|" * len(REF_KEYS)]
    for e in encs:
        cells = []
        for c in REF_KEYS:
            row = piv[c].loc[e]
            cells.append(f"**{row.r_mean:.3f}** · {row.rmse_median:.3f} · {row.share_pct:+.1f}%")
        lines.append(f"| {e} | " + " | ".join(cells) + " |")
    lines += ["", "칸 = **\\|r\\|** · RMSE · 기여 몫", "",
              "참조 설명 R²: "
              + " · ".join(f"{c} {v:.1f}%" for c, v in meta["R2_%"].items())]
    return "\n".join(lines)


def main(config="configs/default.yaml", runs=("SV1_K4_seed42", "K4_seed42"),
         split="val", n=300, outdir="experiments/supervised_noise/outputs"):
    cfg = load_cfg(config)
    os.makedirs(outdir, exist_ok=True)
    os.makedirs(f"{outdir}/figures", exist_ok=True)
    pd.set_option("display.width", 260)
    mds = []
    for run in runs:
        wide, long, meta = build(cfg, run, split, n)
        wide.to_csv(f"{outdir}/table_corr_rmse_{run}.csv", encoding="utf-8-sig")
        long.to_csv(f"{outdir}/table_corr_rmse_{run}_long.csv", index=False,
                    encoding="utf-8-sig")
        fig_table(long, meta, f"{outdir}/figures/table_corr_rmse_{run}.png")
        mds.append(md_table(long, meta))
        print(f"=== {run} (에폭 {meta['epoch']}) · {split} {meta['n_seg']}분절 ===")
        print(wide.to_string())
        print("  R²: " + "  ".join(f"{k} {v:.1f}%" for k, v in meta["R2_%"].items()))
        print(f"  → table_corr_rmse_{run}.csv · figures/table_corr_rmse_{run}.png\n")
    with open(f"{outdir}/table_corr_rmse.md", "w", encoding="utf-8") as f:
        f.write("# 인코더 × 참조 — 상관계수 · RMSE · 기여 몫\n\n")
        f.write("\n\n---\n\n".join(mds))
        f.write("\n\n---\n\n### 각주\n\n```\n" + FOOTNOTE + "```\n")
    print(f"마크다운: {outdir}/table_corr_rmse.md")
    return outdir


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--run", action="append", default=None)
    p.add_argument("--split", default="val")
    p.add_argument("--n", type=int, default=300)
    p.add_argument("--outdir", default="experiments/supervised_noise/outputs")
    a = p.parse_args()
    main(a.config, tuple(a.run) if a.run else ("SV1_K4_seed42", "K4_seed42"),
         a.split, a.n, a.outdir)
