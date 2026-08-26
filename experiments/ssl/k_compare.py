"""T6-FINAL — K 비교 후 최종 모델 확정 (RESEARCH_DESIGN.md §7).

선정 규칙 (D52, 결과 확인 전 등록. **이 순서로 적용**)
  ① 4분할 성공   clean·bw·ma·em에 각각 **다른** 인코더가 1위인가
  ② 재구성 충실도 15–40 Hz 보존율 최상
  ③ clean 배타성  설명 몫 − 잡음 누출 최대

①은 판별이 성립하는지, ②는 성분을 해석할 수 있는지, ③은 마스킹이 통하는지를 묻는다.

    python -m experiments.ssl.k_compare
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
import torch

from src.data.build import load_cfg
from src.data.dataset import REF_KEYS, load
from src.fidelity import BAND, band_power, rpeak_amp_ratio
from src.model import meae
from src.model.meae import enc_label
from src.s4_identify import component_bank, contribution, corr_matrix, load_ckpt
from src.viz import plt

KS = (4, 8, 16)
SEEDS = (42, 202, 2026)


@torch.no_grad()
def measure(cfg, run, ds, idx, device):
    model, ck = load_ckpt(cfg, run)
    model = model.to(device)
    pad, fs = model.pad_each, cfg["data"]["fs"]
    comps, refs = component_bank(model, ds, device, idx)
    cm = corr_matrix(comps, refs).mean(0)
    share = 100 * contribution(comps, refs)[0].mean(0)

    # 재구성
    rec = []
    for s in range(0, len(idx), 100):
        x = meae.pad(ds.tensor(idx[s:s + 100]).to(device), pad)
        rec.append(meae.crop(model(x)[0], pad).squeeze(1).cpu().numpy().astype(np.float64))
    rec = np.concatenate(rec)
    xin = ds.x_noisy[idx].astype(np.float64)
    keep = np.median(band_power(rec, fs, *BAND) / band_power(xin, fs, *BAND))
    rpk = float(np.nanmedian(rpeak_amp_ratio(xin, rec, [ds.rpeaks[g] for g in idx])))

    tops = [int(cm[:, j].argmax()) for j in range(len(REF_KEYS))]
    ce = tops[0]
    return {
        "run": run, "K": model.n_encoders, "epoch": ck["epoch"],
        "4분할": len(set(tops)) == len(REF_KEYS),
        "tops": "-".join(enc_label(t) for t in tops),
        "clean_r": cm[ce, 0], "clean_몫": share[ce, 0],
        "잡음누출": share[ce, 1:].max(),
        "clean_배타성": share[ce, 0] - share[ce, 1:].max(),
        "보존율_15_40": float(keep), "R피크_진폭비": rpk,
        "_cm": cm, "_share": share, "_comps": comps, "_refs": refs,
        "_recon": rec, "_model": model,
    }


def select(rows):
    """D52 규칙을 순서대로 적용."""
    ok = [r for r in rows if r["4분할"]]
    stage = "①4분할 성공"
    if not ok:
        ok, stage = list(rows), "①4분할 성공 모델 없음 → 전체에서 선정"
    best_keep = max(r["보존율_15_40"] for r in ok)
    ok2 = [r for r in ok if r["보존율_15_40"] >= best_keep - 1e-9]
    if len(ok2) == 1:
        return ok2[0], f"{stage} → ②재구성 충실도 최상"
    win = max(ok2, key=lambda r: r["clean_배타성"])
    return win, f"{stage} → ②재구성 동률 → ③clean 배타성"


def fig_summary(tab, win, out):
    """K 비교 한 장 — 4분할 여부 · clean 몫 · 보존율."""
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.4))
    x = np.arange(len(tab))
    names = [r.replace("_seed", "\ns") for r in tab.run]
    colors = ["#2ca02c" if ok else "#bbbbbb" for ok in tab["4분할"]]
    edge = ["#d62728" if r == win["run"] else "none" for r in tab.run]
    for a, col, lab in ((ax[0], "clean_몫", "clean 설명 몫 (%)"),
                        (ax[1], "보존율_15_40", "15–40 Hz 보존율"),
                        (ax[2], "clean_r", "clean |r|")):
        a.bar(x, tab[col], color=colors, edgecolor=edge, linewidth=2.5)
        a.set_xticks(x)
        a.set_xticklabels(names, fontsize=7.5)
        a.set_ylabel(lab, fontsize=9)
        a.grid(alpha=.3, lw=.4, axis="y")
        for i, v in enumerate(tab[col]):
            a.text(i, v, f"{v:.2f}" if v < 10 else f"{v:.0f}", ha="center",
                   va="bottom", fontsize=7)
    ax[1].axhline(0.7, color="#d62728", ls=":", lw=1)
    fig.suptitle("K 비교 — 초록 = 4분할 성공 · 빨간 테두리 = 선정 모델 "
                 f"({win['run']})", fontsize=12)
    fig.tight_layout()
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)


def fig_components(res, ds, idx, out, fs):
    """대표 사례 성분 파형 — 입력·재구성·K개 성분·참조 4종."""
    cm, comps, refs, rec = res["_cm"], res["_comps"], res["_refs"], res["_recon"]
    K = comps.shape[1]
    ce = int(cm[:, 0].argmax())
    p = pd.Series(corr_matrix(comps, refs)[:, ce, 0]).rank(pct=True).values
    seg = int(np.argmin(np.abs(p - .5)))
    m = ds.meta[int(idx[seg])]
    t = np.arange(comps.shape[-1]) / fs
    rows = ([("입력 x_noisy", ds.x_noisy[int(idx[seg])], "#000"),
             ("재구성 x_hat", rec[seg], "#d62728")]
            + [(f"성분 {k+1}", comps[seg, k], "#1f77b4") for k in range(K)]
            + [(f"참조 {c}", refs[seg, i], "#2ca02c" if c == "x_clean" else "#ff7f0e")
               for i, c in enumerate(REF_KEYS)])
    fig, ax = plt.subplots(len(rows), 1, figsize=(12, 0.85 * len(rows)), sharex=True)
    for a, (lb, v, col) in zip(ax, rows):
        a.plot(t, v, lw=.6, color=col)
        note = ""
        if lb.startswith("성분"):
            k = int(lb.split()[1]) - 1
            note = "   " + "  ".join(f"{c} {cm[k, i]:.3f}" for i, c in enumerate(REF_KEYS))
        a.set_title(lb + note, fontsize=7.5, loc="left")
        a.grid(alpha=.2, lw=.3)
        a.tick_params(labelsize=6)
    ax[-1].set_xlabel("시간 (초)")
    fig.suptitle(f"{res['run']} (K={res['K']}, 에폭 {res['epoch']}) · "
                 f"분절 {m['record_id']}_{m['seg_idx']:04d}  "
                 f"(SNR bw {m['snr_bw']:.1f} / ma {m['snr_ma']:.1f} / em {m['snr_em']:.1f} dB)",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)


def main(config="configs/default.yaml", split="val", n=300,
         outdir="experiments/ssl/outputs"):
    cfg = load_cfg(config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(f"{outdir}/figures", exist_ok=True)
    ds = load(cfg, split)
    idx = np.arange(min(n, len(ds)))

    rows = []
    for K in KS:
        for sd in SEEDS:
            run = f"K{K}_seed{sd}"
            try:
                rows.append(measure(cfg, run, ds, idx, device))
            except FileNotFoundError:
                print(f"  [건너뜀] {run} 체크포인트 없음")
    cols = ["run", "K", "epoch", "4분할", "tops", "clean_r", "clean_몫", "잡음누출",
            "clean_배타성", "보존율_15_40", "R피크_진폭비"]
    tab = pd.DataFrame([{c: r[c] for c in cols} for r in rows])
    tab.to_csv(f"{outdir}/k_compare.csv", index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 240)
    print(f"=== K 비교 ({split} {len(idx)}분절) ===")
    print(tab.round(3).to_string(index=False))
    print("\n=== ① 4분할 성공 여부 (시드별) ===")
    t2 = tab.copy()
    t2["seed"] = t2.run.str.extract(r"seed(\d+)")[0].astype(int)
    piv = t2.pivot(index="K", columns="seed", values="4분할")
    piv.columns = [f"seed{c}" for c in piv.columns]
    print(piv.replace({True: "O", False: "X"}).to_string())

    win, why = select(rows)
    print(f"\n=== 최종 모델 = {win['run']} ===")
    print(f"  선정 경로: {why}")
    print(f"  4분할 {win['4분할']} ({win['tops']}) · 보존율 {win['보존율_15_40']:.3f} · "
          f"clean {win['clean_r']:.3f}/{win['clean_몫']:.1f}% · 누출 {win['잡음누출']:.2f}%")
    with open(f"{outdir}/selected.json", "w", encoding="utf-8") as f:
        json.dump({"rule": "D52 ①4분할 → ②재구성 충실도 → ③clean 배타성",
                   "selected": win["run"], "path": why,
                   "table": tab.to_dict("records")}, f, ensure_ascii=False, indent=2,
                  default=float)
    # 아홉 모델 전부 성분 파형을 남긴다 — 선정 근거를 눈으로 대조할 수 있게
    for r in rows:
        mark = "_SELECTED" if r["run"] == win["run"] else ""
        fig_components(r, ds, idx,
                       f"{outdir}/figures/components_{r['run']}{mark}.png",
                       cfg["data"]["fs"])
    fig_summary(tab, win, f"{outdir}/figures/k_compare_summary.png")
    print(f"\n산출물 → {outdir}/")
    return tab, win


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--split", default="val")
    p.add_argument("--n", type=int, default=300)
    p.add_argument("--outdir", default="experiments/ssl/outputs")
    a = p.parse_args()
    main(a.config, a.split, a.n, a.outdir)
