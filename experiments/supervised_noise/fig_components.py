"""성분별 시각화 — 인코더 하나당 그림 한 장.

각 성분이 "무엇인가"를 보기 위한 도구. 성분 하나를 3개 분절에 걸쳐 보여준다.
분절마다 가로 3칸:

  ① 입력 x_noisy + 재구성 x̂     모델이 신호 전체를 얼마나 되돌려놓았나
  ② 참조 + 성분×β                그 성분이 어느 참조를 닮았나 (β = 최소제곱 배율)
  ③ 스펙트럼                      ②의 둘을 주파수축에서. 어느 대역을 놓쳤나

분절 선택은 대상 성분–최적 참조 |r|의 백분위로 상·중·하 3개를 고정해 뽑는다
(잘 나온 것만 고르지 않기 위함). 각 칸 제목에 그 분절의 주입 SNR을 적는다.

    python -m experiments.supervised_noise.fig_components --run SV1_K4_seed42
"""
import argparse
import os

import numpy as np
import pandas as pd
import torch
from scipy.signal import welch

from src.data.build import load_cfg
from src.data.dataset import REF_KEYS, load
from src.model import meae
from src.model.meae import enc_label
from src.s4_identify import component_bank, corr_matrix, load_ckpt
from src.viz import plt

REF_KO = {"x_clean": "clean (심장)", "bw": "bw (기저선 변동)",
          "ma": "ma (근전도)", "em": "em (전극 움직임)"}


def beta(comp, ref):
    c = comp - comp.mean()
    r = ref - ref.mean()
    d = float(c @ c)
    return float(c @ r) / d if d > 0 else 0.0


@torch.no_grad()
def reconstruct(model, ds, device, idx, batch=100):
    """(n, 3600) 재구성. 성분과 같은 분절 순서."""
    pad = model.pad_each
    out = []
    for s in range(0, len(idx), batch):
        x = meae.pad(ds.tensor(idx[s:s + batch]).to(device), pad)
        y = meae.crop(model(x)[0], pad).squeeze(1)
        out.append(y.cpu().numpy().astype(np.float64))
    return np.concatenate(out)


def fig_one(comps, refs, recon, x_noisy, cm, k, segs, ds, idx, out, fs, run, note=""):
    """성분 k 한 개를 3개 분절에 걸쳐. 분절마다 [입력·재구성] [참조·성분] [스펙트럼]."""
    j = int(cm.mean(0)[k].argmax())
    n = len(segs)
    fig, ax = plt.subplots(n, 3, figsize=(19, 3.0 * n), squeeze=False,
                           gridspec_kw={"width_ratios": [2.2, 2.2, 1.0]})
    for row, (tag, s) in enumerate(segs):
        m = ds.meta[int(idx[s])]
        t = np.arange(comps.shape[-1]) / fs
        snr = (f"주입 SNR  bw {m['snr_bw']:.1f} / ma {m['snr_ma']:.1f} "
               f"/ em {m['snr_em']:.1f} dB")

        # ① 입력 + 재구성
        a = ax[row][0]
        a.plot(t, x_noisy[s], lw=.7, color="#000000", label="입력 x_noisy")
        a.plot(t, recon[s], lw=.7, color="#d62728", alpha=.85, label="재구성 x̂")
        a.set_title(f"{tag}  ·  {m['record_id']}_{m['seg_idx']:04d}   ({snr})",
                    fontsize=9, loc="left")
        a.set_ylabel("mV", fontsize=8)
        a.legend(fontsize=7, loc="upper right")

        # ② 참조 + 성분(배율 보정)
        c = comps[s, k] - comps[s, k].mean()
        r = refs[s, j] - refs[s, j].mean()
        b = beta(comps[s, k], refs[s, j])
        a2 = ax[row][1]
        a2.plot(t, r, lw=.8, color="#2ca02c", label=f"참조 {REF_KO[REF_KEYS[j]]}")
        a2.plot(t, b * c, lw=.8, color="#1f77b4", alpha=.9,
                label=f"성분 {k+1} × β({b:.2f})")
        a2.set_title(f"성분 {k+1} vs 참조 {REF_KEYS[j]}   ·   |r| = {cm[s, k, j]:.3f}",
                     fontsize=9, loc="left")
        a2.legend(fontsize=7, loc="upper right")

        # ③ 스펙트럼
        f_, Pr = welch(r, fs=fs, nperseg=1024)
        _, Pc = welch(b * c, fs=fs, nperseg=1024)
        a3 = ax[row][2]
        a3.semilogy(f_, Pr, lw=1, color="#2ca02c")
        a3.semilogy(f_, Pc, lw=1, color="#1f77b4", alpha=.9)
        a3.set_xlim(0, 60)
        if row == 0:
            a3.set_title("스펙트럼 (②의 둘)", fontsize=9, loc="left")
        for a_ in (a, a2, a3):
            a_.grid(alpha=.25, lw=.4)
            a_.tick_params(labelsize=7)
    ax[-1][0].set_xlabel("시간 (초)")
    ax[-1][1].set_xlabel("시간 (초)")
    ax[-1][2].set_xlabel("주파수 (Hz)")
    mr = cm.mean(0)[k]
    fig.suptitle(f"{run} · 성분 {k+1}{note}   —   전체 평균 |r|  "
                 + "  ".join(f"{c_} {mr[i]:.3f}" for i, c_ in enumerate(REF_KEYS)),
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)


def fig_all(comps, refs, recon, x_noisy, cm, seg, ds, idx, out, fs, run,
            sup_refs=None):
    """한 분절에서 입력·재구성·K개 성분·참조 4종을 한 장에."""
    K = comps.shape[1]
    m = ds.meta[int(idx[seg])]
    t = np.arange(comps.shape[-1]) / fs
    rows = ([("입력 x_noisy", x_noisy[seg], "#000000"),
             ("재구성 x̂", recon[seg], "#d62728")]
            + [(f"성분 {k+1}", comps[seg, k], "#1f77b4") for k in range(K)]
            + [(f"참조 {REF_KO[c]}", refs[seg, i],
                "#2ca02c" if c == "x_clean" else "#ff7f0e")
               for i, c in enumerate(REF_KEYS)])
    fig, ax = plt.subplots(len(rows), 1, figsize=(12, 1.05 * len(rows)), sharex=True)
    for a, (lb, v, col) in zip(ax, rows):
        a.plot(t, v, lw=.6, color=col)
        note = ""
        if lb.startswith("성분"):
            k = int(lb.split()[1]) - 1
            note = ("   " + "  ".join(f"{c_} {cm.mean(0)[k, i]:.3f}"
                                     for i, c_ in enumerate(REF_KEYS)))
            if sup_refs is not None:
                note += ("   [지도: " + sup_refs[k] + "]") if k < len(sup_refs) else "   [지도 없음]"
        a.set_title(lb + note, fontsize=8, loc="left")
        a.grid(alpha=.2, lw=.3)
        a.tick_params(labelsize=6)
    ax[-1].set_xlabel("시간 (초)")
    fig.suptitle(f"{run} · 분절 {m['record_id']}_{m['seg_idx']:04d}   "
                 f"(주입 SNR  bw {m['snr_bw']:.1f} / ma {m['snr_ma']:.1f} "
                 f"/ em {m['snr_em']:.1f} dB)", fontsize=12)
    fig.tight_layout()
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)


def main(config="configs/default.yaml", run="SV1_K4_seed42", split="val", n=300,
         outdir=None, sup_refs=("bw", "ma", "em")):
    cfg = load_cfg(config)
    fs = cfg["data"]["fs"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    outdir = outdir or f"experiments/supervised_noise/outputs/figures/by_component_{run}"
    os.makedirs(outdir, exist_ok=True)

    model, ck = load_ckpt(cfg, run)
    model = model.to(device)
    ds = load(cfg, split)
    idx = np.arange(min(n, len(ds)))
    comps, refs = component_bank(model, ds, device, idx)
    recon = reconstruct(model, ds, device, idx)
    x_noisy = ds.x_noisy[idx].astype(np.float64)
    cm = corr_matrix(comps, refs)
    K = comps.shape[1]
    is_sv = str(run).startswith("SV1")
    sup = sup_refs if is_sv else None

    print(f"=== {run} (에폭 {ck['epoch']}) · {split} {len(idx)}분절 ===")
    for k in range(K):
        j = int(cm.mean(0)[k].argmax())
        p = pd.Series(cm[:, k, j]).rank(pct=True).values
        segs = [("상위 (백분위 90)", int(np.argmin(np.abs(p - .90)))),
                ("중앙 (백분위 50)", int(np.argmin(np.abs(p - .50)))),
                ("하위 (백분위 10)", int(np.argmin(np.abs(p - .10))))]
        note = "" if not is_sv else (
            f"  [지도: {sup_refs[k]}]" if k < len(sup_refs) else "  [지도 없음]")
        out = f"{outdir}/{enc_label(k)}.png"
        fig_one(comps, refs, recon, x_noisy, cm, k, segs, ds, idx, out, fs, run, note)
        print(f"  {enc_label(k)}{note}  최적 참조 {REF_KEYS[j]}  "
              + " ".join(f"{c_} {cm.mean(0)[k][i]:.3f}" for i, c_ in enumerate(REF_KEYS))
              + f"  → {os.path.basename(out)}")

    # 전 성분을 한 장에 (대표 분절 = clean 최고 인코더 |r| 백분위 50)
    ce = int(cm.mean(0)[:, 0].argmax())
    p = pd.Series(cm[:, ce, 0]).rank(pct=True).values
    seg = int(np.argmin(np.abs(p - .50)))
    fig_all(comps, refs, recon, x_noisy, cm, seg, ds, idx,
            f"{outdir}/_all_components.png", fs, run, sup)
    print(f"  전 성분 한 장 → _all_components.png")
    print(f"\n{outdir}/ 에 저장")
    return outdir


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--run", default="SV1_K4_seed42")
    p.add_argument("--split", default="val")
    p.add_argument("--n", type=int, default=300)
    p.add_argument("--outdir", default=None)
    a = p.parse_args()
    main(a.config, a.run, a.split, a.n, a.outdir)
