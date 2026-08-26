"""후보 에폭 나란히 비교 — 배율 확정 전 실물 대조용.

비용 함수 ②의 1단계 배율에 따라 선택 에폭이 갈릴 때, 후보들의 실제 차이를 확인한다.
가중치는 `results/01_train/<run>/pool/` 에 보관돼 있으므로 재학습하지 않는다.

    python -m src.compare_epochs --run K8_seed42 --epochs 48 88 --split val --n 900

산출물 (results/00_rehearsal/epoch_compare/)
  recon.csv          전대역·대역별 보존율, R-피크 진폭비, 잔차 상관, 기울기차
  corr.csv           인코더 x 참조 |r| (에폭별)
  rmse_norm.csv      인코더 x 참조 RMSE_norm (에폭별)
  mad.csv            인코더 x 참조 MAD (에폭별)
  contribution.csv   기여 분해 (에폭별)
  summary.csv        clean 최대 인코더의 |r| 과 잡음 상관
  figures/
    zoom.png           확대 파형 + 잔차 (에폭 겹쳐 그림)
    keep_curve.png     주파수별 보존 곡선
    corr_heatmap.png   상관 행렬 나란히
    components_seg####.png   동일 분절 성분 파형 (에폭별 1장)
"""
import argparse
import os

import numpy as np
import pandas as pd
import torch

from .data.build import load_cfg
from .data.dataset import REF_KEYS, load
from .fidelity import check, verdict
from .model.meae import enc_label
from .s4_identify import (component_bank, contribution, load_ckpt,
                          metric_matrices)
from .spectral import band_keep, crossover, keep_curve, psd_pair, slope
from .viz import plt


def ckpt_path(run, epoch):
    p = os.path.join("results", "01_train", run, "pool", f"ep{epoch:04d}.pt")
    if not os.path.exists(p):
        raise FileNotFoundError(f"{p} 없음 — pool 에 보관된 에폭만 비교할 수 있다")
    return p


def fig_zoom_pair(inp, recs, out, fs, i=0, t0=2.0, dur=1.5):
    """확대 파형 — 입력 위에 후보 에폭 재구성을 겹치고, 잔차는 따로 쌓는다."""
    a, b = int(t0 * fs), int((t0 + dur) * fs)
    t = np.arange(a, b) / fs
    cols = ["#d62728", "#1f77b4", "#2ca02c", "#9467bd"]
    fig, ax = plt.subplots(1 + len(recs), 1, figsize=(11, 2.6 * (1 + len(recs))), sharex=True)
    ax[0].plot(t, inp[i][a:b], lw=1.0, color="#000", label="입력 x_noisy")
    for c, (lab, r) in zip(cols, recs.items()):
        ax[0].plot(t, r[i][a:b], lw=.9, color=c, alpha=.85, label=f"재구성 {lab}")
    ax[0].legend(fontsize=8)
    ax[0].set_title("① 확대 — 입력과 후보 에폭 재구성", fontsize=10, loc="left")
    for a_, c, (lab, r) in zip(ax[1:], cols, recs.items()):
        a_.plot(t, inp[i][a:b] - r[i][a:b], lw=.8, color=c)
        a_.set_title(f"② 잔차 (입력 - 재구성) · {lab}", fontsize=10, loc="left")
    for a_ in ax:
        a_.grid(alpha=.25, lw=.4)
        a_.set_ylabel("mV", fontsize=8)
    ax[-1].set_xlabel("시간 (초)")
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def fig_keep_pair(f, curves, out, fmax=90):
    fig, ax = plt.subplots(figsize=(9, 4.6))
    for lab, c in curves.items():
        ax.plot(f, c, lw=1.2, label=lab)
    ax.axhline(1.0, color="#888", ls=":", lw=.8)
    ax.axhline(0.7, color="#d62728", ls=":", lw=.8)
    ax.set_xlim(0, fmax)
    ax.set_ylim(0, 1.35)
    ax.set_xlabel("주파수 (Hz)")
    ax.set_ylabel("보존율 P(x_hat)/P(x_noisy)")
    ax.legend(fontsize=9)
    ax.grid(alpha=.3, lw=.4)
    ax.set_title("주파수별 보존 곡선 — 후보 에폭 비교", fontsize=11)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def fig_corr_pair(cms, out):
    labs = list(cms)
    K = cms[labs[0]].shape[1]
    vmax = max(0.6, max(c.mean(0).max() for c in cms.values()))
    fig, ax = plt.subplots(1, len(labs), figsize=(4.2 * len(labs), 0.62 * K + 2))
    ax = np.atleast_1d(ax)
    for a, lab in zip(ax, labs):
        m, sd = cms[lab].mean(0), cms[lab].std(0)
        im = a.imshow(m, cmap="magma", vmin=0, vmax=vmax, aspect="auto")
        a.set_xticks(range(len(REF_KEYS)))
        a.set_xticklabels(list(REF_KEYS))
        a.set_yticks(range(K))
        a.set_yticklabels([enc_label(k) for k in range(K)])
        for k in range(K):
            for r in range(len(REF_KEYS)):
                a.text(r, k, f"{m[k, r]:.2f}\n±{sd[k, r]:.2f}", ha="center", va="center",
                       fontsize=6.5, color="white" if m[k, r] < vmax * .6 else "black")
        for r in range(len(REF_KEYS)):
            a.add_patch(plt.Rectangle((r - .5, m[:, r].argmax() - .5), 1, 1,
                                      fill=False, ec="cyan", lw=2))
        a.set_title(lab, fontsize=10)
        fig.colorbar(im, ax=a, shrink=.8)
    fig.suptitle("인코더 × 참조 |r| 평균±SD — 후보 에폭 비교", fontsize=11)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def fig_components(comps, refs, x_noisy, i, out, title, fs):
    K = comps.shape[1]
    rows = ([("입력 x_noisy", x_noisy[i], "#000")]
            + [(f"성분 {k+1}", comps[i, k], "#1f77b4") for k in range(K)]
            + [(f"참조 {n}", refs[i, list(REF_KEYS).index(n)],
                "#2ca02c" if n == "x_clean" else "#d62728") for n in REF_KEYS])
    t = np.arange(comps.shape[-1]) / fs
    fig, ax = plt.subplots(len(rows), 1, figsize=(11, 0.95 * len(rows)), sharex=True)
    for a, (lb, v, c) in zip(ax, rows):
        a.plot(t, v, lw=.6, color=c)
        a.set_title(lb, fontsize=8, loc="left")
        a.grid(alpha=.25, lw=.4)
        a.tick_params(labelsize=7)
    ax[-1].set_xlabel("시간 (초)")
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main(config="configs/default.yaml", run="K8_seed42", epochs=(48, 88),
         split="val", n=900, seg=None,
         outdir="results/00_rehearsal/epoch_compare"):
    cfg = load_cfg(config)
    fs = cfg["data"]["fs"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    figdir = os.path.join(outdir, "figures")
    os.makedirs(figdir, exist_ok=True)
    ds = load(cfg, split)
    idx = np.arange(min(n, len(ds)))
    labs = [f"에폭 {e}" for e in epochs]

    recon_rows, corr_rows, rmse_rows, mad_rows, con_rows, sum_rows = [], [], [], [], [], []
    recs, curves, cms, banks = {}, {}, {}, {}
    f = None

    for e, lab in zip(epochs, labs):
        model, ck = load_ckpt(cfg, ckpt_path(run, e))
        model = model.to(device)

        # ---- 재구성
        df, rec, inp = check(model, ds, device, fs, len(idx))
        v = verdict(df)
        f, Pi, Pr = psd_pair(model, ds, device, fs, len(idx))
        c = keep_curve(f, Pi, Pr)
        si, sr = slope(f, Pi), slope(f, Pr)
        recon_rows.append({
            "에폭": e, "val_recon(학습기록)": ck.get("val_recon"), "S(학습기록)": ck.get("S"),
            "보존율_전대역": v["keep_full"], **band_keep(f, Pi, Pr),
            "꺾임_0.7": crossover(f, c, 0.7), "꺾임_0.5": crossover(f, c, 0.5),
            "기울기_입력": si, "기울기_재구성": sr, "기울기차": sr - si,
            "RMSE_대_noisy": v["rmse_to_noisy"], "RMSE_대_clean": v["rmse_to_clean"],
            "디노이징지수_충족비": v["denoise_ok_frac"],
            "R피크_진폭비": v["dx_rpeak_ratio"],
            "잔차_clean": v["dx_resid_clean"], "잔차_bw": v["dx_resid_bw"],
            "잔차_ma": v["dx_resid_ma"], "잔차_em": v["dx_resid_em"],
        })
        recs[lab], curves[lab] = rec, c

        # ---- 분리
        comps, refs = component_bank(model, ds, device, idx)
        cm, rn_m, mad = metric_matrices(comps, refs)
        share, r2 = contribution(comps, refs)
        K = comps.shape[1]
        cms[lab], banks[lab] = cm, (comps, refs)
        ix = [enc_label(k) for k in range(K)]
        for name, arr, sink in (("corr", cm, corr_rows), ("rmse_norm", rn_m, rmse_rows),
                                ("mad", mad, mad_rows)):
            t = pd.DataFrame(arr.mean(0), columns=list(REF_KEYS), index=ix)
            t.insert(0, "에폭", e)
            t.index.name = "인코더"
            sink.append(t.reset_index())
        ct = pd.DataFrame(100 * share.mean(0), columns=list(REF_KEYS), index=ix)
        ct.insert(0, "에폭", e)
        ct.index.name = "인코더"
        con_rows.append(ct.reset_index())

        m = cm.mean(0)
        kc = int(m[:, 0].argmax())                       # clean 최대 상관 인코더
        sum_rows.append({
            "에폭": e, "clean_최대_인코더": enc_label(kc),
            "clean_|r|": m[kc, 0], "그_인코더_bw": m[kc, 1], "그_인코더_ma": m[kc, 2],
            "그_인코더_em": m[kc, 3], "잡음_최대": m[kc, 1:].max(),
            "clean_기여%": 100 * share.mean(0)[kc, 0],
            "그_인코더_잡음기여합%": float(100 * share.mean(0)[kc, 1:].sum()),
            "잡음_최대_인코더": "/".join(enc_label(int(m[:, j].argmax())) for j in (1, 2, 3)),
            "R2_clean%": float(100 * r2.mean(0)[0]),
        })

    # ---- 저장
    rec_df = pd.DataFrame(recon_rows)
    rec_df.to_csv(f"{outdir}/recon.csv", index=False, encoding="utf-8-sig")
    pd.concat(corr_rows).to_csv(f"{outdir}/corr.csv", index=False, encoding="utf-8-sig")
    pd.concat(rmse_rows).to_csv(f"{outdir}/rmse_norm.csv", index=False, encoding="utf-8-sig")
    pd.concat(mad_rows).to_csv(f"{outdir}/mad.csv", index=False, encoding="utf-8-sig")
    pd.concat(con_rows).to_csv(f"{outdir}/contribution.csv", index=False, encoding="utf-8-sig")
    sum_df = pd.DataFrame(sum_rows)
    sum_df.to_csv(f"{outdir}/summary.csv", index=False, encoding="utf-8-sig")

    # ---- 그림
    inp_arr = ds.x_noisy[idx].astype(np.float64)
    fig_zoom_pair(inp_arr, recs, f"{figdir}/zoom.png", fs)
    fig_keep_pair(f, curves, f"{figdir}/keep_curve.png")
    fig_corr_pair(cms, f"{figdir}/corr_heatmap.png")

    # 동일 분절 성분 파형 — 지정이 없으면 첫 에폭의 clean 상관 중앙 분절
    if seg is None:
        m0 = cms[labs[0]].mean(0)
        kc0 = int(m0[:, 0].argmax())
        v = cms[labs[0]][:, kc0, 0]
        seg = int(np.argsort(v)[len(v) // 2])
    md = ds.meta[int(idx[seg])]
    for e, lab in zip(epochs, labs):
        comps, refs = banks[lab]
        mm = cms[lab].mean(0)
        kc = int(mm[:, 0].argmax())
        fig_components(
            comps, refs, ds.x_noisy[idx], seg,
            f"{figdir}/components_seg{seg:04d}_ep{e:04d}.png",
            f"{run} 에폭 {e} · 분절 {md['record_id']}_{md['seg_idx']:04d} "
            f"(SNR bw {md['snr_bw']:.1f} / ma {md['snr_ma']:.1f} / em {md['snr_em']:.1f} dB) · "
            f"clean 최대 = {enc_label(kc)} |r| {cms[lab][seg, kc, 0]:.3f}", fs)

    # ---- 콘솔
    pd.set_option("display.width", 260)
    print(f"=== {run} 후보 에폭 비교 · {split} {len(idx)}분절 ===\n")
    print("[재구성]")
    print(rec_df.round(4).to_string(index=False), "\n")
    print("[분리 요약] clean 최대 상관 인코더 기준")
    print(sum_df.round(4).to_string(index=False), "\n")
    for lab, t in zip(labs, corr_rows):
        print(f"[인코더 × 참조 |r|] {lab}")
        print(t.round(3).to_string(index=False), "\n")
    for lab, t in zip(labs, rmse_rows):
        print(f"[인코더 × 참조 RMSE_norm] {lab}   S4-02, 낮을수록 유사")
        print(t.round(3).to_string(index=False), "\n")
    for lab, t in zip(labs, mad_rows):
        print(f"[인코더 × 참조 MAD] {lab}   국소 최대 편차, |ρ|과 독립")
        print(t.round(3).to_string(index=False), "\n")
    for lab, t in zip(labs, con_rows):
        print(f"[기여 분해 %] {lab}")
        print(t.round(2).to_string(index=False), "\n")
    print(f"동일 분절 성분 파형: 분절 {seg} ({md['record_id']}_{md['seg_idx']:04d})")
    print(f"산출물 → {outdir}/")
    return rec_df, sum_df


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--run", default="K8_seed42")
    p.add_argument("--epochs", type=int, nargs="+", default=[48, 88])
    p.add_argument("--split", default="val")
    p.add_argument("--n", type=int, default=900)
    p.add_argument("--seg", type=int, default=None)
    a = p.parse_args()
    main(a.config, a.run, tuple(a.epochs), a.split, a.n, a.seg)
