"""재구성 충실도 진단 (RESEARCH_DESIGN.md §7.5).

**관문이 아니라 서술 지표다.** 선행이 명시한 재구성–분리 트레이드오프
("인코딩이 너무 작으면 좋은 재구성이 불가하고, 너무 크면 인코더가 특화 대신 전체 특징 공간으로
일반화된다")를 어디쯤에서 취했는지 보고하기 위한 산출물이며, 합격/불합격 수치를 두지 않는다.

산출 지표
  · **보존율** `P(x̂) / P(x_noisy)` — 전대역과 대역별(5–15 / 15–25 / 25–40 / 40–60 / 60–90 Hz).
    1.0이면 완전 보존, 0이면 완전 소실. 대역을 나누는 이유: 저주파 포락선만 재현해도
    전대역 보존율은 높게 나온다. 40–60·60–90 Hz는 **59–61 Hz 노치를 제외**하고 산출한다
    (전원 간섭. MIT-BIH 원본 유래이며 우리 주입 잡음과 무관하다).
  · **꺾임 지점** — 보존율이 0.7 / 0.5 아래로 처음 내려가는 주파수.
  · **로그–로그 PSD 기울기** 입력·재구성과 그 차이 (10–60 Hz 회귀).
  · **R-피크 진폭비** — R-피크 ±30샘플 첨두간 진폭의 재구성/입력 비.
    1보다 크게 작으면 심장 신호의 날카로운 부분이 뭉개진 것이다.

**디노이징 지수와 잔차 상관은 산출하지 않는다.**

    python -m src.fidelity --run K8_seed42 --split val --n 900
"""
import argparse
import os

import numpy as np
import pandas as pd
import torch
from scipy.signal import welch

from .data.build import load_cfg
from .data.dataset import load
from .model import meae
from .s4_identify import load_ckpt
from .spectral import band_keep, crossover, keep_curve, psd_pair, slope
from .viz import plt

RPEAK_WIN = 30          # R-피크 좌우 샘플 (약 ±83 ms)
BAND = (15.0, 40.0)     # 보조 기록(experiments/)이 참조하는 단일 대역


def band_power(x, fs, lo=None, hi=None):
    f, P = welch(x, fs=fs, nperseg=1024, axis=-1)
    if lo is None:
        return P.sum(-1)
    m = (f >= lo) & (f <= hi)
    return P[..., m].sum(-1)


def rpeak_amp_ratio(inp, rec, peaks_list, win=RPEAK_WIN):
    """R-피크 주변 첨두간 진폭의 재구성/입력 비 (분절별 중앙값)."""
    out = []
    for i, peaks in enumerate(peaks_list):
        r = []
        for p in peaks:
            a, b = max(0, p - win), min(inp.shape[-1], p + win)
            di = inp[i, a:b].max() - inp[i, a:b].min()
            dr = rec[i, a:b].max() - rec[i, a:b].min()
            if di > 0:
                r.append(dr / di)
        out.append(np.median(r) if r else np.nan)
    return np.array(out)


@torch.no_grad()
def reconstruct(model, ds, device, idx, batch=100):
    """(n, 3600) 입력과 재구성. 모두 crop 후 중앙 구간이다."""
    pad = model.pad_each
    rec = []
    for s in range(0, len(idx), batch):
        j = idx[s:s + batch]
        x = meae.pad(ds.tensor(j).to(device), pad)
        rec.append(meae.crop(model(x)[0], pad).squeeze(1).cpu().numpy().astype(np.float64))
    return ds.x_noisy[idx].astype(np.float64), np.concatenate(rec)


def fig_zoom(inp, rec, out, fs, i=0, t0=2.0, dur=1.5):
    a, b = int(t0 * fs), int((t0 + dur) * fs)
    t = np.arange(a, b) / fs
    fig, ax = plt.subplots(figsize=(11, 3.4))
    ax.plot(t, inp[i][a:b], lw=.9, color="#000", label="입력 x_noisy")
    ax.plot(t, rec[i][a:b], lw=.9, color="#d62728", alpha=.85, label="재구성 x_hat")
    ax.legend(fontsize=8)
    ax.set_ylabel("mV", fontsize=8)
    ax.set_xlabel("시간 (초)")
    ax.grid(alpha=.25, lw=.4)
    ax.set_title("확대 — 잔떨림(고주파 변동) 재현 여부", fontsize=10, loc="left")
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def fig_spectrum(f, Pi, Pr, out):
    fig, ax = plt.subplots(figsize=(9, 4.6))
    ax.loglog(f[1:], np.median(Pi, 0)[1:], lw=1.2, color="#000", label="입력 x_noisy")
    ax.loglog(f[1:], np.median(Pr, 0)[1:], lw=1.2, color="#d62728", label="재구성 x_hat")
    ax.set_xlabel("주파수 (Hz)")
    ax.set_ylabel("PSD (분절 중앙값)")
    ax.grid(alpha=.3, lw=.4, which="both")
    ax.legend(fontsize=8)
    ax.set_title("PSD — 입력 대비 재구성", fontsize=10, loc="left")
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def fig_keep(f, curve, out, fmax=90):
    m = f <= fmax
    fig, ax = plt.subplots(figsize=(9, 4.6))
    ax.plot(f[m], curve[m], lw=1.4, color="#1f77b4")
    ax.axhline(1.0, color="k", lw=.8, ls="--", alpha=.6)
    ax.axhline(0.7, color="#d62728", lw=.9, ls=":")
    ax.axvspan(59, 61, color="#999", alpha=.25)
    ax.set_xlim(0, fmax)
    ax.set_ylim(0, 1.35)
    ax.set_xlabel("주파수 (Hz)")
    ax.set_ylabel("보존율  P(x_hat) / P(x_noisy)  — 분절 중앙값")
    ax.grid(alpha=.3, lw=.4)
    ax.set_title("주파수별 보존율 — 어디서부터 무너지는가 (회색 = 59–61 Hz 제외 구간)",
                 fontsize=10, loc="left")
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def main(config="configs/default.yaml", run="K8_seed42", split="val", n=900,
         outdir=None):
    cfg = load_cfg(config)
    fs = cfg["data"]["fs"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    outdir = outdir or os.path.join("results", "01_train", run, "fidelity")
    figdir = os.path.join(outdir, "figures")
    os.makedirs(figdir, exist_ok=True)

    model, ck = load_ckpt(cfg, run)
    model = model.to(device)
    ds = load(cfg, split)
    idx = np.arange(min(n, len(ds)))

    inp, rec = reconstruct(model, ds, device, idx)
    f, Pi, Pr = psd_pair(model, ds, device, fs, len(idx))
    curve = keep_curve(f, Pi, Pr)
    row = {
        "run": run, "epoch": ck["epoch"], "split": split, "n_seg": len(idx),
        "보존율_전대역": float(np.median(band_power(rec, fs) / band_power(inp, fs))),
        **band_keep(f, Pi, Pr),
        "꺾임_0.7Hz": crossover(f, curve, 0.7),
        "꺾임_0.5Hz": crossover(f, curve, 0.5),
        "기울기_입력": slope(f, Pi), "기울기_재구성": slope(f, Pr),
        "기울기차": slope(f, Pr) - slope(f, Pi),
        "R피크_진폭비": float(np.nanmedian(
            rpeak_amp_ratio(inp, rec, [ds.rpeaks[k] for k in idx]))),
    }
    out = pd.DataFrame([row])
    out.to_csv(f"{outdir}/fidelity.csv", index=False, encoding="utf-8-sig")
    np.savez(f"{outdir}/keep_curve.npz", f=f, curve=curve)
    with open(f"{outdir}/fidelity_note.txt", "w", encoding="utf-8") as fn:
        fn.write(
            "재구성 충실도 진단 — 관문이 아니라 서술 지표다. 합격/불합격 수치를 두지 않는다.\n"
            "보존율 = P(x_hat)/P(x_noisy) 의 분절 중앙값. 1.0 = 완전 보존.\n"
            "* 붙은 대역(40-60, 60-90 Hz)은 59-61 Hz 노치를 제외하고 산출했다\n"
            "  (전원 간섭. MIT-BIH 원본 유래이며 주입 잡음과 무관하다).\n"
            "기울기는 로그-로그 PSD 를 10-60 Hz 에서 회귀한 분절 중앙값.\n"
            "R-피크 진폭비 = R-피크 +-30샘플 첨두간 진폭의 재구성/입력 비.\n"
            "디노이징 지수와 잔차 상관은 산출하지 않는다.\n")

    fig_zoom(inp, rec, f"{figdir}/zoom.png", fs)
    fig_spectrum(f, Pi, Pr, f"{figdir}/spectrum.png")
    fig_keep(f, curve, f"{figdir}/keep_curve.png")

    pd.set_option("display.width", 220)
    print(f"=== 재구성 충실도 — 서술 지표 ({run} · {split} {len(idx)}분절) ===")
    print(out.round(4).to_string(index=False))
    print(f"\n산출물 → {outdir}/")
    return out


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--run", default="K8_seed42")
    p.add_argument("--split", default="val")
    p.add_argument("--n", type=int, default=900)
    p.add_argument("--outdir", default=None)
    a = p.parse_args()
    main(a.config, a.run, a.split, a.n, a.outdir)
