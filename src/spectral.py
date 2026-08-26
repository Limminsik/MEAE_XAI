"""충실도 진단 — 스펙트럼 정밀화 (재학습 없음).

15–40 Hz 단일 보존율은 **대역 내 에너지가 저주파에 편중되어 고주파 소실을 가린다**.
같은 0.59라도 15 Hz 부근만 살아 있고 35 Hz는 없는 경우와, 대역 전체가 고르게 0.59인 경우가
구분되지 않는다. 아래 셋으로 실태를 드러낸다.

  ① 대역별 보존율 — 5–15 / 15–25 / 25–40 / 40–60 / 60–90 Hz 각각 `P(x_hat)/P(x_noisy)`
  ② 스펙트럼 기울기 차 — 로그–로그 PSD를 10–60 Hz에서 선형 회귀한 기울기의
     (재구성 − 입력). 음수일수록 재구성이 더 가파르다 = 저역통과가 강하다
  ③ 주파수별 보존율 곡선 — 대역 경계 없이 전 주파수에서 비를 그려
     **어디서부터 무너지는지** 본다

**기준값은 정하지 않는다.** 실태 확인이 목적이다.
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
from .viz import plt

BANDS = [(5, 15), (15, 25), (25, 40), (40, 60), (60, 90)]
SLOPE_RANGE = (10.0, 60.0)
NOTCH = (59.0, 61.0)     # 전원 간섭. MIT-BIH 원본 유래이며 우리 주입 잡음과 무관하다.


def _mask(f, lo, hi, drop_notch=True):
    m = (f >= lo) & (f < hi)
    if drop_notch:
        m &= ~((f >= NOTCH[0]) & (f <= NOTCH[1]))
    return m


@torch.no_grad()
def psd_pair(model, ds, device, fs, n=300, batch=100):
    """입력·재구성의 분절별 PSD. 반환 (f, P_in, P_rec) — P는 (n, F)."""
    pad = model.pad_each
    idx = np.arange(min(n, len(ds)))
    Pi, Pr = [], []
    for s in range(0, len(idx), batch):
        j = idx[s:s + batch]
        x = meae.pad(ds.tensor(j).to(device), pad)
        y = meae.crop(model(x)[0], pad).squeeze(1).cpu().numpy().astype(np.float64)
        xi = ds.x_noisy[j].astype(np.float64)
        f, pi = welch(xi, fs=fs, nperseg=1024, axis=-1)
        _, pr = welch(y, fs=fs, nperseg=1024, axis=-1)
        Pi.append(pi)
        Pr.append(pr)
    return f, np.concatenate(Pi), np.concatenate(Pr)


def band_keep(f, Pi, Pr, bands=BANDS):
    """대역별 보존율의 분절 중앙값. 60 Hz 노치(59–61)는 제외한다."""
    out = {}
    for lo, hi in bands:
        m = _mask(f, lo, hi)
        label = f"{lo}-{hi}Hz" + ("*" if lo <= NOTCH[1] and hi > NOTCH[0] else "")
        out[label] = float(np.median(Pr[:, m].sum(1) / Pi[:, m].sum(1)))
    return out


def slope(f, P, lo=SLOPE_RANGE[0], hi=SLOPE_RANGE[1]):
    """로그–로그 PSD 기울기 (분절별로 회귀 후 중앙값)."""
    m = (f >= lo) & (f <= hi) & (f > 0) & ~((f >= NOTCH[0]) & (f <= NOTCH[1]))
    x = np.log10(f[m])
    y = np.log10(np.maximum(P[:, m], 1e-30))
    xc = x - x.mean()
    return np.median((y - y.mean(1, keepdims=True)) @ xc / (xc @ xc))


def keep_curve(f, Pi, Pr):
    """주파수별 보존율 곡선 (분절 중앙값)."""
    return np.median(Pr / np.maximum(Pi, 1e-30), axis=0)


def crossover(f, curve, level=0.7, lo=5.0):
    """보존율이 level 아래로 처음 내려가는 주파수 — 꺾임 지점의 단일 수치."""
    m = f >= lo
    ff, cc = f[m], curve[m]
    below = np.where(cc < level)[0]
    return float(ff[below[0]]) if len(below) else float(ff[-1])


def fig_curves(f, curves, out, fmax=90, base=None):
    m = f <= fmax
    fig, ax = plt.subplots(figsize=(10, 5.2))
    for run, c in curves.items():
        hl = base is not None and run != base
        ax.plot(f[m], c[m], lw=2.2 if hl else 1.3, alpha=1.0 if hl else .55,
                label=run, zorder=3 if hl else 2)
    ax.axvspan(*NOTCH, color="#999", alpha=.25, zorder=1)
    ax.axhline(1.0, color="k", lw=.8, ls="--", alpha=.6)
    ax.axhline(0.7, color="#d62728", lw=.9, ls=":", label="현행 기준 0.7 (참고선)")
    for lo, hi in BANDS:
        ax.axvline(lo, color="#bbb", lw=.5)
    ax.set_yscale("log")
    ax.set_xlabel("주파수 (Hz)")
    ax.set_ylabel("보존율  P(x_hat) / P(x_noisy)  — 분절 중앙값")
    ax.set_xlim(0, fmax)
    ax.grid(alpha=.3, lw=.4)
    ax.legend(fontsize=8, ncol=2)
    ax.set_title("주파수별 보존율 — 어디서부터 무너지는가", fontsize=11, loc="left")
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def main(config="configs/default.yaml", runs=None, split="val", n=300,
         outdir="results/02_separation/spectral", figdir="results/02_separation/spectral/figures", base=None):
    cfg = load_cfg(config)
    fs = cfg["data"]["fs"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(figdir, exist_ok=True)
    ds = load(cfg, split)
    runs = runs or ["K8_seed42"]          # 본 실험 단일 모델. --runs 로 덮어쓴다

    rows, curves = [], {}
    for run in runs:
        model, ck = load_ckpt(cfg, run)
        model = model.to(device)
        f, Pi, Pr = psd_pair(model, ds, device, fs, n)
        si, sr = slope(f, Pi), slope(f, Pr)
        c = keep_curve(f, Pi, Pr)
        rows.append({"run": run, "epoch": ck["epoch"], **band_keep(f, Pi, Pr),
                     "꺾임_0.7Hz": crossover(f, c, 0.7),
                     "꺾임_0.5Hz": crossover(f, c, 0.5),
                     "기울기_입력": si, "기울기_재구성": sr, "기울기차": sr - si})
        curves[run] = c

    out = pd.DataFrame(rows)
    out.to_csv(f"{outdir}/band_keep.csv", index=False, encoding="utf-8-sig")
    np.savez(f"{outdir}/keep_curves.npz", f=f, **{r: c for r, c in curves.items()})
    fig_curves(f, curves, f"{figdir}/keep_curve.png", base=base)
    pd.set_option("display.width", 220)
    print(f"=== 대역별 보존율 · 스펙트럼 기울기 ({split} {n}분절) ===")
    print("보존율 = P(x_hat)/P(x_noisy) 분절 중앙값. 기울기 = 로그-로그 PSD 10–60 Hz 회귀\n")
    print(out.round(3).to_string(index=False))
    return out, f, curves


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--split", default="val")
    p.add_argument("--n", type=int, default=300)
    p.add_argument("--runs", nargs="*", default=None)
    a = p.parse_args()
    main(a.config, runs=a.runs, split=a.split, n=a.n)
