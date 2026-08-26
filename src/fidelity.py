"""T6.5 — 재구성 충실도 점검 (RESEARCH_DESIGN.md §6.5).

**위계**: 재구성 = 모델의 목표, 분리 = 부산물, 판별 = 우리 기여.
따라서 합격 여부는 **재구성 충실도 단일 축**으로만 판정한다. 분리가 잘 되어 보여도
재구성이 부실하면 그 성분은 해석 대상이 아니다.

합격 기준 (둘 다 충족)
  ① **보존율 ≥ 0.7** — 전대역과 15–40 Hz 대역 각각.
     `보존율 = P(x̂) / P(x_noisy)`. 1.0이면 완전 보존, 0이면 완전 소실.
     15–40 Hz를 따로 보는 이유: 저주파 포락선만 재현해도 전대역 보존율은 높게 나온다.
  ② **디노이징 지수** — `RMSE(x̂, x_noisy) < RMSE(x̂, clean)`.
     모델의 목표는 x_noisy다(§0 원칙 3). 자기 목표보다 clean에 더 가깝다면 요청하지 않은
     잡음 제거를 하고 있다는 뜻이고, 그때 "성분"은 학습된 분해가 아니라 평활화의 부산물이다.

진단 지표 (합격 관문 아님 — 미달 시 원인 규명과 처방 선택에 사용)
  · **잔차 상관** `|r(x_noisy − x̂, ref)|` — 못 담은 것이 무엇인지 알려준다.
    파일럿에서 잔차–clean 0.362 ≫ 잔차–ma 0.051이 나와, 소실된 고주파의 정체가
    잡음이 아니라 **QRS 모서리**임을 밝혀냈다.
  · **R-피크 진폭비** — R-피크 주변 첨두간 진폭의 재구성/입력 비. 1보다 크게 작으면
    심장 신호의 날카로운 부분이 뭉개진 것이다.
"""
import argparse
import os

import numpy as np
import pandas as pd
import torch
from scipy.signal import welch

from .data.build import load_cfg
from .data.dataset import REF_KEYS, load
from .model import meae
from .s4_identify import load_ckpt
from .viz import plt

BAND = (15.0, 40.0)
PASS_RATIO = 0.7
RPEAK_WIN = 30          # R-피크 좌우 샘플 (약 ±83 ms)


def band_power(x, fs, lo=None, hi=None):
    f, P = welch(x, fs=fs, nperseg=1024, axis=-1)
    if lo is None:
        return P.sum(-1)
    m = (f >= lo) & (f <= hi)
    return P[..., m].sum(-1)


def _corr(a, b):
    a = a - a.mean(-1, keepdims=True)
    b = b - b.mean(-1, keepdims=True)
    d = np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1)
    return np.where(d > 0, np.abs((a * b).sum(-1)) / np.maximum(d, 1e-12), 0.0)


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
def check(model, ds, device, fs, n=300, batch=100):
    pad = model.pad_each
    idx = np.arange(min(n, len(ds)))
    rows, recon_all, input_all = [], [], []
    for s in range(0, len(idx), batch):
        j = idx[s:s + batch]
        x = meae.pad(ds.tensor(j).to(device), pad)
        y = meae.crop(model(x)[0], pad).squeeze(1).cpu().numpy().astype(np.float64)
        xi = ds.x_noisy[j].astype(np.float64)
        cl = ds.refs["x_clean"][j].astype(np.float64)
        resid = xi - y
        rows.append(pd.DataFrame({
            "keep_full": band_power(y, fs) / band_power(xi, fs),
            "keep_band": band_power(y, fs, *BAND) / band_power(xi, fs, *BAND),
            "rmse_to_noisy": np.sqrt(((y - xi) ** 2).mean(-1)),
            "rmse_to_clean": np.sqrt(((y - cl) ** 2).mean(-1)),
            "rpeak_ratio": rpeak_amp_ratio(xi, y, [ds.rpeaks[k] for k in j]),
            **{f"resid_{k}": _corr(resid, ds.refs[k][j].astype(np.float64)) for k in REF_KEYS},
        }))
        recon_all.append(y)
        input_all.append(xi)
    df = pd.concat(rows, ignore_index=True)
    df["denoise_index"] = df.rmse_to_clean - df.rmse_to_noisy      # > 0 이어야 합격
    return df, np.concatenate(recon_all), np.concatenate(input_all)


def verdict(df):
    kf, kb = float(df.keep_full.median()), float(df.keep_band.median())
    frac = float((df.denoise_index > 0).mean())
    ok_keep = kf >= PASS_RATIO and kb >= PASS_RATIO
    ok_den = float(df.rmse_to_noisy.median()) < float(df.rmse_to_clean.median())
    return {
        # ---- 합격 기준 (재구성 충실도 단일 축)
        "keep_full": kf, "keep_band": kb,
        "rmse_to_noisy": float(df.rmse_to_noisy.median()),
        "rmse_to_clean": float(df.rmse_to_clean.median()),
        "denoise_ok_frac": frac,
        "pass_keep": ok_keep, "pass_denoise": ok_den, "PASS": bool(ok_keep and ok_den),
        # ---- 진단 지표 (관문 아님)
        "dx_resid_clean": float(df.resid_x_clean.median()),
        "dx_resid_bw": float(df.resid_bw.median()),
        "dx_resid_ma": float(df.resid_ma.median()),
        "dx_resid_em": float(df.resid_em.median()),
        "dx_rpeak_ratio": float(np.nanmedian(df.rpeak_ratio)),
    }


def fig_zoom(inp, rec, out, fs, i=0, t0=2.0, dur=1.5):
    a, b = int(t0 * fs), int((t0 + dur) * fs)
    t = np.arange(a, b) / fs
    fig, ax = plt.subplots(2, 1, figsize=(11, 5.5), sharex=True)
    ax[0].plot(t, inp[i][a:b], lw=.8, color="#000", label="입력 x_noisy")
    ax[0].plot(t, rec[i][a:b], lw=.8, color="#d62728", label="재구성 x_hat", alpha=.85)
    ax[0].legend(fontsize=8)
    ax[0].set_title("① 확대 — 잔떨림(고주파 변동) 재현 여부", fontsize=10, loc="left")
    ax[1].plot(t, inp[i][a:b] - rec[i][a:b], lw=.8, color="#1f77b4")
    ax[1].set_title("② 잔차 (입력 − 재구성) — 못 담은 것의 정체", fontsize=10, loc="left")
    for a_ in ax:
        a_.grid(alpha=.25, lw=.4)
        a_.set_ylabel("mV", fontsize=8)
    ax[-1].set_xlabel("시간 (초)")
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def fig_spectrum(inp, rec, out, fs):
    f, Pi = welch(inp, fs=fs, nperseg=1024, axis=-1)
    _, Pr = welch(rec, fs=fs, nperseg=1024, axis=-1)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.semilogy(f, Pi.mean(0), lw=1.2, color="#000", label="입력 x_noisy")
    ax.semilogy(f, Pr.mean(0), lw=1.2, color="#d62728", label="재구성 x_hat")
    ax.axvspan(*BAND, color="orange", alpha=.15, label=f"판정 대역 {BAND[0]:.0f}–{BAND[1]:.0f} Hz")
    ax.set_xlim(0, 60)
    ax.set_xlabel("주파수 (Hz)")
    ax.set_ylabel("PSD")
    ax.legend(fontsize=8)
    ax.grid(alpha=.3, lw=.4)
    ax.set_title("③ 평균 스펙트럼 — 고주파 소실 여부", fontsize=10, loc="left")
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


PASS_COLS = ["run", "epoch", "keep_full", "keep_band", "rmse_to_noisy", "rmse_to_clean",
             "denoise_ok_frac", "pass_keep", "pass_denoise", "PASS"]
DIAG_COLS = ["run", "dx_resid_clean", "dx_resid_bw", "dx_resid_ma", "dx_resid_em",
             "dx_rpeak_ratio"]


def main(config="configs/default.yaml", runs=None, split="val", n=300,
         outdir="results/02_separation/fidelity", figdir="results/02_separation/fidelity/figures"):
    cfg = load_cfg(config)
    fs = cfg["data"]["fs"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(figdir, exist_ok=True)
    ds = load(cfg, split)
    runs = runs or ["K8_seed42"]          # 본 실험 단일 모델. --runs 로 덮어쓴다

    rows = []
    for run in runs:
        model, ck = load_ckpt(cfg, run)
        model = model.to(device)
        df, rec, inp = check(model, ds, device, fs, n)
        v = verdict(df)
        v["run"], v["epoch"] = run, ck["epoch"]
        rows.append(v)
        fig_zoom(inp, rec, f"{figdir}/{run}_zoom.png", fs)
        fig_spectrum(inp, rec, f"{figdir}/{run}_spectrum.png", fs)

    out = pd.DataFrame(rows)
    out[PASS_COLS].to_csv(f"{outdir}/fidelity.csv", index=False, encoding="utf-8-sig")
    out[DIAG_COLS].to_csv(f"{outdir}/diagnostics.csv", index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 220)
    print(f"=== 재구성 충실도 — 서술 지표 ({split} {n}분절) ===")
    print("관문이 아니다. pass_* 열은 참고용 이력이며 합격/불합격 판정에 쓰지 않는다.")
    print("")
    print(out[PASS_COLS].round(3).to_string(index=False))
    print("")
    print("--- 진단 지표 ---")
    print(out[DIAG_COLS].round(3).to_string(index=False))
    return out


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--split", default="val")
    p.add_argument("--n", type=int, default=300)
    p.add_argument("--runs", nargs="*", default=None)
    a = p.parse_args()
    main(a.config, runs=a.runs, split=a.split, n=a.n)
