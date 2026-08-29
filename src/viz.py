"""그림 생성 공통 (RESEARCH_DESIGN.md §13: 300 dpi PNG, 한글 폰트)."""
import json
import os

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

for _f in ("Malgun Gothic", "NanumGothic", "AppleGothic", "DejaVu Sans"):
    if any(_f == f.name for f in matplotlib.font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = _f
        break
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["savefig.dpi"] = 300

ROWS = [("x_clean", "① 깨끗한 원본 x_clean", "#1f77b4"),
        ("bw", "② 기저선 변동 bw", "#2ca02c"),
        ("ma", "③ 근전도 ma", "#ff7f0e"),
        ("em", "④ 전극 움직임 em", "#d62728"),
        ("x_noisy", "⑤ 모델 입력 x_noisy = ①+②+③+④", "#000000")]


def spotcheck_overlay(npz_path, out_path, fs=360):
    """[겹침 그림] x_clean 과 x_noisy 를 **한 축에** 올린다.

    적층 그림은 행마다 y 범위가 달라 "잡음이 얼마나 얹혔나"를 눈으로 재기 어렵다.
    같은 축에 겹쳐 놓으면 주입량이 그대로 보인다. 아래 칸에 잡음 합(x_noisy − x_clean)을
    같은 y 범위로 함께 둔다.
    """
    z = np.load(npz_path, allow_pickle=False)
    meta = json.loads(str(z["meta"])) if "meta" in z.files else {}
    clean, noisy = z["x_clean"], z["x_noisy"]
    added = noisy - clean
    t = np.arange(len(clean)) / fs
    lim = max(np.abs(clean).max(), np.abs(noisy).max()) * 1.08

    fig, ax = plt.subplots(2, 1, figsize=(12, 6), sharex=True, sharey=True)
    ax[0].plot(t, noisy, lw=0.7, color="#000", alpha=.75, label="⑤ x_noisy (모델 입력)")
    ax[0].plot(t, clean, lw=0.9, color="#1f77b4", label="① x_clean (원본)")
    ax[0].legend(fontsize=8, ncol=2, loc="upper right")
    ax[0].set_title("x_clean 과 x_noisy 겹침 — 같은 y 범위", fontsize=9, loc="left")

    ax[1].plot(t, added, lw=0.7, color="#d62728")
    ax[1].set_title("주입된 잡음 합 x_noisy - x_clean = bw + ma + em", fontsize=9,
                    loc="left")
    for a_ in ax:
        a_.set_ylim(-lim, lim)
        a_.set_ylabel("mV", fontsize=8)
        a_.grid(alpha=.25, lw=.4)
    for r in z["rpeaks"]:
        ax[0].axvline(r / fs, color="crimson", lw=.5, alpha=.3)
    ax[-1].set_xlabel("시간 (초)")
    fig.suptitle(f"기록 {meta.get('record_id')} · 분절 {meta.get('seg_idx')} "
                 f"· {meta.get('split')} — 주입 SNR "
                 f"bw {meta.get('snr_bw', float('nan')):.1f} / "
                 f"ma {meta.get('snr_ma', float('nan')):.1f} / "
                 f"em {meta.get('snr_em', float('nan')):.1f} dB", fontsize=11)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def spotcheck(npz_path, out_path, fs=360):
    z = np.load(npz_path, allow_pickle=False)
    meta = json.loads(str(z["meta"])) if "meta" in z.files else {}
    t = np.arange(len(z["x_clean"])) / fs
    fig, ax = plt.subplots(5, 1, figsize=(11, 9), sharex=True)
    for a, (key, label, c) in zip(ax, ROWS):
        a.plot(t, z[key], lw=0.6, color=c)
        a.set_ylabel("mV", fontsize=8)
        a.grid(alpha=.25, lw=.4)
        note = ""
        if key in ("bw", "ma", "em"):
            note = f"  (SNR {meta.get(f'snr_{key}', float('nan')):.1f} dB, " \
                   f"배율 {meta.get(f'gain_{key}', float('nan')):.3f}, " \
                   f"시작 {meta.get(f'start_{key}', -1):,})"
        a.set_title(label + note, fontsize=9, loc="left")
    for r in z["rpeaks"]:
        ax[0].axvline(r / fs, color="crimson", lw=.5, alpha=.5)
        ax[4].axvline(r / fs, color="crimson", lw=.5, alpha=.35)
    ax[0].set_title(ROWS[0][1] + f"  (R-피크 {len(z['rpeaks'])}개, 세로선)", fontsize=9, loc="left")
    ax[-1].set_xlabel("시간 (초)")
    fig.suptitle(f"기록 {meta.get('record_id')} · 분절 {meta.get('seg_idx')} "
                 f"· {meta.get('split')} (잡음 풀: {meta.get('noise_pool')})", fontsize=11)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path
