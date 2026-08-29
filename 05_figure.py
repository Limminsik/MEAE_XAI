"""05 — 보고용 겹침 그림.

    python 05_figure.py --run K4_seed42 --split val

**같은 분절**에 대해 두 장면을 한 그림에 둔다.

    위 칸   x_clean 과 x_noisy 겹침          처리 전 — 무엇이 얹혀 있었나
    아래 칸  x_clean 과 A 성분차감 겹침       처리 후 — 무엇이 남았나

두 칸의 y 범위를 같게 잡는다. 그래야 "잡음이 얼마나 줄었나"를 눈금이 아니라 파형으로
읽을 수 있다. 원하면 `--residual` 로 각 칸 아래에 잔차를 같은 범위로 덧붙인다.

A 성분차감은 04와 **같은 정의**다.

    A = x_noisy - ŝ_bw - ŝ_ma - ŝ_em

성분 ŝ_k 는 `model.component(x, k)` — 다른 인코딩을 0으로 치환해 디코드한 것이고,
배정은 `loss.supervise` (enc1 x_clean · enc2 bw · enc3 ma · enc4 em) 를 따른다.

분절 선정은 **복원의 corr 순위**로 한다 — 결과를 보고 고른 사례이므로 제목에 그 사실과
순위·백분위를 함께 적는다. 기본은 상위 4 · 중간 4 · 하위 4, 모두 12장이다.
중간 4는 중앙값을 가운데 두고 연속한 4개를 쓴다.

산출물  results/05_figure/<run>/<split>/
    seg_<분위>_<기록>_<분절>.png     그림
    segments.csv                     그림에 쓴 분절의 corr·RMSE·ΔSNR
"""
import argparse
import os

import numpy as np
import pandas as pd
import torch

from src import metrics
from src.core import load_ckpt
from src.data.build import load_cfg
from src.data.dataset import load
from src.model import meae
from src.viz import plt

LABELS = {"B": "B 재구성 (잡음 인코딩 마스킹)", "A": "A 성분차감"}
N_EACH = 4        # 구간마다 몇 장을 뽑을지
BANDS = ("상위", "중간", "하위")     # 복원의 corr 순으로 상위 N · 중앙 N · 하위 N


def _corr(a, b, eps=1e-30):
    """마지막 축 기준 피어슨 절댓값."""
    a = a - a.mean(-1, keepdims=True)
    b = b - b.mean(-1, keepdims=True)
    d = np.sqrt((a ** 2).sum(-1) * (b ** 2).sum(-1))
    return np.abs((a * b).sum(-1) / np.maximum(d, eps))


@torch.no_grad()
def restore(model, ds, device, idx, k_clean, k_noise, batch=100, method="B"):
    """복원 신호. 04 의 A·B 와 **같은 계산**이다.

        B (기본)  잡음 인코딩 3개를 0으로 치환한 재구성 = D(z_clean, 0, 0, 0)
        A         x_noisy - s_bw - s_ma - s_em

    **B 를 기본으로 둔다.** A 는 원본 x_noisy 를 그대로 유지한 채 모델 추정치만 빼므로,
    모델이 아무것도 못 뽑아도 최소한 x_noisy 만큼은 보장된다. 신호 전체를 디코더가 새로
    그려야 하는 B 와 출발선이 다르다. 나란히 두고 비교하려면 B 다.
    """
    pad, K = model.pad_each, model.n_encoders
    out = np.zeros((len(idx), ds.x_noisy.shape[1]))
    for s in range(0, len(idx), batch):
        j = idx[s:s + batch]
        x = meae.pad(ds.tensor(j).to(device), pad)
        # [version5] 모델 경로를 쓴다 — 잔차 연결이 있으면 함께 마스킹돼야 한다.
        cut = lambda y: meae.crop(y, pad).squeeze(1).cpu().numpy().astype(np.float64)
        if method == "B":
            out[s:s + len(j)] = cut(model.masked_reconstruct(x, k_noise))
        else:
            noise = np.zeros((len(j), ds.x_noisy.shape[1]))
            for k in k_noise:
                noise += cut(model.component(x, k))
            out[s:s + len(j)] = ds.x_noisy[j].astype(np.float64) - noise
    return out


def fig_pair(clean, noisy, rest, t, out, head, residual=False, label="B 재구성"):
    """위 칸 처리 전, 아래 칸 처리 후. 전 칸 같은 y 범위."""
    lim = max(np.abs(clean).max(), np.abs(noisy).max(), np.abs(rest).max()) * 1.08
    panes = [("처리 전 — x_clean 과 x_noisy", noisy, "#000", "x_noisy (모델 입력)"),
             (f"처리 후 — x_clean 과 {label}", rest, "#c44e52", label)]
    n = len(panes) * (2 if residual else 1)
    fig, ax = plt.subplots(n, 1, figsize=(13, 2.7 * n), sharex=True, sharey=True)
    ax = np.atleast_1d(ax)
    for p, (title, v, c, label) in enumerate(panes):
        a = ax[p * (2 if residual else 1)]
        a.plot(t, v, lw=0.75, color=c, alpha=.8, label=label)
        a.plot(t, clean, lw=0.95, color="#1f77b4", label="x_clean (원본)")
        a.legend(fontsize=8, ncol=2, loc="upper right")
        a.set_title(f"{title}   |r| {float(_corr(clean, v)):.3f} · "
                    f"RMSE {float(np.sqrt(((clean - v) ** 2).mean())):.3f} mV",
                    fontsize=9, loc="left")
        if residual:
            b = ax[p * 2 + 1]
            b.plot(t, clean - v, lw=0.7, color="#777")
            b.set_title(f"잔차 x_clean - {label}", fontsize=9, loc="left")
    for a in ax:
        a.set_ylim(-lim, lim)
        a.set_ylabel("mV", fontsize=8)
        a.grid(alpha=.28, lw=.4)
        a.tick_params(labelsize=7)
    ax[-1].set_xlabel("시간 (초)", fontsize=8)
    fig.suptitle(head + f"   ·   전 칸 같은 y 범위 (±{lim:.2f} mV)", fontsize=11)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, bbox_inches="tight", dpi=130)
    plt.close(fig)


def main(config="configs/default.yaml", run="K4_seed42", split="val", n=None,
         seg=None, residual=False, outdir=None, method="B"):
    cfg = load_cfg(config)
    fs = cfg["data"]["fs"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    outdir = outdir or os.path.join("results", "05_figure", run, split)
    os.makedirs(outdir, exist_ok=True)

    model, ck = load_ckpt(cfg, run)
    model = model.to(device).eval()
    ds = load(cfg, split)
    idx = np.arange(len(ds) if n is None else min(n, len(ds)))

    sup = list(cfg["loss"]["supervise"])
    k_clean = sup.index("x_clean")
    k_noise = [k for k in range(model.n_encoders) if k != k_clean]

    clean = ds.refs["x_clean"][idx].astype(np.float64)
    noisy = ds.x_noisy[idx].astype(np.float64)
    rest = restore(model, ds, device, idx, k_clean, k_noise, method=method)
    r_a = _corr(clean, rest)          # 복원 신호와 x_clean 의 |r|
    t = np.arange(clean.shape[1]) / fs

    # ---- 분절 선정. 이름을 직접 주면 그것만, 아니면 복원의 corr 분위수 세 곳
    if seg:
        names = {f"{m['record_id']}_{m['seg_idx']:04d}": i for i, m in
                 enumerate(ds.meta[int(idx[0]):int(idx[-1]) + 1])}
        jobs = []
        for s in seg:
            if s not in names:
                print(f"  분절 {s} 없음 — 건너뛴다")
                continue
            jobs.append((s, names[s]))
    else:
        order = np.argsort(r_a)                 # 오름차순 — 뒤가 corr 높은 쪽
        m = len(order) // 2
        picks = {"상위": order[-N_EACH:][::-1],          # 가장 높은 쪽부터
                 "중간": order[m - N_EACH // 2:m - N_EACH // 2 + N_EACH],
                 "하위": order[:N_EACH]}                 # 가장 낮은 쪽부터
        jobs = [(f"{b}{i + 1}", int(v)) for b in BANDS for i, v in enumerate(picks[b])]

    rows = []
    for lab, i in jobs:
        m = ds.meta[int(idx[i])]
        name = f"{m['record_id']}_{m['seg_idx']:04d}"
        pct = float((r_a < r_a[i]).mean() * 100)
        head = (f"[05] {run} (에폭 {ck['epoch']}) · {split} · 분절 {name} — "
                f"복원의 corr {r_a[i]:.3f}  [{lab}]  {split} {len(idx)}분절 중 상위 "
                f"{100 - pct:.1f}%")
        fig_pair(clean[i], noisy[i], rest[i], t,
                 os.path.join(outdir, f"seg_{lab}_{name}.png"), head, residual,
                 label=LABELS[method])
        snr_in = float(metrics.snr_db_vec(clean[i:i + 1], noisy[i:i + 1])[0])
        snr_a = float(metrics.snr_db_vec(clean[i:i + 1], rest[i:i + 1])[0])
        rows.append({"구간": lab, "분절": name, "상위백분위": round(100 - pct, 2),
                     "corr_noisy": float(_corr(clean[i], noisy[i])),
                     "corr_복원": float(r_a[i]),
                     "RMSE_noisy": float(np.sqrt(((clean[i] - noisy[i]) ** 2).mean())),
                     "RMSE_복원": float(np.sqrt(((clean[i] - rest[i]) ** 2).mean())),
                     "SNR_noisy": snr_in, "SNR_복원": snr_a, "dSNR": snr_a - snr_in,
                     "주입SNR_bw": m.get("snr_bw"), "주입SNR_ma": m.get("snr_ma"),
                     "주입SNR_em": m.get("snr_em")})
    tab = pd.DataFrame(rows).round(4)
    tab.to_csv(os.path.join(outdir, "segments.csv"), index=False, encoding="utf-8-sig")
    print("")
    print(f"[05 겹침 그림] {run} (에폭 {ck['epoch']}) · {split} {len(idx)}분절")
    print(tab.to_string(index=False))
    print(f"산출물 → {outdir}/")
    return tab


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--run", default="K4_seed42")
    p.add_argument("--split", default="val")
    p.add_argument("--n", type=int, default=None)
    p.add_argument("--seg", nargs="*", default=None,
                   help="분절 이름 지정 (예: 100_0036). 없으면 상위4·중간4·하위4")
    p.add_argument("--residual", action="store_true", help="각 칸 아래에 잔차를 덧붙인다")
    p.add_argument("--method", default="B", choices=["B", "A"],
                   help="복원 방식. B = 마스킹 재구성(기본), A = 성분차감")
    p.add_argument("--outdir", default=None)
    a = p.parse_args()
    main(a.config, a.run, a.split, a.n, a.seg, a.residual, a.outdir, a.method)
