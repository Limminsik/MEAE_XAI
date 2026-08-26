"""보조 실험 — 분리 성분 제거(remove-some) 대 성분 유지(keep-one) 대조.

**본 노선이 아니다.** 04는 remove-some(일부 인코딩을 0으로) 만 수행한다.
여기서는 그 연산이 선행의 추론 절차(keep-one, 하나만 남김)와 어떻게 다른지를
산출로 확인한다. 디노이징 전략을 정하기 위한 탐색이며, 결과는 본 실험 표에 섞지 않는다.

    python experiments/masking_strategy/diagnose.py --run K8_seed42 --split val

산출  experiments/masking_strategy/outputs/<run>/<split>/
  decoder_blocks.csv · .npz · figures/decoder_blocks.png
        디코더 각 층 가중치의 K×K 블록 구조와 비대각/대각 비
  keep_one.csv                성분만 남겼을 때의 지표 5종 (스케일 보정 전/후)
  removesome_vs_keepone.csv   같은 인코더에 대한 두 연산 대조
  figures/mask_vs_keep.png    한 분절의 파형 대조
  subtract/                   ⑤ 신호 공간 뺄셈 (--subtract)
  note.txt

해석과 처방은 붙이지 않는다. 산출과 보고만.
"""
import argparse
import itertools
import os
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src import metrics                                        # noqa: E402
from src.core import enc_names, load_ckpt                      # noqa: E402
from src.data.build import load_cfg                            # noqa: E402
from src.data.dataset import load                              # noqa: E402
from src.model import meae                                     # noqa: E402
from src.model.meae import enc_label                           # noqa: E402
from src.viz import plt                                        # noqa: E402

LOWER_BETTER = {"SSD": True, "MAD": True, "PRD": True,
                "CosSim": False, "SNR": False}


def combo_label(mask, K):
    """마스킹 조합 이름 — 끈 인코더를 1-based 로 나열한다."""
    return "-".join(enc_label(k) for k in mask) if mask else "M0(없음)"


# ================================================================
# 진단 — remove-some 이 선행의 keep-one 과 다른 연산이라는 점을 산출로 확인한다.
#   ① 디코더 블록 구조 (선행 Fig.4·12 방식)
#   ② keep-one 디노이징 평가 (재학습 없음)
#   ③ 스케일 보정 유무 대조
#   ④ 마스킹 후 복원 파형과 성분 파형 대조
# 해석·처방은 붙이지 않는다. 산출과 보고만.
# ================================================================
def decoder_blocks(model):
    """디코더 각 층 가중치의 |W| 를 공간축으로 합산해 K×K 블록으로 접는다.

    선행 논문 Fig.4·12 방식. 출력 채널을 K등분(행), 입력 채널을 K등분(열)하고
    블록 안의 평균 |W| 를 값으로 쓴다. 대각 블록은 같은 인코딩 안에서의 연결,
    비대각 블록은 **인코딩 사이의 섞임**이다. sparse mixing 손실이 벌하는 대상이
    바로 비대각 블록이다(출력층 제외).
    """
    K = model.n_encoders
    out = []
    for name, w in model.net.decoder.named_parameters():
        if name.split(".")[-1] != "weight" or len(w.shape) not in (2, 3, 4):
            continue
        a = w.detach().abs()
        while a.dim() > 2:                      # 공간축(커널) 합산
            a = a.sum(-1)
        o, i = a.shape[0] // K, a.shape[1] // K
        if o == 0 or i == 0:
            continue
        blk = np.zeros((K, K))
        for r in range(K):
            for c in range(K):
                blk[r, c] = a[o * r:o * (r + 1), i * c:i * (c + 1)].mean().item()
        out.append((name, tuple(w.shape), blk))
    return out


def fig_decoder_blocks(blocks, out, run):
    n = len(blocks)
    ncol = min(4, n)
    nrow = int(np.ceil(n / ncol))
    fig, ax = plt.subplots(nrow, ncol, figsize=(3.6 * ncol, 3.4 * nrow))
    ax = np.atleast_1d(ax).ravel()
    for a, (name, shape, blk) in zip(ax, blocks):
        d = np.diag(blk).mean()
        off = (blk.sum() - np.trace(blk)) / (blk.size - blk.shape[0])
        im = a.imshow(blk, cmap="magma")
        a.set_title(f"{name.replace('decoder.', '')}  {shape}\n"
                    f"비대각/대각 = {off / d:.3f}", fontsize=8)
        a.set_xticks(range(blk.shape[1]))
        a.set_yticks(range(blk.shape[0]))
        a.set_xticklabels([str(k + 1) for k in range(blk.shape[1])], fontsize=6)
        a.set_yticklabels([str(k + 1) for k in range(blk.shape[0])], fontsize=6)
        fig.colorbar(im, ax=a, shrink=.8)
    for a in ax[n:]:
        a.axis("off")
    fig.suptitle(f"디코더 블록 구조 — {run}\n"
                 "층별 |W| 를 공간축으로 합산 후 K×K 블록 평균. "
                 "행 = 출력 인코딩, 열 = 입력 인코딩", fontsize=12)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


@torch.no_grad()
def keep_one(model, ds, device, idx, batch=100):
    """keep-one — k번째 인코딩만 남긴 성분 x̂_k 를 clean 과 비교한다.

    스케일 보정 α = <x̂_k, x_noisy> / ‖x̂_k‖² 는 x_noisy 만 쓰므로 정답을 보지 않는다.
    보정 전/후를 모두 산출해 스케일이 지표에 얼마나 영향을 주는지 함께 본다.
    """
    K, pad = model.n_encoders, model.pad_each
    names = list(metrics.S5_METRICS)
    acc = {t: {k: np.zeros((K, len(idx))) for k in names} for t in ("raw", "scaled")}
    alpha = np.zeros((K, len(idx)))
    for s in range(0, len(idx), batch):
        j = idx[s:s + batch]
        x = meae.pad(ds.tensor(j).to(device), pad)
        clean = ds.refs["x_clean"][j].astype(np.float64)
        noisy = ds.x_noisy[j].astype(np.float64)
        for k in range(K):
            c = meae.crop(model.component(x, k), pad).squeeze(1).cpu().numpy().astype(np.float64)
            den = (c ** 2).sum(-1)
            a = np.where(den > 0, (c * noisy).sum(-1) / np.maximum(den, 1e-30), 0.0)
            alpha[k, s:s + len(j)] = a
            for tag, est in (("raw", c), ("scaled", a[:, None] * c)):
                for name, v in metrics.s5_score(clean, est).items():
                    acc[tag][name][k, s:s + len(j)] = v
    return acc, alpha


def fig_mask_vs_keep(model, ds, device, seg_global, out, fs, title):
    """마스킹 후 복원과 keep-one 성분을 나란히 본다 (같은 분절, clean 겹침)."""
    K, pad = model.n_encoders, model.pad_each
    j = np.array([seg_global])
    with torch.no_grad():
        x = meae.pad(ds.tensor(j).to(device), pad)
        zs = model.encode(x)
        zeros = [torch.zeros_like(z) for z in zs]
        rem = [meae.crop(model.decode([zeros[i] if i == k else zs[i] for i in range(K)]),
                         pad).squeeze().cpu().numpy() for k in range(K)]
        kep = [meae.crop(model.component(x, k), pad).squeeze().cpu().numpy()
               for k in range(K)]
        m0 = meae.crop(model(x)[0], pad).squeeze().cpu().numpy()
    clean = ds.refs["x_clean"][seg_global].astype(np.float64)
    noisy = ds.x_noisy[seg_global].astype(np.float64)
    t = np.arange(len(clean)) / fs

    fig, ax = plt.subplots(K + 1, 2, figsize=(15, 1.7 * (K + 1)), sharex=True)
    for c, (lb, v) in enumerate((("입력 x_noisy", noisy), ("M0 복원 (마스킹 없음)", m0))):
        a = ax[0, c]
        a.plot(t, clean, lw=.7, color="#2ca02c", label="clean")
        a.plot(t, v, lw=.7, color="#000" if c == 0 else "#d62728", alpha=.85, label=lb)
        a.set_title(lb, fontsize=8, loc="left")
        a.legend(fontsize=6.5, loc="upper right")
    for k in range(K):
        al = float((kep[k] * noisy).sum() / max((kep[k] ** 2).sum(), 1e-30))
        pairs = ((f"remove-some — {enc_label(k)} 제거 복원", rem[k]),
                 (f"keep-one — {enc_label(k)} 만 남김 (a={al:.2f} 보정)", al * kep[k]))
        for c, (lb, v) in enumerate(pairs):
            a = ax[k + 1, c]
            a.plot(t, clean, lw=.7, color="#2ca02c")
            a.plot(t, v, lw=.7, color="#1f77b4", alpha=.85)
            a.set_title(f"{lb}   SNR {metrics.snr_db(clean, v):+.2f} dB   "
                        f"CosSim {float(metrics.cossim(clean, v)):.3f}",
                        fontsize=7.5, loc="left")
    for a in ax.ravel():
        a.grid(alpha=.25, lw=.3)
        a.tick_params(labelsize=6.5)
        a.set_ylabel("mV", fontsize=6.5)
    for a in ax[-1]:
        a.set_xlabel("시간 (초)")
    fig.suptitle(title + "\n초록 = clean 참조. 왼쪽 = 하나를 끈 복원, 오른쪽 = 하나만 남긴 성분",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)


def diagnose(config="configs/default.yaml", run="K8_seed42", split="val",
             n=None, seg=None, outdir=None):
    cfg = load_cfg(config)
    fs = cfg["data"]["fs"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    outdir = outdir or os.path.join("experiments", "masking_strategy", "outputs",
                                    run, split)
    figdir = os.path.join(outdir, "figures")
    os.makedirs(figdir, exist_ok=True)

    model, ck = load_ckpt(cfg, run)
    model = model.to(device)
    ds = load(cfg, split)
    idx = np.arange(len(ds) if n is None else min(n, len(ds)))
    K = model.n_encoders
    ix = enc_names(K)
    print(f"[보조실험] {run} (에폭 {ck['epoch']}) · {split} {len(idx)}분절")

    # ---- ① 디코더 블록 구조
    blocks = decoder_blocks(model)
    rows = []
    for name, shape, blk in blocks:
        d = float(np.diag(blk).mean())
        off = float((blk.sum() - np.trace(blk)) / (blk.size - blk.shape[0]))
        rows.append({"층": name, "shape": str(shape), "대각_평균": d,
                     "비대각_평균": off, "비대각_대각_비": off / d})
    blk_tab = pd.DataFrame(rows).round(5)
    blk_tab.to_csv(f"{outdir}/decoder_blocks.csv", index=False, encoding="utf-8-sig")
    np.savez(f"{outdir}/decoder_blocks.npz",
             **{nm.replace(".", "_"): b for nm, _, b in blocks})
    fig_decoder_blocks(blocks, f"{figdir}/decoder_blocks.png", run)

    # ---- ②③ keep-one, 스케일 보정 전/후
    acc, alpha = keep_one(model, ds, device, idx)
    krows = []
    for k in range(K):
        r = {"인코더": ix[k], "alpha_중앙": float(np.median(alpha[k]))}
        for tag in ("raw", "scaled"):
            for name in metrics.S5_METRICS:
                r[f"{name}_{tag}"] = float(np.median(acc[tag][name][k]))
        krows.append(r)
    keep_tab = pd.DataFrame(krows).round(4)
    keep_tab.to_csv(f"{outdir}/keep_one.csv", index=False, encoding="utf-8-sig")

    # ---- remove-some(단독 마스킹)과 나란히
    single = os.path.join("results", "04_masked_denoising", run, split,
                          "single_mask.csv")
    comp = None
    if os.path.exists(single):
        rs = pd.read_csv(single, encoding="utf-8-sig")
        comp = pd.DataFrame({"인코더": ix})
        for name in metrics.S5_METRICS:
            comp[f"{name}_removesome"] = rs[f"{name}_중앙"].values
            comp[f"{name}_keepone"] = keep_tab[f"{name}_scaled"].values
        comp = comp.round(4)
        comp.to_csv(f"{outdir}/removesome_vs_keepone.csv", index=False,
                    encoding="utf-8-sig")

    # ---- ④ 파형 대조
    if seg is None:
        seg = int(idx[0])
    m = ds.meta[int(seg)]
    fig_mask_vs_keep(model, ds, device, int(seg), f"{figdir}/mask_vs_keep.png", fs,
                     f"{run} · {split} · 분절 {m['record_id']}_{m['seg_idx']:04d}")

    with open(f"{outdir}/note.txt", "w", encoding="utf-8") as f:
        f.write(
            "04 진단 — remove-some 과 keep-one 을 나란히 산출한 것이다.\n\n"
            "1) decoder_blocks.csv / figures/decoder_blocks.png\n"
            "   디코더 각 층 가중치 |W| 를 공간축으로 합산해 K x K 블록으로 접었다.\n"
            "   행 = 출력 인코딩, 열 = 입력 인코딩. 블록 값은 그 블록의 평균 |W|.\n"
            "   비대각_대각_비 = 비대각 블록 평균 / 대각 블록 평균.\n"
            "   sparse mixing 손실이 벌하는 대상이 비대각 블록이다(출력층 제외).\n\n"
            "2) keep_one.csv\n"
            "   k번째 인코딩만 남긴 성분을 clean 과 비교했다. 지표 5종, 분절 중앙값.\n"
            "   raw    = 성분 그대로\n"
            "   scaled = alpha = <x_hat_k, x_noisy> / ||x_hat_k||^2 로 크기만 맞춘 뒤\n"
            "            (x_noisy 만 쓰므로 정답을 보지 않는다)\n\n"
            "3) removesome_vs_keepone.csv\n"
            "   같은 인코더에 대해 remove-some(그 인코더만 끈 복원)과\n"
            "   keep-one(그 인코더만 남긴 성분, 보정 후)을 나란히 둔 표.\n\n"
            "4) figures/mask_vs_keep.png\n"
            "   한 분절에서 두 연산의 파형을 clean 과 겹쳐 그린 것.\n\n"
            "해석과 처방은 붙이지 않는다.\n")

    pd.set_option("display.width", 260)
    print("\n[1] 디코더 블록 — 비대각/대각 비")
    print(blk_tab.to_string(index=False), "\n")
    print("[2][3] keep-one — 스케일 보정 전(raw) / 후(scaled), 분절 중앙값")
    print(keep_tab.to_string(index=False), "\n")
    if comp is not None:
        print("[대조] remove-some(단독 마스킹) vs keep-one(보정 후)")
        print(comp.to_string(index=False), "\n")
    print(f"산출물 → {outdir}/")
    return blk_tab, keep_tab, comp


# ================================================================
# ⑤ 신호 공간 뺄셈 — 마스킹으로 사라진 몫을 입력에서 직접 뺀다.
#
#   d_S            = x̂_M0 − x̂_{mask S}          마스킹된 인코더들이 재구성에 기여하던 몫
#   x_denoised     = x_noisy − β · d_S
#
#   β = 1 고정판과, x_noisy 기반 최적 스케일판 두 가지를 산출한다.
#   β* = <x_noisy, d_S> / ‖d_S‖²  — x_noisy 만 쓰므로 정답을 보지 않는다.
#
#   참고로 실제로 얼마나 뺐는지도 남긴다:
#   뺀양비 = ‖β·d_S‖ / ‖x_noisy − x_clean‖   (분모는 주입 잡음 총량 Σ α_t n_t)
#
# 해석·처방은 붙이지 않는다. 산출과 보고만.
# ================================================================
@torch.no_grad()
def subtract_sweep(model, ds, device, idx, combos, batch=100):
    """조합별 뺄셈 결과. 반환 {β모드: {지표: (조합, 분절)}} 와 뺀양비."""
    K, pad = model.n_encoders, model.pad_each
    names = list(metrics.S5_METRICS)
    acc = {b: {m: np.zeros((len(combos), len(idx))) for m in names}
           for b in ("beta1", "beta_opt")}
    ratio = {b: np.zeros((len(combos), len(idx))) for b in ("beta1", "beta_opt")}
    beta = np.zeros((len(combos), len(idx)))
    m0_acc = {m: np.zeros(len(idx)) for m in names}     # ⓑ M0 복원 vs clean
    for s in range(0, len(idx), batch):
        j = idx[s:s + batch]
        x = meae.pad(ds.tensor(j).to(device), pad)
        clean = ds.refs["x_clean"][j].astype(np.float64)
        noisy = ds.x_noisy[j].astype(np.float64)
        n_tot = np.linalg.norm(noisy - clean, axis=-1)      # ‖Σ α_t n_t‖
        zs = model.encode(x)
        zeros = [torch.zeros_like(z) for z in zs]
        m0 = meae.crop(model.decode(zs), pad).squeeze(1).cpu().numpy().astype(np.float64)
        for name, v in metrics.s5_score(clean, m0).items():
            m0_acc[name][s:s + len(j)] = v
        for ci, mk in enumerate(combos):
            ms = set(mk)
            y = model.decode([zeros[i] if i in ms else zs[i] for i in range(K)])
            y = meae.crop(y, pad).squeeze(1).cpu().numpy().astype(np.float64)
            d = m0 - y
            den = (d ** 2).sum(-1)
            b_opt = np.where(den > 0, (noisy * d).sum(-1) / np.maximum(den, 1e-30), 0.0)
            beta[ci, s:s + len(j)] = b_opt
            for tag, bb in (("beta1", np.ones(len(j))), ("beta_opt", b_opt)):
                est = noisy - bb[:, None] * d
                for name, v in metrics.s5_score(clean, est).items():
                    acc[tag][name][ci, s:s + len(j)] = v
                ratio[tag][ci, s:s + len(j)] = (
                    np.abs(bb) * np.linalg.norm(d, axis=-1) / np.maximum(n_tot, 1e-30))
    return acc, ratio, beta, m0_acc


def subtract(config="configs/default.yaml", run="K8_seed42", split="val",
             n=None, outdir=None):
    cfg = load_cfg(config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    outdir = outdir or os.path.join("experiments", "masking_strategy", "outputs",
                                    run, split, "subtract")
    figdir = os.path.join(outdir, "figures")
    os.makedirs(figdir, exist_ok=True)

    model, ck = load_ckpt(cfg, run)
    model = model.to(device)
    ds = load(cfg, split)
    idx = np.arange(len(ds) if n is None else min(n, len(ds)))
    K = model.n_encoders
    combos = [tuple(c) for r in range(K + 1) for c in itertools.combinations(range(K), r)]
    print(f"[보조실험 ⑤] {run} (에폭 {ck['epoch']}) · {split} {len(idx)}분절 · "
          f"조합 {len(combos)}개")

    clean = ds.refs["x_clean"][idx].astype(np.float64)
    noisy = ds.x_noisy[idx].astype(np.float64)
    base_a = metrics.s5_score(clean, noisy)

    acc, ratio, beta, m0_acc = subtract_sweep(model, ds, device, idx, combos)

    rows = []
    for ci, mk in enumerate(combos):
        row = {"조합": combo_label(mk, K), "끈_인코더수": len(mk),
               "마스크비트": "".join("1" if k in set(mk) else "0" for k in range(K)),
               "beta_opt_중앙": float(np.median(beta[ci]))}
        for tag in ("beta1", "beta_opt"):
            for name in metrics.S5_METRICS:
                v = acc[tag][name][ci]
                row[f"{name}_{tag}"] = float(np.median(v))
                row[f"{name}_{tag}_vs_noisy"] = float(
                    np.median(v) - np.median(base_a[name]))
            row[f"뺀양비_{tag}_중앙"] = float(np.median(ratio[tag][ci]))
        rows.append(row)
    tab = pd.DataFrame(rows).round(4)
    tab.to_csv(f"{outdir}/subtract_exhaustive.csv", index=False, encoding="utf-8-sig")

    # 지표별 최고 조합 — 선정하지 않고 나열만 한다
    brows = []
    for name, lower in LOWER_BETTER.items():
        for tag in ("beta1", "beta_opt"):
            col = f"{name}_{tag}"
            i = tab[col].idxmin() if lower else tab[col].idxmax()
            brows.append({"지표": name, "beta": tag,
                          "방향": "낮을수록 유사" if lower else "높을수록 유사",
                          "지목_조합": tab.loc[i, "조합"],
                          "끈_인코더수": int(tab.loc[i, "끈_인코더수"]),
                          "값": float(tab.loc[i, col]),
                          "vs_ⓐnoisy": float(tab.loc[i, f"{col}_vs_noisy"]),
                          "뺀양비": float(tab.loc[i, f"뺀양비_{tag}_중앙"])})
    best = pd.DataFrame(brows).round(4)
    best.to_csv(f"{outdir}/subtract_best.csv", index=False, encoding="utf-8-sig")

    base = pd.DataFrame([
        {"상태": "ⓐ x_noisy (처리 전)",
         **{f"{k}_중앙": float(np.median(v)) for k, v in base_a.items()}},
        {"상태": "ⓑ M0 복원 (마스킹 없음)",
         **{f"{k}_중앙": float(np.median(v)) for k, v in m0_acc.items()}}])
    base.to_csv(f"{outdir}/subtract_baseline.csv", index=False, encoding="utf-8-sig")

    # enc4 포함/제외 분포 — 256조합을 둘로 갈라 지표를 요약한다
    k4 = 3
    has = tab["마스크비트"].str[k4] == "1"
    srows = []
    for lab, g in (("enc4 포함", tab[has]), ("enc4 제외", tab[~has])):
        for tag in ("beta1", "beta_opt"):
            r = {"구분": lab, "beta": tag, "조합수": int(len(g))}
            for m in metrics.S5_METRICS:
                c = f"{m}_{tag}"
                r[f"{m}_중앙값"] = float(g[c].median())
                r[f"{m}_최소"] = float(g[c].min())
                r[f"{m}_최대"] = float(g[c].max())
            r["뺀양비_중앙값"] = float(g[f"뺀양비_{tag}_중앙"].median())
            srows.append(r)
    sp = pd.DataFrame(srows).round(4)
    sp.to_csv(f"{outdir}/subtract_enc4_split.csv", index=False, encoding="utf-8-sig")

    nm = len(metrics.S5_METRICS)
    fig, ax = plt.subplots(2, nm, figsize=(3.0 * nm, 8))
    for row, tag in enumerate(("beta1", "beta_opt")):
        for a, m in zip(np.atleast_2d(ax)[row], metrics.S5_METRICS):
            c = f"{m}_{tag}"
            a.boxplot([tab.loc[~has, c].values, tab.loc[has, c].values],
                      labels=["제외", "포함"], widths=.55)
            a.set_title(f"{m}  ({tag})", fontsize=9)
            a.grid(alpha=.3, lw=.4, axis="y")
            a.tick_params(labelsize=8)
            if m in ("SSD", "PRD"):
                a.set_yscale("log")
    fig.suptitle("⑤ 신호 공간 뺄셈 — enc4 포함 여부에 따른 전수 256조합 지표 분포",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(f"{figdir}/subtract_enc4_split.png", bbox_inches="tight")
    plt.close(fig)

    with open(f"{outdir}/note.txt", "w", encoding="utf-8") as f:
        f.write(
            "보조 실험 ⑤ — 신호 공간 뺄셈. 본 노선(04)이 아니다.\n\n"
            "  d_S        = x_hat_M0 - x_hat_{mask S}\n"
            "  x_denoised = x_noisy - beta * d_S\n\n"
            "beta1     beta = 1 고정\n"
            "beta_opt  beta* = <x_noisy, d_S> / ||d_S||^2  (x_noisy 만 쓴다)\n\n"
            "뺀양비 = ||beta * d_S|| / ||x_noisy - x_clean||\n"
            "  분모는 그 분절에 실제로 주입한 잡음 총량 Sum_t alpha_t * n_t 이다.\n"
            "  1 이면 주입량과 같은 크기를 뺐다는 뜻이고, 방향이 맞는지는 말하지 않는다.\n\n"
            "지표 5종은 clean 기준 mV 원단위, 분절 중앙값. 기준선은 x_noisy.\n"
            "최적 조합 선정은 하지 않는다. 해석과 처방도 붙이지 않는다.\n")

    pd.set_option("display.width", 280)
    print("\n[enc4 포함/제외 분포] 전수 256조합")
    print(sp.to_string(index=False), "\n")
    print("[기준선] clean 대비")
    print(base.round(4).to_string(index=False), "\n")
    print("[지표별 지목 조합] 선정하지 않는다 — 나열만")
    print(best.to_string(index=False), "\n")
    show = ["조합", "끈_인코더수", "beta_opt_중앙"] + \
        [f"{m}_beta1" for m in metrics.S5_METRICS] + ["뺀양비_beta1_중앙"]
    print("[단독 마스킹 8개 · beta=1]")
    print(tab[tab["끈_인코더수"] == 1][show].to_string(index=False), "\n")
    show2 = ["조합", "끈_인코더수", "beta_opt_중앙"] + \
        [f"{m}_beta_opt" for m in metrics.S5_METRICS] + ["뺀양비_beta_opt_중앙"]
    print("[단독 마스킹 8개 · beta*]")
    print(tab[tab["끈_인코더수"] == 1][show2].to_string(index=False), "\n")
    print(f"산출물 → {outdir}/")
    return tab, best


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--run", default="K8_seed42")
    p.add_argument("--split", default="val")
    p.add_argument("--n", type=int, default=None)
    p.add_argument("--seg", type=int, default=None, help="파형 대조에 쓸 분절 번호")
    p.add_argument("--outdir", default=None)
    p.add_argument("--subtract", action="store_true",
                   help="⑤ 신호 공간 뺄셈 평가")
    a = p.parse_args()
    if a.subtract:
        subtract(a.config, a.run, a.split, a.n, a.outdir)
    else:
        diagnose(a.config, a.run, a.split, a.n, a.seg, a.outdir)
