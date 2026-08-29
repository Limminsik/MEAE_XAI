"""03 — 블라인드 소스 분리 결과의 참조 대응 평가 (RESEARCH_DESIGN.md §8).

**여기는 최종 선정 모델 하나 전용이다.**

02 에서 여러 설정을 돌려 본 결과(성분 정렬 지표·복원·그림)는 모두 그 런 폴더 안
`experiments/02_model/<그룹>/<run>/metrics/` 에 있다. `results/` 는 최종 하나 전용이다. 그것들을 보고 **모델 하나를 고른 뒤**,
여기서 그 하나만 test 로 확인한다. 이후 04·05·06·07 은 모두 이 모델을 쓴다.

    python 03_bss.py --run d1_1차차분/K4_g150 --split test --final

`--final` 을 주면 선정 기록(`selected.json`)을 남긴다 — 어느 런의 몇 번째 에폭인지,
언제 고정했는지. 04 이후는 그 파일이 가리키는 모델을 쓰면 된다.

성분 `x̂_k = D(0,…,z_k,…,0)` 를 뽑아 **실제 주입한 참조 4종**과의 대응을 표로 만든다.

선행 연구는 실제 소스 파형을 알 수 없어 심박수 같은 파생 지표로 간접 평가했다.
본 연구는 주입 성분을 개별 보존하므로 **파형을 직접 대조**할 수 있다.

────────────────────────────────────────────────────────────────────────
주 지표 4종 — 성분과 참조를 각각 **분절 내 표준화**(평균 0, SD 1)한 뒤 산출
────────────────────────────────────────────────────────────────────────
[S4-01] |r|        분절 내 Pearson 절댓값 → 분절 간 평균 ± SD (ddof=1)
                   부호 반전은 무관이 아니라 반대 위상의 일치이므로 절댓값을 쓴다.
[S4-02] RMSE_norm  표준화 신호 차이의 RMS `sqrt(mean_i (ã−r̃)²)` → 평균 ± SD
                   부호 정렬을 하지 않으므로 반대 위상은 값이 커진다.
[S4-03] MAD        같은 차이 신호의 최댓값 `max_i |ã−r̃|` → 평균 ± SD. 단위는 표준편차.
                   참조 파형의 첨도에 영향받으므로 열 내 비교에 적합하다.

원값 RMSE는 폐기했다 — 성분의 절대 크기가 임의 스케일이라 순위가 뒤집힌다.
SSD·PRD는 RMSE와 순위가 같으므로 여기서는 싣지 않는다(04에서 원단위로 쓴다).

**표시는 열별**이다. |r|는 열 상위 2, RMSE·MAD는 열 하위 2. `[1]`=1위, `[2]`=2위.
잡음 3종을 "잡음 최대"로 요약하지 않고 4열을 그대로 싣는다.

**하지 않는 것**: 표본 수(N=3600) 기반 p값(시간적 자기상관으로 독립 관측 가정 불성립) ·
인코더에 참조 이름 붙이기 · 값에 대한 해석·판정 · 순열 검정 · 코히런스 · 대역 분해.

구간은 패딩 제외 **중앙 N=3600**, 분절은 해당 split 전체.

────────────────────────────────────────────────────────────────────────
그림의 분절 선정 — **지표 상위 3개**
────────────────────────────────────────────────────────────────────────
분절마다 `(1/4) Σ_r max_k |ρ_k(r)|` 를 구해 큰 순으로 3개를 쓴다. 모델 선택에 쓴 S와
같은 형태다. 한 장만 실으면 그 사례가 얼마나 특수한지 알 수 없어 상위 몇 개를 함께 본다.
**결과를 보고 고른 사례**이므로 그림 제목과 note 에 그 사실을 명시한다.

────────────────────────────────────────────────────────────────────────
산출물  results/03_bss/<run>/<split>/
────────────────────────────────────────────────────────────────────────
  corr_matrix.csv · rmse_norm_matrix.csv · mad_matrix.csv   각 8×4, 평균과 SD 나란히
  metric_agreement.csv                                      세 지표의 행별 지목 일치 여부
  figure_segments.csv                                       그림 3분절의 개별 수치 (보조)
  note.txt · console.log
  figures/components_top{1,2,3}.png  입력·재구성·성분 K개·참조 4종 (지표 상위 3분절)
  figures/overlay_top{1,2,3}.png     성분 × 참조 8×4 겹침 격자 (같은 분절)
  figures/correspondence.png         지표 3종 히트맵
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
import torch

from src.core import (NOISE_REFS, aggregate, component_bank, enc_names,
                      load_ckpt, mad_matrix, mark, pearson, reconstruct,
                      render, rmse_norm_matrix, top_idx, znorm)
from src.data.build import load_cfg
from src import metrics
from src.data.dataset import REF_KEYS, load
from src.model.meae import crop as meae_crop, enc_label, pad as meae_pad
from src.viz import plt


def pick_segments(cm, n=3):
    """그림에 쓸 분절 — **지표 상위 n개**. 모델 선택에 쓴 S와 같은 형태다.

    분절마다 (1/4) Σ_r max_k |ρ_k(r)| 를 구해 큰 순으로 n개를 고른다.
    한 장만 실으면 그 사례가 얼마나 특수한지 알 수 없으므로 상위 몇 개를 함께 본다.
    결과를 보고 고르는 것이므로 그림 제목과 note 에 그 사실을 밝힌다.
    """
    per_seg = cm.max(1).mean(1)          # (분절,) — 참조별 최고 인코더의 |r| 를 평균
    return list(np.argsort(-per_seg)[:n]), per_seg


def agreement(rbar, rn_bar, md_bar):
    """세 지표가 **행별로** 같은 참조를 지목하는지. (지목 참조 3종, 일치 여부) 표."""
    names = list(REF_KEYS)
    rows = []
    for k in range(rbar.shape[0]):
        picks = (names[int(rbar[k].argmax())],
                 names[int(rn_bar[k].argmin())],
                 names[int(md_bar[k].argmin())])
        rows.append({"|r|_지목": picks[0], "RMSE_지목": picks[1], "MAD_지목": picks[2],
                     "일치": "일치" if len(set(picks)) == 1 else "불일치"})
    return pd.DataFrame(rows, index=pd.Index(
        [enc_label(k) for k in range(rbar.shape[0])], name="인코더"))

def fig_correspondence(cm, out, run, rn=None, mad=None):
    """색은 ρ̄ = mean_s|ρ|, 칸 안에 RMSE_norm·MAD를 병기한다.

    잡음 3종을 "잡음 최대"로 요약하지 않고 4열을 그대로 싣는다.
    표시는 **열별**이며 지표마다 방향이 다르다 — |ρ|는 상위 2, RMSE_norm·MAD는 하위 2."""
    m, sd, _ = aggregate(cm)          # cm 은 |ρ| 또는 ρ 어느 쪽이 와도 동일한 결과다
    mr, mm = rn.mean(0), mad.mean(0)
    K = m.shape[0]
    # 지표마다 **열별** 표시 대상이 다르다 — |ρ|는 상위 2, RMSE_norm·MAD는 하위 2
    flags = [mark(m, 2, True), mark(mr, 2, False), mark(mm, 2, False)]
    lines = [lambda k, r: f"|ρ| {m[k, r]:.3f}±{sd[k, r]:.3f}",
             lambda k, r: f"RMSEn {mr[k, r]:.3f}",
             lambda k, r: f"MAD {mm[k, r]:.2f}"]
    fig, ax = plt.subplots(figsize=(7.0, 0.95 * K + 2.4))
    im = ax.imshow(m, cmap="magma", vmin=0, vmax=max(0.6, m.max()), aspect="auto")
    ax.set_xticks(range(len(REF_KEYS)))
    ax.set_xticklabels(list(REF_KEYS))
    ax.set_yticks(range(K))
    ax.set_yticklabels([enc_label(k) for k in range(K)])
    for k in range(K):
        for r in range(len(REF_KEYS)):
            col = "white" if m[k, r] < m.max() * .6 else "black"
            for j, (dy, fl, txt) in enumerate(zip((-.27, 0.0, .27), flags, lines)):
                hit = fl[k, r] > 0
                ax.text(r, k + dy, txt(k, r) + ("  ●" if fl[k, r] == 1 else
                                                "  ○" if fl[k, r] == 2 else ""),
                        ha="center", va="center", fontsize=7.2 if hit else 6.4,
                        color=col, fontweight="bold" if hit else "normal")
    for r, ks in enumerate(top_idx(m, 2, largest=True).T):   # 색 기준(|ρ|) 열별 상위 2
        for rank, k in enumerate(ks):
            ax.add_patch(plt.Rectangle((r - .5, k - .5), 1, 1, fill=False, ec="cyan",
                                       lw=2.5, ls="-" if rank == 0 else (0, (4, 2))))
    ax.set_title(f"① 인코더 × 참조 대응 — {run}\n"
                 "표준화 후 mean|ρ| ± SD (색 = mean|ρ|) · RMSE_norm · MAD\n"
                 "굵은 글씨 = 지표별 열 상위 2 (● 1위 / ○ 2위) — |ρ|는 큰 값, RMSE·MAD는 작은 값\n"
                 "테두리 = |ρ| 열 상위 2 (실선 1위 / 파선 2위)",
                 fontsize=9)
    fig.colorbar(im, ax=ax, shrink=.8, label="mean|ρ|")
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)

def fig_s4ext(rbar, rq, f1, be, bnames, refs_c, ix, out, run, split, band):
    """[S4 확장] 세 지표를 한 장에 — 역할이 다르므로 나란히 두고 읽는다.

    ① F1        박동이 있는가          (막대, 열 없음 — clean 하나에 대한 값)
    ② r_QRS     형태가 맞는가          (히트맵, 참조 4열)
    ③ 대역에너지 어느 대역을 점유하는가  (누적 막대)
    비교를 위해 전대역 |r| 도 함께 싣는다.
    """
    K = len(ix)
    fig, ax = plt.subplots(1, 4, figsize=(19, 0.55 * K + 3.4))
    for a, (m, name, vmax) in zip(ax[:2], [(rbar, "전대역 |r|", None),
                                           (rq, f"r_QRS ({band[0]:g}–{band[1]:g} Hz)", None)]):
        im = a.imshow(m, cmap="magma", vmin=0, vmax=vmax or max(0.6, m.max()),
                      aspect="auto")
        a.set_xticks(range(len(refs_c))); a.set_xticklabels(refs_c, fontsize=8)
        a.set_yticks(range(K)); a.set_yticklabels(ix, fontsize=8)
        for k in range(K):
            for r in range(len(refs_c)):
                a.text(r, k, f"{m[k, r]:.3f}", ha="center", va="center", fontsize=7.5,
                       color="white" if m[k, r] < m.max() * .6 else "black")
        for r, ks in enumerate(top_idx(m, 2, largest=True).T):
            for rank, k in enumerate(ks):
                a.add_patch(plt.Rectangle((r - .5, k - .5), 1, 1, fill=False, ec="cyan",
                                          lw=2.2, ls="-" if rank == 0 else (0, (4, 2))))
        a.set_title(name, fontsize=10, loc="left")
        fig.colorbar(im, ax=a, shrink=.75)

    a = ax[2]
    o = np.argsort(-f1)
    a.barh(range(K), f1[o], color=["#d62728" if i < 1 else "#8fa6c4" for i in range(K)])
    a.set_yticks(range(K)); a.set_yticklabels([ix[i] for i in o], fontsize=8)
    a.invert_yaxis(); a.set_xlim(0, 1)
    for i, v in enumerate(f1[o]):
        a.text(v + .01, i, f"{v:.3f}", va="center", fontsize=7.5)
    a.set_title("R-피크 검출 F1 (참조 = x_clean)", fontsize=10, loc="left")
    a.set_xlabel("F1", fontsize=8); a.grid(alpha=.3, lw=.4, axis="x")

    a = ax[3]
    cols = {"vlf": "#4c72b0", "lf": "#55a868", "qrs": "#c44e52", "hf": "#8172b2"}
    left = np.zeros(K)
    for j, b in enumerate(bnames):
        a.barh(range(K), be[:, j], left=left, label=b,
               color=cols.get(b, None), height=.72)
        left = left + be[:, j]
    a.set_yticks(range(K)); a.set_yticklabels(ix, fontsize=8)
    a.invert_yaxis(); a.set_xlim(0, 1)
    a.set_title("대역 에너지 비율 (Welch)", fontsize=10, loc="left")
    a.set_xlabel("비율", fontsize=8); a.legend(fontsize=7.5, ncol=4)

    fig.suptitle(f"[S4 확장] {run} · {split} — 역할이 다른 지표를 나란히\n"
                 "F1 = 박동이 있는가 · r_QRS = 형태가 맞는가 · 대역 = 어느 대역을 점유하는가",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def fig_components(comps, refs, x_noisy, i, out, title, fs, recon=None):
    """**기초 성분 그림** — 입력 · 재구성 · 성분 K개 · 참조 4종을 세로로 쌓는다.

    03에서 가장 먼저 봐야 하는 그림이다. 분해가 무엇을 만들어 냈는지를 먼저 보고,
    그 다음에 대응표(fig1)와 겹침 격자(fig_overlay_grid)로 넘어간다.

    성분·참조는 분절 내 표준화 후 **공통 y축**으로 그린다 — 성분마다 축을 따로 잡으면
    크기가 작은 성분이 크게 보여 서로 비교가 안 된다. 입력과 재구성만 원값(mV)이다.
    """
    K = comps.shape[1]
    zc, zr = znorm(comps[i]), znorm(refs[i])
    rows = [("입력 x_noisy  [원값, mV]", x_noisy[i], "#000", False)]
    if recon is not None:
        rows.append(("재구성 x_hat  [원값, mV]", recon[i], "#d62728", False))
    rows += ([(f"성분 {k+1}", zc[k], "#1f77b4", True) for k in range(K)]
             + [(f"참조 {n}", zr[list(REF_KEYS).index(n)], "#ff7f0e", True) for n in NOISE_REFS]
             + [("참조 clean", zr[0], "#2ca02c", True)])
    lim = max(np.abs(v).max() for _, v, _, z in rows if z) * 1.05
    mv = max(np.abs(v).max() for _, v, _, z in rows if not z) * 1.05
    t = np.arange(comps.shape[-1]) / fs
    fig, ax = plt.subplots(len(rows), 1, figsize=(11, 1.0 * len(rows)), sharex=True)
    for a, (lb, v, c, z) in zip(ax, rows):
        a.plot(t, v, lw=.6, color=c)
        a.set_title(lb, fontsize=8, loc="left")
        a.grid(alpha=.25, lw=.4)
        a.tick_params(labelsize=7)
        a.set_ylim(*((-lim, lim) if z else (-mv, mv)))
        a.set_ylabel("z" if z else "mV", fontsize=7)
    ax[-1].set_xlabel("시간 (초)")
    fig.suptitle(title + "\n성분·참조는 분절 내 표준화 후 공통 y축(단위 없음). "
                 "입력·재구성만 원값 mV", fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)

def fig_overlay(comps, refs, i, cm, rn, mad, out, fs, title):
    """S4 마무리 그림 — 성분 × 참조 8×4 격자. 한 패널에 표준화한 두 파형을 겹쳐 그린다.

    패널마다 그 분절의 세 지표를 병기하고, 전 패널이 공통 y축을 쓴다.
    """
    zc, zr = znorm(comps[i]), znorm(refs[i])
    K, R = zc.shape[0], zr.shape[0]
    lim = max(np.abs(zc).max(), np.abs(zr).max()) * 1.05
    t = np.arange(zc.shape[-1]) / fs
    fig, ax = plt.subplots(K, R, figsize=(4.0 * R, 1.65 * K), sharex=True, sharey=True)
    ax = np.atleast_2d(ax)
    for k in range(K):
        for r in range(R):
            a = ax[k, r]
            a.plot(t, zr[r], lw=.7, color="#d62728", alpha=.85, label="참조")
            a.plot(t, zc[k], lw=.7, color="#1f77b4", alpha=.85, label="성분")
            a.set_ylim(-lim, lim)
            a.grid(alpha=.2, lw=.3)
            a.tick_params(labelsize=6)
            a.set_title(f"|r| {cm[i, k, r]:.3f}  RMSE {rn[i, k, r]:.3f}  "
                        f"MAD {mad[i, k, r]:.2f}", fontsize=7, loc="left")
            if k == 0:
                a.text(.5, 1.35, list(REF_KEYS)[r], transform=a.transAxes,
                       ha="center", fontsize=11)
            if r == 0:
                a.set_ylabel(f"{enc_label(k)}\n(z)", fontsize=8)
    for a in ax[-1]:
        a.set_xlabel("시간 (초)")
    ax[0, 0].legend(fontsize=6, loc="lower left")
    fig.suptitle(title + "\n성분(파랑)·참조(빨강) 모두 분절 내 표준화 후 공통 y축 (단위 없음)",
                 fontsize=12, y=1.005)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main(config="configs/default.yaml", run="K8_seed42", split="val", outdir=None,
         final=False):
    cfg = load_cfg(config)
    fs = cfg["data"]["fs"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    outdir = outdir or os.path.join("results", "03_bss", run, split)
    figdir = os.path.join(outdir, "figures")
    os.makedirs(figdir, exist_ok=True)

    model, ck = load_ckpt(cfg, run)
    model = model.to(device)
    ds = load(cfg, split)
    idx = np.arange(len(ds))
    comps, refs = component_bank(model, ds, device, idx)
    K = comps.shape[1]
    ix = enc_names(K)
    refs_c = list(REF_KEYS)

    # ---- 지표 3종. 모두 분절 내 표준화 후 산출한다
    cm = np.abs(pearson(comps, refs))
    rn = rmse_norm_matrix(comps, refs)
    mad = mad_matrix(comps, refs)
    rbar, rsd, _ = aggregate(cm)
    stats = {"corr": (rbar, rsd, True),
             "rmse_norm": (rn.mean(0), rn.std(0, ddof=1), False),
             "mad": (mad.mean(0), mad.std(0, ddof=1), False)}

    # 표시는 열별 — |r|는 상위 2, RMSE·MAD는 하위 2
    for name, (mu, sd, high) in stats.items():
        rows = []
        for k in range(K):
            cell = {}
            for j, c in enumerate(refs_c):
                cell[c] = mu[k, j]
                cell[f"{c}_sd"] = sd[k, j]
            rows.append({"인코더": ix[k], **cell})
        pd.DataFrame(rows).round(4).to_csv(
            f"{outdir}/{name}_matrix.csv", index=False, encoding="utf-8-sig")

    agree = agreement(rbar, stats["rmse_norm"][0], stats["mad"][0])
    agree.to_csv(f"{outdir}/metric_agreement.csv", encoding="utf-8-sig")

    # ================================================================
    # [S4 확장] r_QRS · F1 · 대역 에너지 — 역할이 다른 지표를 나란히 둔다
    # ================================================================
    sup_keys_ = list(cfg["loss"].get("supervise", []))
    s4 = cfg.get("s4", {})
    clean = refs[:, refs_c.index("x_clean")]
    band = tuple(s4.get("qrs_band", metrics.QRS_BAND))
    bands = {k: tuple(v) for k, v in s4.get("psd_bands", metrics.PSD_BANDS).items()}
    norm = tuple(s4.get("psd_norm", metrics.PSD_NORM))

    rq = metrics.r_qrs_matrix(comps, refs, fs, band)
    pd.DataFrame([{"인코더": ix[k],
                   **{c: rq[:, k, j].mean() for j, c in enumerate(refs_c)},
                   **{f"{c}_sd": rq[:, k, j].std(ddof=1)
                      for j, c in enumerate(refs_c)}} for k in range(K)]
                 ).round(4).to_csv(f"{outdir}/r_qrs_matrix.csv", index=False,
                                   encoding="utf-8-sig")

    print("[F1] 참조 R-피크 검출 후 성분별 매칭 — 분절마다 K회 검출이라 시간이 걸린다")
    ref_pk = metrics.reference_peaks(clean, fs, progress=300)
    f1 = metrics.f1_vector(comps, clean, fs, ref_pk,
                           tol_ms=s4.get("f1_tol_ms", metrics.S4_TOL_MS),
                           min_peaks=s4.get("f1_min_ref_peaks", metrics.MIN_REF_PEAKS),
                           progress=300)
    used = int((~np.isnan(f1[:, 0])).sum())
    pd.DataFrame([{"인코더": ix[k], "F1": np.nanmean(f1[:, k]),
                   "F1_sd": np.nanstd(f1[:, k], ddof=1),
                   "집계분절": used, "전체분절": len(idx)} for k in range(K)]
                 ).round(4).to_csv(f"{outdir}/f1_matrix.csv", index=False,
                                   encoding="utf-8-sig")

    be_c, bnames = metrics.band_energy(comps, fs, bands, norm)
    rows = [{"대상": ix[k], "구분": "성분", "집계분절": len(idx),
             **{b: float(be_c[:, k, j].mean()) for j, b in enumerate(bnames)}}
            for k in range(K)]
    be_r, _ = metrics.band_energy(refs, fs, bands, norm)
    for j, c in enumerate(refs_c):
        rows.append({"대상": c, "구분": "참조", "집계분절": len(idx),
                     **{b: float(be_r[:, j, t].mean()) for t, b in enumerate(bnames)}})
    _, recon_all = reconstruct(model, ds, device, idx)
    for name, sig in (("입력 x_noisy", ds.x_noisy[idx]), ("M0 재구성", recon_all)):
        be, _ = metrics.band_energy(sig, fs, bands, norm)
        rows.append({"대상": name, "구분": "기준", "집계분절": len(idx),
                     **{b: float(be[:, j].mean()) for j, b in enumerate(bnames)}})
    pd.DataFrame(rows).round(4).to_csv(f"{outdir}/band_energy.csv", index=False,
                                       encoding="utf-8-sig")
    fig_s4ext(rbar, rq.mean(0), np.nanmean(f1, 0), be_c.mean(0), bnames, refs_c, ix,
              f"{figdir}/s4_extended.png", run, split, band)

    # ---- [version5] 감시 — 잔차를 끈 성분과 켠 성분이 얼마나 다른가
    # 두 값이 크게 다르면 정보가 잔차 쪽에 실린 것이고, 인코딩이 비어간다는 신호다.
    if getattr(model, "skip_levels", None) and model.skip_weight:
        import torch as _t
        pad = model.pad_each
        off = np.zeros_like(comps)
        with _t.no_grad():
            for s0 in range(0, len(idx), 100):
                j = idx[s0:s0 + 100]
                xb = meae_pad(ds.tensor(j).to(device), pad)
                for k in range(K):
                    y = model.component(xb, k, use_skip=False)
                    off[s0:s0 + len(j), k] = (meae_crop(y, pad).squeeze(1)
                                              .cpu().numpy().astype(np.float64))
        rows = []
        for k in range(K):
            a_, b_ = comps[:, k], off[:, k]
            a0 = a_ - a_.mean(-1, keepdims=True)
            b0 = b_ - b_.mean(-1, keepdims=True)
            d_ = np.sqrt((a0 ** 2).sum(-1) * (b0 ** 2).sum(-1))
            rows.append({"인코더": ix[k], "배정참조": sup_keys_[k] if sup_keys_ else "",
                         "잔차유무_corr": float(np.abs((a0 * b0).sum(-1)
                                                    / np.maximum(d_, 1e-30)).mean()),
                         "잔차기여_RMS": float(np.sqrt(((a_ - b_) ** 2).mean(-1)).mean()),
                         "성분_SD": float(a_.std(-1).mean())})
        w = pd.DataFrame(rows)
        w["잔차기여_비율"] = (w["잔차기여_RMS"] / w["성분_SD"]).round(4)
        w.round(4).to_csv(f"{outdir}/skip_watch.csv", index=False, encoding="utf-8-sig")
        print("")
        print("[감시] 잔차 유무 성분 차이 — 크면 정보가 잔차로 새고 있다는 뜻")
        print(w.round(4).to_string(index=False))

    # ---- [version4] 배정 대각 요약
    # 지도학습은 인코더-참조 배정이 미리 정해져 있으므로 **대각이 주 지표**다.
    # 비대각 최댓값을 함께 실어 누출을 본다 (배정 아닌 참조를 얼마나 잡고 있는가).
    sup_keys = list(cfg["loss"]["supervise"])
    diag = []
    for k, key in enumerate(sup_keys):
        j = refs_c.index(key)
        off = [t for t in range(len(refs_c)) if t != j]
        jo = off[int(np.argmax(rbar[k, off]))]
        # 주 지표 4종 → 보조(누출비·r_QRS) 순으로 싣는다.
        # r_QRS 는 QRS 대역만 통과시키므로 **심장 행에서만** 뜻이 있다.
        diag.append({"인코더": ix[k], "배정참조": key,
                     "corr": rbar[k, j], "corr_sd": rsd[k, j],
                     "rmse_norm": stats["rmse_norm"][0][k, j],
                     "mad": stats["mad"][0][k, j],
                     "F1": np.nanmean(f1[:, k]),
                     "누출_최대참조": refs_c[jo], "누출_corr": rbar[k, jo],
                     "누출비": rbar[k, jo] / max(rbar[k, j], 1e-12),
                     "r_QRS": (rq[:, k, j].mean() if key == "x_clean" else np.nan)})
    pd.DataFrame(diag).round(4).to_csv(f"{outdir}/assignment_diagonal.csv",
                                       index=False, encoding="utf-8-sig")
    print("")
    print("[배정 대각] 인코더-참조 배정 쌍의 지표와 최대 누출")
    print(pd.DataFrame(diag).round(4).to_string(index=False))

    # ---- 그림. 분절은 **지표 상위 3개** — 결과를 보고 고른 사례임을 제목에 밝힌다
    segs, per_seg = pick_segments(cm, 3)
    _, recon = reconstruct(model, ds, device, idx)
    picked, seg_rows = [], []
    for rank, seg in enumerate(segs, 1):
        m = ds.meta[int(idx[seg])]
        name = f"{m['record_id']}_{m['seg_idx']:04d}"
        picked.append((rank, name, float(per_seg[seg])))
        head = (f"{run} · {split} · 분절 {name} — 지표 상위 {rank}위 "
                f"(분절 점수 {per_seg[seg]:.3f}, {split} {len(idx)}분절 중)")
        fig_components(comps, refs, ds.x_noisy[idx], seg,
                       f"{figdir}/components_top{rank}.png", head, fs, recon=recon)
        fig_overlay(comps, refs, seg, cm, rn, mad,
                    f"{figdir}/overlay_top{rank}.png", fs, head)
        for k in range(K):
            for j, c in enumerate(refs_c):
                seg_rows.append({"순위": rank, "분절": name, "분절점수": per_seg[seg],
                                 "인코더": ix[k], "참조": c,
                                 "corr": cm[seg, k, j], "rmse_norm": rn[seg, k, j],
                                 "mad": mad[seg, k, j]})
    # 그림에 쓴 3분절의 개별 수치 — 본 표(전체 집계)의 보조다
    pd.DataFrame(seg_rows).round(4).to_csv(
        f"{outdir}/figure_segments.csv", index=False, encoding="utf-8-sig")
    fig_correspondence(cm, f"{figdir}/correspondence.png", run, rn, mad)

    if final:
        # 최종 고정 — 04 이후 단계가 이 파일 하나만 보면 되게 한다
        import datetime
        sel = {"run": run, "epoch": int(ck["epoch"]), "split": split,
               "n_segments": int(len(idx)), "K": int(K),
               "supervise": list(cfg["loss"]["supervise"]),
               "lambda_sup": cfg["loss"].get("lambda_sup"),
               "gamma_sup": cfg["loss"].get("gamma_sup"),
               "gamma2_sup": cfg["loss"].get("gamma2_sup"),
               "checkpoint": os.path.join("results", "02_model", run,
                                          f"{os.path.basename(run)}.pt"),
               "fixed_at": datetime.datetime.now().isoformat(timespec="seconds"),
               "note": "04·05·06·07 은 이 모델을 쓴다"}
        with open(os.path.join("results", "03_bss", "selected.json"), "w",
                  encoding="utf-8") as f:
            json.dump(sel, f, ensure_ascii=False, indent=2)
        print("")
        print(f"[최종 선정] {run} · 에폭 {ck['epoch']} → results/03_bss/selected.json")

    with open(f"{outdir}/note.txt", "w", encoding="utf-8") as f:
        f.write(
            f"03 — 성분 <-> 참조 대응 평가.  {run} (에폭 {ck['epoch']}) · {split} "
            f"{len(idx)}분절\n\n"
            "지표 3종 — 성분과 참조를 각각 분절 내 표준화(평균 0, SD 1)한 뒤 산출한다.\n"
            "  corr        분절 내 Pearson 절댓값 -> 분절 간 평균 (높을수록 유사)\n"
            "  rmse_norm   표준화 신호 차이의 RMS -> 분절 간 평균 (낮을수록 유사)\n"
            "  mad         그 차이의 최댓값       -> 분절 간 평균 (낮을수록 유사).\n"
            "              단위는 표준편차. 참조 파형의 첨도에 영향받으므로 열 내 비교에 적합.\n"
            "각 참조마다 평균 열 옆에 SD 열(<참조>_sd, ddof=1)을 나란히 둔다.\n"
            "구간은 패딩 제외 중앙 3600 표본.\n\n"
            "표시는 열별이다 — 한 참조를 어느 인코더가 가장 잘 잡는지 열 안에서 비교한다.\n"
            "  |r| 은 열 상위 2개, RMSE_norm 과 MAD 는 열 하위 2개.\n"
            "세 지표가 행별로 같은 참조를 지목하는지는 metric_agreement.csv 에 있다.\n\n"
            f"그림의 분절 — **지표 최고 분절**이다. 분절마다 (1/4)*sum_r max_k |rho_k(r)| 를\n"
            f"구해 가장 큰 분절을 골랐다. {m['record_id']}_{m['seg_idx']:04d}, "
            f"점수 {per_seg[seg]:.4f}.\n"
            "결과를 보고 고른 사례이므로 원고에서도 그 사실을 밝혀야 한다.\n\n"
            "하지 않는 것: 표본 수 기반 p값 · 인코더에 참조 이름 붙이기 ·\n"
            "값에 대한 해석과 판정 · 순열 검정 · 코히런스 · 대역 분해.\n")

    # ---- 콘솔
    pd.set_option("display.width", 260)
    print(f"=== 03 {run} (에폭 {ck['epoch']}) · {split} {len(idx)}분절 ===")
    print("성분·참조를 분절 내 표준화한 뒤 산출. 표시는 열별 — [1] 1위, [2] 2위\n")
    titles = {"corr": "① |r| 평균±SD  (높을수록 유사, 열 상위 2)",
              "rmse_norm": "② RMSE_norm 평균±SD  (낮을수록 유사, 열 하위 2)",
              "mad": "③ MAD 평균±SD  (낮을수록 유사, 열 하위 2. 단위 표준편차)"}
    for name, (mu, sd, high) in stats.items():
        print(titles[name])
        print(pd.DataFrame(render(mu, sd, mark(mu, 2, largest=high)),
                           columns=refs_c, index=ix).to_string(), "\n")
    print("④ 세 지표의 행별 지목 일치 여부")
    print(agree.to_string(), "\n")
    print("그림 분절 — 지표 상위 3개")
    for r, nm, sc in picked:
        print(f"  {r}위  {nm}  점수 {sc:.4f}")
    print(f"산출물 → {outdir}/")
    return stats, agree


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--run", default="K8_seed42")
    p.add_argument("--split", default="val")
    p.add_argument("--outdir", default=None)
    p.add_argument("--final", action="store_true",
                   help="선정 기록(results/03_bss/selected.json)을 남긴다")
    a = p.parse_args()
    main(a.config, a.run, a.split, a.outdir, a.final)
