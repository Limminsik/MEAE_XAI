"""03 — 블라인드 소스 분리 실험 및 참조 대응 분석 (RESEARCH_DESIGN.md §8).

학습된 MEAE에서 성분 `x̂_k = D(0,…,z_k,…,0)` 를 뽑아, 각 성분이 **실제 주입한 잡음**과
얼마나 대응하는지 정량 비교한다. 선행 연구는 실제 소스 파형을 알 수 없어 심박수 같은 파생
지표로 간접 평가했지만, 본 연구는 주입 성분을 개별 보존하므로 **파형을 직접 대조**할 수 있다.

    python 03_bss.py --run K8_seed42 --split val

────────────────────────────────────────────────────────────────────────
지표 3종 — 성분과 참조를 각각 **분절 내 표준화**(평균 0, SD 1)한 뒤 산출
────────────────────────────────────────────────────────────────────────
[S4-01] |r|   분절 내 Pearson 절댓값 → 900분절 평균 ± SD (ddof=1)
              부호 반전은 무관이 아니라 반대 위상의 일치이므로 절댓값을 쓰고,
              원 부호 분포는 corr_sign.csv 에 따로 남긴다.
[S4-02] RMSE  표준화 신호 차이의 RMS `sqrt(mean_i (ã−r̃)²)` → 평균 ± SD
              부호 정렬을 하지 않으므로 반대 위상은 값이 커진다.
[S4-03] MAD   같은 차이 신호의 최댓값 `max_i |ã−r̃|` → 평균 ± SD. 단위는 표준편차.
              참조 파형의 첨도에 영향받으므로 열 내 비교에 적합하다.

원값 RMSE는 폐기했다 — 성분의 절대 크기가 임의 스케일이라 순위가 뒤집힌다.
SSD·PRD는 RMSE와 순위가 같으므로 여기서는 싣지 않는다(S5에서 원단위로 쓴다).

**표시는 열별**이다. |r|는 열 상위 2, RMSE·MAD는 열 하위 2. `[1]`=1위, `[2]`=2위.
세 지표가 행별로 같은 참조를 지목하는지는 metric_agreement.csv 에 따로 싣는다.

**하지 않는 것**: 표본 수(N=3600) 기반 p값(시간적 자기상관으로 독립 관측 가정 불성립) ·
인코더에 참조 이름 붙이기 · 잡음 3종의 "잡음 최대" 요약 · 값에 대한 해석·판정 ·
순열 검정 · 코히런스 · 대역 분해.

구간은 패딩 제외 **중앙 N=3600**, 분절은 해당 split 전체.

────────────────────────────────────────────────────────────────────────
산출물  results/03_bss/<run>/<split>/
────────────────────────────────────────────────────────────────────────
  corr_matrix.csv · rmse_norm_matrix.csv · mad_matrix.csv     K×4, 각 칸 평균 ± SD
  metric_matrix.csv (+ _note.txt)                             셋을 합친 통합표
  metric_agreement.csv                                        세 지표 지목 일치 여부
  corr_persegment.csv · rmse_norm_persegment.csv · mad_persegment.csv   분절별 원값
  corr_sign.csv                                               원 부호 양수 비율
  reference_correlation.csv · reference_spectrum.csv          참조 간 상관·중심주파수
  contribution.csv                                            기여 분해 (다중 회귀)
  figures/  fig1_correspondence · fig_overlay_grid · components_bw_{high,mid,low}
            hist_enc* · mad_argmax_hist
"""
import argparse
import os

import numpy as np
import pandas as pd
import torch

from src.core import (NOISE_REFS, aggregate, component_bank, enc_names, load_ckpt,
                      mad_matrix, mark, pearson, persegment, render,
                      rmse_norm_matrix, top_idx, znorm, _center)
from src.data.build import load_cfg
from src.data.dataset import REF_KEYS, load
from src.model.meae import enc_label
from src.viz import plt


def contribution(comps, refs):
    """분절별 다중 회귀 기여 분해. 반환 (n, K, R) 몫, (n, R) R²."""
    n, K, T = comps.shape
    R = refs.shape[1]
    X = _center(comps).transpose(0, 2, 1)          # (n, T, K)
    share = np.zeros((n, K, R))
    r2 = np.zeros((n, R))
    for i in range(n):
        Xi = X[i]
        for j in range(R):
            y = _center(refs[i, j])
            vy = float(y @ y)
            if vy <= 0:
                continue
            beta, *_ = np.linalg.lstsq(Xi, y, rcond=None)
            share[i, :, j] = beta * (Xi.T @ y) / vy      # Σ = R²
            resid = y - Xi @ beta
            r2[i, j] = 1.0 - float(resid @ resid) / vy
    return share, r2


def reference_correlation(refs):
    """참조끼리의 |r| — 분리 한계의 독립 근거."""
    r = _center(refs)
    rn = np.linalg.norm(r, axis=-1)
    out = {}
    names = list(REF_KEYS)
    for a in range(len(names)):
        for b in range(a + 1, len(names)):
            den = rn[:, a] * rn[:, b]
            v = np.where(den > 0, np.abs(np.einsum("nt,nt->n", r[:, a], r[:, b])) / np.maximum(den, 1e-12), 0.0)
            out[f"{names[a]}–{names[b]}"] = v
    return out


def spectral_centroid(refs, fs):
    from scipy.signal import welch
    out = {}
    for j, name in enumerate(REF_KEYS):
        v = []
        for i in range(min(len(refs), 200)):
            f, P = welch(refs[i, j], fs=fs, nperseg=1024)
            v.append(float((f * P).sum() / P.sum()))
        out[name] = np.array(v)
    return out


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


def fig_components(comps, refs, x_noisy, i, out, title, fs):
    """성분·참조를 z-정규화해 **공통 y축**으로 그린다.

    성분마다 y축을 따로 잡으면 크기가 작은 성분이 크게 보여 서로 비교가 안 된다.
    입력 x_noisy만 원값(mV)이므로 축을 따로 둔다."""
    K = comps.shape[1]
    zc, zr = znorm(comps[i]), znorm(refs[i])
    rows = ([("입력 x_noisy  [원값, mV]", x_noisy[i], "#000", False)]
            + [(f"성분 {k+1}", zc[k], "#1f77b4", True) for k in range(K)]
            + [(f"참조 {n}", zr[list(REF_KEYS).index(n)], "#d62728", True) for n in NOISE_REFS]
            + [("참조 clean", zr[0], "#2ca02c", True)])
    lim = max(np.abs(v).max() for _, v, _, z in rows if z) * 1.05
    t = np.arange(comps.shape[-1]) / fs
    fig, ax = plt.subplots(len(rows), 1, figsize=(11, 1.0 * len(rows)), sharex=True)
    for a, (lb, v, c, z) in zip(ax, rows):
        a.plot(t, v, lw=.6, color=c)
        a.set_title(lb, fontsize=8, loc="left")
        a.grid(alpha=.25, lw=.4)
        a.tick_params(labelsize=7)
        if z:
            a.set_ylim(-lim, lim)
            a.set_ylabel("z", fontsize=7)
        else:
            a.set_ylabel("mV", fontsize=7)
    ax[-1].set_xlabel("시간 (초)")
    fig.suptitle(title + "\n성분·참조는 z-정규화 후 공통 y축 (단위 없음). 입력만 원값 mV",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


def pick_segment(ds, idx):
    """대표 분절 **사전 규칙** — 주입 SNR 3종의 평균이 val 중앙값에 가장 가까운 분절.

    모델 출력과 무관한 데이터 속성만으로 정한다. 지표를 보고 고르지 않기 위해서다.
    """
    snr = np.array([np.mean([ds.meta[int(i)][f"snr_{t}"] for t in NOISE_REFS]) for i in idx])
    return int(np.argmin(np.abs(snr - np.median(snr)))), snr


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


def fig_mad_argmax(pos, out, fs, T, bins=40):
    """S4-03 — 최대 편차 발생 시점의 분포. MAD가 분절의 어느 구간에서 나는지 확인용."""
    K, R = pos.shape[1], pos.shape[2]
    t = pos / fs
    fig, ax = plt.subplots(K, R, figsize=(3.0 * R, 1.5 * K), sharex=True, sharey="row")
    ax = np.atleast_2d(ax)
    for k in range(K):
        for r in range(R):
            a = ax[k, r]
            a.hist(t[:, k, r], bins=bins, range=(0, T / fs), color="#1f77b4", alpha=.8)
            a.grid(alpha=.25, lw=.3)
            a.tick_params(labelsize=6)
            if k == 0:
                a.set_title(list(REF_KEYS)[r], fontsize=9)
            if r == 0:
                a.set_ylabel(enc_label(k), fontsize=8)
    for a in ax[-1]:
        a.set_xlabel("최대 편차 발생 시점 (초)", fontsize=7)
    fig.suptitle("MAD 최대 편차가 발생한 시점의 분절별 분포 (val 전체)", fontsize=11)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def fig_hist(cm, k, out):
    fig, ax = plt.subplots(figsize=(8, 4))
    for j, name in enumerate(REF_KEYS):
        ax.hist(cm[:, k, j], bins=40, alpha=.55,
                label=f"{name} (평균 {cm[:, k, j].mean():.3f})")
    ax.set_xlabel(f"{enc_label(k)} 성분과 참조의 |ρ| (분절 단위)")
    ax.set_ylabel("분절 수")
    ax.legend(fontsize=8)
    ax.grid(alpha=.3, lw=.4)
    ax.set_title(f"{enc_label(k)}의 분절별 |ρ| 분포 — 평균값 뒤에 가려진 산포", fontsize=10)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def main(config="configs/default.yaml", run="K8_seed42", split="val", outdir=None):
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

    # ① 대응표 — S4-01 명세. 모두 z-정규화 후 값이다
    rho = pearson(comps, refs)                       # (S, K, R) 부호 있는 원값
    cm = np.abs(rho)
    rn = rmse_norm_matrix(comps, refs)
    mad, mad_arg = mad_matrix(comps, refs, with_argmax=True)
    rbar, rsd, pos = aggregate(rho)
    ix = [enc_label(k) for k in range(K)]

    rn_bar, rn_sd = rn.mean(0), rn.std(0, ddof=1)
    md_bar, md_sd = mad.mean(0), mad.std(0, ddof=1)
    # 표식은 **열별**이다. |r| 은 열 상위 2, RMSE·MAD 는 열 하위 2 (** = 1위, * = 2위)
    f_r = mark(rbar, 2, largest=True)
    f_n = mark(rn_bar, 2, largest=False)
    f_m = mark(md_bar, 2, largest=False)

    # corr_matrix.csv — K x 4, 각 칸 rho_bar +- sigma
    pd.DataFrame(render(rbar, rsd, f_r, "{:.4f}"), columns=list(REF_KEYS),
                 index=pd.Index(ix, name="인코더")).to_csv(
        f"{outdir}/corr_matrix.csv", encoding="utf-8-sig")

    # corr_persegment.csv — 분절별 원값, 부호 포함
    persegment(rho, "rho", idx, ix, ds).to_csv(
        f"{outdir}/corr_persegment.csv", index=False, encoding="utf-8-sig")

    # corr_sign.csv — 원 부호의 분절별 분포 (양수 비율)
    pd.DataFrame(pos.round(4), columns=list(REF_KEYS),
                 index=pd.Index(ix, name="인코더")).to_csv(
        f"{outdir}/corr_sign.csv", encoding="utf-8-sig")

    # ---- S4-02 정규화 RMSE. 낮을수록 유사하므로 **열별 하위 2**를 표시한다
    pd.DataFrame(render(rn_bar, rn_sd, f_n, "{:.4f}"), columns=list(REF_KEYS),
                 index=pd.Index(ix, name="인코더")).to_csv(
        f"{outdir}/rmse_norm_matrix.csv", encoding="utf-8-sig")
    persegment(rn, "rmse_norm", idx, ix, ds).to_csv(
        f"{outdir}/rmse_norm_persegment.csv", index=False, encoding="utf-8-sig")

    # ---- S4-03 MAD. 단위는 표준편차. 역시 **열별 하위 2**를 표시한다
    pd.DataFrame(render(md_bar, md_sd, f_m, "{:.4f}"), columns=list(REF_KEYS),
                 index=pd.Index(ix, name="인코더")).to_csv(
        f"{outdir}/mad_matrix.csv", encoding="utf-8-sig")
    md_per = persegment(mad, "mad", idx, ix, ds)
    md_per["argmax_표본"] = mad_arg.reshape(-1)
    md_per["argmax_초"] = (mad_arg.reshape(-1) / fs).round(4)
    md_per.to_csv(f"{outdir}/mad_persegment.csv", index=False, encoding="utf-8-sig")
    fig_mad_argmax(mad_arg, f"{figdir}/mad_argmax_hist.png", fs, comps.shape[-1])

    # ---- 세 지표의 행별 지목 일치 여부
    agree = agreement(rbar, rn_bar, md_bar)
    agree.to_csv(f"{outdir}/metric_agreement.csv", encoding="utf-8-sig")

    mat = pd.DataFrame(rbar, columns=list(REF_KEYS), index=ix)
    sd = pd.DataFrame(rsd, columns=[f"{c}_sd" for c in REF_KEYS], index=ix)
    r2c = pd.DataFrame((cm ** 2).mean(0), columns=[f"{c}_r2" for c in REF_KEYS], index=ix)
    rnm = pd.DataFrame(rn.mean(0), columns=[f"{c}_rmse_norm" for c in REF_KEYS], index=ix)
    rnsd = pd.DataFrame(rn.std(0, ddof=1), columns=[f"{c}_rmse_norm_sd" for c in REF_KEYS], index=ix)
    mdm = pd.DataFrame(mad.mean(0), columns=[f"{c}_mad" for c in REF_KEYS], index=ix)
    mdsd = pd.DataFrame(mad.std(0, ddof=1), columns=[f"{c}_mad_sd" for c in REF_KEYS], index=ix)
    energy = (comps ** 2).mean(-1).mean(0)
    mat_out = pd.concat([mat.round(4), sd.round(4), r2c.round(4),
                         rnm.round(4), rnsd.round(4), mdm.round(4), mdsd.round(4)], axis=1)
    mat_out["energy_ratio"] = (energy / energy.sum()).round(4)
    mat_out.to_csv(f"{outdir}/metric_matrix.csv", encoding="utf-8-sig")
    with open(f"{outdir}/metric_matrix_note.txt", "w", encoding="utf-8") as fnote:
        fnote.write(
            "[S4-01] 상관계수 산출 명세\n"
            "1단계 분절 내 Pearson — 평균·표준편차는 해당 분절 안에서만 구한다.\n"
            "  분절을 이어붙여 일괄 계산하지 않는다(분절마다 다른 잡음 구간이 주입되었고,\n"
            "  진폭 큰 분절이 결과를 지배한다). 패딩 제외 중앙 N=3600 표본.\n"
            "2단계 분절 간 집계 — rho_bar = mean_s|rho|, sigma = std_s|rho| (ddof=1).\n"
            "  절댓값: 인코딩과 디코더 가중치가 동시에 부호 반전되어도 재구성이 불변하므로\n"
            "  성분이 참조와 반대 위상으로 수렴할 수 있다. 부호 반전은 무관이 아니라\n"
            "  반대 위상의 일치다. 원 부호 분포는 corr_sign.csv 에 양수 비율로 기록한다.\n"
            "\n"
            "[S4-02] 정규화 RMSE 산출 명세  (rmse_norm_matrix.csv / _persegment.csv)\n"
            "구조는 S4-01과 같다. 한 분절에서 한 성분을 참조 4종과 각각 비교하고\n"
            "900분절 반복 후 평균 +- SD 로 집계한다.\n"
            "1단계 분절 내 표준화 — a~ = (comp - mean)/std, r~ = (ref - mean)/std.\n"
            "  성분 진폭은 비선형 디코더의 임의 출력이고 참조 4종의 RMS도 서로 다르다\n"
            "  (clean 0.204, 잡음 0.113~0.119 mV). 표준화 없이는 크기 차이가 값을 지배한다.\n"
            "2단계 분절 내 RMSE — sqrt(mean_i (a~[i] - r~[i])^2). 부호 정렬을 하지 않으므로\n"
            "  반대 위상은 값이 커진다.\n"
            "3단계 분절 간 집계 — mean_s RMSE, std_s RMSE (ddof=1).\n"
            "표 렌더링은 행별 **최솟값**을 표시한다 (낮을수록 유사).\n"
            "\n"
            "[S4-03] MAD 산출 명세  (mad_matrix.csv / mad_persegment.csv)\n"
            "1단계 표준화는 S4-02와 같다. 2단계에서 RMS 대신 max_i |a~[i] - r~[i]| 를 취하고,\n"
            "3단계로 mean_s / std_s (ddof=1) 집계한다. 단위는 표준편차.\n"
            "표 렌더링은 행별 **최솟값**을 표시한다.\n"
            "각주: 값이 낮을수록 유사하며, 참조 파형의 첨도에 영향받으므로 열 내 비교에 적합하다.\n"
            "정규화 후에도 rho 와 독립인 유일한 지표로 국소 최대 편차를 포착한다.\n"
            "최대 편차가 발생한 시점의 분포는 figures/mad_argmax_hist.png 에 싣는다\n"
            "  (분절별 argmax 위치 히스토그램. MAD가 어느 구간에서 나는지 확인용).\n"
            "원값 RMSE는 폐기했다 — 성분의 절대 크기가 임의 스케일이라 순위가 뒤집힌다.\n"
            "\n"
            "표본 수(N=3600) 기반 p값은 산출하지 않는다 — 시간적 자기상관 때문에\n"
            "독립 관측 가정이 성립하지 않는다.\n"
            "잡음 3종을 '잡음 최대'로 요약하지 않고 4열을 그대로 싣는다.\n")
    fig_correspondence(cm, f"{figdir}/fig1_correspondence.png", run, rn, mad)

    # ---- S4 마무리 그림 — 성분 x 참조 8x4 겹침 격자 (대표 분절, 사전 규칙)
    seg, snr = pick_segment(ds, idx)
    m = ds.meta[int(idx[seg])]
    fig_overlay(comps, refs, seg, cm, rn, mad, f"{figdir}/fig_overlay_grid.png", fs,
                f"{run} · 분절 {m['record_id']}_{m['seg_idx']:04d} "
                f"(주입 SNR bw {m['snr_bw']:.1f} / ma {m['snr_ma']:.1f} / em {m['snr_em']:.1f} dB, "
                f"평균 {snr[seg]:.2f} dB · val 중앙값 {np.median(snr):.2f} dB)")

    # ② 참조 간 상관 + 스펙트럼
    rc = reference_correlation(refs)
    sc = spectral_centroid(refs, fs)
    ref_rows = [{"쌍": k, "mean": v.mean(), "median": np.median(v),
                 "p75": np.percentile(v, 75), "max": v.max()} for k, v in rc.items()]
    ref_df = pd.DataFrame(ref_rows).round(4)
    ref_df.to_csv(f"{outdir}/reference_correlation.csv", index=False, encoding="utf-8-sig")
    spec_df = pd.DataFrame({"참조": list(sc), "중심주파수_Hz_중앙": [np.median(v) for v in sc.values()]}).round(3)
    spec_df.to_csv(f"{outdir}/reference_spectrum.csv", index=False, encoding="utf-8-sig")

    # ③ 기여 분해
    share, r2 = contribution(comps, refs)
    con = pd.DataFrame(100 * share.mean(0), columns=list(REF_KEYS),
                       index=[enc_label(k) for k in range(K)])
    con_sd = pd.DataFrame(100 * share.std(0), columns=[f"{c}_sd" for c in REF_KEYS],
                          index=con.index)
    out = pd.concat([con.round(2), con_sd.round(2)], axis=1)
    out.loc["합계 (R²·%)"] = list((100 * r2.mean(0)).round(2)) + list((100 * r2.std(0)).round(2))
    out.to_csv(f"{outdir}/contribution.csv", encoding="utf-8-sig")

    # ④ 시각 — bw 최대 상관 인코더를 기준 인코더로
    k_bw = int(cm.mean(0)[:, list(REF_KEYS).index("bw")].argmax())
    v = cm[:, k_bw, list(REF_KEYS).index("bw")]
    order = np.argsort(v)
    picks = {"high": int(order[-1]), "mid": int(order[len(order) // 2]), "low": int(order[0])}
    for tag, i in picks.items():
        fig_components(comps, refs, ds.x_noisy, i,
                       f"{figdir}/components_bw_{tag}.png",
                       f"{enc_label(k_bw)}–bw |ρ| {v[i]:.3f} ({tag}) · "
                       f"{ds.meta[i]['record_id']}_{ds.meta[i]['seg_idx']:04d}", fs)
    fig_hist(cm, k_bw, f"{figdir}/hist_{enc_label(k_bw)}.png")

    # ---- 콘솔 보고
    print(f"=== {run} · {split} {len(ds)}분절 · best epoch {ck['epoch']} ===\n")
    print("[S4-01] 분절 내 Pearson → 분절 간 mean|ρ| ± std|ρ|(ddof=1). * = 행별 최댓값\n")
    print("표식은 열별. ** = 열 1위, * = 열 2위 (|r|은 상위 2, RMSE·MAD는 하위 2)\n")
    print("① 인코더 × 참조 ρ̄ ± σ  [r² 병기]")
    print(pd.DataFrame([[f"{c}  (r² {r2c.loc[ix[k], list(REF_KEYS)[j]+'_r2']:.3f})"
                         for j, c in enumerate(row)]
                        for k, row in enumerate(render(rbar, rsd, f_r))],
                       columns=list(REF_KEYS), index=ix).to_string(), "\n")
    print("①-보조 원 부호 양수 비율 (부호 반전 여부 기록용. |ρ| 산출과 무관)")
    print(pd.DataFrame(pos.round(3), columns=list(REF_KEYS), index=ix).to_string(), "\n")
    print("[S4-02] 분절 내 표준화 → 분절 내 RMSE → 분절 간 평균±SD(ddof=1). 낮을수록 유사\n")
    print("② 인코더 × 참조 RMSE_norm 평균 (±SD)")
    print(pd.DataFrame(render(rn_bar, rn_sd, f_n), columns=list(REF_KEYS),
                       index=ix).to_string(), "\n")
    print("[S4-03] 분절 내 표준화 → 분절 내 max|차이| → 분절 간 평균±SD(ddof=1). "
          "단위는 표준편차\n")
    print("③ 인코더 × 참조 MAD 평균 (±SD)")
    print(pd.DataFrame(render(md_bar, md_sd, f_m), columns=list(REF_KEYS),
                       index=ix).to_string())
    print("   각주: 값이 낮을수록 유사하며, 참조 파형의 첨도에 영향받으므로 열 내 비교에 적합.")
    print("   최대 편차 발생 시점 분포 → figures/mad_argmax_hist.png\n")
    print("③-보조 세 지표의 행별 지목 일치 여부")
    print(agree.to_string(), "\n")
    print("④ 참조 간 |r| — 분리 한계의 독립 근거")
    print(ref_df.to_string(index=False))
    print(spec_df.to_string(index=False), "\n")
    print("⑤ 기여 분해 — 참조 에너지 중 각 성분이 설명하는 몫 (%)")
    print(out[list(REF_KEYS)].to_string(), "\n")
    print(f"⑥ 시각 → {figdir}/  (fig1_correspondence · fig_overlay_grid · "
          f"components_bw_* · hist · mad_argmax_hist)")
    return mat_out, ref_df, out


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--run", default="K8_seed42")
    p.add_argument("--split", default="val")
    p.add_argument("--outdir", default=None)
    a = p.parse_args()
    main(a.config, a.run, a.split, a.outdir)
