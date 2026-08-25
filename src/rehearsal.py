"""T6.9 — val 전체 리허설 (test 봉인 해제 전 필수 관문).

T7-1~T7-6과 T8(M0–M5·전수 지도·통계)의 **모든 표·그림·CSV를 val에서 실제 생성**한다.
형식·축·라벨·집계 방식까지 원고 게재 수준으로 만든다.
여기서 처음 만들어지는 산출물이 있다면 그것이 곧 발견이다 — test에서 즉흥 제작될 뻔한 것.

산출 위치: `results/validation/`, `figures/validation/`
"""
import argparse
import itertools
import json
import os

import numpy as np
import pandas as pd
import torch

from . import stats
from .data.build import load_cfg
from .data.dataset import REF_KEYS, load
from .model import meae
from .s4_identify import (component_bank, contribution, corr_matrix, fig_components,
                          fig_hist, load_ckpt, reference_correlation, spectral_centroid)
from .s5_restore import M2_THRESHOLD, exhaustive_map, mask_sets, run_conditions
from .viz import plt

NOISE_REFS = ("bw", "ma", "em")
MAIN_K, ROBUST_K = 8, 4           # D34: K=8 주 분석, K=4 강건성
SEEDS = (42, 202, 2026)
COND_ORDER = ["INPUT", "M0", "M2", "M3", "M4", "M5"]


# ---------------------------------------------------------------- 역할 정렬
def role_order(cm_mean):
    """시드 간 집계를 위한 역할 정렬 (§7 ①).

    clean 최대 상관 인코더를 첫 행, 나머지는 bw 상관 내림차순.
    인코더 번호는 시드마다 의미가 다르므로(순열 불변성) 번호로 평균내면 뭉개진다.
    """
    ci, bi = list(REF_KEYS).index("x_clean"), list(REF_KEYS).index("bw")
    clean_enc = int(cm_mean[:, ci].argmax())
    rest = sorted((k for k in range(cm_mean.shape[0]) if k != clean_enc),
                  key=lambda k: -cm_mean[k, bi])
    return [clean_enc] + rest


ROLE_NAMES = ["R0 (clean 최대)"] + [f"R{i} (잡음 {i}위)" for i in range(1, 9)]


# ---------------------------------------------------------------- 그림
def fig1_correspondence(mean, sd, out, title):
    K = mean.shape[0]
    fig, ax = plt.subplots(figsize=(5.8, 0.68 * K + 2.2))
    im = ax.imshow(mean, cmap="magma", vmin=0, vmax=max(0.6, mean.max()), aspect="auto")
    ax.set_xticks(range(len(REF_KEYS)))
    ax.set_xticklabels(["clean", "bw", "ma", "em"])
    ax.set_yticks(range(K))
    ax.set_yticklabels(ROLE_NAMES[:K], fontsize=8)
    for k in range(K):
        for r in range(len(REF_KEYS)):
            ax.text(r, k, f"{mean[k, r]:.2f}\n±{sd[k, r]:.2f}", ha="center", va="center",
                    fontsize=7, color="white" if mean[k, r] < mean.max() * .6 else "black")
    for r in range(len(REF_KEYS)):
        ax.add_patch(plt.Rectangle((r - .5, mean[:, r].argmax() - .5), 1, 1,
                                   fill=False, ec="cyan", lw=2))
    ax.set_title(title, fontsize=10)
    fig.colorbar(im, ax=ax, shrink=.8, label="|피어슨 r|")
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def fig3_ablation(tab, out, title, base="M0"):
    """인코더 단독 마스킹(M1) 효과 — 기준 조건 대비 SNR 변화."""
    d = tab.sort_values("enc")
    colors = ["#d62728" if v < 0 else "#2ca02c" for v in d.delta]
    fig, ax = plt.subplots(figsize=(9, 4.6))
    ax.bar(d.enc.astype(str), d.delta, color=colors, alpha=.85)
    for x, (v, r) in enumerate(zip(d.delta, d.role)):
        ax.text(x, v + (0.05 if v >= 0 else -0.05), f"{v:+.2f}", ha="center",
                va="bottom" if v >= 0 else "top", fontsize=8)
        ax.text(x, ax.get_ylim()[0], r, ha="center", va="bottom", fontsize=7, rotation=90,
                color="#555")
    ax.axhline(0, color="k", lw=.8)
    ax.set_xlabel("단독 마스킹한 인코더")
    ax.set_ylabel(f"SNR 변화 (dB, {base} 대비)")
    ax.grid(alpha=.3, lw=.4, axis="y")
    ax.set_title(title, fontsize=10, loc="left")
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def fig_mask_map(emap, marks, out, title):
    fig, ax = plt.subplots(figsize=(9, 4.6))
    ax.scatter(emap.n_masked, emap.snr_median, s=14, alpha=.45, color="#888",
               label=f"전수 조합 {len(emap)}개")
    for name, (nm, snr) in marks.items():
        ax.scatter([nm], [snr], s=90, marker="D", zorder=5,
                   label=f"{name} (백분위 {emap.snr_percentile[(emap.snr_median - snr).abs().idxmin()]:.0f})")
    ax.set_xlabel("마스킹한 인코더 수")
    ax.set_ylabel("SNR 중앙값 (dB)")
    ax.grid(alpha=.3, lw=.4)
    ax.legend(fontsize=8)
    ax.set_title(title, fontsize=10, loc="left")
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def fig2_ladder(tab, out, title):
    d = tab[tab.cond.isin(COND_ORDER)].set_index("cond").reindex(COND_ORDER).dropna(how="all")
    fig, ax = plt.subplots(figsize=(8, 4.6))
    ax.errorbar(range(len(d)), d.snr_median, yerr=[d.snr_median - d.snr_q1, d.snr_q3 - d.snr_median],
                fmt="o-", capsize=4, lw=1.4, color="#1f77b4")
    ax.set_xticks(range(len(d)))
    ax.set_xticklabels(d.index)
    ax.axhline(float(d.loc["INPUT", "snr_median"]) if "INPUT" in d.index else 0,
               color="#d62728", ls="--", lw=1, label="입력 x_noisy")
    ax.set_ylabel("SNR 중앙값 [IQR] (dB)")
    ax.grid(alpha=.3, lw=.4)
    ax.legend(fontsize=8)
    ax.set_title(title, fontsize=10, loc="left")
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)



# ---------------------------------------------------------------- 표1 · 그림1 (메인 결과)
def beta_scale(comp, ref):
    """최소제곱 배율 β = <ref, comp> / ||comp||². 분절별 (n, K, R)."""
    c = comp - comp.mean(-1, keepdims=True)
    r = ref - ref.mean(-1, keepdims=True)
    num = np.einsum("nkt,nrt->nkr", c, r)
    den = (c ** 2).sum(-1)[:, :, None]
    return np.where(den > 0, num / np.maximum(den, 1e-12), 0.0)


def rmse_matrix(comps, refs):
    """인코더×참조 RMSE 두 버전. 반환 (원값, 배율보정), 각 (n, K, R)."""
    c = comps - comps.mean(-1, keepdims=True)
    r = refs - refs.mean(-1, keepdims=True)
    n, K, T = c.shape
    R = r.shape[1]
    raw = np.sqrt(((c[:, :, None, :] - r[:, None, :, :]) ** 2).mean(-1))
    b = beta_scale(comps, refs)
    scaled = np.sqrt(((b[:, :, :, None] * c[:, :, None, :] - r[:, None, :, :]) ** 2).mean(-1))
    return raw, scaled


TABLE1_FOOTNOTE = """표 1 각주
· |r| = 성분과 참조의 **형태** 일치도 (분절 단위 절대 피어슨 상관, 평균±SD).
· RMSE = 성분과 참조의 **절대 차이** (원값, 평균 제거 후). 성분 진폭은 비선형 디코더의
  산출이므로 참조와의 크기 대응이 보장되지 않는다 — RMSE는 그 전제 위에서 읽는다.
· 중앙차 = (해당 잡음 |r| − clean |r|)의 분절 단위 중앙값. **부호가 방향이다.**
· p = 위 차이에 대한 Wilcoxon 부호순위 검정, Holm 보정(잡음 3종 × 인코더 K개 일괄), 양측.
  `*`는 α=.05에서 유의, `↑`는 잡음 쪽이 큼, `↓`는 clean 쪽이 큼을 뜻한다.
  양측 검정이므로 **clean 전담 인코더도 `↓` 방향으로 유의하게 나온다** — p만 보면 오독한다.
· 효과크기 r = Z/√N.
· 배율 보정 RMSE는 `RMSE(β·x̂, ref) = RMS(ref)·√(1−r²)` 로 |r|의 단조 변환임이 수치로
  확인되어 표에서 제외했다. 산출물 `supp_rmse_scaled_*.csv` 에 보존한다."""


def table1(cm, raw, order, K, alpha=0.05):
    """표 1 — 인코더 × 참조. |r| · 원값 RMSE · 잡음별 Wilcoxon(잡음 − clean) Holm p."""
    ci = list(REF_KEYS).index("x_clean")
    out = pd.DataFrame(index=[ROLE_NAMES[i] for i in range(K)])
    out.insert(0, "encoder", [f"enc{e}" for e in order])
    for j, c in enumerate(REF_KEYS):
        out[f"{c}_r"] = [f"{cm[:, e, j].mean():.3f}±{cm[:, e, j].std():.3f}" for e in order]
        out[f"{c}_RMSE"] = [f"{raw[:, e, j].mean():.4f}" for e in order]

    # 잡음 3종 × K개 인코더에 대해 (잡음 |r| − clean |r|) 검정 후 Holm 일괄 보정
    cells, pv = [], []
    for j, c in enumerate(REF_KEYS):
        if c == "x_clean":
            continue
        for e in order:
            w = stats.wilcoxon_pair(cm[:, e, j], cm[:, e, ci])
            cells.append((c, e, w))
            pv.append(w["p"])
    adj, sig = stats.holm(pv, alpha)
    tab = {}
    for (c, e, w), a, sg in zip(cells, adj, sig):
        tab.setdefault(c, {})[e] = (w, a, sg)
    for c in [k for k in REF_KEYS if k != "x_clean"]:
        # 방향 없는 p는 오독을 부른다 — clean 인코더도 (잡음 - clean) < 0 으로 유의하다.
        # 중앙차를 p 바로 옆에 두고, 부호를 화살표로 표기한다.
        out[f"{c}_중앙차"] = [f"{tab[c][e][0]['median_diff']:+.3f}" for e in order]
        out[f"{c}_p_holm"] = [
            f"{tab[c][e][1]:.1e}" + ("*" if tab[c][e][2] else "")
            + ("↑" if tab[c][e][0]["median_diff"] > 0 else "↓") for e in order]
        out[f"{c}_효과크기r"] = [f"{tab[c][e][0]['r']:.3f}" for e in order]
    return out


def pick_representative_segment(cm, order):
    """D42 — 각 참조의 매칭 인코더 |r| 평균 백분위가 중앙에 가장 가까운 분절."""
    n = cm.shape[0]
    pcts = []
    for j in range(len(REF_KEYS)):
        e = int(cm.mean(0)[:, j].argmax())
        v = cm[:, e, j]
        pcts.append(pd.Series(v).rank(pct=True).values)
    mean_pct = np.mean(pcts, 0)
    return int(np.argmin(np.abs(mean_pct - 0.5))), mean_pct


def fig1_visual(comps, refs, cm, seg, out, title, fs, with_spectrum=True):
    """그림 1 — 성분 파형 ↕ 참조 파형 시각 대조 (배율 보정 후 겹쳐 그림)."""
    from scipy.signal import welch
    b = beta_scale(comps[seg:seg + 1], refs[seg:seg + 1])[0]
    t = np.arange(comps.shape[-1]) / fs
    ncol = 2 if with_spectrum else 1
    fig, ax = plt.subplots(len(REF_KEYS), ncol, figsize=(13 if with_spectrum else 9, 9),
                           squeeze=False,
                           gridspec_kw={"width_ratios": [2.6, 1]} if with_spectrum else None)
    for j, name in enumerate(REF_KEYS):
        e = int(cm.mean(0)[:, j].argmax())
        c = b[e, j] * (comps[seg, e] - comps[seg, e].mean())
        r = refs[seg, j] - refs[seg, j].mean()
        a = ax[j][0]
        a.plot(t, r, lw=.8, color="#d62728", label=f"참조 {name}")
        a.plot(t, c, lw=.8, color="#1f77b4", alpha=.85, label=f"성분 enc{e} (β={b[e, j]:.2f})")
        a.set_title(f"{name}  ·  enc{e}  ·  |r| = {cm[seg, e, j]:.3f}", fontsize=9, loc="left")
        a.grid(alpha=.25, lw=.4)
        a.legend(fontsize=7, loc="upper right")
        a.set_ylabel("mV", fontsize=8)
        a.tick_params(labelsize=7)
        if with_spectrum:
            f_, Pr = welch(r, fs=fs, nperseg=1024)
            _, Pc = welch(c, fs=fs, nperseg=1024)
            s_ = ax[j][1]
            s_.semilogy(f_, Pr, lw=1, color="#d62728")
            s_.semilogy(f_, Pc, lw=1, color="#1f77b4", alpha=.85)
            s_.set_xlim(0, 60)
            s_.grid(alpha=.25, lw=.4)
            s_.tick_params(labelsize=7)
            if j == 0:
                s_.set_title("스펙트럼", fontsize=9, loc="left")
    ax[-1][0].set_xlabel("시간 (초)")
    if with_spectrum:
        ax[-1][1].set_xlabel("주파수 (Hz)")
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- 본체
def analyse_model(cfg, run, ds, idx, device, outdir, figdir, make_figs):
    fs, seg_len = cfg["data"]["fs"], cfg["data"]["fs"] * cfg["data"]["seg_sec"]
    model, ck = load_ckpt(cfg, run)
    model = model.to(device)
    pad = model.pad_each
    K = model.n_encoders

    comps, refs = component_bank(model, ds, device, idx)
    cm = corr_matrix(comps, refs)
    raw, scaled = rmse_matrix(comps, refs)
    share, r2 = contribution(comps, refs)
    share_pct = 100 * share.mean(0)
    sets = mask_sets(share_pct, M2_THRESHOLD)

    conds = {c: sets[c] for c in ("M0", "M2", "M3", "M4", "M5")}
    conds.update({f"M1_e{k}": [k] for k in range(K)})
    df = run_conditions(model, ds, device, idx, conds, fs, seg_len, pad)
    base = [{"seg": int(g), "cond": "INPUT", "record": ds.meta[g]["record_id"],
             **metrics_score(ds, g, fs, seg_len)} for g in idx]
    df = pd.concat([pd.DataFrame(base), df], ignore_index=True)
    emap = exhaustive_map(model, ds, device, idx, pad)

    if make_figs:
        order = role_order(cm.mean(0))
        fig1_correspondence(cm.mean(0)[order], cm.std(0)[order],
                            f"{figdir}/fig1_correspondence_{os.path.basename(run)}.png",
                            f"그림1 · 인코더×참조 |r| (역할 정렬) — {run}")
        k_bw = int(cm.mean(0)[:, list(REF_KEYS).index("bw")].argmax())
        v = cm[:, k_bw, list(REF_KEYS).index("bw")]
        o = np.argsort(v)
        for tag, i in (("high", int(o[-1])), ("mid", int(o[len(o) // 2])), ("low", int(o[0]))):
            fig_components(comps, refs, ds.x_noisy, i, f"{figdir}/components_{os.path.basename(run)}_{tag}.png",
                           f"{run} · enc{k_bw}–bw |r| {v[i]:.3f} ({tag})", fs)
        fig_hist(cm, k_bw, f"{figdir}/hist_{os.path.basename(run)}_enc{k_bw}.png")
    return dict(run=run, K=K, epoch=ck["epoch"], cm=cm, share_pct=share_pct,
                rmse_raw=raw, rmse_scaled=scaled, comps=comps,
                val_sep=float(ck.get("val_separation", np.nan)),
                r2=100 * r2.mean(0), sets=sets, df=df, emap=emap, refs=refs)


def metrics_score(ds, g, fs, seg_len):
    from . import metrics as M
    return M.score(ds.refs["x_clean"][g].astype(np.float64),
                   ds.x_noisy[g].astype(np.float64), ds.rpeaks[g], fs, seg_len)


def summarise_conditions(df):
    rows = []
    for c, g in df.groupby("cond"):
        s = stats.describe(g.snr_db)
        rows.append({"cond": c, "n": s["n"], "snr_median": s["median"],
                     "snr_q1": s["q1"], "snr_q3": s["q3"],
                     "rmse_median": float(np.median(g.rmse)),
                     "f1_median": float(np.median(g.f1)),
                     "sdnn_err_median": float(np.nanmedian(g.sdnn_abs_err_ms))})
    return pd.DataFrame(rows)


def main(config="configs/default.yaml", split="val", n=1000,
         outdir="results/validation", figdir="figures/validation", run_prefix=""):
    cfg = load_cfg(config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(outdir, exist_ok=True)
    os.makedirs(figdir, exist_ok=True)
    ds = load(cfg, split)
    idx = np.arange(min(n, len(ds)))
    print(f"[리허설] {split} {len(idx)}분절 · 6개 모델")

    res = {}
    for K in (MAIN_K, ROBUST_K):
        for sd in SEEDS:
            run = f"{run_prefix}K{K}_seed{sd}"
            print(f"  {run} …", flush=True)
            res[run] = analyse_model(cfg, run, ds, idx, device, outdir, figdir,
                                     make_figs=(K == MAIN_K and sd == SEEDS[0]))

    # ================= 메인 결과 (D38–D46) =================
    # 대표 모델 = T6.5 충실도 관문 통과 모델 중 검증 분리 품질 최고 (D43)
    # 위계 일관성: 재구성이 미달인 모델의 성분은 해석 대상이 아니다 (D26·D30).
    fid = pd.read_csv("results/t6_5_fidelity_final.csv")
    passed = set(fid.loc[fid.PASS.astype(str).str.lower() == "true", "run"])
    pool = {r: v for r, v in res.items() if os.path.basename(r) in passed}
    if not pool:
        print("  ! 충실도 통과 모델이 없어 전체에서 선정한다 (D43 예외)")
        pool = res
    rep_run = max(pool, key=lambda r: pool[r]["val_sep"])
    fid_row = fid[fid.run == os.path.basename(rep_run)].iloc[0]
    diag = pd.read_csv("results/t6_5_diagnostics_final.csv")
    diag_row = diag[diag.run == os.path.basename(rep_run)].iloc[0]
    R = res[rep_run]
    order = role_order(R["cm"].mean(0))
    t1 = table1(R["cm"], R["rmse_raw"], order, R["K"])
    t1.to_csv(f"{outdir}/table1_main_{os.path.basename(rep_run)}.csv", encoding="utf-8-sig")
    with open(f"{outdir}/table1_footnote.txt", "w", encoding="utf-8") as f:
        f.write(TABLE1_FOOTNOTE + f"""

대표 모델: {os.path.basename(rep_run)} (에폭 {R['epoch']}) — T6.5 충실도 관문 통과 모델 중
검증 분리 품질 최고 (D43). 통과 모델은 6개 중 1개다.
  · 15–40 Hz 보존율 {fid_row.keep_band:.3f} (기준 0.7) · 전대역 {fid_row.keep_full:.3f}
  · R-피크 진폭비 {diag_row.dx_rpeak_ratio:.3f}
  · 잔차–clean |r| {diag_row.dx_resid_clean:.3f} (잡음 쪽 {diag_row.dx_resid_bw:.3f}/"""
                f"""{diag_row.dx_resid_ma:.3f}/{diag_row.dx_resid_em:.3f})
나머지 5개 모델의 판별 구조 재현성은 보조 자료로 제시한다 (D46).""")
    # 보정 RMSE는 표에서 제외하고 산출물로만 보존 (|r|의 단조 변환)
    pd.DataFrame(R["rmse_scaled"].mean(0), columns=list(REF_KEYS),
                 index=[f"enc{k}" for k in range(R["K"])]).round(4).to_csv(
        f"{outdir}/supp_rmse_scaled_{os.path.basename(rep_run)}.csv", encoding="utf-8-sig")
    seg, mean_pct = pick_representative_segment(R["cm"], order)
    fig1_visual(R["comps"], R["refs"], R["cm"], seg,
                f"{figdir}/fig1_visual_{os.path.basename(rep_run)}.png",
                f"그림1 · 성분 ↕ 참조 시각 대조 — {rep_run} · 분절 "
                f"{ds.meta[int(idx[seg])]['record_id']}_{ds.meta[int(idx[seg])]['seg_idx']:04d} "
                f"(대표성 백분위 {100*mean_pct[seg]:.0f})", cfg["data"]["fs"])
    with open(f"{outdir}/representative.json", "w", encoding="utf-8") as f:
        json.dump({"rule_model": "T6.5 충실도 관문 통과 모델 중 val 분리 품질 최고 (D43)",
                   "rule_segment": "참조별 매칭 인코더 |r| 평균 백분위가 50%에 최근접 (D42)",
                   "run": rep_run, "val_sep": R["val_sep"], "epoch": R["epoch"],
                   "passed_fidelity": sorted(passed),
                   "candidate_val_sep": {r: pool[r]["val_sep"] for r in pool},
                   "all_val_sep": {r: v["val_sep"] for r, v in res.items()},
                   "segment_index": int(idx[seg]),
                   "segment_id": f"{ds.meta[int(idx[seg])]['record_id']}_"
                                 f"{ds.meta[int(idx[seg])]['seg_idx']:04d}",
                   "segment_percentile": float(mean_pct[seg])},
                  f, ensure_ascii=False, indent=2)
    print(f"  대표 모델 = {rep_run} (val_sep {R['val_sep']:.4f}), "
          f"대표 분절 = {ds.meta[int(idx[seg])]['record_id']}_"
          f"{ds.meta[int(idx[seg])]['seg_idx']:04d}")

    # ---- 보조: 기여 분해 + 시드 집계 (D41로 강등)
    for K in (MAIN_K, ROBUST_K):
        runs = [f"{run_prefix}K{K}_seed{s}" for s in SEEDS]
        cms, shs = [], []
        for r in runs:
            o = role_order(res[r]["cm"].mean(0))
            cms.append(res[r]["cm"].mean(0)[o])
            shs.append(res[r]["share_pct"][o])
        cm_m, cm_s = np.mean(cms, 0), np.std(cms, 0)
        sh_m, sh_s = np.mean(shs, 0), np.std(shs, 0)
        idxn = ROLE_NAMES[:K]
        ts = pd.DataFrame({f"{c}_r": [f"{cm_m[k, j]:.3f}±{cm_s[k, j]:.3f}" for k in range(K)]
                           for j, c in enumerate(REF_KEYS)}, index=idxn)
        for j, c in enumerate(REF_KEYS):
            ts[f"{c}_기여%"] = [f"{sh_m[k, j]:.1f}±{sh_s[k, j]:.1f}" for k in range(K)]
        ts.to_csv(f"{outdir}/supp_seed_aggregate_K{K}.csv", encoding="utf-8-sig")
        fig1_correspondence(cm_m, cm_s, f"{figdir}/fig1_correspondence_K{K}_seedmean.png",
                            f"그림1 · K={K} 시드 3개 평균 (역할 정렬)")

    # ---- 참조 간 상관 + 스펙트럼 (모델 무관)
    refs = res[f"{run_prefix}K{MAIN_K}_seed{SEEDS[0]}"]["refs"]
    rc = reference_correlation(refs)
    pd.DataFrame([{"쌍": k, "mean": v.mean(), "median": np.median(v),
                   "p75": np.percentile(v, 75), "max": v.max()}
                  for k, v in rc.items()]).round(4).to_csv(
        f"{outdir}/reference_correlation.csv", index=False, encoding="utf-8-sig")
    sc = spectral_centroid(refs, cfg["data"]["fs"])
    pd.DataFrame({"참조": list(sc), "중심주파수_Hz_중앙": [np.median(v) for v in sc.values()]}
                 ).round(3).to_csv(f"{outdir}/reference_spectrum.csv", index=False,
                                   encoding="utf-8-sig")

    # ---- 표2: M0–M5 SNR 계단 + 전수 지도 백분위 + 통계
    lad_all, stat_all, abl_all = [], [], []
    for run, r in res.items():
        s = summarise_conditions(r["df"])
        s.insert(0, "run", run)
        emap = r["emap"]
        pct = {}
        for c in ("M0", "M2", "M3", "M4", "M5"):
            key = "-".join(map(str, r["sets"][c])) if r["sets"][c] else "(none)"
            row = emap[emap["mask"] == key]
            pct[c] = float(row.snr_percentile.iloc[0]) if len(row) else np.nan
        s["mask_percentile"] = s.cond.map(pct)
        s["mask_set"] = s.cond.map({c: str(r["sets"][c]) for c in ("M0", "M2", "M3", "M4", "M5")})
        lad_all.append(s)

        st = stats.paired_table(r["df"][r["df"].cond.isin(COND_ORDER)], "snr_db", "cond", "M0")
        st.insert(0, "run", run)
        stat_all.append(st)

        m0 = r["df"][r["df"].cond == "M0"].set_index("seg").snr_db
        order = role_order(r["cm"].mean(0))
        rolemap = {e: ROLE_NAMES[i] for i, e in enumerate(order)}
        for k in range(r["K"]):
            mk = r["df"][r["df"].cond == f"M1_e{k}"].set_index("seg").snr_db
            abl_all.append({"run": run, "enc": k, "role": rolemap[k],
                            "delta": float(np.median(mk - m0)),
                            **{f"w_{a}": b for a, b in
                               stats.wilcoxon_pair(mk.values, m0.values).items()}})
    lad = pd.concat(lad_all, ignore_index=True)
    lad.to_csv(f"{outdir}/table2_ladder.csv", index=False, encoding="utf-8-sig")
    pd.concat(stat_all, ignore_index=True).to_csv(
        f"{outdir}/stats_pairwise.csv", index=False, encoding="utf-8-sig")
    abl = pd.DataFrame(abl_all)
    abl["p_holm"], abl["유의"] = stats.holm(abl.w_p.values)
    abl.to_csv(f"{outdir}/ablation.csv", index=False, encoding="utf-8-sig")

    main_run = f"{run_prefix}K{MAIN_K}_seed{SEEDS[0]}"
    fig2_ladder(lad[lad.run == main_run], f"{figdir}/fig2_ladder_{os.path.basename(main_run)}.png",
                f"표2 그림 · M0–M5 SNR 계단 — {main_run}")
    fig3_ablation(abl[abl.run == main_run], f"{figdir}/fig3_ablation_{os.path.basename(main_run)}.png",
                  f"그림3 · 인코더 단독 마스킹(M1) 효과 — {main_run}")
    r = res[main_run]
    marks = {c: (len(r["sets"][c]),
                 float(lad[(lad.run == main_run) & (lad.cond == c)].snr_median.iloc[0]))
             for c in ("M2", "M3", "M4")}
    fig_mask_map(r["emap"], marks, f"{figdir}/fig_mask_map_{os.path.basename(main_run)}.png",
                 f"전수 2^{r['K']} 마스킹 지도 — {main_run}")
    for run, rr in res.items():
        rr["emap"].to_csv(f"{outdir}/mask_map_{os.path.basename(run)}.csv", index=False, encoding="utf-8-sig")
        rr["df"].to_csv(f"{outdir}/conditions_{os.path.basename(run)}.csv", index=False, encoding="utf-8-sig")
        with open(f"{outdir}/mask_sets_{os.path.basename(run)}.json", "w", encoding="utf-8") as f:
            json.dump({k: v for k, v in rr["sets"].items()}, f, ensure_ascii=False, indent=2)
    print(f"[리허설] 완료 → {outdir}/ , {figdir}/")
    return res, lad, abl, rep_run


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--split", default="val")
    p.add_argument("--n", type=int, default=1000)
    a = p.parse_args()
    main(a.config, a.split, a.n)
