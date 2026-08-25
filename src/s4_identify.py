"""S4 — 인코더–참조 대응 분석 (RESEARCH_DESIGN.md §7).

T5.6 산출물 패키지 v1
  ① 인코더 × 참조 상관 행렬        stage1_matrix.csv        fig1_correspondence.png
  ② 참조 간 상관표                 reference_correlation.csv
  ③ 기여 분해 (다중 회귀)          stage1_contribution.csv
  ④ 시각 — bw 상관 상위/중앙/하위 분절 성분 그림 + 대상 인코더 |r| 히스토그램

**기여 분해**: 참조 r을 K개 성분에 다중 회귀하고, 설명 분산을 성분별로 쪼갠다.

    r ≈ Σ_k β_k · x̂_k        (분절마다, 평균 제거 후)
    R²      = 1 − ‖r − Σβ_k x̂_k‖² / ‖r‖²
    몫_k    = β_k · cov(x̂_k, r) / var(r)        (Σ_k 몫_k = R²)

단일 성분 상관(|r|)은 "모양이 얼마나 닮았나"만 말한다. 기여 분해는 **"그 참조의 에너지 중
몇 %를 그 성분이 설명하나"** 에 답한다. 비선형 디코더에서 성분의 절대 크기는 임의 스케일이라
β로 배율을 보정한 뒤 해부해야 한다. 몫은 억제(suppression) 상황에서 음수가 될 수 있으며,
그 자체가 "다른 성분과 겹쳐 상쇄 역할을 한다"는 정보다.

모든 계산은 **crop 후 중앙 3600 구간**에서 한다 (§5A).
"""
import argparse
import os

import numpy as np
import pandas as pd
import torch

from .data.build import load_cfg
from .data.dataset import REF_KEYS, load
from .model import meae
from .viz import plt

NOISE_REFS = ("bw", "ma", "em")


def load_ckpt(cfg, run):
    name = os.path.basename(run)
    cand = [os.path.join("runs", run, f"{name}.pt"),          # 표준 위치
            os.path.join("archive", run, f"{name}.pt"),        # 폐기된 실행
            os.path.join("runs", f"{run}.pt")]
    path = next((c for c in cand if os.path.exists(c)), cand[0])
    ck = torch.load(path, map_location="cpu", weights_only=False)
    model = meae.build(cfg, ck["n_encoders"])
    model.load_state_dict(ck["model"])
    return model.eval(), ck


@torch.no_grad()
def component_bank(model, ds, device, idx, batch=100):
    """(n, K, 3600) 성분과 (n, R, 3600) 참조를 numpy로 돌려준다."""
    K, pad = model.n_encoders, model.pad_each
    comps, refs = [], []
    for s in range(0, len(idx), batch):
        j = idx[s:s + batch]
        x = meae.pad(ds.tensor(j).to(device), pad)
        c = torch.stack([meae.crop(model.component(x, k), pad).squeeze(1)
                         for k in range(K)], 1)                      # (B, K, T)
        r = torch.stack([ds.ref_tensor(k, j).to(device).squeeze(1)
                         for k in REF_KEYS], 1)                      # (B, R, T)
        comps.append(c.cpu().numpy().astype(np.float64))
        refs.append(r.cpu().numpy().astype(np.float64))
    return np.concatenate(comps), np.concatenate(refs)


def _center(a):
    return a - a.mean(-1, keepdims=True)


def corr_matrix(comps, refs):
    """(n, K, R) 분절별 |피어슨 r|."""
    c, r = _center(comps), _center(refs)
    cn = np.linalg.norm(c, axis=-1)[:, :, None]
    rn = np.linalg.norm(r, axis=-1)[:, None, :]
    num = np.einsum("nkt,nrt->nkr", c, r)
    den = cn * rn
    return np.where(den > 0, np.abs(num) / np.maximum(den, 1e-12), 0.0)


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


# ---------------------------------------------------------------- 그림
def fig_correspondence(cm, out, run):
    m = cm.mean(0)
    sd = cm.std(0)
    K = m.shape[0]
    fig, ax = plt.subplots(figsize=(5.6, 0.66 * K + 2))
    im = ax.imshow(m, cmap="magma", vmin=0, vmax=max(0.6, m.max()), aspect="auto")
    ax.set_xticks(range(len(REF_KEYS)))
    ax.set_xticklabels(list(REF_KEYS))
    ax.set_yticks(range(K))
    ax.set_yticklabels([f"인코더 {k}" for k in range(K)])
    for k in range(K):
        for r in range(len(REF_KEYS)):
            ax.text(r, k, f"{m[k, r]:.2f}\n±{sd[k, r]:.2f}", ha="center", va="center",
                    fontsize=7, color="white" if m[k, r] < m.max() * .6 else "black")
    for r in range(len(REF_KEYS)):
        ax.add_patch(plt.Rectangle((r - .5, m[:, r].argmax() - .5), 1, 1,
                                   fill=False, ec="cyan", lw=2))
    ax.set_title(f"① 인코더 × 참조 |r| 평균±SD — {run}", fontsize=10)
    fig.colorbar(im, ax=ax, shrink=.8)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def fig_components(comps, refs, x_noisy, i, out, title, fs):
    K = comps.shape[1]
    rows = ([("입력 x_noisy", x_noisy[i], "#000")]
            + [(f"성분 {k}", comps[i, k], "#1f77b4") for k in range(K)]
            + [(f"참조 {n}", refs[i, list(REF_KEYS).index(n)], "#d62728") for n in NOISE_REFS]
            + [("참조 clean", refs[i, 0], "#2ca02c")])
    t = np.arange(comps.shape[-1]) / fs
    fig, ax = plt.subplots(len(rows), 1, figsize=(11, 1.0 * len(rows)), sharex=True)
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


def fig_hist(cm, k, out):
    fig, ax = plt.subplots(figsize=(8, 4))
    for j, name in enumerate(REF_KEYS):
        ax.hist(cm[:, k, j], bins=40, alpha=.55, label=f"{name} (중앙 {np.median(cm[:, k, j]):.3f})")
    ax.set_xlabel(f"인코더 {k} 성분과 참조의 |r| (분절 단위)")
    ax.set_ylabel("분절 수")
    ax.legend(fontsize=8)
    ax.grid(alpha=.3, lw=.4)
    ax.set_title(f"④ 인코더 {k}의 분절별 |r| 분포 — 평균값 뒤에 가려진 산포", fontsize=10)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def main(config="configs/default.yaml", run="K8_seed42_lam0.01", split="val",
         outdir="analysis/s4", figdir="analysis/s4/figures"):
    cfg = load_cfg(config)
    fs = cfg["data"]["fs"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(outdir, exist_ok=True)
    os.makedirs(figdir, exist_ok=True)

    model, ck = load_ckpt(cfg, run)
    model = model.to(device)
    ds = load(cfg, split)
    idx = np.arange(len(ds))
    comps, refs = component_bank(model, ds, device, idx)
    K = comps.shape[1]

    # ① 상관 행렬
    cm = corr_matrix(comps, refs)
    mat = pd.DataFrame(cm.mean(0), columns=list(REF_KEYS), index=[f"enc{k}" for k in range(K)])
    sd = pd.DataFrame(cm.std(0), columns=[f"{c}_sd" for c in REF_KEYS],
                      index=mat.index)
    energy = (comps ** 2).mean(-1).mean(0)
    mat_out = pd.concat([mat.round(4), sd.round(4)], axis=1)
    mat_out["energy_ratio"] = (energy / energy.sum()).round(4)
    mat_out.to_csv(f"{outdir}/stage1_matrix.csv", encoding="utf-8-sig")
    fig_correspondence(cm, f"{figdir}/fig1_correspondence.png", run)

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
                       index=[f"enc{k}" for k in range(K)])
    con_sd = pd.DataFrame(100 * share.std(0), columns=[f"{c}_sd" for c in REF_KEYS],
                          index=con.index)
    out = pd.concat([con.round(2), con_sd.round(2)], axis=1)
    out.loc["합계 (R²·%)"] = list((100 * r2.mean(0)).round(2)) + list((100 * r2.std(0)).round(2))
    out.to_csv(f"{outdir}/stage1_contribution.csv", encoding="utf-8-sig")

    # ④ 시각 — bw 최대 상관 인코더를 기준 인코더로
    k_bw = int(cm.mean(0)[:, list(REF_KEYS).index("bw")].argmax())
    v = cm[:, k_bw, list(REF_KEYS).index("bw")]
    order = np.argsort(v)
    picks = {"high": int(order[-1]), "mid": int(order[len(order) // 2]), "low": int(order[0])}
    for tag, i in picks.items():
        fig_components(comps, refs, ds.x_noisy, i,
                       f"{figdir}/components_bw_{tag}.png",
                       f"인코더 {k_bw}–bw |r| {v[i]:.3f} ({tag}) · "
                       f"{ds.meta[i]['record_id']}_{ds.meta[i]['seg_idx']:04d}", fs)
    fig_hist(cm, k_bw, f"{figdir}/hist_enc{k_bw}.png")

    # ---- 콘솔 보고
    print(f"=== {run} · {split} {len(ds)}분절 · best epoch {ck['epoch']} ===\n")
    print("① 인코더 × 참조 |r| 평균 (±SD)")
    print(pd.DataFrame({c: [f"{mat.loc[i, c]:.3f}±{sd.loc[i, c+'_sd']:.3f}" for i in mat.index]
                        for c in REF_KEYS}, index=mat.index).to_string(), "\n")
    print("② 참조 간 |r| — 분리 한계의 독립 근거")
    print(ref_df.to_string(index=False))
    print(spec_df.to_string(index=False), "\n")
    print("③ 기여 분해 — 참조 에너지 중 각 성분이 설명하는 몫 (%)")
    print(out[list(REF_KEYS)].to_string(), "\n")
    print(f"④ 기준 인코더 = enc{k_bw} (bw 최대 상관). 시각 4종 → {figdir}/")
    return mat_out, ref_df, out


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--run", default="K8_seed42_lam0.01")
    p.add_argument("--split", default="val")
    a = p.parse_args()
    main(a.config, a.run, a.split)
