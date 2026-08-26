"""S4 — 인코더–참조 대응 분석 (RESEARCH_DESIGN.md §8).

선행 연구는 실제 소스 파형을 알 수 없어 심박수 같은 파생 지표로 간접 평가했다. 본 연구는
주입한 잡음 성분을 개별 보존하므로 **성분과 참조 파형을 직접 대조하는 정량 평가**가 가능하다.

산출물
  ① 인코더 × 참조 대응표           stage1_matrix.csv        fig1_correspondence.png
  ② 참조 간 상관표                 reference_correlation.csv
  ③ 기여 분해 (다중 회귀)          stage1_contribution.csv
  ④ 시각 — bw 상관 상위/중앙/하위 분절 성분 그림 + 대상 인코더 |r| 히스토그램

**확정 지표 — 성분과 참조를 각각 z-정규화한 뒤 산출한다.**

| 지표 | 정의 | 성격 |
|---|---|---|
| ① 상관 r (+r²) | 피어슨 상관. 부호는 인코더–디코더 가중치의 임의 부호이므로 절대값 | 형태 유사도 |
| ② RMSE_norm | z-정규화·부호 정렬 후 `RMSE = √(2(1−ρ))` — **ρ의 함수**다. 해석 편의로 병기 | ①과 같은 정보의 다른 표현 |
| ③ MAD | z-정규화 후 `max |ẑ_k − r̂|`. 정규화 후에도 ρ와 독립인 유일한 지표 | 국소 최대 편차 (보완적) |

**원값 RMSE는 폐기했다** — 성분의 절대 크기가 임의 스케일이라 순위가 뒤집힌다
(에폭 48 실측: enc3이 bw \|r\| 0.602로 1위인데 원값 RMSE는 0.4936으로 최하위).

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
from .model.meae import enc_label
from .viz import plt

NOISE_REFS = ("bw", "ma", "em")


def load_ckpt(cfg, run):
    """run 은 실행 이름이거나 체크포인트 파일 경로다.
    후자를 허용하는 이유: pool/ 에 보관한 후보 에폭을 재학습 없이 그대로 불러 비교하기 위해서다."""
    if run.endswith(".pt") and os.path.exists(run):
        ck = torch.load(run, map_location="cpu", weights_only=False)
        model = meae.build(ck.get("cfg", cfg), ck["n_encoders"])
        model.load_state_dict(ck["model"])
        return model.eval(), ck
    name = os.path.basename(run)
    cand = [os.path.join("results", "01_train", run, f"{name}.pt"),  # 본 실험
            os.path.join("_work", "runs", run, f"{name}.pt"),         # 보조 실행
            os.path.join("_work", "archive", run, f"{name}.pt")]
    path = next((c for c in cand if os.path.exists(c)), cand[0])
    ck = torch.load(path, map_location="cpu", weights_only=False)
    # 체크포인트에 저장된 config를 우선한다 — hidden 등 구조 오버라이드가 반영돼 있다
    model = meae.build(ck.get("cfg", cfg), ck["n_encoders"])
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


def znorm(a):
    """마지막 축 기준 z-정규화. 상수 신호는 0으로 둔다."""
    s = a.std(-1, keepdims=True)
    return np.where(s > 0, _center(a) / np.maximum(s, 1e-12), 0.0)


def metric_matrices(comps, refs):
    """확정 지표 3종을 (n, K, R) 로 돌려준다: (|r|, RMSE_norm, MAD).

    성분과 참조를 각각 z-정규화한 뒤 계산한다. 성분의 절대 크기는 임의 스케일이므로
    원값 RMSE는 순위가 뒤집힌다(폐기).

      RMSE_norm = sqrt(2 * (1 - |rho|))     정규화·부호 정렬 후. rho 의 함수다
      MAD       = max |z_comp * sign(rho) - z_ref|   정규화 후에도 rho 와 독립
    """
    zc, zr = znorm(comps), znorm(refs)
    n, K, T = zc.shape
    R = zr.shape[1]
    rho = np.einsum("nkt,nrt->nkr", zc, zr) / T          # 부호 있는 상관
    corr = np.abs(rho)
    rmse_norm = np.sqrt(np.maximum(2.0 * (1.0 - corr), 0.0))
    mad = np.empty((n, K, R))
    for k in range(K):                    # (n,K,R,T) 를 한 번에 만들면 수백 MB가 된다
        sgn = np.sign(rho[:, k, :])[:, :, None]
        sgn[sgn == 0] = 1.0
        mad[:, k, :] = np.abs(zc[:, k, None, :] * sgn - zr).max(-1)
    return corr, rmse_norm, mad


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
def fig_correspondence(cm, out, run, rn=None, mad=None):
    """색은 |r|, 칸 안에 RMSE_norm·MAD를 병기한다. 모두 z-정규화 후 값이다."""
    m, sd = cm.mean(0), cm.std(0)
    mr = rn.mean(0) if rn is not None else None
    mm = mad.mean(0) if mad is not None else None
    K = m.shape[0]
    fig, ax = plt.subplots(figsize=(6.6, 0.86 * K + 2))
    im = ax.imshow(m, cmap="magma", vmin=0, vmax=max(0.6, m.max()), aspect="auto")
    ax.set_xticks(range(len(REF_KEYS)))
    ax.set_xticklabels(list(REF_KEYS))
    ax.set_yticks(range(K))
    ax.set_yticklabels([enc_label(k) for k in range(K)])
    for k in range(K):
        for r in range(len(REF_KEYS)):
            lab = f"|r| {m[k, r]:.2f}±{sd[k, r]:.2f}"
            if mr is not None:
                lab += f"\nRMSEn {mr[k, r]:.2f}"
            if mm is not None:
                lab += f"\nMAD {mm[k, r]:.2f}"
            ax.text(r, k, lab, ha="center", va="center",
                    fontsize=6.5, color="white" if m[k, r] < m.max() * .6 else "black")
    for r in range(len(REF_KEYS)):
        ax.add_patch(plt.Rectangle((r - .5, m[:, r].argmax() - .5), 1, 1,
                                   fill=False, ec="cyan", lw=2))
    ax.set_title(f"① 인코더 × 참조 대응 — {run}\n"
                 "z-정규화 후 |r| (색) · RMSE_norm · MAD", fontsize=10)
    fig.colorbar(im, ax=ax, shrink=.8, label="|r|")
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


def fig_hist(cm, k, out):
    fig, ax = plt.subplots(figsize=(8, 4))
    for j, name in enumerate(REF_KEYS):
        ax.hist(cm[:, k, j], bins=40, alpha=.55, label=f"{name} (중앙 {np.median(cm[:, k, j]):.3f})")
    ax.set_xlabel(f"인코더 {k+1} 성분과 참조의 |r| (분절 단위)")
    ax.set_ylabel("분절 수")
    ax.legend(fontsize=8)
    ax.grid(alpha=.3, lw=.4)
    ax.set_title(f"④ 인코더 {k+1}의 분절별 |r| 분포 — 평균값 뒤에 가려진 산포", fontsize=10)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def main(config="configs/default.yaml", run="K8_seed42", split="val",
         outdir="results/02_separation", figdir="results/02_separation/figures"):
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

    # ① 대응표 — 확정 지표 3종. 모두 z-정규화 후 값이다
    cm, rn, mad = metric_matrices(comps, refs)
    ix = [enc_label(k) for k in range(K)]
    mat = pd.DataFrame(cm.mean(0), columns=list(REF_KEYS), index=ix)
    sd = pd.DataFrame(cm.std(0), columns=[f"{c}_sd" for c in REF_KEYS], index=ix)
    r2c = pd.DataFrame((cm ** 2).mean(0), columns=[f"{c}_r2" for c in REF_KEYS], index=ix)
    rnm = pd.DataFrame(rn.mean(0), columns=[f"{c}_rmse_norm" for c in REF_KEYS], index=ix)
    rnsd = pd.DataFrame(rn.std(0), columns=[f"{c}_rmse_norm_sd" for c in REF_KEYS], index=ix)
    mdm = pd.DataFrame(mad.mean(0), columns=[f"{c}_mad" for c in REF_KEYS], index=ix)
    mdsd = pd.DataFrame(mad.std(0), columns=[f"{c}_mad_sd" for c in REF_KEYS], index=ix)
    energy = (comps ** 2).mean(-1).mean(0)
    mat_out = pd.concat([mat.round(4), sd.round(4), r2c.round(4),
                         rnm.round(4), rnsd.round(4), mdm.round(4), mdsd.round(4)], axis=1)
    mat_out["energy_ratio"] = (energy / energy.sum()).round(4)
    mat_out.to_csv(f"{outdir}/stage1_matrix.csv", encoding="utf-8-sig")
    with open(f"{outdir}/stage1_matrix_note.txt", "w", encoding="utf-8") as fnote:
        fnote.write(
            "지표는 성분과 참조를 각각 z-정규화한 뒤 산출한다.\n"
            "① |r| — 형태 유사도. 부호는 인코더–디코더 가중치의 임의 부호이므로 절대값.\n"
            "② RMSE_norm — 정규화·부호 정렬 후 RMSE = sqrt(2(1-|r|)) 로 |r|의 함수다.\n"
            "   ①과 같은 정보의 다른 표현이며 해석 편의를 위해 병기한다. 독립 근거가 아니다.\n"
            "③ MAD — 정규화 후 max|z_comp - z_ref|. 정규화 후에도 |r|과 독립인 유일한 지표로\n"
            "   국소 최대 편차를 포착한다.\n"
            "원값 RMSE는 폐기했다 — 성분의 절대 크기가 임의 스케일이라 순위가 뒤집힌다.\n")
    fig_correspondence(cm, f"{figdir}/fig1_correspondence.png", run, rn, mad)

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
    out.to_csv(f"{outdir}/stage1_contribution.csv", encoding="utf-8-sig")

    # ④ 시각 — bw 최대 상관 인코더를 기준 인코더로
    k_bw = int(cm.mean(0)[:, list(REF_KEYS).index("bw")].argmax())
    v = cm[:, k_bw, list(REF_KEYS).index("bw")]
    order = np.argsort(v)
    picks = {"high": int(order[-1]), "mid": int(order[len(order) // 2]), "low": int(order[0])}
    for tag, i in picks.items():
        fig_components(comps, refs, ds.x_noisy, i,
                       f"{figdir}/components_bw_{tag}.png",
                       f"{enc_label(k_bw)}–bw |r| {v[i]:.3f} ({tag}) · "
                       f"{ds.meta[i]['record_id']}_{ds.meta[i]['seg_idx']:04d}", fs)
    fig_hist(cm, k_bw, f"{figdir}/hist_{enc_label(k_bw)}.png")

    # ---- 콘솔 보고
    print(f"=== {run} · {split} {len(ds)}분절 · best epoch {ck['epoch']} ===\n")
    print("지표는 성분·참조를 각각 z-정규화한 뒤 산출한다. 원값 RMSE는 폐기.\n")
    print("① 인코더 × 참조 |r| 평균 (±SD)  [r² 병기]")
    print(pd.DataFrame({c: [f"{mat.loc[i, c]:.3f}±{sd.loc[i, c+'_sd']:.3f} "
                            f"(r² {r2c.loc[i, c+'_r2']:.3f})" for i in mat.index]
                        for c in REF_KEYS}, index=mat.index).to_string(), "\n")
    print("② 인코더 × 참조 RMSE_norm 평균 (±SD)   = √(2(1−|r|)) — |r|의 함수, 병기용")
    print(pd.DataFrame({c: [f"{rnm.loc[i, c+'_rmse_norm']:.3f}±{rnsd.loc[i, c+'_rmse_norm_sd']:.3f}"
                            for i in mat.index] for c in REF_KEYS},
                       index=mat.index).to_string(), "\n")
    print("③ 인코더 × 참조 MAD 평균 (±SD)   국소 최대 편차 — |r|과 독립")
    print(pd.DataFrame({c: [f"{mdm.loc[i, c+'_mad']:.3f}±{mdsd.loc[i, c+'_mad_sd']:.3f}"
                            for i in mat.index] for c in REF_KEYS},
                       index=mat.index).to_string(), "\n")
    print("④ 참조 간 |r| — 분리 한계의 독립 근거")
    print(ref_df.to_string(index=False))
    print(spec_df.to_string(index=False), "\n")
    print("⑤ 기여 분해 — 참조 에너지 중 각 성분이 설명하는 몫 (%)")
    print(out[list(REF_KEYS)].to_string(), "\n")
    print(f"⑥ 기준 인코더 = {enc_label(k_bw)} (bw 최대 상관). 시각 4종 → {figdir}/")
    return mat_out, ref_df, out


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--run", default="K8_seed42")
    p.add_argument("--split", default="val")
    a = p.parse_args()
    main(a.config, a.run, a.split)
