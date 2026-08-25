"""T5 파일럿 보고 4항목 생성 (RESEARCH_DESIGN.md §6, T5).

  ① 성분 파형 적층 그림      → logs/<run>/plots/ (학습 중 저장) + best 체크포인트 기준 1장
  ② 4개 손실 항 에폭별 크기  → figures/pilot/loss_terms.png
  ③ 경계 120샘플 아티팩트    → figures/pilot/boundary.png + 수치
  ④ 비공식 대응 행렬 히트맵  → figures/pilot/corr_heatmap.png  ★go/no-go 관문

④는 S4가 아니다. S4는 test셋에서 δ·ε 임계와 통계 검정까지 갖춰 수행한다.
여기서는 검증 200분절로 "잡음마다 두드러진 인코더가 있는가"만 눈으로 확인한다.
"""
import argparse
import os

import numpy as np
import pandas as pd
import torch

from .data.build import load_cfg
from .data.dataset import REF_KEYS, load
from .model import meae
from .train import NOISE_REFS, _corr, _plot_components
from .viz import plt


def load_ckpt(cfg, run):
    ck = torch.load(os.path.join("checkpoints", f"{run}.pt"), map_location="cpu",
                    weights_only=False)
    model = meae.build(cfg, ck["n_encoders"])
    model.load_state_dict(ck["model"])
    return model.eval(), ck


@torch.no_grad()
def correspondence(model, val, device, n_seg=200, batch=100):
    """인코더 x 참조 |r| 행렬 + 성분·경계 통계."""
    K, pad_each = model.n_encoders, model.pad_each
    n = min(n_seg, len(val))
    acc = np.zeros((K, len(REF_KEYS)))
    energy = np.zeros(K)
    edge, mid = [], []
    for s in range(0, n, batch):
        idx = np.arange(s, min(s + batch, n))
        x = meae.pad(val.tensor(idx).to(device), pad_each)
        comps = torch.stack([meae.crop(model.component(x, k), pad_each).squeeze(1)
                             for k in range(K)])
        refs = torch.stack([val.ref_tensor(r, idx).to(device).squeeze(1) for r in REF_KEYS])
        for k in range(K):
            for r in range(len(REF_KEYS)):
                acc[k, r] += _corr(comps[k], refs[r]).abs().sum().item()
        energy += comps.pow(2).mean(-1).sum(-1).cpu().numpy()
        # 경계 아티팩트: 재구성 신호의 양끝 pad_each 구간 vs 중앙
        recon = meae.crop(model(x)[0], pad_each).squeeze(1)
        edge.append(torch.cat([recon[:, :pad_each], recon[:, -pad_each:]], 1).abs().cpu())
        mid.append(recon[:, pad_each:-pad_each].abs().cpu())
    return acc / n, energy / n, torch.cat(edge), torch.cat(mid), n


def fig_heatmap(corr, run, out):
    K = corr.shape[0]
    fig, ax = plt.subplots(figsize=(5.2, 0.62 * K + 2))
    im = ax.imshow(corr, cmap="magma", vmin=0, vmax=max(0.6, corr.max()), aspect="auto")
    ax.set_xticks(range(len(REF_KEYS)))
    ax.set_xticklabels(["clean", "bw", "ma", "em"])
    ax.set_yticks(range(K))
    ax.set_yticklabels([f"인코더 {k}" for k in range(K)])
    for k in range(K):
        top = corr[k].argmax()
        for r in range(len(REF_KEYS)):
            ax.text(r, k, f"{corr[k, r]:.2f}", ha="center", va="center", fontsize=8,
                    color="white" if corr[k, r] < corr.max() * .6 else "black",
                    fontweight="bold" if r == top else "normal")
    for r, name in enumerate(REF_KEYS):
        ax.add_patch(plt.Rectangle((r - .5, corr[:, r].argmax() - .5), 1, 1,
                                   fill=False, ec="cyan", lw=2))
    ax.set_title(f"비공식 대응 행렬 |r| — {run}\n(검증 분절, 청록 테두리 = 참조별 최대)",
                 fontsize=10)
    fig.colorbar(im, ax=ax, shrink=.8, label="|피어슨 r| 평균")
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def fig_loss_terms(hist, out):
    terms = [("train_recon", "재구성 (MSE)"), ("train_mixing", "sparse mixing (원값)"),
             ("train_zero_recon", "zero reconstruction (원값)"), ("train_z_l2", "인코딩 L2 (원값)")]
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    for key, label in terms:
        ax[0].plot(hist.epoch, hist[key], lw=1.2, label=label)
    ax[0].set_yscale("log")
    ax[0].set_title("① 손실 항 원값 (가중치 적용 전)", fontsize=10, loc="left")
    w = {"train_recon": 1.0, "train_mixing": 1e-3, "train_zero_recon": 1e-2, "train_z_l2": 1e-2}
    for key, label in terms:
        ax[1].plot(hist.epoch, hist[key] * w[key], lw=1.2, label=label)
    ax[1].set_yscale("log")
    ax[1].set_title("② 총 손실에 실제로 기여하는 크기 (가중치 적용 후)", fontsize=10, loc="left")
    for a in ax:
        a.set_xlabel("에폭")
        a.grid(alpha=.3, lw=.4)
        a.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def fig_boundary(model, val, device, out):
    pad_each = model.pad_each
    with torch.no_grad():
        x = meae.pad(val.tensor(np.arange(3)).to(device), pad_each)
        recon = meae.crop(model(x)[0], pad_each).squeeze(1).cpu().numpy()
    fig, ax = plt.subplots(3, 1, figsize=(11, 6), sharex=True)
    for a, i in zip(ax, range(3)):
        a.plot(recon[i], lw=.6, color="#555")
        a.plot(val.x_noisy[i], lw=.5, color="#000", alpha=.4, label="입력")
        a.axvspan(0, pad_each, color="crimson", alpha=.12)
        a.axvspan(len(recon[i]) - pad_each, len(recon[i]), color="crimson", alpha=.12)
        a.grid(alpha=.25, lw=.4)
        a.tick_params(labelsize=7)
    ax[0].set_title(f"③ 경계 {pad_each}샘플 아티팩트 확인 (붉은 구간 = 패딩에 인접한 양끝)",
                    fontsize=10, loc="left")
    ax[0].legend(fontsize=8)
    ax[-1].set_xlabel("샘플")
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def main(config="configs/default.yaml", run="K8_seed42", n_seg=200):
    cfg = load_cfg(config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    outdir = "figures/pilot"
    os.makedirs(outdir, exist_ok=True)

    model, ck = load_ckpt(cfg, run)
    model = model.to(device)
    val = load(cfg, "val")
    hist = pd.read_csv(os.path.join("logs", run, "history.csv"))

    corr, energy, edge, mid, n = correspondence(model, val, device, n_seg)
    fig_heatmap(corr, run, f"{outdir}/corr_heatmap.png")
    fig_loss_terms(hist, f"{outdir}/loss_terms.png")
    fig_boundary(model, val, device, f"{outdir}/boundary.png")
    _plot_components(model, val, cfg, device, f"{outdir}/components.png", idx=0)

    df = pd.DataFrame(corr, columns=list(REF_KEYS),
                      index=[f"enc{k}" for k in range(model.n_encoders)])
    df["energy_ratio"] = energy / energy.sum()
    df.to_csv(f"{outdir}/corr_matrix.csv", encoding="utf-8-sig")

    print(f"=== {run} · best epoch {ck['epoch']} · 검증 {n}분절 ===\n")
    print("① 대응 행렬 |r| (행=인코더, 열=참조)")
    print(df.round(3).to_string(), "\n")
    print("② 참조별 최대 상관 인코더")
    for r in NOISE_REFS + ("x_clean",):
        c = list(REF_KEYS).index(r)
        k = int(corr[:, c].argmax())
        srt = np.sort(corr[:, c])[::-1]
        print(f"   {r:>7}: 인코더 {k}  |r|={srt[0]:.3f}  (2위 {srt[1]:.3f}, 격차 {srt[0]-srt[1]:.3f})")
    print(f"\n③ 경계 {model.pad_each}샘플 |진폭| 평균 {edge.mean():.4f} vs "
          f"중앙 {mid.mean():.4f}  (비 {edge.mean()/mid.mean():.2f}x)")
    print(f"④ 에폭당 소요 {hist.sec.mean():.1f}s (총 {len(hist)}에폭, "
          f"{hist.sec.sum()/60:.1f}분), 최고 분리품질 {ck['val_separation']:.4f}")
    return df


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--run", default="K8_seed42")
    p.add_argument("--n-seg", type=int, default=200)
    a = p.parse_args()
    main(a.config, a.run, a.n_seg)
