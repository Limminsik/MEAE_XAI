"""S3 — 학습 (RESEARCH_DESIGN.md §6).

자기지도: 입력도 목표도 x_noisy 하나다. clean·bw·ma·em은 손실에 절대 들어가지 않고,
검증 로깅과 체크포인트 선택에만 쓰인다 (§0 원칙 3·4).

매 에폭 3층위 로깅
  학습 상태 : 총 손실 + 4개 항 개별, 검증 재구성 손실
  분리 품질 : 인코더 x 참조 상관 행렬, **잡음별 최대상관의 평균**
  붕괴 감지 : 인코더별 성분 에너지 비율, 인코더 간 성분 상관

조기 종료 : 검증 재구성 손실, patience 10 (자기지도 손실만 사용)
체크포인트: 검증 **분리 품질** 최고 에폭. 선행이 수동으로 하던 체크포인트 고르기를
            정답 잡음으로 자동화한 것이며 손실에는 관여하지 않는다.
"""
import argparse
import json
import os
import random
import time

import numpy as np
import pandas as pd
import torch

from .data.build import load_cfg
from .data.dataset import REF_KEYS, load
from .model import losses, meae

NOISE_REFS = ("bw", "ma", "em")


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _corr(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """마지막 축 기준 피어슨 상관. (B, L) x (B, L) → (B,)"""
    a = a - a.mean(-1, keepdim=True)
    b = b - b.mean(-1, keepdim=True)
    denom = a.norm(dim=-1) * b.norm(dim=-1)
    return torch.where(denom > 0, (a * b).sum(-1) / denom.clamp_min(1e-12),
                       torch.zeros_like(denom))


@torch.no_grad()
def evaluate(model, crit, val, cfg, device, batch: int):
    """검증 재구성 손실 + 분리 품질 + 붕괴 지표를 한 번에 낸다."""
    model.eval()
    pad_each, K = model.pad_each, model.n_encoders
    n = len(val)
    recon_sum, seen = 0.0, 0
    corr_sum = torch.zeros(K, len(REF_KEYS))          # 인코더 x 참조 (|r| 합)
    energy_sum = torch.zeros(K)
    pair_sum = torch.zeros(K, K)

    for s in range(0, n, batch):
        idx = np.arange(s, min(s + batch, n))
        x = meae.pad(val.tensor(idx).to(device), pad_each)
        y, zs = model(x)
        recon_sum += torch.nn.functional.mse_loss(y, x, reduction="sum").item()
        seen += x.numel()

        comps = torch.stack([meae.crop(model.component(x, k), pad_each).squeeze(1)
                             for k in range(K)])              # (K, B, 3600)
        refs = torch.stack([val.ref_tensor(r, idx).to(device).squeeze(1)
                            for r in REF_KEYS])                # (R, B, 3600)
        for k in range(K):
            for r in range(len(REF_KEYS)):
                corr_sum[k, r] += _corr(comps[k], refs[r]).abs().sum().cpu()
        energy_sum += comps.pow(2).mean(-1).sum(-1).cpu()
        for i in range(K):
            for j in range(K):
                pair_sum[i, j] += _corr(comps[i], comps[j]).abs().sum().cpu()

    corr = (corr_sum / n).numpy()
    energy = (energy_sum / n).numpy()
    pair = (pair_sum / n).numpy()
    off = pair[~np.eye(K, dtype=bool)]

    # 분리 품질 = 잡음별 최대 |r| 평균 - penalty x 중복 수.
    # 덜 학습된 모델은 모든 성분이 완만한 곡선이라 bw/ma와 폭넓게 상관돼 앞항이 높게 나온다.
    # 그때는 세 잡음의 최고 인코더가 한 곳으로 몰리므로, 중복 페널티가 이를 걸러낸다.
    # 헝가리안 1:1 강제 배정은 쓰지 않는다 — 분리 실패를 강제로 가려버리기 때문이다.
    noise_cols = [REF_KEYS.index(r) for r in NOISE_REFS]
    base = float(np.mean([corr[:, c].max() for c in noise_cols]))
    tops = [int(corr[:, c].argmax()) for c in noise_cols]
    n_dup = len(NOISE_REFS) - len(set(tops))
    penalty = cfg["train"]["separation_duplicate_penalty"]
    model.train()
    return {
        "val_recon": recon_sum / seen,
        "val_separation": base - penalty * n_dup,
        "sep_base": base, "n_dup": n_dup, "tops": tops,
        "corr": corr, "energy": energy, "pair": pair,
        "energy_ratio": (energy / max(energy.sum(), 1e-12)),
        "max_pair_corr": float(off.max()) if off.size else 0.0,
    }


def _plot_components(model, val, cfg, device, path, idx=0):
    """성분 파형 적층 그림 (선행 저장소 방식, 불안정성 디버깅에 필수)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from .viz import plt as _  # 한글 폰트 설정 재사용

    pad_each, K = model.pad_each, model.n_encoders
    fs = cfg["data"]["fs"]
    model.eval()
    with torch.no_grad():
        x = meae.pad(val.tensor(np.array([idx])).to(device), pad_each)
        comps = [meae.crop(model.component(x, k), pad_each).squeeze().cpu().numpy()
                 for k in range(K)]
        recon = meae.crop(model(x)[0], pad_each).squeeze().cpu().numpy()
    model.train()

    rows = ([("입력 x_noisy", val.x_noisy[idx], "#000000"),
             ("재구성 x_hat", recon, "#555555")]
            + [(f"성분 {k}", c, "#1f77b4") for k, c in enumerate(comps)]
            + [(f"참조 {r}", val.refs[r][idx], "#d62728") for r in NOISE_REFS]
            + [("참조 clean", val.refs["x_clean"][idx], "#2ca02c")])
    t = np.arange(len(rows[0][1])) / fs
    fig, ax = plt.subplots(len(rows), 1, figsize=(11, 1.05 * len(rows)), sharex=True)
    for a, (label, v, c) in zip(ax, rows):
        a.plot(t, v, lw=0.6, color=c)
        a.set_title(label, fontsize=8, loc="left")
        a.grid(alpha=.25, lw=.4)
        a.tick_params(labelsize=7)
    ax[-1].set_xlabel("시간 (초)")
    fig.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def train(cfg, n_encoders: int, seed: int, tag: str = "", plot_every: int = 5,
          max_epoch: int = None, lambda_mixing: float = None):
    if lambda_mixing is not None:                 # λ 스윕 전용 오버라이드
        cfg = json.loads(json.dumps(cfg))
        cfg["loss"]["lambda_mixing"] = lambda_mixing
    tr_cfg = cfg["train"]
    max_epoch = max_epoch or tr_cfg["max_epoch"]
    batch = tr_cfg["batch"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    run = tag or (f"K{n_encoders}_seed{seed}"
                  + (f"_lam{cfg['loss']['lambda_mixing']:g}" if lambda_mixing is not None else ""))

    set_seed(seed)
    train_set, val_set = load(cfg, "train"), load(cfg, "val")
    model = meae.build(cfg, n_encoders).to(device)
    crit = losses.build(cfg, n_encoders).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=tr_cfg["lr"],
                           weight_decay=tr_cfg["weight_decay"])
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=tr_cfg["lr_step_size"], gamma=0.1)

    ckpt_dir, log_dir = "checkpoints", os.path.join("logs", run)
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    print(f"[{run}] device={device} train={len(train_set)} val={len(val_set)} "
          f"K={n_encoders} max_epoch={max_epoch}")

    rng = np.random.default_rng(seed)
    history, best_sep, best_recon, bad = [], -np.inf, np.inf, 0
    for epoch in range(1, max_epoch + 1):
        t0 = time.time()
        order = rng.permutation(len(train_set))
        sums, nb = {}, 0
        for s in range(0, len(order), batch):
            x = meae.pad(train_set.tensor(order[s:s + batch]).to(device), model.pad_each)
            y, zs = model(x)
            d = crit(model, x, y, zs)          # 목표도 x_noisy — 자기지도
            opt.zero_grad(set_to_none=True)
            d["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), tr_cfg["gradient_clip_val"])
            opt.step()
            for k, v in d.items():
                sums[k] = sums.get(k, 0.0) + float(v.detach())
            nb += 1
        sched.step()
        train_terms = {f"train_{k}": v / nb for k, v in sums.items()}
        ev = evaluate(model, crit, val_set, cfg, device, batch)
        dt = time.time() - t0

        row = {"epoch": epoch, **train_terms, "val_recon": ev["val_recon"],
               "val_separation": ev["val_separation"],
               "sep_base": ev["sep_base"], "n_dup": ev["n_dup"],
               "tops": "".join(str(t) for t in ev["tops"]),   # bw/ma/em 최고 인코더
               "clean_max_corr": float(ev["corr"][:, REF_KEYS.index("x_clean")].max()),
               "max_pair_corr": ev["max_pair_corr"],
               "min_energy_ratio": float(ev["energy_ratio"].min()),
               "n_dead": int((ev["energy_ratio"] < 0.01).sum()),
               "lr": opt.param_groups[0]["lr"], "sec": dt}
        for r_i, r in enumerate(REF_KEYS):
            for k in range(n_encoders):
                row[f"corr_e{k}_{r}"] = float(ev["corr"][k, r_i])
        history.append(row)
        pd.DataFrame(history).to_csv(os.path.join(log_dir, "history.csv"),
                                     index=False, encoding="utf-8-sig")

        print(f"  ep{epoch:3d} tot={train_terms['train_total']:.5f} "
              f"rec={train_terms['train_recon']:.5f} mix={train_terms['train_mixing']:.4f} "
              f"zero={train_terms['train_zero_recon']:.5f} zl2={train_terms['train_z_l2']:.4f} "
              f"| val_rec={ev['val_recon']:.5f} "
              f"sep={ev['val_separation']:.3f}(base {ev['sep_base']:.3f} dup {ev['n_dup']}"
              f" tops {''.join(str(t) for t in ev['tops'])}) "
              f"clean={row['clean_max_corr']:.3f} dead={row['n_dead']} {dt:.0f}s")

        if ev["val_separation"] > best_sep:                    # 체크포인트: 분리 품질
            best_sep = ev["val_separation"]
            torch.save({"model": model.state_dict(), "epoch": epoch, "cfg": cfg,
                        "n_encoders": n_encoders, "seed": seed,
                        "val_separation": best_sep, "corr": ev["corr"].tolist()},
                       os.path.join(ckpt_dir, f"{run}.pt"))
        if ev["val_recon"] < best_recon - 1e-6:                # 조기 종료: 재구성 손실
            best_recon, bad = ev["val_recon"], 0
        else:
            bad += 1
            if bad >= tr_cfg["early_stop_patience"]:
                print(f"  조기 종료 (val_recon {tr_cfg['early_stop_patience']}에폭 정체)")
                break
        if epoch % plot_every == 0 or epoch == 1:
            _plot_components(model, val_set, cfg, device,
                             os.path.join(log_dir, "plots", f"ep{epoch:03d}.png"))

    print(f"[{run}] 완료. 최고 분리 품질 {best_sep:.4f} → checkpoints/{run}.pt")
    return pd.DataFrame(history)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--k", type=int, required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--max-epoch", type=int, default=None)
    p.add_argument("--plot-every", type=int, default=5)
    p.add_argument("--lambda-mixing", type=float, default=None,
                   help="λ 스윕 전용. 지정하면 실행명에 기록된다.")
    a = p.parse_args()
    train(load_cfg(a.config), a.k, a.seed, max_epoch=a.max_epoch,
          plot_every=a.plot_every, lambda_mixing=a.lambda_mixing)
