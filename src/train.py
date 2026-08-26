"""S3 — 학습 (RESEARCH_DESIGN.md §6).

자기지도: 입력도 목표도 x_noisy 하나다. clean·bw·ma·em은 손실에 들어가지 않는다 (§0 원칙 3).

────────────────────────────────────────────────────────────────────────
비용 함수 ① — 학습 손실
────────────────────────────────────────────────────────────────────────
    L = MSE(x̂, x_noisy) + λ_m·L_mix + λ_o·‖D(0)‖² + λ_z·Σ_k ‖z_k‖²/h

    λ_m = 1e-2   sparse mixing (디코더 비대각 가중치 L1, 선행 구현 Alternative)
    λ_o = 1e-2   zero reconstruction (전영 인코딩 → 출력 0)
    λ_z = 1e-3   인코딩 L2

────────────────────────────────────────────────────────────────────────
비용 함수 ② — 체크포인트 선택 기준
────────────────────────────────────────────────────────────────────────
    x̂_k = D(0,…,z_k,…,0)                      k번째 인코딩만 남긴 재구성, 중앙 3600 크롭
    ρ_k(t) = median_{s∈V} |ρ(x̂_k, x_clean)|    |V| = 300 고정
    S(t)   = max_k ρ_k(t)

    1단계  C  = { t : L_recon^val(t) ≤ ratio × min_τ L_recon^val(τ) },  ratio = 1.5
    2단계  t* = argmax_{t∈C} S(t)

    2에폭 간격 산출 · 학습 종료 후 전체 이력 일괄 판정 · 후보 구간 가중치 보관.
    x_clean은 이 선택에만 쓰이고 가중치 갱신에는 관여하지 않는다 (§0 원칙 4).

산출물: `results/01_train/<run>/` — `<run>.pt` · `history.csv` · `selection.json` ·
        `console.log` · `pool/`(후보 구간 가중치) · `plots/`
"""
import argparse
import json
import os
import random
import shutil
import time

import numpy as np
import pandas as pd
import torch

from .data.build import load_cfg
from .data.dataset import load
from .model import losses, meae

SENSITIVITY_RATIOS = (1.2, 1.5, 2.0)      # 민감도 보고용 (원고 부록)


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
    d = a.norm(dim=-1) * b.norm(dim=-1)
    return torch.where(d > 0, (a * b).sum(-1) / d.clamp_min(1e-12), torch.zeros_like(d))


@torch.no_grad()
def evaluate(model, val, device, batch: int, n_v: int, with_rho: bool):
    """L_recon^val 과 (필요 시) ρ_k · S. V는 val의 앞 n_v 분절로 고정한다."""
    model.eval()
    pad, K = model.pad_each, model.n_encoders
    idx_all = np.arange(min(n_v, len(val)))
    recon_sum, seen = 0.0, 0
    rho = [[] for _ in range(K)] if with_rho else None

    for s in range(0, len(idx_all), batch):
        idx = idx_all[s:s + batch]
        x = meae.pad(val.tensor(idx).to(device), pad)
        y, zs = model(x)
        recon_sum += torch.nn.functional.mse_loss(y, x, reduction="sum").item()
        seen += x.numel()
        if with_rho:
            clean = val.ref_tensor("x_clean", idx).to(device).squeeze(1)
            for k in range(K):
                comp = meae.crop(model.decode(model._mask(zs, [k])), pad).squeeze(1)
                rho[k].append(_corr(comp, clean).abs().cpu())

    out = {"val_recon": recon_sum / seen}
    if with_rho:
        r = np.array([float(torch.cat(v).median()) for v in rho])
        out.update(rho=r, S=float(r.max()), argmax_k=int(r.argmax()))
    model.train()
    return out


def _plot_components(model, val, cfg, device, path, idx=0):
    """성분 파형 적층 그림 (학습 중 확인용)."""
    import matplotlib
    matplotlib.use("Agg")
    from .viz import plt

    pad, K = model.pad_each, model.n_encoders
    fs = cfg["data"]["fs"]
    model.eval()
    with torch.no_grad():
        x = meae.pad(val.tensor(np.array([idx])).to(device), pad)
        comps = [meae.crop(model.component(x, k), pad).squeeze().cpu().numpy()
                 for k in range(K)]
        recon = meae.crop(model(x)[0], pad).squeeze().cpu().numpy()
    model.train()

    rows = ([("입력 x_noisy", val.x_noisy[idx], "#000"), ("재구성 x_hat", recon, "#d62728")]
            + [(f"성분 {k+1}", c, "#1f77b4") for k, c in enumerate(comps)]
            + [(f"참조 {r}", val.refs[r][idx], "#ff7f0e") for r in ("bw", "ma", "em")]
            + [("참조 clean", val.refs["x_clean"][idx], "#2ca02c")])
    t = np.arange(len(rows[0][1])) / fs
    fig, ax = plt.subplots(len(rows), 1, figsize=(11, 0.95 * len(rows)), sharex=True)
    for a, (label, v, c) in zip(ax, rows):
        a.plot(t, v, lw=0.6, color=c)
        a.set_title(label, fontsize=8, loc="left")
        a.grid(alpha=.25, lw=.4)
        a.tick_params(labelsize=7)
    ax[-1].set_xlabel("시간 (초)")
    fig.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def select(history, ratio):
    """비용 함수 ②. C를 만들고 그 안에서 S 최대 에폭을 고른다."""
    ev = [h for h in history if h.get("S") is not None]
    if not ev:
        return None, []
    lo = min(h["val_recon"] for h in ev)
    C = [h for h in ev if h["val_recon"] <= ratio * lo]
    return max(C, key=lambda h: h["S"]), C


def train(cfg, n_encoders: int, seed: int, tag: str = "", plot_every: int = 25,
          max_epoch: int = None, hidden: int = None):
    if hidden is not None:
        cfg = json.loads(json.dumps(cfg))
        cfg["model"]["hidden"] = hidden

    tr = cfg["train"]
    ck = tr["checkpoint"]
    max_epoch = max_epoch or tr["max_epoch"]
    batch, n_v, every, ratio = tr["batch"], ck["val_n"], ck["eval_every"], ck["recon_ratio"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    run = tag or f"K{n_encoders}_seed{seed}"

    set_seed(seed)
    train_set, val_set = load(cfg, "train"), load(cfg, "val")
    model = meae.build(cfg, n_encoders).to(device)
    crit = losses.build(cfg, n_encoders).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=tr["lr"], weight_decay=tr["weight_decay"])
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=tr["lr_step_size"], gamma=0.1)

    run_dir = os.path.join("results", "01_train", run)
    pool_dir = os.path.join(run_dir, "pool")
    os.makedirs(pool_dir, exist_ok=True)
    console = open(os.path.join(run_dir, "console.log"), "w", encoding="utf-8")

    def say(msg):
        print(msg, flush=True)
        console.write(msg + "\n")
        console.flush()

    lo = cfg["loss"]
    say(f"[{run}] device={device} K={n_encoders} seed={seed} hidden={cfg['model']['hidden']} "
        f"train={len(train_set)} val={len(val_set)} V={n_v}")
    say(f"  L = MSE + {lo['lambda_mixing']:g}*L_mix + {lo['lambda_zero_recon']:g}*|D(0)|^2 "
        f"+ {lo['lambda_z_l2']:g}*sum|z|^2/h")
    say(f"  C = {{ L_recon^val <= {ratio} x min }},  t* = argmax S,  {every}에폭 간격")

    rng = np.random.default_rng(seed)
    history, best_recon, bad = [], np.inf, 0
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
            torch.nn.utils.clip_grad_norm_(model.parameters(), tr["gradient_clip_val"])
            opt.step()
            for k, v in d.items():
                sums[k] = sums.get(k, 0.0) + float(v.detach())
            nb += 1
        sched.step()

        judge = (epoch % every == 0) or (epoch == 1)
        ev = evaluate(model, val_set, device, batch, n_v, with_rho=judge)
        row = {"epoch": epoch, **{f"train_{k}": v / nb for k, v in sums.items()},
               "val_recon": ev["val_recon"], "S": ev.get("S"),
               "argmax_k": ev.get("argmax_k"), "lr": opt.param_groups[0]["lr"],
               "sec": time.time() - t0}
        if judge:
            for k, v in enumerate(ev["rho"]):
                row[f"rho_e{k}"] = float(v)
            torch.save({"model": model.state_dict(), "epoch": epoch, "cfg": cfg,
                        "n_encoders": n_encoders, "seed": seed,
                        "val_recon": ev["val_recon"], "S": ev["S"]},
                       os.path.join(pool_dir, f"ep{epoch:04d}.pt"))
        history.append(row)
        pd.DataFrame(history).to_csv(os.path.join(run_dir, "history.csv"),
                                     index=False, encoding="utf-8-sig")
        say(f"  ep{epoch:3d} L={row['train_total']:.5f} rec={row['train_recon']:.5f} "
            f"mix={row['train_mixing']:.3f} | val_rec={ev['val_recon']:.5f}"
            + (f" S={ev['S']:.3f}(enc{ev['argmax_k']+1})" if judge else "")
            + f" {row['sec']:.0f}s")

        if ev["val_recon"] < best_recon - 1e-6:
            best_recon, bad = ev["val_recon"], 0
        else:
            bad += 1
            if bad >= tr["early_stop_patience"]:
                say(f"  조기 종료 (val_recon {tr['early_stop_patience']}에폭 정체)")
                break
        if epoch % plot_every == 0:
            _plot_components(model, val_set, cfg, device,
                             os.path.join(run_dir, "plots", f"ep{epoch:04d}.png"))

    # ---- 학습 종료 후 일괄 판정
    best, C = select(history, ratio)
    shutil.copyfile(os.path.join(pool_dir, f"ep{best['epoch']:04d}.pt"),
                    os.path.join(run_dir, f"{run}.pt"))
    keep = {h["epoch"] for h in C}
    for f in os.listdir(pool_dir):
        if int(f[2:6]) not in keep:
            os.remove(os.path.join(pool_dir, f))

    sens = {}
    for r in SENSITIVITY_RATIOS:
        b, c = select(history, r)
        sens[str(r)] = {"n_candidates": len(c), "selected_epoch": b["epoch"],
                        "S": b["S"], "val_recon": b["val_recon"],
                        "epoch_range": [min(h["epoch"] for h in c),
                                        max(h["epoch"] for h in c)]}
    ev_all = [h for h in history if h.get("S") is not None]
    info = {"run": run, "K": n_encoders, "seed": seed,
            "hidden": cfg["model"]["hidden"], "ratio": ratio,
            "eval_every": every, "val_n": n_v,
            "n_epochs": len(history), "n_evaluated": len(ev_all),
            "min_val_recon": min(h["val_recon"] for h in ev_all),
            "n_candidates": len(C),
            "candidate_range": [min(h["epoch"] for h in C), max(h["epoch"] for h in C)],
            "selected_epoch": best["epoch"], "S": best["S"],
            "argmax_k": best["argmax_k"], "val_recon": best["val_recon"],
            "sensitivity": sens, "loss": cfg["loss"]}
    with open(os.path.join(run_dir, "selection.json"), "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)

    say(f"[{run}] 완료. {len(history)}에폭, 판정 {len(ev_all)}개")
    say(f"  min L_recon^val = {info['min_val_recon']:.5f}")
    say(f"  C (ratio {ratio}) = {len(C)}개, 에폭 {info['candidate_range'][0]}"
        f"~{info['candidate_range'][1]}")
    say(f"  t* = 에폭 {best['epoch']}  S = {best['S']:.4f}  "
        f"(enc{best['argmax_k']+1})  val_recon {best['val_recon']:.5f}")
    for r, v in sens.items():
        say(f"    민감도 ratio {r}: C {v['n_candidates']}개 "
            f"(에폭 {v['epoch_range'][0]}~{v['epoch_range'][1]}) -> "
            f"t* {v['selected_epoch']}  S {v['S']:.4f}")
    console.close()
    return pd.DataFrame(history), info


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--k", type=int, required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--max-epoch", type=int, default=None)
    p.add_argument("--plot-every", type=int, default=25)
    p.add_argument("--hidden", type=int, default=None)
    a = p.parse_args()
    train(load_cfg(a.config), a.k, a.seed, max_epoch=a.max_epoch,
          plot_every=a.plot_every, hidden=a.hidden)
