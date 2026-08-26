"""02 — 모델·학습 (RESEARCH_DESIGN.md §5–7).

    python 02_model.py --k 8 --seed 42

구조는 `src/model/meae.py`, 손실은 `src/model/losses.py`. 이 파일은 학습 루프와
두 비용 함수의 운용(선택 기준·후보 보관·민감도)을 담는다.

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

산출물: `results/02_model/<run>/` — `<run>.pt` · `history.csv` · `selection.json` ·
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

from scipy.signal import welch

from src.core import load_ckpt, reconstruct
from src.data.build import load_cfg
from src.data.dataset import load
from src.model import losses, meae
from src.spectral import band_keep, crossover, keep_curve, psd_pair, slope
from src.viz import plt

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

    run_dir = os.path.join("results", "02_model", run)
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


# ================================================================
# 재구성 충실도 진단 — 관문이 아니라 서술 지표다. 합격/불합격 수치를 두지 않는다.
#   보존율 P(x_hat)/P(x_noisy) (전대역·대역별) · 꺾임 지점 · log-log PSD 기울기 ·
#   R-피크 진폭비.  **디노이징 지수와 잔차 상관은 산출하지 않는다.**
# ================================================================
RPEAK_WIN = 30          # R-피크 좌우 샘플 (약 ±83 ms)


def band_power(x, fs, lo=None, hi=None):
    f, P = welch(x, fs=fs, nperseg=1024, axis=-1)
    if lo is None:
        return P.sum(-1)
    m = (f >= lo) & (f <= hi)
    return P[..., m].sum(-1)


def rpeak_amp_ratio(inp, rec, peaks_list, win=RPEAK_WIN):
    """R-피크 주변 첨두간 진폭의 재구성/입력 비 (분절별 중앙값)."""
    out = []
    for i, peaks in enumerate(peaks_list):
        r = []
        for p in peaks:
            a, b = max(0, p - win), min(inp.shape[-1], p + win)
            di = inp[i, a:b].max() - inp[i, a:b].min()
            dr = rec[i, a:b].max() - rec[i, a:b].min()
            if di > 0:
                r.append(dr / di)
        out.append(np.median(r) if r else np.nan)
    return np.array(out)


def fig_zoom(inp, rec, out, fs, i=0, t0=2.0, dur=1.5):
    a, b = int(t0 * fs), int((t0 + dur) * fs)
    t = np.arange(a, b) / fs
    fig, ax = plt.subplots(figsize=(11, 3.4))
    ax.plot(t, inp[i][a:b], lw=.9, color="#000", label="입력 x_noisy")
    ax.plot(t, rec[i][a:b], lw=.9, color="#d62728", alpha=.85, label="재구성 x_hat")
    ax.legend(fontsize=8)
    ax.set_ylabel("mV", fontsize=8)
    ax.set_xlabel("시간 (초)")
    ax.grid(alpha=.25, lw=.4)
    ax.set_title("확대 — 잔떨림(고주파 변동) 재현 여부", fontsize=10, loc="left")
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def fig_spectrum(f, Pi, Pr, out):
    fig, ax = plt.subplots(figsize=(9, 4.6))
    ax.loglog(f[1:], np.median(Pi, 0)[1:], lw=1.2, color="#000", label="입력 x_noisy")
    ax.loglog(f[1:], np.median(Pr, 0)[1:], lw=1.2, color="#d62728", label="재구성 x_hat")
    ax.set_xlabel("주파수 (Hz)")
    ax.set_ylabel("PSD (분절 중앙값)")
    ax.grid(alpha=.3, lw=.4, which="both")
    ax.legend(fontsize=8)
    ax.set_title("PSD — 입력 대비 재구성", fontsize=10, loc="left")
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def fig_keep(f, curve, out, fmax=90):
    m = f <= fmax
    fig, ax = plt.subplots(figsize=(9, 4.6))
    ax.plot(f[m], curve[m], lw=1.4, color="#1f77b4")
    ax.axhline(1.0, color="k", lw=.8, ls="--", alpha=.6)
    ax.axhline(0.7, color="#d62728", lw=.9, ls=":")
    ax.axvspan(59, 61, color="#999", alpha=.25)
    ax.set_xlim(0, fmax)
    ax.set_ylim(0, 1.35)
    ax.set_xlabel("주파수 (Hz)")
    ax.set_ylabel("보존율  P(x_hat) / P(x_noisy)  — 분절 중앙값")
    ax.grid(alpha=.3, lw=.4)
    ax.set_title("주파수별 보존율 — 어디서부터 무너지는가 (회색 = 59–61 Hz 제외 구간)",
                 fontsize=10, loc="left")
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def diagnose(config="configs/default.yaml", run="K8_seed42", split="val", n=900,
             outdir=None):
    cfg = load_cfg(config)
    fs = cfg["data"]["fs"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    outdir = outdir or os.path.join("results", "02_model", run, "fidelity")
    figdir = os.path.join(outdir, "figures")
    os.makedirs(figdir, exist_ok=True)

    model, ck = load_ckpt(cfg, run)
    model = model.to(device)
    ds = load(cfg, split)
    idx = np.arange(min(n, len(ds)))

    inp, rec = reconstruct(model, ds, device, idx)
    f, Pi, Pr = psd_pair(model, ds, device, fs, len(idx))
    curve = keep_curve(f, Pi, Pr)
    row = {
        "run": run, "epoch": ck["epoch"], "split": split, "n_seg": len(idx),
        "보존율_전대역": float(np.median(band_power(rec, fs) / band_power(inp, fs))),
        **band_keep(f, Pi, Pr),
        "꺾임_0.7Hz": crossover(f, curve, 0.7),
        "꺾임_0.5Hz": crossover(f, curve, 0.5),
        "기울기_입력": slope(f, Pi), "기울기_재구성": slope(f, Pr),
        "기울기차": slope(f, Pr) - slope(f, Pi),
        "R피크_진폭비": float(np.nanmedian(
            rpeak_amp_ratio(inp, rec, [ds.rpeaks[k] for k in idx]))),
    }
    out = pd.DataFrame([row])
    out.to_csv(f"{outdir}/fidelity.csv", index=False, encoding="utf-8-sig")
    np.savez(f"{outdir}/keep_curve.npz", f=f, curve=curve)
    with open(f"{outdir}/fidelity_note.txt", "w", encoding="utf-8") as fn:
        fn.write(
            "재구성 충실도 진단 — 관문이 아니라 서술 지표다. 합격/불합격 수치를 두지 않는다.\n"
            "보존율 = P(x_hat)/P(x_noisy) 의 분절 중앙값. 1.0 = 완전 보존.\n"
            "* 붙은 대역(40-60, 60-90 Hz)은 59-61 Hz 노치를 제외하고 산출했다\n"
            "  (전원 간섭. MIT-BIH 원본 유래이며 주입 잡음과 무관하다).\n"
            "기울기는 로그-로그 PSD 를 10-60 Hz 에서 회귀한 분절 중앙값.\n"
            "R-피크 진폭비 = R-피크 +-30샘플 첨두간 진폭의 재구성/입력 비.\n"
            "디노이징 지수와 잔차 상관은 산출하지 않는다.\n")

    fig_zoom(inp, rec, f"{figdir}/zoom.png", fs)
    fig_spectrum(f, Pi, Pr, f"{figdir}/spectrum.png")
    fig_keep(f, curve, f"{figdir}/keep_curve.png")

    pd.set_option("display.width", 220)
    print(f"=== 재구성 충실도 — 서술 지표 ({run} · {split} {len(idx)}분절) ===")
    print(out.round(4).to_string(index=False))
    print(f"\n산출물 → {outdir}/")
    return out


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--k", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--max-epoch", type=int, default=None)
    p.add_argument("--plot-every", type=int, default=25)
    p.add_argument("--diagnose", action="store_true",
                   help="학습 대신 재구성 충실도 진단만 수행")
    p.add_argument("--run", default="K8_seed42")
    p.add_argument("--split", default="val")
    p.add_argument("--n", type=int, default=900)
    p.add_argument("--hidden", type=int, default=None)
    a = p.parse_args()
    if a.diagnose:
        diagnose(a.config, a.run, a.split, a.n)
    else:
        if a.k is None or a.seed is None:
            p.error("학습에는 --k 와 --seed 가 필요하다 (진단만 하려면 --diagnose)")
        train(load_cfg(a.config), a.k, a.seed, max_epoch=a.max_epoch,
              plot_every=a.plot_every, hidden=a.hidden)
