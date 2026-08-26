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
    x̂_k  = D(0,…,z_k,…,0)                  k번째 인코딩만 남긴 재구성, 중앙 3600 크롭
    ρ_k(r) = mean_{s∈V} |ρ(x̂_k^(s), r^(s))|   V = 검증 전체 900분절
    S(t)   = mean over 참조 4종 of [ max_k ρ_k(r) ]

    1단계  C  = { t : L_recon^val(t) ≤ ratio × min_τ L_recon^val(τ) },  ratio = 1.5
    2단계  t* = argmax_{t∈C} S(t)

    S는 심장(clean) 하나가 아니라 **참조 4종 전부**를 반영한다 — 그 에폭의 모델이
    네 신호를 전반적으로 얼마나 잡아내는가다. 집계 구간·상관 정의는 03 보고 표와 같다.

    1단계는 학습 중 `eval_every`(2)에폭 간격으로 기록하고, 2단계는 **학습 종료 후**
    후보 전체에 대해 `--epoch-metrics` 로 일괄 산출한다.
    x_clean·bw·ma·em은 이 선택에만 쓰이고 가중치 갱신에는 관여하지 않는다 (§0 원칙 4).

산출물: `results/02_model/<run>/`
    학습        `history.csv` · `stage1.json` · `console.log` · `pool/`(후보 가중치) · `plots/`
    --epoch-metrics   `epoch_metrics/`(후보별 평가 표) · `selection.json` · `<run>.pt`
    --diagnose        `fidelity/`(재구성 충실도 진단)
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

from src.core import (aggregate, component_bank, enc_names, load_ckpt,
                      mad_matrix, pearson, reconstruct, rmse_norm_matrix)
from src.data.build import load_cfg
from src.data.dataset import REF_KEYS, load
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


def _plot_components(model, val, cfg, device, path, idx=0, epoch=None):
    """성분 파형 적층 그림 — 입력·재구성·성분 K개·참조 4종.

    학습 중 **모든 에폭**에 대해 저장한다(plot_every=1). 성분이 에폭에 따라 어떻게
    자리를 잡아 가는지가 체크포인트 선택의 근거를 눈으로 보게 해 준다."""
    import matplotlib
    matplotlib.use("Agg")
    from src.viz import plt

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
    if epoch is not None:
        m = val.meta[idx]
        fig.suptitle(f"에폭 {epoch}  ·  val 분절 {m['record_id']}_{m['seg_idx']:04d} "
                     f"(주입 SNR bw {m['snr_bw']:.1f} / ma {m['snr_ma']:.1f} / "
                     f"em {m['snr_em']:.1f} dB)", fontsize=11)
    fig.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def select(history, ratio):
    """비용 함수 ② **1단계** — 재구성 손실 관문으로 후보 집합 C를 만든다.

    2단계(S 최대화)는 학습 중에 할 수 없다. S가 검증 전체에서 성분을 다 뽑아야 나오는
    값이라 매 에폭 계산하면 학습이 크게 느려지기 때문이다. 후보 가중치를 보관해 두고
    학습이 끝난 뒤 `epoch_metrics()` 가 2단계를 적용한다.
    """
    ev = [h for h in history if h.get("val_recon") is not None]
    if not ev:
        return None, []
    lo = min(h["val_recon"] for h in ev)
    C = [h for h in ev if h["val_recon"] <= ratio * lo]
    return min(C, key=lambda h: h["val_recon"]), C


def train(cfg, n_encoders: int, seed: int, tag: str = "", plot_every: int = 1,
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
    say(f"  1단계 C = {{ L_recon^val <= {ratio} x min }},  {every}에폭 간격")
    say("  2단계(S 최대화)는 학습 종료 후 --epoch-metrics 로 적용한다")

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
        if plot_every and epoch % plot_every == 0:
            _plot_components(model, val_set, cfg, device,
                             os.path.join(run_dir, "plots", f"ep{epoch:04d}.png"),
                             epoch=epoch)

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
    with open(os.path.join(run_dir, "stage1.json"), "w", encoding="utf-8") as f:
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


# ================================================================
# 후보 에폭별 성분↔참조 평가 표 — 선정 근거를 눈으로 볼 수 있게 저장한다.
#   pool/ 에 보관한 후보 구간(배율 1.5) 체크포인트를 하나씩 불러
#   03과 **같은 조건**(val 전체, 같은 지표 정의)으로 8×4 표를 만든다.
#   지표 하나 = 표 하나. 한 에폭이 인코더 8행을 차지하고 열은 참조 4종이다.
#
#   이 표는 서술·투명성용이며 **선정 규칙을 바꾸지 않는다.**
#   표를 보고 에폭을 손으로 고르면 기준 이동이 된다.
# ================================================================
METRICS = ("corr", "rmse_norm", "mad")


def _pool_epochs(run_dir):
    """pool/ 에 남아 있는 에폭 번호 (오름차순)."""
    d = os.path.join(run_dir, "pool")
    if not os.path.isdir(d):
        raise SystemExit(f"[02] {d} 없음 — 먼저 학습해야 한다")
    return sorted(int(f[2:6]) for f in os.listdir(d) if f.endswith(".pt"))


def epoch_metrics(config="configs/default.yaml", run="K8_seed42", split="val",
                  n=None, outdir=None):
    cfg = load_cfg(config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    run_dir = os.path.join("results", "02_model", run)
    outdir = outdir or os.path.join(run_dir, "epoch_metrics")
    figdir = os.path.join(outdir, "figures")
    os.makedirs(figdir, exist_ok=True)

    hist = pd.read_csv(os.path.join(run_dir, "history.csv"), encoding="utf-8-sig")
    p1 = os.path.join(run_dir, "stage1.json")
    if not os.path.exists(p1):                       # 구 파일명 호환
        p1 = os.path.join(run_dir, "selection.json")
    with open(p1, encoding="utf-8") as f:
        sel = json.load(f)
    ds = load(cfg, split)
    idx = np.arange(len(ds) if n is None else min(n, len(ds)))
    epochs = _pool_epochs(run_dir)
    refs_c = list(REF_KEYS)
    print(f"[02] 후보 에폭 {len(epochs)}개 × {split} {len(idx)}분절 — 지표 3종 산출")

    rows_of = {m: [] for m in METRICS}
    summary = []
    for e in epochs:
        model, _ = load_ckpt(cfg, os.path.join(run_dir, "pool", f"ep{e:04d}.pt"))
        model = model.to(device)
        comps, refs = component_bank(model, ds, device, idx)
        K = comps.shape[1]
        ix = enc_names(K)
        rbar, rsd, _ = aggregate(pearson(comps, refs))
        # 비용 함수 ② 의 S — 참조 4종 각각에서 최고 인코더의 평균 상관을 구하고, 그 넷을 평균한다.
        # 심장 하나가 아니라 **모델 전체의 분리 상태**를 반영한다.
        per_ref = rbar.max(0)                                  # 참조별 max_k [mean_s |ρ_k|]
        S_new = float(per_ref.mean())
        vals = {"corr": (rbar, rsd)}
        for name, arr in (("rmse_norm", rmse_norm_matrix(comps, refs)),
                          ("mad", mad_matrix(comps, refs))):
            vals[name] = (arr.mean(0), arr.std(0, ddof=1))
        for m, (mu, sd) in vals.items():
            for k in range(K):
                # 평균과 SD 를 한 행에 나란히 둔다 — 값과 산포는 같이 봐야 한다
                cell = {}
                for j, c in enumerate(refs_c):
                    cell[c] = mu[k, j]
                    cell[f"{c}_sd"] = sd[k, j]
                rows_of[m].append({"에폭": e, "인코더": ix[k], **cell})

        h = hist[hist.epoch == e]
        row = {"에폭": e,
               "val_recon": float(h.val_recon.iloc[0]) if len(h) else np.nan,
               "S": S_new,
               **{f"S기여_{c}": float(per_ref[j]) for j, c in enumerate(refs_c)}}
        # 참조별로 어느 인코더가 1위인지 — 지표마다 방향이 다르다
        for m, (mu, _sd) in vals.items():
            best = mu.argmax(0) if m == "corr" else mu.argmin(0)
            for j, c in enumerate(refs_c):
                row[f"{m}_{c}_최고인코더"] = ix[int(best[j])]
                row[f"{m}_{c}_값"] = float(mu[int(best[j]), j])
        summary.append(row)
        print(f"  ep{e:4d}  S {S_new:.4f}   " +
              "  ".join(f"{c} {per_ref[j]:.3f}" for j, c in enumerate(refs_c)), flush=True)

    for m in METRICS:
        pd.DataFrame(rows_of[m]).round(4).to_csv(
            f"{outdir}/{m}_by_epoch.csv", index=False, encoding="utf-8-sig")
    summ = pd.DataFrame(summary).round(4)

    # ---- 비용 함수 ② 2단계. 1단계 관문(배율 1.5)은 pool 구성으로 이미 반영돼 있다.
    win = summ.loc[summ["S"].idxmax()]
    summ["선택여부"] = ["선택" if e == int(win["에폭"]) else "" for e in summ["에폭"]]
    summ.to_csv(f"{outdir}/epoch_summary.csv", index=False, encoding="utf-8-sig")

    scols = ["에폭", "val_recon", "S"] + [f"S기여_{c}" for c in refs_c]         + [f"corr_{c}_최고인코더" for c in refs_c]
    top5 = summ.nlargest(5, "S")[scols]
    top5.to_csv(f"{outdir}/top5_by_S.csv", index=False, encoding="utf-8-sig")

    # 선택된 체크포인트를 본 가중치로 확정한다 — 03·04·05 가 이 파일을 읽는다
    best_pt = os.path.join(run_dir, "pool", f"ep{int(win['에폭']):04d}.pt")
    shutil.copyfile(best_pt, os.path.join(run_dir, f"{run}.pt"))

    chosen = {
        "run": run, "split": split, "n_seg": int(len(idx)),
        "rule": ("S(t) = mean over 참조 4종 of [ max_k ( mean_s |rho_k(r)| ) ], "
                 "검증 전체"),
        "stage1": f"val_recon <= {sel['ratio']} x min  (pool 구성에 반영돼 있다)",
        "n_candidates": int(len(summ)),
        "candidate_range": [int(summ["에폭"].min()), int(summ["에폭"].max())],
        "selected_epoch": int(win["에폭"]),
        "S": float(win["S"]),
        "S기여": {c: float(win[f"S기여_{c}"]) for c in refs_c},
        "val_recon": float(win["val_recon"]),
        "checkpoint": f"{run}.pt  (pool/ep{int(win['에폭']):04d}.pt 복사)",
        "top5": top5.to_dict("records")}
    with open(os.path.join(run_dir, "selection.json"), "w", encoding="utf-8") as f:
        json.dump(chosen, f, ensure_ascii=False, indent=2)

    fig_epoch_metrics(summ, hist, sel, f"{figdir}/epoch_metrics.png",
                      new_best=int(win["에폭"]),
                      corr_tab=pd.DataFrame(rows_of["corr"]))

    note = """후보 에폭별 성분<->참조 평가 표와 체크포인트 선택 — 비용 함수 ② 2단계.

1단계(재구성 손실 관문)는 학습 중에 적용해 후보 가중치를 pool/ 에 남긴다.
2단계는 여기서 한다:

    rho_k(r) = mean_s |rho( x_hat_k, r )|          검증 {nseg}분절, 성분·참조를 분절 내 표준화
    S(t)     = mean over 참조 4종 of [ max_k rho_k(r) ]
    t*       = argmax_(t in C) S(t)

S 는 심장(clean) 하나가 아니라 참조 4종 전부를 반영한다 — 그 에폭의 모델이 네 신호를
전반적으로 얼마나 잡아내는가다. 집계 구간과 상관 정의는 03 보고 표와 같다.
선택된 체크포인트는 {run}.pt 로 복사되며 03·04·05 가 그 파일을 읽는다.

대상   pool/ 의 배율 {ratio} 후보 {ncand}개 (에폭 {e0}~{e1}, {every}에폭 간격)
구간   {split} {nseg}분절, 패딩 제외 중앙 3600 표본

표 — 지표 하나 = 파일 하나. 행 (에폭, 인코더) x 열 (참조 4종의 평균과 SD, ddof=1)
  corr_by_epoch.csv        분절 내 Pearson 절댓값 -> 분절 간 평균 (높을수록 유사)
  rmse_norm_by_epoch.csv   표준화 신호 차이의 RMS -> 분절 간 평균 (낮을수록 유사)
  mad_by_epoch.csv         그 차이의 최댓값       -> 분절 간 평균 (낮을수록 유사)
  epoch_summary.csv        에폭당 1행 — S, 참조별 S 기여, 참조별 최고 인코더와 값
  top5_by_S.csv            S 상위 5개
  ../selection.json        선택 결과와 규칙 (가중치 옆)
  figures/epoch_metrics.png
""".format(nseg=len(idx), run=run, ratio=sel["ratio"], ncand=len(epochs),
           e0=epochs[0], e1=epochs[-1], every=sel["eval_every"], split=split)
    with open(f"{outdir}/epoch_metrics_note.txt", "w", encoding="utf-8") as f:
        f.write(note)

    pd.set_option("display.width", 260)
    print(f"\n산출물 → {outdir}/")
    return summ


def fig_epoch_metrics(summ, hist, sel, out, new_best, corr_tab=None):
    """에폭축 3행 그림 — 관문 / 참조별 최고 |r| / 인코더별 clean |r|."""
    ratio = sel["ratio"]
    best = new_best
    lo = sel["min_val_recon"]
    cols = {"x_clean": "#2ca02c", "bw": "#d62728", "ma": "#ff7f0e", "em": "#9467bd"}
    fig, ax = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    a = ax[0]
    a.plot(hist.epoch, hist.val_recon, lw=1, color="#1f77b4", label="val_recon")
    a.axhline(lo, color="#888", ls=":", lw=1, label=f"최소 {lo:.5f}")
    a.axhline(ratio * lo, color="#d62728", ls="--", lw=1.2,
              label=f"{ratio}× 관문 {ratio * lo:.5f}")
    a.axvspan(summ["에폭"].min(), summ["에폭"].max(), color="#2ca02c", alpha=.08,
              label=f"후보 구간 ({len(summ)}개)")
    a.set_ylabel("L_recon^val")
    a.set_title("① 1단계 관문 — 재구성 손실", fontsize=10, loc="left")
    a.legend(fontsize=7.5, ncol=2)

    a = ax[1]
    for c in REF_KEYS:
        a.plot(summ["에폭"], summ[f"corr_{c}_값"], "o-", ms=3, lw=1.1,
               color=cols[c], label=f"{c} 최고 |r|")
    a.set_ylabel("|r|")
    a.set_title("② 참조별 최고 인코더의 |r| — 검증 전체", fontsize=10, loc="left")
    a.legend(fontsize=7.5, ncol=3)

    a = ax[2]
    if corr_tab is not None:
        for enc, g in corr_tab.groupby("인코더", sort=False):
            a.plot(g["에폭"], g["x_clean"], lw=1, label=enc)
    a.set_ylabel("clean |r| (평균)")
    a.set_xlabel("에폭")
    a.set_title("③ 인코더별 clean 상관 — 어느 인코더가 언제 심장 성분을 잡는가",
                fontsize=10, loc="left")
    a.legend(fontsize=7, ncol=4)

    for a in ax:
        a.axvline(best, color="#d62728", lw=1.2, ls="--", alpha=.55)
        a.grid(alpha=.3, lw=.4)
        a.tick_params(labelsize=8)
    ax[0].annotate(f"선택 에폭 {best}", (best, ax[0].get_ylim()[1]),
                   xytext=(4, -12), textcoords="offset points",
                   fontsize=9, color="#d62728", alpha=.8)
    fig.suptitle(f"후보 에폭별 선정 근거 — {sel['run']} (배율 {ratio}, 후보 {len(summ)}개)\n"
                 "S(t) = 참조 4종 각각의 max_k [ mean_s |ρ_k(r)| ] 를 평균,  검증 전체",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--k", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--max-epoch", type=int, default=None)
    p.add_argument("--plot-every", type=int, default=1,
                   help="성분 파형 그림 저장 간격(에폭). 1 = 모든 에폭")
    p.add_argument("--diagnose", action="store_true",
                   help="학습 대신 재구성 충실도 진단만 수행")
    p.add_argument("--epoch-metrics", dest="epoch_metrics", action="store_true",
                   help="후보 에폭별 성분<->참조 평가 표를 산출한다 (학습 없음)")
    p.add_argument("--run", default="K8_seed42")
    p.add_argument("--split", default="val")
    p.add_argument("--n", type=int, default=900)
    p.add_argument("--hidden", type=int, default=None)
    a = p.parse_args()
    if a.epoch_metrics:
        epoch_metrics(a.config, a.run, a.split, a.n)
    elif a.diagnose:
        diagnose(a.config, a.run, a.split, a.n)
    else:
        if a.k is None or a.seed is None:
            p.error("학습에는 --k 와 --seed 가 필요하다 (진단만 하려면 --diagnose)")
        train(load_cfg(a.config), a.k, a.seed, max_epoch=a.max_epoch,
              plot_every=a.plot_every, hidden=a.hidden)
