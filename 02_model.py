"""02 — 모델·학습 (RESEARCH_DESIGN.md §5–7).

    python 02_model.py --k 8 --seed 42

구조는 `src/model/meae.py`, 손실은 `src/model/losses.py`. 이 파일은 학습 루프와
두 비용 함수의 운용(선택 기준·후보 보관·민감도)을 담는다.

**지도학습.** 이 폴더만 참조 4종을 손실에 직접 넣는다 — §0 원칙 3(자기지도)을 의도적으로
벗어난 대조 실험이다. 인코더마다 참조를 하나씩 배정한다: enc1 x_clean · enc2 bw · enc3 ma ·
enc4 em (`loss.supervise`). 결과를 자기지도 결과와 같은 줄에 놓지 않는다.

────────────────────────────────────────────────────────────────────────
비용 함수 ① — 학습 손실
────────────────────────────────────────────────────────────────────────
    L     = (1/L)‖x̂ − x_noisy‖² + λ_sup · L_sup                        λ_sup = 1.0

    L_sup = (1/K) Σ_k [ (1/L)‖ŝ_k − r_k‖²
                      + γ₁·(1/(L−1))‖Δŝ_k  − Δr_k‖²                     γ₁ = gamma_sup
                      + γ₂·(1/(L−2))‖Δ²ŝ_k − Δ²r_k‖² ]                   γ₂ = gamma2_sup

    (Δx)[t] = x[t] − x[t−1] (길이 L−1),  (Δ²x)[t] = x[t] − 2x[t−1] + x[t−2] (길이 L−2).

    파형 일치만으로는 참조의 국소 고주파가 학습되지 않아 변화량 일치를 함께 건다.
    1차가 기울기라면 2차는 곡률이다. 세 항을 각자 길이로 나눠 γ 가 상대 비중 그대로가
    되게 한다.

    ŝ_k 는 `model.component(x, k)` — 다른 인코딩을 0으로 두고 디코드하는, 04가 쓰는 것과
    같은 마스킹 경로다. 크롭은 기존과 같이 중앙 3600.

    정칙화 3항(λ_m 스파스 혼합 · λ_o 전영 재구성 · λ_z 인코딩 L2)은 **끈다**(0.0).
    "어떻게 나눌지 모를 때 돕는" 항인데 여기서는 답을 직접 주므로 불필요하고, 목표와
    충돌할 수 있다.

────────────────────────────────────────────────────────────────────────
비용 함수 ② — 체크포인트 선택 기준
────────────────────────────────────────────────────────────────────────
    **선택 기준은 성분 정렬 손실 자체다** — 답을 직접 주므로 대리 지표가 필요 없다.

        C  = { t : L_sup^val(t) ≤ ratio × min_τ L_sup^val(τ) },  ratio = 1.5   (후보 보관용)
        t* = argmin_t L_sup^val(t)

    S(성분 <-> x_clean 상관의 최댓값)와 val_recon 은 계속 기록하지만 선택에는 쓰지 않는다.

산출물: `<out-root>/02_model/<그룹>/<run>/` — `--group` 으로 묶는다
        탐색은 전부 experiments(기본값)다. `results/` 는 **최종 선정 모델 하나** 전용이라
        확정한 뒤에만 `--out-root results` 로 돌린다.
    학습        `<run>.pt` · `history.csv` · `stage1.json` · `console.log` ·
                `pool/`(후보 가중치) · `plots/`(에폭 그림 2종: 적층 · 배정 쌍 겹침)
    자동 평가    `metrics/`(성분 정렬) · `restore/`(복원)  — 학습이 끝나면 항상 함께 낸다.
                 `--no-eval` 로만 끈다.
    --diagnose  `fidelity/`(재구성 충실도 진단 — 서술 지표)

`results/` 전체가 **최종 선정 모델 하나** 전용이다. 탐색 단계의 학습·지표·그림은 모두
`experiments/` 안에 있고, 각 런의 지표는 그 런 폴더 안(`metrics/`·`restore/`)에 있다.
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
from src.data.dataset import REF_KEYS, Segments, load
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
def evaluate(model, val, device, batch: int, n_v: int, with_rho: bool, sup_keys=None,
             gamma: float = 0.0, gamma2: float = 0.0, beta: float = 0.0,
             sigmas=None):
    """L_recon^val · **L_sup^val** 과 (필요 시) ρ_k · S. V는 val의 앞 n_v 분절로 고정한다.

    L_sup^val 이 선택 기준이다. 학습 손실과 같은 정의(성분별 MSE의 합)로 계산한다.
    """
    model.eval()
    pad, K = model.pad_each, model.n_encoders
    idx_all = np.arange(min(n_v, len(val)))
    recon_sum, seen = 0.0, 0
    sup_sum, sup_n = np.zeros(K), 0
    rho = [[] for _ in range(K)] if with_rho else None

    for s in range(0, len(idx_all), batch):
        idx = idx_all[s:s + batch]
        x = meae.pad(val.tensor(idx).to(device), pad)
        y, zs = model(x)
        recon_sum += torch.nn.functional.mse_loss(y, x, reduction="sum").item()
        seen += x.numel()
        if sup_keys:
            for k, key in enumerate(sup_keys):
                r = val.ref_tensor(key, idx).to(device).squeeze(1)
                comp = meae.crop(model.component(x, k), pad).squeeze(1)
                d = ((comp - r) ** 2).mean(dim=-1)                       # 파형
                if gamma:                                                # 1차 차분
                    e1 = (losses._diff(comp, 1) - losses._diff(r, 1)) ** 2
                    d = d + gamma * e1.mean(dim=-1)
                if gamma2:                                               # 2차 차분
                    e2 = (losses._diff(comp, 2) - losses._diff(r, 2)) ** 2
                    d = d + gamma2 * e2.mean(dim=-1)
                if beta:                                                 # |FFT| 크기
                    ef = (losses._fft_mag(comp) - losses._fft_mag(r)) ** 2
                    d = d + beta * ef.mean(dim=-1)
                if sigmas is not None:                                   # 소스별 정규화
                    d = d / (sigmas[k] ** 2)
                sup_sum[k] += float(d.sum())
            sup_n += len(idx)
        if with_rho:
            clean = val.ref_tensor("x_clean", idx).to(device).squeeze(1)
            for k in range(K):
                comp = meae.crop(model.decode(model._mask(zs, [k])), pad).squeeze(1)
                rho[k].append(_corr(comp, clean).abs().cpu())

    out = {"val_recon": recon_sum / seen}
    if sup_keys:
        per = sup_sum / max(sup_n, 1)
        out["val_sup"] = float(per.mean())          # 손실과 같은 정의 — k 에 대해 평균
        out["val_sup_per"] = per
    if with_rho:
        r = np.array([float(torch.cat(v).median()) for v in rho])
        out.update(rho=r, S=float(r.max()), argmax_k=int(r.argmax()))
    model.train()
    return out


def _plot_components(model, val, cfg, device, path, idx=0, epoch=None):
    """성분 파형 적층 그림 — 입력·재구성·성분 K개·참조 4종.

    학습 중 **모든 에폭**에 대해 저장한다(plot_every=1). 성분이 에폭에 따라 어떻게
    자리를 잡아 가는지가 체크포인트 선택의 근거를 눈으로 보게 해 준다.

    참조는 **배정 순서**(`loss.supervise` = x_clean · bw · ma · em)로 놓는다. 성분 k 와
    그 아래 참조가 같은 순서라 눈으로 짝을 맞출 수 있다.

    같은 에폭·같은 분절의 배정 쌍 겹침 그림은 `_plot_pairs` 가 `*_pairs.png` 로 따로 낸다.
    """
    import matplotlib
    matplotlib.use("Agg")
    from src.viz import plt

    pad, K = model.pad_each, model.n_encoders
    fs = cfg["data"]["fs"]
    sup_keys = list(cfg["loss"]["supervise"])
    comps, recon = _components_of(model, val, device, idx, pad, K)

    rows = ([("입력 x_noisy", val.x_noisy[idx], "#000"), ("재구성 x_hat", recon, "#d62728")]
            + [(f"성분 {k+1}", c, "#1f77b4") for k, c in enumerate(comps)]
            + [(f"참조 {r}", val.refs[r][idx],
                "#2ca02c" if r == "x_clean" else "#ff7f0e") for r in sup_keys])
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


def _components_of(model, val, device, idx, pad, K):
    """성분 K개와 재구성을 한 번에. 두 그림이 같은 값을 쓰게 하려고 따로 뺐다."""
    model.eval()
    with torch.no_grad():
        x = meae.pad(val.tensor(np.array([idx])).to(device), pad)
        comps = [meae.crop(model.component(x, k), pad).squeeze().cpu().numpy()
                 for k in range(K)]
        recon = meae.crop(model(x)[0], pad).squeeze().cpu().numpy()
    model.train()
    return comps, recon


def _plot_pairs(model, val, cfg, device, path, idx=0, epoch=None):
    """[추가 그림] 배정 쌍(성분 k ↔ 배정 참조)을 **한 축에 겹쳐** 그린다.

    적층 그림은 행마다 y 범위가 달라 "닮아지고 있는가"를 눈으로 판단하기 어렵다.
    여기서는 쌍의 y 범위를 참조와 성분이 함께 들어가도록 잡고, 제목에 그 쌍의 |r| 을
    적어 눈과 수치를 같이 본다.

    **쌍끼리는 y 범위가 다르다.** 참조의 진폭 자체가 다르기 때문이다 — val 900분절
    중앙값으로 첨두간 x_clean 2.85 mV 대 bw 0.51 · ma 0.78 · em 0.74 mV 로, R파 때문에
    clean 이 3.6~5.6배 크다. 전 행을 한 축에 묶으면 잡음 3행이 그만큼 눌려 형태가
    보이지 않는다. 대신 행마다 y 범위를 제목에 mV 로 적어 규모 차이를 수치로 남긴다.
    """
    import matplotlib
    matplotlib.use("Agg")
    from src.viz import plt

    pad, K = model.pad_each, model.n_encoders
    fs = cfg["data"]["fs"]
    sup_keys = list(cfg["loss"]["supervise"])
    comps, recon = _components_of(model, val, device, idx, pad, K)
    t = np.arange(len(recon)) / fs

    def corr(a, b):
        a = a - a.mean(); b = b - b.mean()
        d = np.sqrt((a ** 2).sum() * (b ** 2).sum())
        return float(abs((a * b).sum() / d)) if d > 0 else np.nan

    fig, ax = plt.subplots(2 + K, 1, figsize=(11, 1.15 * (2 + K)), sharex=True)
    x_in = val.x_noisy[idx]
    ax[0].plot(t, x_in, lw=0.6, color="#000")
    ax[1].plot(t, x_in, lw=0.6, color="#bbb", label="입력")
    ax[1].plot(t, recon, lw=0.6, color="#d62728", label="재구성")
    ax[1].legend(fontsize=6.5, ncol=2, loc="upper right")
    lim01 = max(np.abs(x_in).max(), np.abs(recon).max()) * 1.1
    for a in ax[:2]:
        a.set_ylim(-lim01, lim01)
    ax[0].set_title(f"입력 x_noisy   y ±{lim01:.2f} mV", fontsize=8, loc="left")
    ax[1].set_title(f"재구성 x_hat  (입력과 |r| {corr(x_in, recon):.3f})   "
                    f"y ±{lim01:.2f} mV", fontsize=8, loc="left")

    for k in range(K):
        a, key = ax[2 + k], sup_keys[k]
        ref = val.refs[key][idx]
        a.plot(t, ref, lw=1.1, color="#ff7f0e", alpha=.75, label=f"참조 {key}")
        a.plot(t, comps[k], lw=0.7, color="#1f77b4", label=f"성분 {k + 1}")
        lim = max(np.abs(ref).max(), np.abs(comps[k]).max()) * 1.1
        a.set_ylim(-lim, lim)          # 쌍 안에서 공통 — 크기 차이가 그대로 보인다
        a.set_title(f"성분 {k + 1} ↔ 참조 {key}   |r| {corr(ref, comps[k]):.3f}   "
                    f"y ±{lim:.2f} mV", fontsize=8, loc="left")
        a.legend(fontsize=6.5, ncol=2, loc="upper right")

    for a in ax:
        a.grid(alpha=.25, lw=.4)
        a.tick_params(labelsize=7)
    ax[-1].set_xlabel("시간 (초)")
    if epoch is not None:
        m = val.meta[idx]
        fig.suptitle(f"에폭 {epoch}  ·  val 분절 {m['record_id']}_{m['seg_idx']:04d} "
                     f"(주입 SNR bw {m['snr_bw']:.1f} / ma {m['snr_ma']:.1f} / "
                     f"em {m['snr_em']:.1f} dB)  ·  배정 쌍 겹침 "
                     f"(y 범위는 쌍 안에서만 공통, 행 제목에 표기)", fontsize=10)
    fig.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def select(history, ratio):
    """성분 정렬 손실 L_sup^val 로 후보 집합 C를 만들고 최소점을 고른다."""
    # 후보는 **체크포인트가 저장된 에폭**(판정 에폭)뿐이다. 판정하지 않은 에폭은
    # pool/ 에 가중치가 없어 뽑아도 불러올 수 없다.
    ev = [h for h in history if h.get("S") is not None]
    if not ev:
        return None, []
    lo = min(h["val_sup"] for h in ev)
    C = [h for h in ev if h["val_sup"] <= ratio * lo]
    return min(C, key=lambda h: h["val_sup"]), C


def train(cfg, n_encoders: int, seed: int, tag: str = "", plot_every: int = 1,
          max_epoch: int = None, hidden: int = None, gamma: float = None,
          gamma2: float = None, beta: float = None, sup_norm: str = None,
          channels=None, dilations=None, skip_levels=None, skip_weight=None,
          group: str = "", out_root: str = "experiments", evaluate_after: bool = True):
    if any(v is not None for v in (hidden, gamma, gamma2, beta, sup_norm,
                                   channels, dilations, skip_levels, skip_weight)):
        cfg = json.loads(json.dumps(cfg))
        if hidden is not None:
            cfg["model"]["hidden"] = hidden
        if gamma is not None:
            cfg["loss"]["gamma_sup"] = float(gamma)
        if gamma2 is not None:
            cfg["loss"]["gamma2_sup"] = float(gamma2)
        if beta is not None:
            cfg["loss"]["beta_sup"] = float(beta)
        if sup_norm is not None:
            cfg["loss"]["sup_normalize"] = sup_norm
        if channels is not None:
            cfg["model"]["channels"] = [int(v) for v in str(channels).split(",")]
        if dilations is not None:
            cfg["model"]["dilations"] = [int(v) for v in str(dilations).split(",")]
        if skip_levels is not None:
            cfg["model"]["skip_levels"] = [int(v) for v in str(skip_levels).split(",")]
        if skip_weight is not None:
            cfg["model"]["skip_weight"] = float(skip_weight)

    tr = cfg["train"]
    ck = tr["checkpoint"]
    max_epoch = max_epoch or tr["max_epoch"]
    batch, n_v, every, ratio = tr["batch"], ck["val_n"], ck["eval_every"], ck["recon_ratio"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # 그룹 폴더를 run 이름 앞에 붙인다 — `load_ckpt` 가 경로를 그대로 받는다
    run = os.path.join(group, tag or f"K{n_encoders}_seed{seed}") if group         else (tag or f"K{n_encoders}_seed{seed}")
    name = os.path.basename(run)

    sup_keys = list(cfg["loss"]["supervise"])
    gamma = float(cfg["loss"].get("gamma_sup", 0.0))
    gamma2 = float(cfg["loss"].get("gamma2_sup", 0.0))
    beta = float(cfg["loss"].get("beta_sup", 0.0))

    set_seed(seed)
    # 지도학습이므로 **학습셋에도 참조를 올린다**. 기존 load()는 train 에서 참조를 빼
    # 자기지도를 구조로 강제하는데, 이 실험은 그 원칙을 벗어나는 것이 목적이다.
    train_set = Segments(cfg, "train", with_refs=True)
    val_set = load(cfg, "val")
    model = meae.build(cfg, n_encoders).to(device)
    sigmas = (losses.source_sigmas(train_set, sup_keys)
              if cfg["loss"].get("sup_normalize") == "source_std" else None)
    crit = losses.build_supervised(cfg, n_encoders, model.pad_each, sigmas).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=tr["lr"], weight_decay=tr["weight_decay"])
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=tr["lr_step_size"], gamma=0.1)

    run_dir = os.path.join(out_root, "02_model", run)
    os.makedirs(run_dir, exist_ok=True)
    pool_dir = os.path.join(run_dir, "pool")
    os.makedirs(pool_dir, exist_ok=True)
    console = open(os.path.join(run_dir, "console.log"), "w", encoding="utf-8")

    def say(msg):
        print(msg, flush=True)
        console.write(msg + "\n")
        console.flush()

    lo = cfg["loss"]
    rf, rf_sec = meae.receptive_field(model.dilations, cfg["data"]["fs"])
    say(f"[{run}] device={device} K={n_encoders} seed={seed} hidden={cfg['model']['hidden']} "
        f"train={len(train_set)} val={len(val_set)} V={n_v}")
    say(f"  구조  깊이 {model.depth} · 압축률 {model.downsample} · 인코딩 시간축 "
        f"{(model.input_length // model.downsample)} "
        f"(조각당 {model.downsample / cfg['data']['fs']:.3f}초)")
    say(f"        채널 {list(cfg['model']['channels'])} · dilations {model.dilations}")
    say(f"        수용영역 {rf} 표본 = {rf_sec:.2f}초 · 패딩 {model.input_length}"
        f"(양쪽 {model.pad_each})")
    if model.skip_levels:
        say(f"        잔차 연결 블록 {model.skip_levels} · 가중치 {model.skip_weight:g}"
            "  (성분을 뽑을 때 그 인코더의 잔차만 살린다)")
    else:
        say("        잔차 연결 없음")
    say(f"  L = MSE(x_hat, x_noisy) + {lo['lambda_sup']:g} * L_sup   [지도학습]")
    say(f"  L_sup = (1/K) sum_k [ MSE(s_k, r_k) + {gamma:g}*MSE(d1 s_k, d1 r_k)"
        f" + {gamma2:g}*MSE(d2 s_k, d2 r_k) + {beta:g}*MSE(|F s_k|, |F r_k|) ]")
    say("    d1 = 1차 차분, d2 = 2차 차분, F = rfft(norm='ortho') 크기")
    if sigmas is None:
        say("  소스별 정규화 없음 (sup_normalize: none)")
    else:
        say("  소스별 정규화 sigma_k (훈련셋 전체) — "
            + " · ".join(f"{r} {v:.4f}" for r, v in zip(sup_keys, sigmas)))
    say("  배정  " + " · ".join(f"enc{k + 1}={r}" for k, r in enumerate(sup_keys)))
    say(f"  정칙화 3항 해제 — lambda_mixing={lo['lambda_mixing']:g} "
        f"lambda_zero_recon={lo['lambda_zero_recon']:g} lambda_z_l2={lo['lambda_z_l2']:g}")
    say(f"  선택  C = {{ L_sup^val <= {ratio} x min }} 에서 L_sup^val 최소, {every}에폭 간격")

    rng = np.random.default_rng(seed)
    history, best_recon, bad = [], np.inf, 0
    for epoch in range(1, max_epoch + 1):
        t0 = time.time()
        order = rng.permutation(len(train_set))
        sums, nb = {}, 0
        for s in range(0, len(order), batch):
            idx = order[s:s + batch]
            x = meae.pad(train_set.tensor(idx).to(device), model.pad_each)
            refs = torch.stack([train_set.ref_tensor(r, idx).squeeze(1)
                                for r in sup_keys], dim=1).to(device)   # (B, K, 3600)
            y, zs = model(x)
            d = crit(model, x, y, zs, refs)    # 참조를 직접 준다 — 지도학습
            opt.zero_grad(set_to_none=True)
            d["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), tr["gradient_clip_val"])
            opt.step()
            for k, v in d.items():
                sums[k] = sums.get(k, 0.0) + float(v.detach())
            nb += 1
        sched.step()

        judge = (epoch % every == 0) or (epoch == 1)
        ev = evaluate(model, val_set, device, batch, n_v, with_rho=judge,
                      sup_keys=sup_keys, gamma=gamma, gamma2=gamma2, beta=beta,
                      sigmas=sigmas)
        row = {"epoch": epoch, **{f"train_{k}": v / nb for k, v in sums.items()},
               "val_recon": ev["val_recon"], "val_sup": ev["val_sup"],
               **{f"val_sup_e{k + 1}": float(v) for k, v in enumerate(ev["val_sup_per"])},
               "S": ev.get("S"),
               "argmax_k": ev.get("argmax_k"), "lr": opt.param_groups[0]["lr"],
               "sec": time.time() - t0}
        if judge:
            for k, v in enumerate(ev["rho"]):
                row[f"rho_e{k}"] = float(v)
            torch.save({"model": model.state_dict(), "epoch": epoch, "cfg": cfg,
                        "n_encoders": n_encoders, "seed": seed,
                        "val_recon": ev["val_recon"], "val_sup": ev["val_sup"],
                        "S": ev["S"]},
                       os.path.join(pool_dir, f"ep{epoch:04d}.pt"))
        history.append(row)
        pd.DataFrame(history).to_csv(os.path.join(run_dir, "history.csv"),
                                     index=False, encoding="utf-8-sig")
        say(f"  ep{epoch:3d} L={row['train_total']:.5f} rec={row['train_recon']:.5f} "
            f"sup={row['train_sup']:.5f}(w {row['train_sup_wave']:.5f} "
            f"d1 {row['train_sup_diff']:.5f} d2 {row['train_sup_diff2']:.5f} "
            f"f {row['train_sup_freq']:.5f}) | "
            f"val_rec={ev['val_recon']:.5f} "
            f"val_sup={ev['val_sup']:.5f} ["
            + " ".join(f"{v:.4f}" for v in ev["val_sup_per"]) + "]"
            + (f" S={ev['S']:.3f}(enc{ev['argmax_k']+1})" if judge else "")
            + f" {row['sec']:.0f}s")

        if ev["val_sup"] < best_recon - 1e-6:      # 선택 기준과 같은 양으로 본다
            best_recon, bad = ev["val_sup"], 0
        else:
            bad += 1
            if bad >= tr["early_stop_patience"]:
                say(f"  조기 종료 (val_sup {tr['early_stop_patience']}에폭 정체)")
                break
        if plot_every and epoch % plot_every == 0:
            _plot_components(model, val_set, cfg, device,
                             os.path.join(run_dir, "plots", f"ep{epoch:04d}.png"),
                             epoch=epoch)
            _plot_pairs(model, val_set, cfg, device,
                        os.path.join(run_dir, "plots", f"ep{epoch:04d}_pairs.png"),
                        epoch=epoch)

    # ---- 학습 종료 후 일괄 판정
    best, C = select(history, ratio)
    shutil.copyfile(os.path.join(pool_dir, f"ep{best['epoch']:04d}.pt"),
                    os.path.join(run_dir, f"{name}.pt"))
    keep = {h["epoch"] for h in C}
    for f in os.listdir(pool_dir):
        if int(f[2:6]) not in keep:
            os.remove(os.path.join(pool_dir, f))

    sens = {}
    for r in SENSITIVITY_RATIOS:
        b, c = select(history, r)
        sens[str(r)] = {"n_candidates": len(c), "selected_epoch": b["epoch"],
                        "val_sup": b["val_sup"], "S": b["S"], "val_recon": b["val_recon"],
                        "epoch_range": [min(h["epoch"] for h in c),
                                        max(h["epoch"] for h in c)]}
    ev_all = [h for h in history if h.get("S") is not None]
    info = {"run": run, "K": n_encoders, "seed": seed,
            "hidden": cfg["model"]["hidden"], "ratio": ratio,
            "channels": list(cfg["model"]["channels"]),
            "dilations": model.dilations, "depth": model.depth,
            "skip_levels": model.skip_levels, "skip_weight": model.skip_weight,
            "downsample": model.downsample,
            "receptive_field": int(meae.receptive_field(
                model.dilations, cfg["data"]["fs"])[0]),
            "input_length": model.input_length, "pad_each": model.pad_each,
            "eval_every": every, "val_n": n_v,
            "n_epochs": len(history), "n_evaluated": len(ev_all),
            "min_val_sup": min(h["val_sup"] for h in ev_all),
            "min_val_recon": min(h["val_recon"] for h in ev_all),
            "supervise": list(cfg["loss"]["supervise"]),
            "sup_normalize": cfg["loss"].get("sup_normalize", "none"),
            "sigmas": sigmas,
            "n_candidates": len(C),
            "candidate_range": [min(h["epoch"] for h in C), max(h["epoch"] for h in C)],
            "selected_epoch": best["epoch"], "val_sup": best["val_sup"],
            "val_sup_per": [best[f"val_sup_e{k + 1}"] for k in range(n_encoders)],
            "S": best["S"], "argmax_k": best["argmax_k"],
            "val_recon": best["val_recon"],
            "sensitivity": sens, "loss": cfg["loss"]}
    with open(os.path.join(run_dir, "stage1.json"), "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)

    say(f"[{run}] 완료. {len(history)}에폭, 판정 {len(ev_all)}개")
    say(f"  min L_sup^val = {info['min_val_sup']:.5f}   "
        f"min L_recon^val = {info['min_val_recon']:.5f}")
    say(f"  C (ratio {ratio}) = {len(C)}개, 에폭 {info['candidate_range'][0]}"
        f"~{info['candidate_range'][1]}")
    say(f"  t* = 에폭 {best['epoch']}  val_sup {best['val_sup']:.5f} ["
        + " ".join(f"{v:.4f}" for v in info["val_sup_per"]) + "]  "
        f"val_recon {best['val_recon']:.5f}  S {best['S']:.4f}")
    for r, v in sens.items():
        say(f"    민감도 ratio {r}: C {v['n_candidates']}개 "
            f"(에폭 {v['epoch_range'][0]}~{v['epoch_range'][1]}) -> "
            f"t* {v['selected_epoch']}  val_sup {v['val_sup']:.5f}")
    # ---- 정량 지표는 **학습의 일부**로 항상 낸다. 따로 불러야 나오면 빠뜨리게 된다.
    if evaluate_after:
        say("")
        say("[정량 지표] 학습 종료 후 자동 산출 — 03 성분 정렬 · 04 복원 세 방식")
        for label, fn in (("성분 정렬", _run_stage3), ("복원 세 방식", _run_stage4)):
            try:
                fn(cfg, run, run_dir)
                say(f"  {label} 완료")
            except Exception as e:                      # 지표가 실패해도 학습본은 남긴다
                say(f"  {label} 실패 — {type(e).__name__}: {e}")
    console.close()
    return pd.DataFrame(history), info


def _load_script(name):
    """번호로 시작하는 파일명은 import 문으로 못 부른다. 경로로 직접 올린다."""
    import importlib.util
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), name)
    spec = importlib.util.spec_from_file_location(name[:-3], path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _ckpt_path(run, run_dir):
    """체크포인트 경로를 직접 넘긴다.

    `load_ckpt` 는 이름을 받으면 `results/02_model/...` 만 뒤진다. `--out-root` 로
    experiments 에 낸 런은 그 경로에 없으므로 파일 경로를 그대로 준다 (허용된 형태다).
    """
    return os.path.join(run_dir, f"{os.path.basename(run)}.pt")


def _run_stage3(cfg, run, run_dir, split="val"):
    """성분 정렬 지표 — 확정 체크포인트 하나에 대해 val 전체.

    산출물은 **그 런 폴더 안**(`metrics/`)에 둔다. 03_bss/ 는 최종 선정 모델 전용이다.
    """
    _load_script("03_bss.py").main(run=_ckpt_path(run, run_dir), split=split,
                                   outdir=os.path.join(run_dir, "metrics"))


def _run_stage4(cfg, run, run_dir, split="val"):
    """복원 세 방식(A 성분차감 · B 심장직접 · C 마스킹디코드). 산출물은 `restore/`."""
    _load_script("04_masked_denoising.py").three_ways(
        run=_ckpt_path(run, run_dir), split=split,
        outdir=os.path.join(run_dir, "restore"))


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


def diagnose(config="configs/default.yaml", run="C16_seed42", split="val", n=900,
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
    p.add_argument("--run", default="C16_seed42")
    p.add_argument("--split", default="val")
    p.add_argument("--n", type=int, default=900)
    p.add_argument("--hidden", type=int, default=None)
    p.add_argument("--tag", default="", help="실행 이름. 비우면 K<k>_seed<seed>")
    p.add_argument("--gamma", type=float, default=None,
                   help="γ₁ — loss.gamma_sup 오버라이드 (1차 차분 비중)")
    p.add_argument("--gamma2", type=float, default=None,
                   help="γ₂ — loss.gamma2_sup 오버라이드 (2차 차분 비중)")
    p.add_argument("--beta", type=float, default=None,
                   help="β — loss.beta_sup 오버라이드 (|FFT| 크기 항 비중)")
    p.add_argument("--channels", default=None,
                   help="쉼표로 구분한 채널 목록. 길이가 곧 인코더 깊이(압축률 2^깊이)")
    p.add_argument("--dilations", default=None,
                   help="쉼표로 구분한 블록별 dilation. 층을 줄일 때 수용영역 보전용")
    p.add_argument("--skip-levels", dest="skip_levels", default=None,
                   help="잔차로 쓸 인코더 블록 번호 (0부터). 예: 0")
    p.add_argument("--skip-weight", dest="skip_weight", type=float, default=None,
                   help="잔차 가중치. 0 이면 잔차 없음(기존과 동일)")
    p.add_argument("--sup-normalize", dest="sup_norm", default=None,
                   choices=["none", "source_std"],
                   help="지도항을 소스별 표준편차로 정규화한다")
    p.add_argument("--out-root", dest="out_root", default="experiments",
                   help="산출물 최상위. **기본은 experiments** — results/ 는 최종 선정 "
                        "모델 하나 전용이다. 확정 후에만 --out-root results 로 돌린다")
    p.add_argument("--group", default="",
                   help="results/02_model/<group>/<run>/ 로 묶는다")
    p.add_argument("--no-eval", dest="no_eval", action="store_true",
                   help="학습 후 정량 지표 자동 산출을 건너뛴다")
    a = p.parse_args()
    if a.diagnose:
        diagnose(a.config, a.run, a.split, a.n)
    else:
        if a.k is None or a.seed is None:
            p.error("학습에는 --k 와 --seed 가 필요하다 (진단만 하려면 --diagnose)")
        train(load_cfg(a.config), a.k, a.seed, tag=a.tag, max_epoch=a.max_epoch,
              plot_every=a.plot_every, hidden=a.hidden, gamma=a.gamma,
              gamma2=a.gamma2, beta=a.beta, sup_norm=a.sup_norm,
              channels=a.channels, dilations=a.dilations,
              skip_levels=a.skip_levels, skip_weight=a.skip_weight, group=a.group,
              out_root=a.out_root, evaluate_after=not a.no_eval)
