"""S5 — 마스킹 조건 사다리 M0–M5 + 전수 2^K 지도 (RESEARCH_DESIGN.md §8).

조건 정의
  M0  마스킹 없음 (전 인코더 유지)                     기준선
  M1  인코더 1개씩 단독 마스킹 (K개 조건)              아블레이션
  M2  주력 잡음 인코더만                               최소 개입
  M3  주력 + 잔여 잡음군                               중간 개입
  M4  clean 1위 인코더 제외 전부                       최대 개입
  M5  clean 인코더만                                   음성 대조 (악화해야 함)

선정 기준 (§7 ③ 기여 분해 기준. **val에서 정의하고 test에 적용**)
  잡음 기여 합_k = share_k(bw) + share_k(ma) + share_k(em)
  M2 = { k : 잡음 기여 합_k >= 10% }
  M3 = M2 ∪ { k : 잡음 기여 합_k > clean 기여_k }

모든 지표는 **crop 후 중앙 3600**에서 clean 참조 기준으로 계산한다.
전수 2^K 지도는 인코딩을 한 번만 계산하고 디코더만 2^K번 돌린다.
"""
import argparse
import itertools
import json
import os

import numpy as np
import pandas as pd
import torch

from . import metrics
from .data.build import load_cfg
from .data.dataset import REF_KEYS, load
from .model import meae
from .s4_identify import component_bank, contribution, load_ckpt

NOISE_REFS = ("bw", "ma", "em")
M2_THRESHOLD = 10.0          # 잡음 기여 합 임계 (%). val에서 정의, test에 적용


def mask_sets(share_pct: np.ndarray, threshold: float = M2_THRESHOLD):
    """기여 분해(%)에서 M0–M5의 마스킹 대상 인코더 집합을 정한다."""
    K = share_pct.shape[0]
    ci = list(REF_KEYS).index("x_clean")
    noise_cols = [list(REF_KEYS).index(n) for n in NOISE_REFS]
    noise_sum = share_pct[:, noise_cols].sum(1)
    clean_share = share_pct[:, ci]
    clean_enc = int(clean_share.argmax())

    m2 = sorted(int(k) for k in np.where(noise_sum >= threshold)[0] if k != clean_enc)
    m3 = sorted(set(m2) | {int(k) for k in range(K)
                           if k != clean_enc and noise_sum[k] > clean_share[k]})
    m4 = sorted(k for k in range(K) if k != clean_enc)
    return {
        "M0": [], "M2": m2, "M3": m3, "M4": m4, "M5": [clean_enc],
        "_clean_enc": clean_enc, "_noise_sum": noise_sum.tolist(),
        "_clean_share": clean_share.tolist(), "_threshold": threshold,
    }


@torch.no_grad()
def masked_batch(model, x_pad, mask):
    """mask(인덱스 리스트)를 0으로 치환한 재구성. 인코딩은 호출자가 재사용한다."""
    return model.masked_reconstruct(x_pad, mask) if mask else model(x_pad)[0]


@torch.no_grad()
def run_conditions(model, ds, device, idx, conds, fs, seg_len, pad, batch=100,
                   full_metrics=True):
    """조건별 지표. conds = {이름: 마스킹 인덱스 리스트}"""
    rows = []
    for s in range(0, len(idx), batch):
        j = idx[s:s + batch]
        x = meae.pad(ds.tensor(j).to(device), pad)
        clean = ds.refs["x_clean"][j].astype(np.float64)
        noisy = ds.x_noisy[j].astype(np.float64)
        zs = model.encode(x)
        for name, mk in conds.items():
            keep = [i for i in range(model.n_encoders) if i not in set(mk)]
            y = model.decode(model._mask(zs, keep))
            y = meae.crop(y, pad).squeeze(1).cpu().numpy().astype(np.float64)
            for t, gi in enumerate(j):
                if full_metrics:
                    m = metrics.score(clean[t], y[t], ds.rpeaks[gi], fs, seg_len)
                else:
                    m = {"snr_db": metrics.snr_db(clean[t], y[t]),
                         "rmse": metrics.rmse(clean[t], y[t])}
                rows.append({"seg": int(gi), "cond": name,
                             "record": ds.meta[gi]["record_id"], **m})
        # 입력 자체(x_noisy)도 M0 옆에 참조로 싣는다
        if "INPUT" in conds:
            continue
    return pd.DataFrame(rows)


@torch.no_grad()
def exhaustive_map(model, ds, device, idx, pad, batch=100):
    """전수 2^K 마스킹 지도. SNR·RMSE만 계산한다 (R-피크 검출은 비용이 크다)."""
    K = model.n_encoders
    combos = [tuple(c) for r in range(K + 1) for c in itertools.combinations(range(K), r)]
    snr = np.zeros((len(combos), len(idx)))
    rmse = np.zeros_like(snr)
    for s in range(0, len(idx), batch):
        j = idx[s:s + batch]
        x = meae.pad(ds.tensor(j).to(device), pad)
        clean = ds.refs["x_clean"][j].astype(np.float64)
        zs = model.encode(x)
        zeros = [torch.zeros_like(z) for z in zs]
        for ci, mk in enumerate(combos):
            ms = set(mk)
            y = model.decode([zeros[i] if i in ms else zs[i] for i in range(K)])
            y = meae.crop(y, pad).squeeze(1).cpu().numpy().astype(np.float64)
            d = y - clean
            p = (d ** 2).mean(-1)
            snr[ci, s:s + len(j)] = 10 * np.log10(clean.var(-1) / np.maximum(p, 1e-30))
            rmse[ci, s:s + len(j)] = np.sqrt(p)
    df = pd.DataFrame({
        "mask": ["-".join(map(str, c)) if c else "(none)" for c in combos],
        "n_masked": [len(c) for c in combos],
        "snr_median": np.median(snr, 1), "snr_mean": snr.mean(1),
        "rmse_median": np.median(rmse, 1),
    })
    df["snr_percentile"] = df.snr_median.rank(pct=True) * 100
    return df.sort_values("snr_median", ascending=False).reset_index(drop=True)


def main(config="configs/default.yaml", run="K8_seed42", split="val", n=1000,
         outdir="analysis/validation", figdir="analysis/validation/figures", seed_for_share=None):
    cfg = load_cfg(config)
    fs, seg_len = cfg["data"]["fs"], cfg["data"]["fs"] * cfg["data"]["seg_sec"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(outdir, exist_ok=True)
    os.makedirs(figdir, exist_ok=True)

    tag = os.path.basename(run)          # 하위 폴더 경로가 파일명에 섞이지 않게
    model, ck = load_ckpt(cfg, run)
    model = model.to(device)
    ds = load(cfg, split)
    idx = np.arange(min(n, len(ds)))
    pad = model.pad_each

    # --- 기여 분해로 마스킹 집합 확정 (val 기준)
    comps, refs = component_bank(model, ds, device, idx)
    share, r2 = contribution(comps, refs)
    share_pct = 100 * share.mean(0)
    sets = mask_sets(share_pct)

    conds = {"M0": sets["M0"], "M2": sets["M2"], "M3": sets["M3"],
             "M4": sets["M4"], "M5": sets["M5"]}
    for k in range(model.n_encoders):
        conds[f"M1_e{k}"] = [k]

    df = run_conditions(model, ds, device, idx, conds, fs, seg_len, pad)

    # 입력 자체(마스킹 이전 원신호) 기준선
    base_rows = []
    for gi in idx:
        m = metrics.score(ds.refs["x_clean"][gi].astype(np.float64),
                          ds.x_noisy[gi].astype(np.float64), ds.rpeaks[gi], fs, seg_len)
        base_rows.append({"seg": int(gi), "cond": "INPUT",
                          "record": ds.meta[gi]["record_id"], **m})
    df = pd.concat([pd.DataFrame(base_rows), df], ignore_index=True)
    df.to_csv(f"{outdir}/{tag}_conditions_raw.csv", index=False, encoding="utf-8-sig")

    emap = exhaustive_map(model, ds, device, idx, pad)
    emap.to_csv(f"{outdir}/{tag}_mask_map.csv", index=False, encoding="utf-8-sig")

    with open(f"{outdir}/{tag}_mask_sets.json", "w", encoding="utf-8") as f:
        json.dump({**{k: v for k, v in sets.items()},
                   "run": run, "split": split, "n": int(len(idx)),
                   "epoch": ck["epoch"]}, f, ensure_ascii=False, indent=2)
    return df, emap, sets, share_pct


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--run", default="K8_seed42")
    p.add_argument("--split", default="val")
    p.add_argument("--n", type=int, default=1000)
    a = p.parse_args()
    main(a.config, a.run, a.split, a.n)
