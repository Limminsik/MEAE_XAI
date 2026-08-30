"""분절 데이터셋 구축 — 기록을 10초로 자르고 잡음 3종을 주입한다 (README §2.1).

윈도우 하나가 거치는 과정
  1. mitdb 기록에서 MLII를 3600샘플(10초) 비중첩으로 잘라 x_clean
  2. 해당 분절이 속한 split에 허용된 잡음 구간에서 bw/ma/em 조각을 독립 추첨
  3. 잡음마다 SNR을 독립 추첨하고, 파형은 그대로 둔 채 진폭만 맞춤
  4. x_noisy = x_clean + bw + ma + em 로 합산
  5. 5개 배열 + R-피크 + meta 를 npz 한 개로 저장

재현성
  분절마다 (gen_seed, record, seg_idx)로 독립 난수기를 만든다. 순서를 바꾸거나
  일부만 다시 만들어도 항상 같은 데이터가 나온다.
"""
import argparse
import hashlib
import json
import os

import numpy as np
import pandas as pd
import wfdb
import yaml
from tqdm import tqdm

NOISES = ("bw", "ma", "em")
# 박동 심볼. 'f'(paced+normal fusion)도 실제 박동이므로 포함한다 (217번 260박).
BEAT_SYMBOLS = set("NLRVAFjE/aJSeQf")
EXCLUDED_BEAT_SYMBOLS = set()


def load_cfg(path="configs/default.yaml"):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def seg_rng(gen_seed, record, seg_idx):
    """분절마다 순서에 의존하지 않는 독립 난수기."""
    key = f"{gen_seed}|{record}|{seg_idx}".encode()
    return np.random.default_rng(int.from_bytes(hashlib.sha256(key).digest()[:8], "big"))


def load_noise_pools(cfg):
    """잡음 레코드를 읽고 train / val·test 구간으로 시간 분할한다 — 잡음 풀이 split 간에 겹치지 않게."""
    d, nst = cfg["data"], cfg["paths"]["nstdb"]
    ch, ratio = d["noise_channel"], d["noise_split_ratio"]
    pools = {}
    for n in NOISES:
        sig, fields = wfdb.rdsamp(os.path.join(nst, n))
        assert fields["fs"] == d["fs"], (n, fields["fs"])
        v = sig[:, ch].astype(np.float64)
        cut = int(len(v) * ratio)
        pools[n] = {"train": (v, 0, cut), "eval": (v, cut, len(v))}
    return pools


def draw_noise(pools, name, which, rng, seg_len):
    """허용 구간 안에서만 시작점을 뽑아 seg_len 샘플을 잘라낸다."""
    v, lo, hi = pools[name][which]
    start = int(rng.integers(lo, hi - seg_len + 1))
    return v[start:start + seg_len], start


def read_record(mitdb, record, lead):
    sig, fields = wfdb.rdsamp(os.path.join(mitdb, record))
    ch = fields["sig_name"].index(lead)          # 114번은 MLII가 1번 채널이다
    ann = wfdb.rdann(os.path.join(mitdb, record), "atr")
    beats = np.array([s for i, s in enumerate(ann.symbol)])
    keep = np.array([s in BEAT_SYMBOLS for s in beats])
    drop = int(sum(s in EXCLUDED_BEAT_SYMBOLS for s in beats))
    return sig[:, ch].astype(np.float64), ann.sample[keep], drop


def build_segment(x_clean, pools, which, rng, snr_range):
    """주입 SNR 정의. 신호 전력은 분산 기준, 잡음 전력은 mean(n**2) 기준."""
    p_sig = float(x_clean.var())
    comps, info = {}, {}
    for n in NOISES:
        raw, start = draw_noise(pools, n, which, rng, len(x_clean))
        snr = float(rng.uniform(*snr_range))
        a = np.sqrt(p_sig / (float((raw ** 2).mean()) * 10.0 ** (snr / 10.0)))
        comps[n] = (a * raw).astype(np.float32)
        info[n] = {"snr": snr, "start": start, "gain": float(a)}
    x32 = x_clean.astype(np.float32)
    # float32로 캐스팅한 뒤 더해야 x_noisy == x_clean + bw + ma + em 이 정확히 성립한다
    x_noisy = x32 + comps["bw"] + comps["ma"] + comps["em"]
    return x32, x_noisy, comps, info


def measured_snr(x_clean, comp):
    """검증용 실측 SNR. 주입과 동일하게 신호는 분산, 잡음은 mean(n**2) 기준."""
    return 10.0 * np.log10(np.var(x_clean, dtype=np.float64)
                           / np.mean(np.square(comp, dtype=np.float64)))


def main(config="configs/default.yaml", limit=None):
    cfg = load_cfg(config)
    d, fs = cfg["data"], cfg["data"]["fs"]
    seg_len = fs * d["seg_sec"]
    proc = cfg["paths"]["processed"]
    split = json.load(open(os.path.join(proc, "split.json"), encoding="utf-8"))
    rec2split = {r: s for s in ("train", "val", "test") for r in split[s]}
    gen_seed = d["split"]["split_seed"]        # 생성 시드도 분할 시드와 같은 고정값을 쓴다

    pools = load_noise_pools(cfg)
    out_root = os.path.join(proc, "segments")
    rows = []
    records = sorted(rec2split)
    if limit:
        records = records[:limit]

    for rec in tqdm(records, desc="records"):
        which = "train" if rec2split[rec] == "train" else "eval"
        x, rpeaks, n_dropped = read_record(cfg["paths"]["mitdb"], rec, d["lead"])
        n_seg = len(x) // seg_len
        odir = os.path.join(out_root, rec2split[rec])
        os.makedirs(odir, exist_ok=True)

        for i in range(n_seg):
            s0 = i * seg_len
            x_clean = x[s0:s0 + seg_len]
            rng = seg_rng(gen_seed, rec, i)
            x32, x_noisy, comps, info = build_segment(
                x_clean, pools, which, rng, d["noise_snr_range_db"])
            local_r = (rpeaks[(rpeaks >= s0) & (rpeaks < s0 + seg_len)] - s0).astype(np.int32)

            meta = {"record_id": rec, "seg_idx": i, "split": rec2split[rec],
                    "noise_pool": which, "lead": d["lead"],
                    "noise_channel": d["noise_channel"], "gen_seed": gen_seed,
                    "n_rpeaks": int(len(local_r)),
                    "n_dropped_beats_record": n_dropped,
                    **{f"snr_{n}": info[n]["snr"] for n in NOISES},
                    **{f"start_{n}": info[n]["start"] for n in NOISES},
                    **{f"gain_{n}": info[n]["gain"] for n in NOISES}}
            np.savez_compressed(
                os.path.join(odir, f"{rec}_{i:04d}.npz"),
                x_clean=x32, x_noisy=x_noisy, bw=comps["bw"], ma=comps["ma"],
                em=comps["em"], rpeaks=local_r,
                meta=json.dumps(meta, ensure_ascii=False))
            rows.append({**meta,
                         **{f"snr_meas_{n}": measured_snr(x32, comps[n]) for n in NOISES},
                         "p_signal_var": float(x_clean.var()),
                         "dc_clean": float(x_clean.mean())})

    man = pd.DataFrame(rows)
    man_path = os.path.join(proc, "manifest.csv")
    man.to_csv(man_path, index=False, encoding="utf-8-sig")
    print(f"\n[write] {man_path}  분절 {len(man):,}개")
    print(man.groupby("split").size().to_string())
    for n in NOISES:
        err = (man[f"snr_meas_{n}"] - man[f"snr_{n}"]).abs()
        print(f"  {n}: SNR 오차 최대 {err.max():.2e} dB, "
              f"주입 SNR {man[f'snr_{n}'].min():.2f}~{man[f'snr_{n}'].max():.2f} dB")
    return man


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--limit", type=int, default=None, help="기록 수 제한 (스모크 테스트용)")
    a = p.parse_args()
    main(a.config, a.limit)
