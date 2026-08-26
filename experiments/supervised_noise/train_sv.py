"""SV1 — 잡음 지도(supervised) 학습.

**기존 자기지도 파이프라인과 분리된 방향 탐색 실험이다.** `src/`는 읽기만 하고 고치지 않는다.
§0 원칙 3(자기지도)을 의도적으로 위반하는 실험이므로 본 노선의 결과와 섞지 않는다.

손실
    L = MSE(x̂, x_noisy)                                재구성 (유지)
      + λ_sup · Σ_{t∈{bw,ma,em}} MSE(x̂_t, α_t·n_t)      잡음 3종 지도 (신설)
      + λ_zero · MSE(decode(0), 0)                      zero reconstruction (유지)
      + λ_z · Σ_k mean(z_k²)                            인코딩 L2 (유지)

sparse mixing은 제외한다 — 정답이 분업을 강제하므로 중복 장치다.
인코더 1·2·3(내부 0·1·2)을 bw·ma·em에 고정 할당하고, **인코더 4는 지도하지 않는다.**
clean은 손실에 넣지 않는다. 심장 성분은 "남은 것"으로만 주어진다.

관심 질문: 지도받지 않은 인코더 4가 심장 성분을 담당하게 되는가.
"""
import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import torch

from src.data.build import load_cfg
from src.data.dataset import Segments
from src.model import losses, meae
from src.train import CheckpointPool, evaluate, set_seed

SUP_REFS = ("bw", "ma", "em")          # 인코더 0·1·2에 이 순서로 고정 할당
RUN_PREFIX = "SV1"


def sup_targets(ds, idx, pad_each, device):
    """성분 목표 (B, 3, 3840). 저장된 배열은 이미 α_t 가 곱해진 값이다."""
    t = torch.stack([ds.ref_tensor(r, idx).squeeze(1) for r in SUP_REFS], 1)
    return meae.pad(t.to(device), pad_each)


def train(cfg, seed, n_encoders=4, lambda_sup=1.0, max_epoch=None, plot_every=25):
    tr = cfg["train"]
    lo = cfg["loss"]
    max_epoch = max_epoch or tr["max_epoch"]
    batch = tr["batch"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    run = f"{RUN_PREFIX}_K{n_encoders}_seed{seed}"

    set_seed(seed)
    # **학습셋에 정답을 올린다** — 기존 노선에서는 금지된 동작이다 (SV1 전용)
    train_set = Segments(cfg, "train", with_refs=True)
    val_set = Segments(cfg, "val", with_refs=True)

    model = meae.build(cfg, n_encoders).to(device)
    crit = losses.build(cfg, n_encoders).to(device)      # zero_recon·z_l2 재사용
    mse = torch.nn.MSELoss()
    opt = torch.optim.Adam(model.parameters(), lr=tr["lr"], weight_decay=tr["weight_decay"])
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=tr["lr_step_size"], gamma=0.1)

    log_dir = os.path.join("_work", "runs", run)
    os.makedirs(log_dir, exist_ok=True)
    console = open(os.path.join(log_dir, "console.log"), "w", encoding="utf-8")

    def say(msg):
        print(msg, flush=True)
        console.write(msg + "\n")
        console.flush()
    pool = CheckpointPool(run, log_dir, log_dir, tr["checkpoint_recon_gate"],
                          tr["checkpoint_pool"], tr["checkpoint_keep_top"])
    pad = model.pad_each
    say(f"[{run}] device={device} K={n_encoders} λ_sup={lambda_sup} "
          f"train={len(train_set)} val={len(val_set)}")
    say(f"  지도 할당: " + ", ".join(
        f"{meae.enc_label(i)}→{r}" for i, r in enumerate(SUP_REFS))
        + f", {meae.enc_label(len(SUP_REFS))}→지도 없음")

    rng = np.random.default_rng(seed)
    history, best_recon, bad = [], np.inf, 0
    for epoch in range(1, max_epoch + 1):
        t0 = time.time()
        order = rng.permutation(len(train_set))
        sums, nb, step_t = {}, 0, 0.0
        for s in range(0, len(order), batch):
            ts = time.time()
            idx = order[s:s + batch]
            x = meae.pad(train_set.tensor(idx).to(device), pad)
            tgt = sup_targets(train_set, idx, pad, device)

            zs = model.encode(x)
            y = model.net.output(model.net.decoder(zs))          # 전체 재구성
            recon = mse(y, x)

            sup = x.new_zeros(())
            for k, _ in enumerate(SUP_REFS):                     # 성분별 추가 디코더 통과
                keep = [k]
                comp = model.decode([z if i in keep else torch.zeros_like(z)
                                     for i, z in enumerate(zs)])
                sup = sup + mse(comp.squeeze(1), tgt[:, k])

            z_l2 = sum(torch.mean(z ** 2) for z in zs)
            zeros = model.zero_encoding(1, zs[0].shape[-1], device)
            x_zero = model.decode(zeros, zeros_train=True)
            zero_recon = mse(x_zero, torch.zeros_like(x_zero))

            total = (recon + lambda_sup * sup
                     + lo["lambda_zero_recon"] * zero_recon
                     + lo["lambda_z_l2"] * z_l2)
            opt.zero_grad(set_to_none=True)
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), tr["gradient_clip_val"])
            opt.step()
            for k, v in (("total", total), ("recon", recon), ("sup", sup),
                         ("zero_recon", zero_recon), ("z_l2", z_l2)):
                sums[k] = sums.get(k, 0.0) + float(v.detach())
            nb += 1
            step_t += time.time() - ts
        sched.step()
        terms = {f"train_{k}": v / nb for k, v in sums.items()}
        ev = evaluate(model, crit, val_set, cfg, device, batch)
        dt = time.time() - t0

        row = {"epoch": epoch, **terms, "val_recon": ev["val_recon"],
               "val_separation": ev["val_separation"], "sep_base": ev["sep_base"],
               "n_dup": ev["n_dup"], "tops": "".join(str(t) for t in ev["tops"]),
               "clean_max_corr": float(ev["corr"][:, 0].max()),
               "max_pair_corr": ev["max_pair_corr"],
               "sec": dt, "sec_per_step": step_t / nb, "n_steps": nb}
        for r_i, r in enumerate(("x_clean", "bw", "ma", "em")):
            for k in range(n_encoders):
                row[f"corr_e{k}_{r}"] = float(ev["corr"][k, r_i])
        history.append(row)
        pd.DataFrame(history).to_csv(os.path.join(log_dir, "history.csv"),
                                     index=False, encoding="utf-8-sig")
        say(f"  ep{epoch:3d} tot={terms['train_total']:.5f} rec={terms['train_recon']:.5f} "
              f"sup={terms['train_sup']:.5f} | val_rec={ev['val_recon']:.5f} "
              f"clean={row['clean_max_corr']:.3f} tops {row['tops']} "
              f"{dt:.0f}s ({row['sec_per_step']*1000:.0f}ms/step)")

        pool.update(epoch, ev["val_separation"], ev["val_recon"],
                    {"model": model.state_dict(), "epoch": epoch, "cfg": cfg,
                     "n_encoders": n_encoders, "seed": seed,
                     "val_separation": ev["val_separation"], "val_recon": ev["val_recon"],
                     "corr": ev["corr"].tolist(), "sv1": True,
                     "lambda_sup": lambda_sup, "sup_refs": list(SUP_REFS)})
        if ev["val_recon"] < best_recon - 1e-6:
            best_recon, bad = ev["val_recon"], 0
        else:
            bad += 1
            if bad >= tr["early_stop_patience"]:
                say(f"  조기 종료 (val_recon {tr['early_stop_patience']}에폭 정체)")
                break

    chosen = pool.finalize()
    with open(os.path.join(log_dir, "checkpoints.json"), "w", encoding="utf-8") as f:
        json.dump({"gate": tr["checkpoint_recon_gate"], "min_val_recon": pool.best_recon,
                   "chosen": chosen, "sv1": True, "lambda_sup": lambda_sup}, f,
                  ensure_ascii=False, indent=2)
    h = pd.DataFrame(history)
    say(f"[{run}] 완료. 최소 val_recon {pool.best_recon:.5f}")
    for c in chosen:
        say(f"    rank{c['rank']} 에폭 {c['epoch']:3d}  sep {c['sep']:.4f}  "
              f"val_recon {c['val_recon']:.5f}  → {c['file']}")
    say(f"    총 {len(h)}에폭, {h.sec.sum()/60:.1f}분, "
          f"에폭당 {h.sec.mean():.1f}s, 스텝당 {h.sec_per_step.mean()*1000:.0f}ms "
          f"({int(h.n_steps.iloc[0])}스텝/에폭)")
    console.close()
    return h


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--k", type=int, default=4)
    p.add_argument("--lambda-sup", type=float, default=1.0)
    p.add_argument("--max-epoch", type=int, default=None)
    a = p.parse_args()
    train(load_cfg(a.config), a.seed, a.k, a.lambda_sup, a.max_epoch)
