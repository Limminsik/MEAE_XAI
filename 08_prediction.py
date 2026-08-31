"""08 — 하류 과제. 디노이징이 부정맥 분류에 도움이 되는가.

    python 08_prediction.py --run C16_seed42 --task nv

같은 분류기를 **입력만 바꿔** 두 번 따로 학습하고, 두 모델의 성능을 견준다.
디노이징 자체의 지표(04·06)가 아니라 **그 신호를 받아 쓰는 쪽**에서 본 값이다.

    x_noisy    → 1-D CNN → 성능 A          (처리 전)
    x_denoised → 1-D CNN → 성능 B          (처리 후, 04·06 의 B 재구성과 같은 신호)

**두 경로는 완전히 독립이다.** 구조·초기화·최적화·에폭·시드·분할·박동 창이 모두 같고
입력 신호만 다르다. 가중치를 공유하지 않고, 한쪽 결과를 다른 쪽에 쓰지 않는다.

════════════════════════════════════════════════════════════════════════
차용한 규약
════════════════════════════════════════════════════════════════════════
  AAMI EC57 5분류    N · S · V · F · Q      de Chazal et al. 2004
  환자 간 분할       학습 기록과 평가 기록을 겹치지 않게      〃
  1-D CNN 기준선     잔차 블록 5개 + FC 2단                Kachuee et al. 2018
                     (ICHI, doi:10.1109/ICHI.2018.00092)

**표준 DS1/DS2(22/22) 분할은 쓰지 않는다.** 디노이저가 우리 train 32기록을 보고
학습했는데 DS2 에 그 기록이 섞이면, x_denoised 가 "이미 본 기록"이라 부당하게 유리해져
입력 간 비교라는 이 실험의 목적이 깨진다. **디노이저의 기록 분할을 그대로 물려받는다.**

════════════════════════════════════════════════════════════════════════
과제를 왜 5분류로 두지 않는가
════════════════════════════════════════════════════════════════════════
우리 test 9기록의 박동 20,463개를 AAMI 로 세면 F 12박 · Q 2박이다. 그 둘은 재현율이
0 아니면 1 로 튀어 수치에 뜻이 없다. S 도 111박 중 94박이 기록 220 하나에 몰려 있다.

  --task nv   (기본)  N vs V 이진.  V 는 test 에 1,105박이 세 기록(233·215·116)에
                      퍼져 있어 유일하게 제대로 평가된다. PVC 검출은 단독으로도
                      임상적 의미가 있다.
  --task nsv          N/S/V 3분류.  S 는 표본이 얇다 — 클래스별 재현율로만 읽는다.
  F·Q 는 두 과제 모두에서 제외하고 그 근거를 note.txt 에 남긴다.

════════════════════════════════════════════════════════════════════════
박동 잘라내기
════════════════════════════════════════════════════════════════════════
기준점은 **주석의 R-피크**다(검출기가 아니다). 신호마다 검출을 다시 하면 "박동을
찾았는가"와 "찾은 자리의 파형이 분류에 쓸 만한가"가 섞인다. 06 과 같은 설계다.

    창 = R + [-250 ms, +400 ms] = 360 Hz 에서 [-90, +144] → 234 표본
    분절(10초) 밖으로 나가는 박동은 뺀다 — 두 입력에서 **같은 박동만** 남는다.

정규화는 하지 않는다. 디노이저가 mV 원값을 그대로 받도록 학습됐고, 여기서 표준화하면
"진폭이 얼마나 살아났는가"라는 차이가 지워진다.

════════════════════════════════════════════════════════════════════════
읽는 법
════════════════════════════════════════════════════════════════════════
클래스가 심하게 치우쳐 있어 **정확도는 보지 않는다** (N 만 찍어도 94%다).
주 지표는 **macro-F1** 이고, 소수 클래스의 재현율을 함께 싣는다.
시드 5개를 돌려 평균±SD 로 낸다 — 한 번 돌린 값은 불균형 때문에 흔들린다.

**한계 하나를 밝혀 둔다.** 디노이저는 분류기의 train 기록을 보고 학습했으므로
그 구간의 x_denoised 는 "본 기록"이고, test 구간은 "못 본 기록"이다. 분류기 쪽에서는
학습·평가의 입력 분포가 그만큼 어긋난다. note.txt 에 두 구간의 복원 품질을 함께 적어
얼마나 어긋나는지 수치로 남긴다.

────────────────────────────────────────────────────────────────────────
산출물  results/08_prediction/<run>/<task>/
────────────────────────────────────────────────────────────────────────
  beats.csv           split × 클래스 박동 수
  runs.csv            입력 × 시드 × 지표 (원값)
  summary.csv         입력별 평균±SD 와 두 입력의 차이
  per_class.csv       클래스별 정밀도·재현율·F1 (시드 평균)
  confusion.csv       입력별 혼동행렬 (시드 합)
  note.txt
  figures/compare.png     입력 간 성능 — 막대 + 시드별 점
  figures/confusion.png   혼동행렬 두 장

값에 대한 해석·판정은 붙이지 않는다.
"""
import argparse
import json
import os
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import wfdb

from src.core import load_ckpt
from src.data.build import BEAT_SYMBOLS, load_cfg
from src.data.dataset import load as load_split
from src.model import meae
from src.viz import plt

SEG = 3600
NL = chr(10)
# AAMI EC57 (de Chazal et al. 2004). BEAT_SYMBOLS 에 없는 심볼은 애초에 박동이 아니다.
AAMI = {**{s: "N" for s in "NLRej"}, **{s: "S" for s in "AaJS"},
        **{s: "V" for s in "VE"}, "F": "F", **{s: "Q" for s in "/fQ"}}
TASKS = {"nv": ("N", "V"), "nsv": ("N", "S", "V")}
WIN = (-90, 144)              # R 기준 [-250, +400] ms @ 360 Hz → 234 표본
BEAT_LEN = WIN[1] - WIN[0]
SEEDS = (0, 1, 2, 3, 4)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------- 라벨
def beat_labels(mitdb, record):
    """기록의 박동 위치와 AAMI 클래스. `build.py` 와 **같은 필터**를 쓴다.

    저장된 `rpeaks` 가 이 위치를 분절 좌표로 옮긴 것과 정확히 일치하는지는
    `_check_alignment` 가 확인한다 — 어긋나면 라벨이 엉뚱한 박동에 붙는다.
    """
    ann = wfdb.rdann(os.path.join(mitdb, record), "atr")
    keep = [i for i, s in enumerate(ann.symbol) if s in BEAT_SYMBOLS]
    return ann.sample[keep], [AAMI[ann.symbol[i]] for i in keep]


def _check_alignment(ds, mitdb, n=50):
    """분절에 저장된 rpeaks 와 주석에서 유도한 위치가 같은지 확인한다."""
    by_rec = {}
    bad = 0
    for i in range(min(n, len(ds))):
        m = ds.meta[i]
        r = m["record_id"]
        if r not in by_rec:
            by_rec[r] = beat_labels(mitdb, r)[0]
        pos = by_rec[r]
        s0 = m["seg_idx"] * SEG
        want = pos[(pos >= s0) & (pos < s0 + SEG)] - s0
        if not np.array_equal(np.sort(ds.rpeaks[i]), np.sort(want)):
            bad += 1
    if bad:
        raise SystemExit(f"[08] 박동 위치가 {bad}/{min(n, len(ds))} 분절에서 어긋난다. "
                         "라벨이 엉뚱한 박동에 붙으므로 진행하지 않는다.")


# ---------------------------------------------------------------- 신호
@torch.no_grad()
def denoise(model, ds, device, batch=100):
    """B 재구성 — 04·06 과 **같은 계산**이다 (잡음 인코딩 3개를 0으로)."""
    pad = model.pad_each
    k_clean, K = 0, model.n_encoders
    k_noise = [k for k in range(K) if k != k_clean]
    out = np.zeros((len(ds), ds.x_noisy.shape[1]), dtype=np.float32)
    for s in range(0, len(ds), batch):
        j = np.arange(s, min(s + batch, len(ds)))
        x = meae.pad(ds.tensor(j).to(device), pad)
        y = model.masked_reconstruct(x, k_noise)
        out[j] = meae.crop(y, pad).squeeze(1).cpu().numpy()
        if s % 2000 == 0:
            print(f"    {s}/{len(ds)}", flush=True)
    return out


def cut_beats(sig, ds, mitdb, classes):
    """(박동, 234) 배열과 라벨·기록. 창이 분절 밖으로 나가면 그 박동은 버린다.

    두 입력이 **같은 박동 집합**을 보도록 잘라내기는 신호와 무관하게 정해진다.
    """
    cache, X, y, rec = {}, [], [], []
    for i in range(len(ds)):
        m = ds.meta[i]
        r = m["record_id"]
        if r not in cache:
            cache[r] = beat_labels(mitdb, r)
        pos, lab = cache[r]
        s0 = m["seg_idx"] * SEG
        for p, c in zip(pos, lab):
            if not (s0 <= p < s0 + SEG) or c not in classes:
                continue
            q = p - s0
            a, b = q + WIN[0], q + WIN[1]
            if a < 0 or b > SEG:                 # 창이 분절 경계를 넘는다
                continue
            X.append(sig[i, a:b])
            y.append(classes.index(c))
            rec.append(r)
    return (np.asarray(X, np.float32), np.asarray(y, np.int64), np.asarray(rec))


# ---------------------------------------------------------------- 분류기
class ResBlock(nn.Module):
    """Kachuse et al. 2018 의 잔차 블록 — conv-relu-conv + skip, 그리고 풀링."""

    def __init__(self, ch, k=5):
        super().__init__()
        self.c1 = nn.Conv1d(ch, ch, k, padding=k // 2)
        self.c2 = nn.Conv1d(ch, ch, k, padding=k // 2)
        self.act = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool1d(5, stride=2, padding=2)

    def forward(self, x):
        h = self.c2(self.act(self.c1(x)))
        return self.pool(self.act(h + x))


class BeatNet(nn.Module):
    """1-D CNN 기준선. **두 입력이 똑같이 쓰는 구조**라 비교의 통제 변인이다."""

    def __init__(self, n_class, ch=32, n_block=5, length=BEAT_LEN):
        super().__init__()
        self.stem = nn.Conv1d(1, ch, 5, padding=2)
        self.blocks = nn.Sequential(*[ResBlock(ch) for _ in range(n_block)])
        with torch.no_grad():                       # 평탄화 길이를 구조에서 유도한다
            n = self.blocks(self.stem(torch.zeros(1, 1, length))).numel()
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(n, 32), nn.ReLU(True),
                                  nn.Linear(32, n_class))

    def forward(self, x):
        return self.head(self.blocks(self.stem(x)))


def _metrics(y, p, n_class):
    """클래스별 정밀도·재현율·F1 과 macro-F1. 정확도는 참고로만 싣는다."""
    cm = np.zeros((n_class, n_class), dtype=np.int64)
    for t, q in zip(y, p):
        cm[t, q] += 1
    prec, rec, f1 = [], [], []
    for c in range(n_class):
        tp, fp, fn = cm[c, c], cm[:, c].sum() - cm[c, c], cm[c].sum() - cm[c, c]
        pr = tp / (tp + fp) if tp + fp else 0.0
        rc = tp / (tp + fn) if tp + fn else 0.0
        prec.append(pr)
        rec.append(rc)
        f1.append(2 * pr * rc / (pr + rc) if pr + rc else 0.0)
    return {"macro_F1": float(np.mean(f1)), "정확도": float(np.trace(cm) / cm.sum()),
            "precision": prec, "recall": rec, "f1": f1, "cm": cm}


def train_one(Xtr, ytr, Xva, yva, Xte, yte, n_class, seed, device,
              epochs=30, batch=256, lr=1e-3):
    """한 입력·한 시드. val macro-F1 이 가장 높은 에폭의 가중치로 test 를 잰다."""
    set_seed(seed)
    model = BeatNet(n_class).to(device)
    # 클래스 가중치 = 역빈도. 두 입력에 **같은 규칙**을 쓴다 (같은 박동 집합이므로 값도 같다)
    cnt = np.bincount(ytr, minlength=n_class).astype(np.float64)
    w = torch.tensor(cnt.sum() / (n_class * np.maximum(cnt, 1)), dtype=torch.float32)
    crit = nn.CrossEntropyLoss(weight=w.to(device))
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    def as_t(a):
        return torch.from_numpy(a).unsqueeze(1)

    Xtr_t, ytr_t = as_t(Xtr), torch.from_numpy(ytr)
    rng = np.random.default_rng(seed)
    best, best_state = -1.0, None
    for ep in range(1, epochs + 1):
        model.train()
        order = rng.permutation(len(Xtr))
        for s in range(0, len(order), batch):
            j = order[s:s + batch]
            opt.zero_grad(set_to_none=True)
            loss = crit(model(Xtr_t[j].to(device)), ytr_t[j].to(device))
            loss.backward()
            opt.step()
        va = _metrics(yva, predict(model, Xva, device), n_class)["macro_F1"]
        if va > best:
            best = va
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    out = _metrics(yte, predict(model, Xte, device), n_class)
    out["val_macro_F1"] = best
    return out


@torch.no_grad()
def predict(model, X, device, batch=1024):
    model.eval()
    p = np.empty(len(X), dtype=np.int64)
    for s in range(0, len(X), batch):
        x = torch.from_numpy(X[s:s + batch]).unsqueeze(1).to(device)
        p[s:s + batch] = model(x).argmax(1).cpu().numpy()
    return p


# ---------------------------------------------------------------- 그림
def fig_compare(runs, classes, out, run, task, n_te):
    """입력 간 성능 — 막대는 시드 평균, 점은 시드별 값."""
    keys = ["macro_F1"] + [f"recall_{c}" for c in classes]
    names = ["macro-F1"] + [f"{c} 재현율" for c in classes]
    inputs = ["x_noisy(처리 전)", "x_denoised(처리 후)"]
    col = {inputs[0]: "#d9d9d9", inputs[1]: "#a8c8a0"}
    fig, ax = plt.subplots(1, len(keys), figsize=(2.5 * len(keys) + 1, 4.2))
    ax = np.atleast_1d(ax)
    for a, k, nm in zip(ax, keys, names):
        for i, inp in enumerate(inputs):
            v = runs[runs["입력"] == inp][k].to_numpy()
            a.bar(i, v.mean(), yerr=v.std(ddof=1) if len(v) > 1 else 0,
                  width=.6, color=col[inp], edgecolor="#333", lw=.9, capsize=4)
            a.plot(np.full(len(v), i), v, "o", ms=4, color="#333", alpha=.7)
        a.set_xticks([0, 1])
        a.set_xticklabels(["처리 전", "처리 후"], fontsize=8.5)
        a.set_xlim(-.6, 1.6)
        d = runs[runs["입력"] == inputs[1]][k].mean() - runs[runs["입력"] == inputs[0]][k].mean()
        a.set_title(f"{nm}{NL}차 {d:+.4f}", fontsize=9.5, loc="left")
        a.grid(alpha=.3, lw=.4, axis="y")
        a.tick_params(labelsize=8)
    fig.suptitle(f"[08 하류 분류] {run} · {task} · test 박동 {n_te:,}개 · 시드 "
                 f"{len(SEEDS)}개{NL}같은 구조를 입력만 바꿔 따로 학습했다 · "
                 "막대는 시드 평균±SD, 점은 시드별 값", fontsize=11)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, bbox_inches="tight", dpi=130)
    plt.close(fig)


def fig_confusion(cms, classes, out, run, task):
    """혼동행렬 — 시드 합. 행이 참, 열이 예측이고 행별 비율로 칠한다."""
    fig, ax = plt.subplots(1, len(cms), figsize=(4.2 * len(cms), 3.9))
    ax = np.atleast_1d(ax)
    for a, (name, cm) in zip(ax, cms.items()):
        frac = cm / np.maximum(cm.sum(1, keepdims=True), 1)
        a.imshow(frac, cmap="Greens", vmin=0, vmax=1)
        for i in range(len(classes)):
            for j in range(len(classes)):
                a.text(j, i, f"{cm[i, j]:,}{NL}{frac[i, j] * 100:.1f}%",
                       ha="center", va="center", fontsize=8.5,
                       color="#fff" if frac[i, j] > .55 else "#222")
        a.set_xticks(range(len(classes)), classes)
        a.set_yticks(range(len(classes)), classes)
        a.set_xlabel("예측", fontsize=9)
        a.set_ylabel("참", fontsize=9)
        a.set_title(name, fontsize=10, loc="left")
    fig.suptitle(f"[08 혼동행렬] {run} · {task} · 시드 {len(SEEDS)}개 합 · "
                 "칸의 색은 행별 비율", fontsize=11)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight", dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------- 본체
def main(config="configs/default.yaml", run="C16_seed42", task="nv", outdir=None,
         epochs=30, seeds=SEEDS):
    cfg = load_cfg(config)
    classes = list(TASKS[task])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    outdir = outdir or os.path.join("results", "08_prediction", run, task)
    os.makedirs(os.path.join(outdir, "figures"), exist_ok=True)
    mitdb = cfg["paths"]["mitdb"]

    model, ck = load_ckpt(cfg, run)
    model = model.to(device).eval()
    print(f"[08] {run} (에폭 {ck['epoch']}) · 과제 {task} ({'/'.join(classes)}) · "
          f"시드 {len(seeds)}개 · device={device}")

    data, brows = {}, []
    for sp in ("train", "val", "test"):
        ds = load_split(cfg, sp)
        _check_alignment(ds, mitdb)
        print(f"  [{sp}] {len(ds)}분절 — B 재구성")
        den = denoise(model, ds, device)
        noisy = ds.x_noisy.astype(np.float32)
        cut = {"x_noisy(처리 전)": cut_beats(noisy, ds, mitdb, classes),
               "x_denoised(처리 후)": cut_beats(den, ds, mitdb, classes)}
        # 두 입력이 같은 박동을 봐야 비교가 성립한다
        a, b = cut["x_noisy(처리 전)"], cut["x_denoised(처리 후)"]
        assert np.array_equal(a[1], b[1]) and np.array_equal(a[2], b[2]), \
            f"[08] {sp} 의 두 입력이 다른 박동 집합을 본다"
        data[sp] = cut
        c = np.bincount(a[1], minlength=len(classes))
        brows.append({"split": sp, "분절수": len(ds), "박동수": int(c.sum()),
                      **{cl: int(n) for cl, n in zip(classes, c)}})
        print("    " + " · ".join(f"{cl} {n:,}" for cl, n in zip(classes, c)))

    beats = pd.DataFrame(brows)
    beats.to_csv(f"{outdir}/beats.csv", index=False, encoding="utf-8-sig")

    rows, cms = [], {}
    for inp in ("x_noisy(처리 전)", "x_denoised(처리 후)"):
        Xtr, ytr, _ = data["train"][inp]
        Xva, yva, _ = data["val"][inp]
        Xte, yte, _ = data["test"][inp]
        acc = np.zeros((len(classes), len(classes)), dtype=np.int64)
        for sd in seeds:
            m = train_one(Xtr, ytr, Xva, yva, Xte, yte, len(classes), sd, device,
                          epochs=epochs)
            acc += m["cm"]
            rows.append({"입력": inp, "시드": sd, "macro_F1": m["macro_F1"],
                         "val_macro_F1": m["val_macro_F1"], "정확도": m["정확도"],
                         **{f"recall_{c}": m["recall"][i] for i, c in enumerate(classes)},
                         **{f"precision_{c}": m["precision"][i] for i, c in enumerate(classes)},
                         **{f"f1_{c}": m["f1"][i] for i, c in enumerate(classes)}})
            print(f"  [{inp}] seed {sd}  macro-F1 {m['macro_F1']:.4f}  "
                  + " · ".join(f"{c} 재현율 {m['recall'][i]:.4f}"
                               for i, c in enumerate(classes)), flush=True)
        cms[inp] = acc

    runs = pd.DataFrame(rows)
    runs.round(5).to_csv(f"{outdir}/runs.csv", index=False, encoding="utf-8-sig")

    num = [c for c in runs.columns if c not in ("입력", "시드")]
    g = runs.groupby("입력")[num]
    summ = g.mean().add_suffix("_평균").join(g.std(ddof=1).add_suffix("_SD")).reset_index()
    a, b = "x_noisy(처리 전)", "x_denoised(처리 후)"
    diff = {"입력": "차이 (처리 후 − 처리 전)"}
    for c in num:
        diff[f"{c}_평균"] = float(g.mean().loc[b, c] - g.mean().loc[a, c])
    summ = pd.concat([summ, pd.DataFrame([diff])], ignore_index=True)
    summ.round(5).to_csv(f"{outdir}/summary.csv", index=False, encoding="utf-8-sig")

    pc = []
    for inp in cms:
        for i, c in enumerate(classes):
            s = runs[runs["입력"] == inp]
            pc.append({"입력": inp, "클래스": c, "test_박동수": int(cms[inp][i].sum() / len(seeds)),
                       "정밀도": s[f"precision_{c}"].mean(),
                       "재현율": s[f"recall_{c}"].mean(), "F1": s[f"f1_{c}"].mean()})
    pd.DataFrame(pc).round(5).to_csv(f"{outdir}/per_class.csv", index=False,
                                     encoding="utf-8-sig")
    crows = []
    for inp, cm in cms.items():
        for i, c in enumerate(classes):
            crows.append({"입력": inp, "참": c,
                          **{f"예측_{q}": int(cm[i, j]) for j, q in enumerate(classes)}})
    pd.DataFrame(crows).to_csv(f"{outdir}/confusion.csv", index=False,
                               encoding="utf-8-sig")

    n_te = int(beats[beats["split"] == "test"]["박동수"].iloc[0])
    fig_compare(runs, classes, f"{outdir}/figures/compare.png", run, task, n_te)
    fig_confusion(cms, classes, f"{outdir}/figures/confusion.png", run, task)
    _note(outdir, run, ck, task, classes, beats, seeds, epochs)

    pd.set_option("display.width", 250)
    print("")
    print(f"=== 08 하류 분류 — {task} ({'/'.join(classes)}) · test 박동 {n_te:,}개 ===")
    show = ["입력", "macro_F1_평균", "macro_F1_SD"] + \
           [f"recall_{c}_평균" for c in classes]
    print(summ[show].round(4).to_string(index=False))
    print("")
    print("클래스별 (시드 평균)")
    print(pd.DataFrame(pc).round(4).to_string(index=False))
    print(f"{NL}산출물 → {outdir}/")
    return runs, summ, beats


def _note(outdir, run, ck, task, classes, beats, seeds, epochs):
    with open(f"{outdir}/note.txt", "w", encoding="utf-8") as f:
        f.write(
            f"08 하류 분류 — {run} (에폭 {ck['epoch']}), 과제 {task} "
            f"({'/'.join(classes)}).\n\n"
            "같은 분류기를 입력만 바꿔 두 번 따로 학습하고 성능을 견준다.\n"
            "  x_noisy    (처리 전)\n"
            "  x_denoised (처리 후) = B 재구성. 04·06 과 같은 계산이다.\n"
            "두 경로는 완전히 독립이다 - 구조·초기화·최적화·에폭·시드·분할·박동 창이\n"
            "모두 같고 입력 신호만 다르다. 가중치를 공유하지 않는다.\n\n"
            "차용한 규약\n"
            "  AAMI EC57 5분류와 환자 간 분할   de Chazal et al. 2004\n"
            "  1-D CNN(잔차 블록 5개 + FC 2단)  Kachuee et al. 2018,\n"
            "                                   doi:10.1109/ICHI.2018.00092\n\n"
            "표준 DS1/DS2(22/22) 분할은 쓰지 않았다.\n"
            "  디노이저가 train 32기록을 보고 학습했는데 DS2 에 그 기록이 섞이면\n"
            "  x_denoised 가 '이미 본 기록'이라 부당하게 유리해진다. 입력 간 비교라는\n"
            "  이 실험의 목적이 깨지므로 디노이저의 기록 분할을 그대로 물려받았다.\n\n"
            "F·Q 를 뺀 이유\n"
            "  test 9기록에서 F 12박 · Q 2박이다. 재현율이 0 아니면 1 로 튀어\n"
            "  수치에 뜻이 없다. S 도 111박 중 94박이 기록 220 하나에 몰려 있어\n"
            "  --task nsv 에서는 클래스별 재현율로만 읽는다.\n\n"
            "박동 잘라내기\n"
            f"  기준점은 주석의 R-피크다(검출기가 아니다).\n"
            f"  창 = R + [{WIN[0]}, {WIN[1]}] 표본 = [-250, +400] ms @ 360 Hz\n"
            f"  = {BEAT_LEN} 표본. 분절(10초) 밖으로 나가는 박동은 뺐다.\n"
            "  잘라내기는 신호와 무관하게 정해지므로 두 입력이 같은 박동을 본다\n"
            "  (라벨·기록 배열이 같은지 실행 중에 확인한다).\n"
            "  정규화하지 않았다 - 표준화하면 진폭 차이가 지워진다.\n\n"
            "학습\n"
            f"  Adam lr 1e-3, batch 256, {epochs}에폭.\n"
            "  클래스 가중치 = 역빈도. 두 입력에 같은 규칙을 쓴다.\n"
            "  val macro-F1 이 가장 높은 에폭의 가중치로 test 를 잰다.\n"
            f"  시드 {list(seeds)} 를 돌려 평균±SD 로 낸다.\n\n"
            "읽는 법\n"
            "  클래스가 치우쳐 있어 정확도는 보지 않는다 (N 만 찍어도 94%다).\n"
            "  주 지표는 macro-F1 이고 소수 클래스의 재현율을 함께 싣는다.\n\n"
            "한계\n"
            "  디노이저는 분류기의 train 기록을 보고 학습했으므로 그 구간의\n"
            "  x_denoised 는 '본 기록'이고 test 구간은 '못 본 기록'이다.\n"
            "  분류기 쪽에서는 학습·평가의 입력 분포가 그만큼 어긋난다.\n"
            "  중첩 분할로 없앨 수 있으나 이 실험의 범위 밖이다.\n\n"
            "박동 수\n")
        f.write(beats.to_string(index=False) + "\n\n")
        f.write("값에 대한 해석과 판정은 붙이지 않는다.\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--run", default="C16_seed42")
    p.add_argument("--task", default="nv", choices=sorted(TASKS),
                   help="nv = N vs V 이진(기본) · nsv = N/S/V 3분류")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--seeds", type=int, nargs="*", default=list(SEEDS))
    p.add_argument("--outdir", default=None)
    a = p.parse_args()
    main(a.config, a.run, a.task, a.outdir, a.epochs, tuple(a.seeds))
