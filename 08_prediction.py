"""08 — 하류 과제. 디노이징이 부정맥 분류에 도움이 되는가.

    python 08_prediction.py --run C16_seed42

같은 분류기를 **입력만 바꿔** 두 번 따로 학습하고, 두 모델의 성능을 견준다.
디노이징 자체의 지표(04·06)가 아니라 **그 신호를 받아 쓰는 쪽**에서 본 값이다.

    x_noisy    → 1-D CNN → 성능 A          (처리 전)
    x_denoised → 1-D CNN → 성능 B          (처리 후, 04·06 의 B 재구성과 같은 신호)

**두 경로는 완전히 독립이다.** 구조·초기화·최적화·에폭·시드·분할·박동 창이 모두 같고
입력 신호만 다르다. 가중치를 공유하지 않고, 한쪽 결과를 다른 쪽에 쓰지 않는다.

════════════════════════════════════════════════════════════════════════
읽는 법 — 절대 성능이 아니라 **차이**를 본다
════════════════════════════════════════════════════════════════════════
**이 수치를 다른 논문의 부정맥 분류 성능과 비교하지 않는다.** 분할이 다르고(표준
DS1/DS2 가 아니라 디노이저의 기록 분할) 3분류이므로 절대 성능은 뜻이 없다.
여기서 묻는 것은 하나다 — **다른 조건을 모두 고정하고 입력만 바꾸면 얼마나 달라지는가.**

차이가 작을 때 "이 정도가 의미 있는가"라는 물음에 답하려고 **McNemar 검정**을 함께 낸다.
같은 test 박동에 대한 두 모델의 정오를 짝지어 보는 검정이라 이 설계에 정확히 맞는다.

════════════════════════════════════════════════════════════════════════
분류 체계 — AAMI 권고를 따르되 N·S·V 로 제한한다
════════════════════════════════════════════════════════════════════════
AAMI EC57 은 박동을 N·S·V·F·Q 다섯으로 나눈다(de Chazal et al. 2004). 우리는 그 권고에
따라 박동을 분류하되, **사용 기록에서 F·Q 박동이 통계적 평가에 불충분해 N·S·V 로
제한한다.** 두 클래스가 빠지는 사정은 서로 다르다.

  Q  MIT-BIH 전체 8,043박 중 8,010박이 페이스메이커 박동(`/` 7,028 · `f` 982)이고
     순수 분류 불가(`Q`)는 33박뿐이다. AAMI EC57 은 페이스메이커 기록 102·104·107·217
     을 평가에서 제외하도록 권고하며 de Chazal 의 DS1/DS2 도 그 넷을 뺀 44기록이다.
     **Q 제외는 표준을 따른 결과다.**  (우리 분할: train 3,893 · val 0 · test 2)

  F  MIT-BIH 전체 803박 중 735박(91.5%)이 기록 208(373)·213(362) 두 개에 몰려 있고,
     그 둘이 모두 우리 train 에 들어갔다. **분할의 한계다.**
     (우리 분할: train 784 · val 7 · test 12)

AAMI 매핑 (MIT-BIH 주석 기호 → 클래스). 48기록 전 기호를 열거해 확인했다.
     N ← N · L · R · j · e          90,631박
     S ← A · a · J · S               2,781박
     V ← V · E                       7,236박
     F ← F                             803박   (제외)
     Q ← / · f · Q                   8,043박   (제외)

════════════════════════════════════════════════════════════════════════
차용한 규약
════════════════════════════════════════════════════════════════════════
  AAMI EC57 분류 · 환자 간 분할   de Chazal et al. 2004
  1-D CNN(잔차 블록 5개 + FC 2단)  Kachuee et al. 2018 (ICHI,
                                   doi:10.1109/ICHI.2018.00092)
                                   ※ 원 논문은 환자 내 분할이다. 구조만 가져왔다.

**표준 DS1/DS2(22/22)는 쓰지 않는다.** 디노이저가 우리 train 32기록을 보고 학습했는데
DS2 에 그 기록이 섞이면 x_denoised 가 "이미 본 기록"이라 부당하게 유리해져 입력 간
비교라는 이 실험의 목적이 깨진다. **디노이저의 기록 분할을 그대로 물려받는다.**

════════════════════════════════════════════════════════════════════════
박동 잘라내기
════════════════════════════════════════════════════════════════════════
기준점은 **주석의 R-피크**다(검출기가 아니다). 신호마다 검출을 다시 하면 "박동을
찾았는가"와 "찾은 자리의 파형이 분류에 쓸 만한가"가 섞인다. 06 과 같은 설계다.

    창 = R + [-250 ms, +400 ms] = 360 Hz 에서 [-90, +144] → 234 표본
    분절(10초) 밖으로 나가는 박동은 뺀다 — 두 입력에서 **같은 박동만** 남는다.

정규화는 하지 않는다. 디노이저가 mV 원값을 그대로 받도록 학습됐고, 여기서 표준화하면
"진폭이 얼마나 살아났는가"라는 차이가 지워진다.

────────────────────────────────────────────────────────────────────────
산출물  results/08_prediction/<run>/
────────────────────────────────────────────────────────────────────────
  beats.csv       split × 클래스 박동 수
  runs.csv        입력 × 시드 × 지표 (원값)
  summary.csv     입력별 정확도 평균±SD 와 두 입력의 차이
  per_class.csv   클래스별 **민감도·양성예측도** (시드 평균±SD)
  confusion.csv   입력별 혼동행렬 (시드 다수결 예측)
  mcnemar.csv     McNemar 검정 — 시드 다수결과 시드별
  note.txt
  figures/compare.png     정확도와 클래스별 민감도·양성예측도
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
CLASSES = ("N", "S", "V")     # AAMI 권고를 따르되 F·Q 는 제외 (머리말 참조)
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
    """정확도와 클래스별 **민감도·양성예측도**. 혼동행렬도 함께 돌려준다.

        민감도(sensitivity, recall)     = TP / (TP + FN)   그 클래스를 얼마나 잡아내나
        양성예측도(PPV, precision)      = TP / (TP + FP)   잡았다고 한 것이 얼마나 맞나

    임상 문헌의 관례를 따라 재현율·정밀도 대신 민감도·양성예측도로 부른다.
    """
    cm = np.zeros((n_class, n_class), dtype=np.int64)
    for t, q in zip(y, p):
        cm[t, q] += 1
    sens, ppv = [], []
    for c in range(n_class):
        tp = cm[c, c]
        fn = cm[c].sum() - tp
        fp = cm[:, c].sum() - tp
        sens.append(tp / (tp + fn) if tp + fn else 0.0)
        ppv.append(tp / (tp + fp) if tp + fp else 0.0)
    return {"정확도": float(np.trace(cm) / cm.sum()), "민감도": sens,
            "양성예측도": ppv, "cm": cm}


def mcnemar(y, pa, pb):
    """McNemar 검정 — 같은 박동에 대한 두 모델의 정오를 짝지어 본다.

    b = A 만 맞힌 박동 수, c = B 만 맞힌 박동 수. 둘 다 맞히거나 둘 다 틀린 박동은
    두 모델을 가르지 못하므로 검정에서 빠진다 — 그래서 **짝지은** 검정이다.

    b+c 가 작으면 정확 이항검정, 크면 연속성 보정 카이제곱을 쓴다 (관례).
    귀무가설: 두 모델의 오류율이 같다.
    """
    from scipy import stats
    ok_a, ok_b = (pa == y), (pb == y)
    b = int(np.sum(ok_a & ~ok_b))
    c = int(np.sum(~ok_a & ok_b))
    if b + c == 0:
        return {"b_A만맞음": b, "c_B만맞음": c, "검정": "—", "통계량": np.nan, "p": 1.0}
    if b + c < 25:
        p = float(stats.binomtest(b, b + c, 0.5).pvalue)
        return {"b_A만맞음": b, "c_B만맞음": c, "검정": "정확 이항",
                "통계량": np.nan, "p": p}
    chi2 = (abs(b - c) - 1) ** 2 / (b + c)
    return {"b_A만맞음": b, "c_B만맞음": c, "검정": "카이제곱(연속성 보정)",
            "통계량": float(chi2), "p": float(stats.chi2.sf(chi2, 1))}


def vote(preds):
    """시드별 예측의 다수결. 동수면 작은 클래스 번호를 쓴다(결정론적)."""
    P = np.stack(preds)
    return np.array([np.bincount(P[:, i]).argmax() for i in range(P.shape[1])])


def train_one(Xtr, ytr, Xva, yva, Xte, yte, n_class, seed, device,
              epochs=30, batch=256, lr=1e-3):
    """한 입력·한 시드. val 정확도가 가장 높은 에폭의 가중치로 test 를 잰다.

    반환에 **test 예측**을 함께 담는다 — McNemar 가 짝지을 대상이다.
    """
    set_seed(seed)
    model = BeatNet(n_class).to(device)
    # 클래스 가중치 = 역빈도. 두 입력에 **같은 규칙**을 쓴다 (같은 박동 집합이라 값도 같다)
    cnt = np.bincount(ytr, minlength=n_class).astype(np.float64)
    w = torch.tensor(cnt.sum() / (n_class * np.maximum(cnt, 1)), dtype=torch.float32)
    crit = nn.CrossEntropyLoss(weight=w.to(device))
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    Xtr_t, ytr_t = torch.from_numpy(Xtr).unsqueeze(1), torch.from_numpy(ytr)
    rng = np.random.default_rng(seed)
    best, best_state = -1.0, None
    for _ in range(epochs):
        model.train()
        order = rng.permutation(len(Xtr))
        for s in range(0, len(order), batch):
            j = order[s:s + batch]
            opt.zero_grad(set_to_none=True)
            crit(model(Xtr_t[j].to(device)), ytr_t[j].to(device)).backward()
            opt.step()
        va = _metrics(yva, predict(model, Xva, device), n_class)["정확도"]
        if va > best:
            best = va
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    pred = predict(model, Xte, device)
    out = _metrics(yte, pred, n_class)
    out["val_정확도"] = best
    out["pred"] = pred
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
def fig_compare(runs, classes, out, run, n_te, mc):
    """정확도와 클래스별 민감도·양성예측도. 막대는 시드 평균, 점은 시드별 값."""
    A, B = "x_noisy(처리 전)", "x_denoised(처리 후)"
    col = {A: "#d9d9d9", B: "#a8c8a0"}
    panels = ([("정확도", "정확도")]
              + [(f"{c} 민감도", f"민감도_{c}") for c in classes]
              + [(f"{c} 양성예측도", f"PPV_{c}") for c in classes])
    fig, ax = plt.subplots(1, len(panels), figsize=(1.9 * len(panels) + 1, 4.4))
    ax = np.atleast_1d(ax)
    for a_, (nm, k) in zip(ax, panels):
        for i, inp in enumerate((A, B)):
            v = runs[runs["입력"] == inp][k].to_numpy()
            a_.bar(i, v.mean(), yerr=v.std(ddof=1) if len(v) > 1 else 0, width=.6,
                   color=col[inp], edgecolor="#333", lw=.9, capsize=4)
            a_.plot(np.full(len(v), i), v, "o", ms=4, color="#333", alpha=.7)
        d = runs[runs["입력"] == B][k].mean() - runs[runs["입력"] == A][k].mean()
        a_.set_xticks([0, 1])
        a_.set_xticklabels(["전", "후"], fontsize=9)
        a_.set_xlim(-.6, 1.6)
        a_.set_ylim(0, 1.05)
        a_.set_title(f"{nm}{NL}차 {d:+.4f}", fontsize=9, loc="left")
        a_.grid(alpha=.3, lw=.4, axis="y")
        a_.tick_params(labelsize=8)
    fig.suptitle(f"[08 하류 분류 N/S/V] {run} · test 박동 {n_te:,}개 · "
                 f"시드 {len(SEEDS)}개{NL}"
                 "같은 구조를 입력만 바꿔 따로 학습했다 · "
                 f"McNemar p = {mc['p']:.3g} "
                 f"(전만 맞힘 {mc['b_A만맞음']:,} · 후만 맞힘 {mc['c_B만맞음']:,}){NL}"
                 "다른 논문의 성능과 비교하지 않는다 — 분할이 다르고 3분류다",
                 fontsize=10.5)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, bbox_inches="tight", dpi=130)
    plt.close(fig)


def fig_confusion(cms, classes, out, run):
    """혼동행렬 — 시드 다수결 예측. 행이 참, 열이 예측이고 행별 비율로 칠한다."""
    fig, ax = plt.subplots(1, len(cms), figsize=(4.2 * len(cms), 3.9))
    ax = np.atleast_1d(ax)
    for a_, (name, cm) in zip(ax, cms.items()):
        frac = cm / np.maximum(cm.sum(1, keepdims=True), 1)
        a_.imshow(frac, cmap="Greens", vmin=0, vmax=1)
        for i in range(len(classes)):
            for j in range(len(classes)):
                a_.text(j, i, f"{cm[i, j]:,}{NL}{frac[i, j] * 100:.1f}%",
                        ha="center", va="center", fontsize=8.5,
                        color="#fff" if frac[i, j] > .55 else "#222")
        a_.set_xticks(range(len(classes)), classes)
        a_.set_yticks(range(len(classes)), classes)
        a_.set_xlabel("예측", fontsize=9)
        a_.set_ylabel("참", fontsize=9)
        a_.set_title(name, fontsize=10, loc="left")
    fig.suptitle(f"[08 혼동행렬 N/S/V] {run} · 시드 {len(SEEDS)}개 다수결 · "
                 "칸의 색은 행별 비율", fontsize=11)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight", dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------- 본체
def main(config="configs/default.yaml", run="C16_seed42", outdir=None,
         epochs=30, seeds=SEEDS):
    cfg = load_cfg(config)
    classes = list(CLASSES)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    outdir = outdir or os.path.join("results", "08_prediction", run)
    os.makedirs(os.path.join(outdir, "figures"), exist_ok=True)
    mitdb = cfg["paths"]["mitdb"]
    A, B = "x_noisy(처리 전)", "x_denoised(처리 후)"

    model, ck = load_ckpt(cfg, run)
    model = model.to(device).eval()
    print(f"[08] {run} (에폭 {ck['epoch']}) · N/S/V 3분류 · "
          f"시드 {len(seeds)}개 · device={device}")

    data, brows = {}, []
    for sp in ("train", "val", "test"):
        ds = load_split(cfg, sp)
        _check_alignment(ds, mitdb)
        print(f"  [{sp}] {len(ds)}분절 — B 재구성")
        den = denoise(model, ds, device)
        cut = {A: cut_beats(ds.x_noisy.astype(np.float32), ds, mitdb, classes),
               B: cut_beats(den, ds, mitdb, classes)}
        # 두 입력이 같은 박동을 봐야 비교가 성립한다
        same = (np.array_equal(cut[A][1], cut[B][1])
                and np.array_equal(cut[A][2], cut[B][2]))
        if not same:
            raise SystemExit(f"[08] {sp} 의 두 입력이 다른 박동 집합을 본다")
        data[sp] = cut
        c = np.bincount(cut[A][1], minlength=len(classes))
        brows.append({"split": sp, "기록수": len(set(cut[A][2])), "분절수": len(ds),
                      "박동수": int(c.sum()),
                      **{cl: int(n) for cl, n in zip(classes, c)}})
        print("    " + " · ".join(f"{cl} {n:,}" for cl, n in zip(classes, c)))

    beats = pd.DataFrame(brows)
    beats.to_csv(f"{outdir}/beats.csv", index=False, encoding="utf-8-sig")

    yte = data["test"][A][1]
    rows, preds = [], {}
    for inp in (A, B):
        Xtr, ytr, _ = data["train"][inp]
        Xva, yva, _ = data["val"][inp]
        Xte, _, _ = data["test"][inp]
        preds[inp] = []
        for sd in seeds:
            m = train_one(Xtr, ytr, Xva, yva, Xte, yte, len(classes), sd, device,
                          epochs=epochs)
            preds[inp].append(m["pred"])
            rows.append({"입력": inp, "시드": sd, "정확도": m["정확도"],
                         "val_정확도": m["val_정확도"],
                         **{f"민감도_{c}": m["민감도"][i]
                            for i, c in enumerate(classes)},
                         **{f"PPV_{c}": m["양성예측도"][i]
                            for i, c in enumerate(classes)}})
            print(f"  [{inp}] seed {sd}  정확도 {m['정확도']:.4f}  "
                  + " · ".join(f"{c} 민감도 {m['민감도'][i]:.4f}"
                               for i, c in enumerate(classes)), flush=True)

    runs = pd.DataFrame(rows)
    runs.round(5).to_csv(f"{outdir}/runs.csv", index=False, encoding="utf-8-sig")

    # ---- 시드 다수결 예측으로 혼동행렬과 주 McNemar 를 낸다
    vp = {inp: vote(preds[inp]) for inp in (A, B)}
    cms = {inp: _metrics(yte, vp[inp], len(classes))["cm"] for inp in (A, B)}
    mc = mcnemar(yte, vp[A], vp[B])
    mrows = [{"짝": "시드 다수결", **mc}]
    for i, sd in enumerate(seeds):
        mrows.append({"짝": f"시드 {sd}",
                      **mcnemar(yte, preds[A][i], preds[B][i])})
    pd.DataFrame(mrows).to_csv(f"{outdir}/mcnemar.csv", index=False,
                               encoding="utf-8-sig")

    num = [c for c in runs.columns if c not in ("입력", "시드")]
    g = runs.groupby("입력")[num]
    summ = (g.mean().add_suffix("_평균")
            .join(g.std(ddof=1).add_suffix("_SD")).reset_index())
    diff = {"입력": "차이 (처리 후 − 처리 전)"}
    for c in num:
        diff[f"{c}_평균"] = float(g.mean().loc[B, c] - g.mean().loc[A, c])
    summ = pd.concat([summ, pd.DataFrame([diff])], ignore_index=True)
    summ.round(5).to_csv(f"{outdir}/summary.csv", index=False, encoding="utf-8-sig")

    pc = []
    for inp in (A, B):
        r = runs[runs["입력"] == inp]
        for i, c in enumerate(classes):
            pc.append({"입력": inp, "클래스": c, "test_박동수": int((yte == i).sum()),
                       "민감도": r[f"민감도_{c}"].mean(),
                       "민감도_SD": r[f"민감도_{c}"].std(ddof=1),
                       "양성예측도": r[f"PPV_{c}"].mean(),
                       "양성예측도_SD": r[f"PPV_{c}"].std(ddof=1)})
    pcd = pd.DataFrame(pc)
    pcd.round(5).to_csv(f"{outdir}/per_class.csv", index=False, encoding="utf-8-sig")

    crows = []
    for inp, cm in cms.items():
        for i, c in enumerate(classes):
            crows.append({"입력": inp, "참": c,
                          **{f"예측_{q}": int(cm[i, j])
                             for j, q in enumerate(classes)}})
    pd.DataFrame(crows).to_csv(f"{outdir}/confusion.csv", index=False,
                               encoding="utf-8-sig")

    n_te = int(beats[beats["split"] == "test"]["박동수"].iloc[0])
    fig_compare(runs, classes, f"{outdir}/figures/compare.png", run, n_te, mc)
    fig_confusion(cms, classes, f"{outdir}/figures/confusion.png", run)
    _note(outdir, run, ck, beats, seeds, epochs, mc)

    pd.set_option("display.width", 250)
    print("")
    print(f"=== 08 하류 분류 N/S/V · test 박동 {n_te:,}개 ===")
    print(summ[["입력", "정확도_평균", "정확도_SD"]].round(4).to_string(index=False))
    print("")
    print("클래스별 (시드 평균±SD)")
    print(pcd.round(4).to_string(index=False))
    print("")
    print(f"McNemar (시드 다수결) — {mc['검정']} · "
          f"처리 전만 맞힘 {mc['b_A만맞음']:,} · 처리 후만 맞힘 {mc['c_B만맞음']:,} · "
          f"p = {mc['p']:.3g}")
    print("")
    print("이 수치를 다른 논문의 부정맥 분류 성능과 비교하지 않는다 — "
          "분할이 다르고 3분류다.")
    print(f"산출물 → {outdir}/")
    return runs, summ, pcd, mc


def _note(outdir, run, ck, beats, seeds, epochs, mc):
    L = [
        f"08 하류 분류 — {run} (에폭 {ck['epoch']}), N/S/V 3분류.",
        "",
        "같은 분류기를 입력만 바꿔 두 번 따로 학습하고 성능을 견준다.",
        "  x_noisy    (처리 전)",
        "  x_denoised (처리 후) = B 재구성. 04·06 과 같은 계산이다.",
        "두 경로는 완전히 독립이다 - 구조·초기화·최적화·에폭·시드·분할·박동 창이",
        "모두 같고 입력 신호만 다르다. 가중치를 공유하지 않는다.",
        "",
        "읽는 법",
        "  이 수치를 다른 논문의 부정맥 분류 성능과 비교하지 않는다. 분할이 다르고",
        "  (표준 DS1/DS2 가 아니라 디노이저의 기록 분할) 3분류이므로 절대 성능은",
        "  뜻이 없다. 묻는 것은 하나다 - 다른 조건을 모두 고정하고 입력만 바꾸면",
        "  얼마나 달라지는가.",
        "",
        "분류 체계",
        "  AAMI EC57 권고에 따라 박동을 분류하되, 사용 기록에서 F·Q 박동이",
        "  통계적 평가에 불충분해 N·S·V 로 제한했다.",
        "    Q  MIT-BIH 전체 8,043박 중 8,010박이 페이스메이커 박동(/ 7,028,",
        "       f 982)이고 순수 분류 불가(Q)는 33박뿐이다. AAMI EC57 은 페이스",
        "       기록 102·104·107·217 을 평가에서 제외하도록 권고하며 de Chazal 의",
        "       DS1/DS2 도 그 넷을 뺀 44기록이다. Q 제외는 표준을 따른 결과다.",
        "       (우리 분할: train 3,893 · val 0 · test 2)",
        "    F  MIT-BIH 전체 803박 중 735박(91.5%)이 기록 208(373)·213(362) 두 개에",
        "       몰려 있고 그 둘이 모두 우리 train 에 들어갔다. 분할의 한계다.",
        "       (우리 분할: train 784 · val 7 · test 12)",
        "  AAMI 매핑: N <- N,L,R,j,e / S <- A,a,J,S / V <- V,E / F <- F / Q <- /,f,Q",
        "  48기록의 전 주석 기호를 열거해 매핑을 확인했다.",
        "",
        "차용한 규약",
        "  AAMI EC57 분류·환자 간 분할     de Chazal et al. 2004",
        "  1-D CNN(잔차 블록 5개 + FC 2단)  Kachuee et al. 2018,",
        "                                   doi:10.1109/ICHI.2018.00092",
        "                                   (원 논문은 환자 내 분할. 구조만 가져왔다)",
        "",
        "표준 DS1/DS2(22/22)는 쓰지 않았다.",
        "  디노이저가 train 32기록을 보고 학습했는데 DS2 에 그 기록이 섞이면",
        "  x_denoised 가 '이미 본 기록'이라 부당하게 유리해진다. 입력 간 비교라는",
        "  이 실험의 목적이 깨지므로 디노이저의 기록 분할을 그대로 물려받았다.",
        "",
        "박동 잘라내기",
        "  기준점은 주석의 R-피크다(검출기가 아니다).",
        f"  창 = R + [{WIN[0]}, {WIN[1]}] 표본 = [-250, +400] ms @ 360 Hz "
        f"= {BEAT_LEN} 표본.",
        "  분절(10초) 밖으로 나가는 박동은 뺐다. 잘라내기는 신호와 무관하게",
        "  정해지므로 두 입력이 같은 박동을 본다(실행 중 확인한다).",
        "  정규화하지 않았다 - 표준화하면 진폭 차이가 지워진다.",
        "",
        "학습",
        f"  Adam lr 1e-3, batch 256, {epochs}에폭.",
        "  클래스 가중치 = 역빈도. 두 입력에 같은 규칙을 쓴다.",
        "  val 정확도가 가장 높은 에폭의 가중치로 test 를 잰다.",
        f"  시드 {list(seeds)} 를 돌려 평균±SD 로 낸다.",
        "",
        "McNemar 검정",
        "  같은 test 박동에 대한 두 모델의 정오를 짝지어 본다. 둘 다 맞히거나 둘 다",
        "  틀린 박동은 두 모델을 가르지 못하므로 검정에서 빠진다.",
        "  b = 처리 전만 맞힌 박동, c = 처리 후만 맞힌 박동.",
        "  b+c < 25 면 정확 이항검정, 아니면 연속성 보정 카이제곱.",
        "  귀무가설: 두 모델의 오류율이 같다.",
        f"  시드 다수결 예측 기준 - {mc['검정']}, b {mc['b_A만맞음']}, "
        f"c {mc['c_B만맞음']}, p = {mc['p']:.4g}",
        "  시드별 값은 mcnemar.csv 에 있다.",
        "",
        "한계",
        "  디노이저는 분류기의 train 기록을 보고 학습했으므로 그 구간의 x_denoised 는",
        "  '본 기록'이고 test 구간은 '못 본 기록'이다. 분류기 쪽에서는 학습·평가의",
        "  입력 분포가 그만큼 어긋난다 (clean 과의 |r| — train 0.917 vs test 0.880).",
        "  중첩 분할로 없앨 수 있으나 이 실험의 범위 밖이다.",
        "",
        "박동 수",
        beats.to_string(index=False),
        "",
        "값에 대한 해석과 판정은 붙이지 않는다.",
    ]
    with open(f"{outdir}/note.txt", "w", encoding="utf-8") as f:
        f.write(chr(10).join(L) + chr(10))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--run", default="C16_seed42")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--seeds", type=int, nargs="*", default=list(SEEDS))
    p.add_argument("--outdir", default=None)
    a = p.parse_args()
    main(a.config, a.run, a.outdir, a.epochs, tuple(a.seeds))
