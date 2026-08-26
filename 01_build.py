"""01 — 데이터셋 구축 (RESEARCH_DESIGN.md §4). 완료·동결.

MIT-BIH Arrhythmia(참조 ECG)와 NSTDB(실측 잡음)를 내려받아 검증하고, 기록 단위로 분할한 뒤,
10초 비중첩 분절마다 bw·ma·em 세 잡음을 SNR 무작위로 주입해 실험용 데이터셋을 만든다.

    python 01_build.py                       # 전체 (내려받기 → 분할 → 생성 → 스팟체크)
    python 01_build.py --skip-download

스팟체크만 다시 그릴 때 (데이터 재생성 없음)

    python 01_build.py --spotcheck                      # split 마다 무작위 1개
    python 01_build.py --spotcheck 100_0114 231_0018    # 기록_분절 지정
    python 01_build.py --spotcheck --record 100 --from 110 --n 5
                                                        # 한 기록의 연속 window 5개
    python 01_build.py --appendix                       # 주입 부록만 다시 만들기

산출
  data/processed/{train,val,test}.npz   분절당 x_clean·x_noisy·bw·ma·em·rpeaks·meta
  data/processed/split.json             기록 단위 분할 고정
  results/01_build/                     스팟체크 그림 · 주입 부록

구현은 `src/data/{download,split,build}.py` 에 있다 — S1은 동결이므로 그대로 호출만 한다.
"""
import argparse
import glob
import os
import random

from src.data import build, download, split
from src.data.build import load_cfg
from src.viz import spotcheck

SEGDIR = "data/processed/segments"
MANIFEST = "data/processed/manifest.csv"
OUTDIR = "results/01_build"
NOISES = ("bw", "ma", "em")


def find_seg(name):
    """`<기록>_<분절>` 이름으로 npz 를 찾는다. split 은 기록 단위로 갈리므로 하나만 나온다."""
    hits = glob.glob(os.path.join(SEGDIR, "*", f"{name}.npz"))
    if not hits:
        raise SystemExit(f"[01] 분절 {name} 없음")
    return hits[0], os.path.basename(os.path.dirname(hits[0]))


def appendix(cfg):
    """주입 부록 — **검증이 아니라 기술(記述)**이다. 어떻게 주입되었는지를 그대로 싣는다.

    manifest.csv 에 분절마다 기록된 주입 파라미터(추첨 SNR·배율·잡음 원본 시작점)를
    모아 분포와 요약표로 보여 준다. 합격/불합격 판정은 두지 않는다.
    """
    import pandas as pd
    from src.viz import plt

    os.makedirs(OUTDIR, exist_ok=True)
    m = pd.read_csv(MANIFEST, encoding="utf-8-sig")
    d = cfg["data"]
    lo, hi = d["noise_snr_range_db"]

    # `noise_pool` 은 잡음 조각을 **잡음 원본 레코드(30분)의 어느 시간 구간**에서
    # 가져왔는지를 뜻한다. train 분절과 val·test 분절이 같은 잡음 파형을 보지 않게
    # 앞 70% / 뒤 30% 로 갈라 두었다. 표에는 사람이 읽을 문구로 적는다.
    fs = d["fs"]
    ratio = d["noise_split_ratio"]
    POOL = {"train": f"앞 {ratio:.0%} (train 전용)",
            "eval": f"뒤 {1 - ratio:.0%} (val·test 전용)"}

    # ---- 잡음 원본 레코드 표 — 어디서 가져왔는지의 출처
    nst = cfg["paths"]["nstdb"]
    src_rows = []
    for t in NOISES:
        n_all = int(m[f"start_{t}"].max()) + d["seg_sec"] * fs
        try:
            import wfdb
            n_all = wfdb.rdheader(os.path.join(nst, t)).sig_len
        except Exception:
            pass
        b = int(n_all * ratio)
        src_rows.append({
            "잡음": t, "레코드": f"nstdb/{t}", "채널": f"noise{d['noise_channel'] + 1}",
            "표본수": n_all, "길이_분": round(n_all / fs / 60, 2),
            "경계_표본": b, "경계_분": round(b / fs / 60, 2),
            "train_구간": f"0 ~ {b:,}", "eval_구간": f"{b:,} ~ {n_all:,}"})
    src_tab = pd.DataFrame(src_rows)
    src_tab.to_csv(f"{OUTDIR}/noise_source.csv", index=False, encoding="utf-8-sig")

    # ---- 요약표: split × 잡음별 추첨 SNR·배율 기술통계
    rows = []
    for sp, g in m.groupby("split", sort=False):
        for t in NOISES:
            rows.append({
                "split": sp, "잡음": t, "분절수": len(g),
                "잡음원본_구간": POOL.get(g.noise_pool.iloc[0], g.noise_pool.iloc[0]),
                "SNR_중앙": g[f"snr_{t}"].median(), "SNR_최소": g[f"snr_{t}"].min(),
                "SNR_최대": g[f"snr_{t}"].max(),
                "배율a_중앙": g[f"gain_{t}"].median(), "배율a_최소": g[f"gain_{t}"].min(),
                "배율a_최대": g[f"gain_{t}"].max(),
                "원본시작_최소": int(g[f"start_{t}"].min()),
                "원본시작_최대": int(g[f"start_{t}"].max())})
    tab = pd.DataFrame(rows).round(4)
    tab.to_csv(f"{OUTDIR}/injection_summary.csv", index=False, encoding="utf-8-sig")

    # ---- 분할 대장
    sp_rows = []
    for sp, g in m.groupby("split", sort=False):
        sp_rows.append({"split": sp, "기록수": g.record_id.nunique(), "분절수": len(g),
                        "R피크수": int(g.n_rpeaks.sum()),
                        "잡음원본_구간": POOL.get(g.noise_pool.iloc[0], g.noise_pool.iloc[0]),
                        "기록": " ".join(map(str, sorted(g.record_id.unique())))})
    split_tab = pd.DataFrame(sp_rows)
    split_tab.to_csv(f"{OUTDIR}/split_summary.csv", index=False, encoding="utf-8-sig")

    # ---- 그림: 추첨 SNR · 배율 · 잡음 원본 시작점
    fig, ax = plt.subplots(3, 3, figsize=(13, 9))
    cols = {"bw": "#2ca02c", "ma": "#ff7f0e", "em": "#d62728"}
    for j, t in enumerate(NOISES):
        a = ax[0, j]
        a.hist(m[f"snr_{t}"], bins=36, range=(lo, hi), color=cols[t], alpha=.8)
        a.set_title(f"{t} — 추첨 SNR (dB)", fontsize=9, loc="left")
        a.set_xlabel(f"[{lo}, {hi}] dB 균등추첨", fontsize=7)

        a = ax[1, j]
        a.hist(m[f"gain_{t}"], bins=40, color=cols[t], alpha=.8)
        a.set_title(f"{t} — 주입 배율 a", fontsize=9, loc="left")
        a.set_xlabel("신호 전력에 맞춘 크기 조정값", fontsize=7)

        a = ax[2, j]
        for sp, style in (("train", dict(alpha=.75)), ("val", dict(alpha=.55)),
                          ("test", dict(alpha=.4))):
            g = m[m.split == sp]
            if len(g):
                a.hist(g[f"start_{t}"], bins=40,
                       label=f"{sp} — {POOL.get(g.noise_pool.iloc[0], '')}", **style)
        b = int(src_tab.loc[src_tab["잡음"] == t, "경계_표본"].iloc[0])
        a.axvline(b, color="k", ls="--", lw=1)
        a.set_title(f"{t} — 잡음 조각을 가져온 위치", fontsize=9, loc="left")
        a.set_xlabel(f"잡음 원본 레코드 안의 표본 번호 "
                     f"(점선 = {ratio:.0%} 경계, {b:,}표본)", fontsize=7)
        a.legend(fontsize=6.5)
    for a in ax.ravel():
        a.grid(alpha=.3, lw=.4)
        a.tick_params(labelsize=7)
        a.set_ylabel("분절 수", fontsize=7)
    fig.suptitle("주입 부록 — 분절마다 무엇이 어떻게 섞였나 "
                 f"(전체 {len(m):,}분절)\n"
                 "x_noisy = x_clean + a_bw·n_bw + a_ma·n_ma + a_em·n_em,   "
                 "a = √( var(x_clean) / (mean(n²)·10^(SNR/10)) )", fontsize=12)
    fig.tight_layout()
    fig.savefig(f"{OUTDIR}/injection_appendix.png", bbox_inches="tight")
    plt.close(fig)

    note = """주입 부록 — 어떻게 섞였는지에 대한 참조 기록이다. 검증·판정이 아니다.

분절 하나를 만드는 순서
  1. MIT-BIH 기록에서 10초(3600 표본) 비중첩 창을 잘라 x_clean 으로 둔다.
  2. 잡음 3종(bw, ma, em)을 **모두** 주입한다. 강도만 무작위다.
  3. 잡음별로 SNR 을 [{lo}, {hi}] dB 에서 독립 균등추첨한다.
  4. 잡음 레코드(30분)에서 무작위 시작점의 3600 표본을 가져온다.
     train 분절은 앞 {tr}%, val·test 분절은 뒤 {ev}% 구간에서만 가져온다.
  5. 크기를 맞춘다:  a = sqrt( var(x_clean) / (mean(n^2) * 10^(SNR/10)) )
     신호 전력은 **분산**(평균 제거), 잡음 전력은 mean(n^2) 이다.
     mean(x_clean^2) 를 쓰면 MIT-BIH 전극 오프셋(DC)이 전력의 큰 몫을 차지해
     명목 SNR 과 실제 SNR 이 분절마다 어긋난다.
  6. x_noisy = x_clean + a_bw*n_bw + a_ma*n_ma + a_em*n_em

기호
  x_clean   MIT-BIH 원 신호 10초 창 (mV). 필터·정규화를 하지 않은 그대로다.
  n         잡음 원본 레코드에서 잘라 온 같은 길이의 조각 (mV)
  SNR       그 분절·그 잡음에 배정된 목표 신호대잡음비 (dB)
  a         잡음을 목표 SNR 에 맞추기 위한 배율. 파형은 그대로 두고 크기만 바꾼다
  var(·)    표본 분산 (평균을 뺀 뒤 제곱 평균)
  mean(n^2) 잡음 조각의 제곱 평균. 잡음은 평균을 빼지 않는다 —
            기저선 오프셋도 제거 대상 잡음의 일부이기 때문이다

"잡음 원본 구간" 이란
  잡음 레코드는 30분짜리 하나뿐이다. train 분절과 val·test 분절이 **같은 잡음 파형**을
  보지 않도록, 레코드를 시간축에서 앞 {tr}% / 뒤 {ev}% 로 갈라 두었다.
  train 분절은 앞 구간에서만, val·test 분절은 뒤 구간에서만 조각을 가져온다.
  경계 위치와 구간 범위는 noise_source.csv 에 있다.

파일
  noise_source.csv             잡음 원본 레코드의 길이·채널·앞뒤 구간 경계
  injection_summary.csv        split x 잡음별 추첨 SNR·배율 a·원본 시작점 범위
  split_summary.csv            split 별 기록·분절·R피크 수와 기록 목록
  injection_appendix.png       위 세 가지의 분포
  <split>_<기록>_<분절>.png    분절 하나의 5행 스팟체크
""".format(lo=lo, hi=hi, tr=int(d["noise_split_ratio"] * 100),
           ev=int(round((1 - d["noise_split_ratio"]) * 100)))
    with open(f"{OUTDIR}/injection_note.txt", "w", encoding="utf-8") as f:
        f.write(note)

    print(tab.to_string(index=False))
    print()
    print(split_tab.drop(columns=["기록"]).to_string(index=False))
    print(f"\n부록 → {OUTDIR}/")
    return tab, split_tab


def spotcheck_figs(cfg, names=None, record=None, start=0, n=3, seed=42):
    """스팟체크 그림. 지정이 없으면 split 마다 무작위 1개.

    `record` 를 주면 **한 기록의 연속 window** n개를 그린다 — 같은 환자에서 잡음이
    분절마다 어떻게 달라지는지 보기 위한 것이다.
    """
    os.makedirs(OUTDIR, exist_ok=True)
    jobs = []
    if record is not None:
        for i in range(start, start + n):
            jobs.append(f"{record}_{i:04d}")
    elif names:
        jobs = list(names)
    if jobs:
        for name in jobs:
            path, sp = find_seg(name)
            out = os.path.join(OUTDIR, f"{sp}_{name}.png")
            spotcheck(path, out)
            print(f"  {sp}  {name}  →  {out}")
        return
    random.seed(seed)
    for sp in ("train", "val", "test"):
        files = sorted(glob.glob(os.path.join(SEGDIR, sp, "*.npz")))
        if not files:
            continue
        f = random.choice(files)
        name = os.path.splitext(os.path.basename(f))[0]
        out = os.path.join(OUTDIR, f"{sp}_{name}.png")
        spotcheck(f, out)
        print(f"  {sp}  {name}  ({len(files)}분절 중 무작위)  →  {out}")


def main(config="configs/default.yaml", skip_download=False):
    cfg = load_cfg(config)
    if not skip_download:
        print("[01] 원본 내려받기·검증")
        download.download_mitdb(cfg["paths"]["mitdb"])
        if not download.verify(cfg):
            raise SystemExit("[01] 원본 검증 실패 — 진행하지 않는다")
    print("[01] 기록 단위 분할")
    split.main(config)
    print("[01] 분절 생성 + 잡음 주입")
    build.main(config)
    spotcheck_figs(cfg)
    appendix(cfg)
    print("[01] 완료 — data/processed/ · results/01_build/")
    return cfg


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--skip-download", action="store_true")
    p.add_argument("--spotcheck", nargs="*", default=None,
                   help="스팟체크만 다시 그린다. 인자로 `기록_분절` 을 나열할 수 있다")
    p.add_argument("--record", default=None, help="한 기록의 연속 window 를 그린다")
    p.add_argument("--from", dest="start", type=int, default=0, help="시작 분절 번호")
    p.add_argument("--n", type=int, default=3, help="연속 window 개수")
    p.add_argument("--appendix", action="store_true", help="주입 부록만 다시 만든다")
    a = p.parse_args()
    if a.appendix:
        appendix(load_cfg(a.config))
    elif a.spotcheck is not None or a.record is not None:
        spotcheck_figs(load_cfg(a.config), a.spotcheck, a.record, a.start, a.n)
    else:
        main(a.config, a.skip_download)
