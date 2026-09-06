"""s5 — 2채널 활용 가능성 데이터 점검. **학습하지 않는다. 읽기만 한다.**

    PYTHONIOENCODING=utf-8 python experiments/s5_2채널_점검/inspect_2ch.py

지금은 MLII 한 채널만 쓴다. 상한 측정(s4)에서 bw 의 |r| 이 0.65 에서 움직이지 않았고,
그것이 단일 채널 입력이 담은 정보의 한계로 보인다. 두 번째 리드를 넣으면
**"심장은 두 채널에 공통, 잡음은 채널마다 다르다"** 는 새 판별 근거가 생길 수 있다.
그 전제가 데이터에서 성립하는지만 확인한다.

────────────────────────────────────────────────────────────────────────
확인하는 것
────────────────────────────────────────────────────────────────────────
① 기록별 리드 구성   우리가 쓰는 46기록의 채널 1·2 가 각각 무슨 리드인가.
                    두 번째 리드가 기록마다 다르면 일관된 학습이 어렵다.
② NSTDB 잡음의 채널 간 차이   bw·ma·em 각각 두 채널이 얼마나 다른가.
③ x_clean 의 채널 간 차이     심장은 두 채널에 공통이어야 한다.

**판정선** — 채널 간 |r| 이 0.9 이상이면 사실상 같은 신호라 정보가 늘지 않는다.
0.5 이하로 갈리면 채널마다 다른 잡음이 실린 것이라 활용 가치가 있다.
그리고 **심장 상관 > 잡음 상관** 이 성립해야 2채널 접근의 전제가 선다.

────────────────────────────────────────────────────────────────────────
재는 방법
────────────────────────────────────────────────────────────────────────
전 구간 상관 하나로는 부족하다 — 우리가 실제로 다루는 단위는 10초 분절이므로
**분절 단위 분포**를 함께 낸다(모델이 보는 것과 같은 창).

부호를 살린 r 과 |r| 을 **둘 다** 싣는다. MLII 와 V1 은 R 파 극성이 반대인 경우가 있어
r 이 음수로 나올 수 있는데, "두 채널에 같은 사건이 실렸는가"를 묻는 데는 |r| 이 맞고
극성 정보는 r 이 갖고 있다.

대역은 03·04 와 같은 정의를 쓴다 (`src.metrics.PSD_BANDS`).

────────────────────────────────────────────────────────────────────────
산출물  experiments/s5_2채널_점검/
────────────────────────────────────────────────────────────────────────
  leads.csv          ① 기록별 채널 구성 · MLII 위치 · 두 번째 리드
  leads_summary.csv  ① 두 번째 리드의 분포 (split 별)
  noise_channels.csv ② NSTDB 3종 × (상관 · 대역 프로파일 · 진폭비)
  clean_channels.csv ③ MIT-BIH x_clean 의 채널 간 상관 (기록별)
  bands.csv          대역별 두 채널 |r| — 심장 vs 잡음 3종
  verdict.csv        ①②③ 을 판정선과 대조한 한 장짜리 표
  note.txt
  figures/overlay_<기록>.png   기록 하나의 두 채널 겹침 (clean·bw·ma·em)
  figures/corr_dist.png        분절 단위 |r| 분포 — 심장 vs 잡음 3종

값에 대한 해석·판정은 붙이지 않는다. 판정선과의 대조만 표로 낸다.
"""
import json
import os
import sys

import numpy as np
import pandas as pd
import wfdb

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))     # version5_supervised
sys.path.insert(0, ROOT)

from src.data.build import load_cfg                        # noqa: E402
from src.metrics import PSD_BANDS, PSD_NORM                # noqa: E402
from src.viz import plt                                    # noqa: E402

SEG = 3600
NL = chr(10)
NOISES = ("bw", "ma", "em")
HI, LO = 0.9, 0.5          # 판정선 — 0.9 이상이면 같은 신호, 0.5 이하면 갈린다


# ---------------------------------------------------------------- 원시 계산
def seg_corr(a, b, seg=SEG):
    """분절 단위 Pearson r. 겹치지 않는 10초 창마다 하나씩. 상수 구간은 뺀다."""
    n = min(len(a), len(b)) // seg
    out = []
    for i in range(n):
        u, v = a[i * seg:(i + 1) * seg], b[i * seg:(i + 1) * seg]
        u = u - u.mean()
        v = v - v.mean()
        d = np.sqrt((u ** 2).sum() * (v ** 2).sum())
        if d > 0:
            out.append(float((u * v).sum() / d))
    return np.asarray(out)


def band_profile(x, fs, nperseg=1024):
    """대역별 전력 **비율**. 03·04 와 같은 정의(`PSD_BANDS`·`PSD_NORM`)."""
    from scipy.signal import welch
    f, P = welch(np.asarray(x, np.float64), fs=fs, nperseg=nperseg,
                 noverlap=nperseg // 2, window="hann")
    den = P[(f >= PSD_NORM[0]) & (f < PSD_NORM[1])].sum()
    return {k: float(P[(f >= lo) & (f < hi)].sum() / max(den, 1e-30))
            for k, (lo, hi) in PSD_BANDS.items()}


def band_corr(a, b, fs, seg=SEG):
    """대역별 두 채널 |r| 중앙값 — 대역통과 후 분절 단위.

    **전대역 상관 하나로는 오해가 생긴다.** bw 는 에너지의 82%가 vlf 에 있어, 전대역
    상관이 사실상 vlf 상관이 된다. 심장은 에너지가 lf·qrs 에 있다. 서로 다른 대역에
    사는 두 신호를 전대역 한 숫자로 견주면 에너지가 큰 쪽이 답을 지배한다.
    대역을 나눠야 "어느 대역에서 심장이 더 공통인가"를 물을 수 있다.
    """
    from src.metrics import bandpass
    out = {"전대역": float(np.median(np.abs(seg_corr(a, b, seg))))}
    for k, (lo, hi) in PSD_BANDS.items():
        r = seg_corr(bandpass(a, lo, hi, fs), bandpass(b, lo, hi, fs), seg)
        out[k] = float(np.median(np.abs(r)))
    return out


def _stats(r):
    """부호 있는 r 과 |r| 의 요약. 분절 수도 함께."""
    a = np.abs(r)
    return {"분절수": len(r), "r_평균": float(r.mean()), "r_중앙": float(np.median(r)),
            "abs_r_평균": float(a.mean()), "abs_r_중앙": float(np.median(a)),
            "abs_r_SD": float(a.std(ddof=1)) if len(a) > 1 else np.nan,
            "abs_r_p05": float(np.percentile(a, 5)),
            "abs_r_p95": float(np.percentile(a, 95))}


# ---------------------------------------------------------------- ① 리드 구성
def leads(cfg, outdir):
    mitdb, want = cfg["paths"]["mitdb"], cfg["data"]["lead"]
    sp = json.load(open(os.path.join(cfg["paths"]["processed"], "split.json"),
                        encoding="utf-8"))
    where = {r: k for k in ("train", "val", "test") for r in sp[k]}
    rows = []
    for r in sorted(where):
        h = wfdb.rdheader(os.path.join(mitdb, r))
        names = list(h.sig_name)
        i = names.index(want) if want in names else -1
        other = [n for j, n in enumerate(names) if j != i]
        rows.append({"기록": r, "split": where[r], "채널1": names[0],
                     "채널2": names[1] if len(names) > 1 else "",
                     f"{want}_위치": i + 1 if i >= 0 else 0,
                     "두번째리드": other[0] if other else "",
                     "채널수": len(names)})
    d = pd.DataFrame(rows)
    d.to_csv(f"{outdir}/leads.csv", index=False, encoding="utf-8-sig")

    g = (d.groupby(["split", "두번째리드"]).size().rename("기록수").reset_index()
         .pivot(index="두번째리드", columns="split", values="기록수").fillna(0)
         .astype(int))
    g["합"] = g.sum(1)
    g = g.reset_index().sort_values("합", ascending=False)
    g.to_csv(f"{outdir}/leads_summary.csv", index=False, encoding="utf-8-sig")
    return d, g


# ---------------------------------------------------------------- ② 잡음 채널
def noise_channels(cfg, outdir):
    nst, fs = cfg["paths"]["nstdb"], cfg["data"]["fs"]
    rows, dist = [], {}
    for n in NOISES:
        sig, fields = wfdb.rdsamp(os.path.join(nst, n))
        c0, c1 = sig[:, 0].astype(np.float64), sig[:, 1].astype(np.float64)
        r = seg_corr(c0, c1)
        dist[n] = np.abs(r)
        whole = seg_corr(c0, c1, seg=min(len(c0), len(c1)))
        p0, p1 = band_profile(c0, fs), band_profile(c1, fs)
        bc = band_corr(c0, c1, fs)
        rows.append({
            "대상": n, "채널명": ",".join(fields["sig_name"]),
            **{f"band_{k}": v for k, v in bc.items()},
            "전체_r": float(whole[0]) if len(whole) else np.nan,
            **_stats(r),
            "SD_ch1": float(c0.std()), "SD_ch2": float(c1.std()),
            "진폭비_ch2/ch1": float(c1.std() / max(c0.std(), 1e-12)),
            **{f"{k}_ch1": v for k, v in p0.items()},
            **{f"{k}_ch2": v for k, v in p1.items()},
        })
    d = pd.DataFrame(rows)
    d.round(5).to_csv(f"{outdir}/noise_channels.csv", index=False,
                      encoding="utf-8-sig")
    return d, dist


# ---------------------------------------------------------------- ③ clean 채널
def clean_channels(cfg, outdir, leads_df):
    mitdb = cfg["paths"]["mitdb"]
    rows, pool = [], []
    for _, row in leads_df.iterrows():
        if row["채널수"] < 2:
            continue
        sig, fields = wfdb.rdsamp(os.path.join(mitdb, row["기록"]))
        c0, c1 = sig[:, 0].astype(np.float64), sig[:, 1].astype(np.float64)
        r = seg_corr(c0, c1)
        if not len(r):
            continue
        pool.append(np.abs(r))
        bc = band_corr(c0, c1, cfg["data"]["fs"])
        rows.append({"기록": row["기록"], "split": row["split"],
                     "채널1": row["채널1"], "채널2": row["채널2"],
                     **{f"band_{k}": v for k, v in bc.items()},
                     **_stats(r),
                     "진폭비_ch2/ch1": float(c1.std() / max(c0.std(), 1e-12))})
        print(f"  {row['기록']}  |r| 중앙 {np.median(np.abs(r)):.4f}", flush=True)
    d = pd.DataFrame(rows)
    d.round(5).to_csv(f"{outdir}/clean_channels.csv", index=False,
                      encoding="utf-8-sig")
    return d, np.concatenate(pool)


# ---------------------------------------------------------------- 그림
def fig_overlay(cfg, record, outdir, seg_idx=0):
    """기록 하나의 두 채널을 겹쳐 본다 — x_clean 과 잡음 3종."""
    mitdb, nst, fs = (cfg["paths"]["mitdb"], cfg["paths"]["nstdb"],
                      cfg["data"]["fs"])
    sig, fields = wfdb.rdsamp(os.path.join(mitdb, record))
    a = sig[seg_idx * SEG:(seg_idx + 1) * SEG, :2].astype(np.float64)
    panels = [(f"x_clean — 기록 {record}", a, list(fields["sig_name"][:2]))]
    for n in NOISES:
        s, f2 = wfdb.rdsamp(os.path.join(nst, n))
        panels.append((f"{n}", s[seg_idx * SEG:(seg_idx + 1) * SEG, :2]
                       .astype(np.float64), list(f2["sig_name"][:2])))
    t = np.arange(SEG) / fs
    fig, ax = plt.subplots(len(panels), 1, figsize=(12, 2.5 * len(panels)),
                           sharex=True)
    for a_, (title, v, names) in zip(ax, panels):
        r = seg_corr(v[:, 0], v[:, 1], seg=SEG)
        a_.plot(t, v[:, 0], lw=.8, color="#1f77b4", label=f"채널1 {names[0]}")
        a_.plot(t, v[:, 1], lw=.8, color="#d62728", alpha=.8,
                label=f"채널2 {names[1]}")
        a_.set_title(f"{title}   ·   두 채널 r {r[0]:+.3f} (|r| {abs(r[0]):.3f})",
                     fontsize=10, loc="left")
        a_.legend(fontsize=8, ncol=2, loc="upper right")
        a_.grid(alpha=.3, lw=.4)
        a_.set_ylabel("mV", fontsize=8)
        a_.tick_params(labelsize=8)
    ax[-1].set_xlabel("시간 (초)")
    fig.suptitle(f"[s5] 두 채널 겹침 — 기록 {record}, 분절 {seg_idx} (10초){NL}"
                 "잡음 3종은 NSTDB 의 같은 구간이다 (기록과 무관)", fontsize=12)
    fig.tight_layout()
    os.makedirs(f"{outdir}/figures", exist_ok=True)
    fig.savefig(f"{outdir}/figures/overlay_{record}.png", bbox_inches="tight",
                dpi=130)
    plt.close(fig)


def fig_dist(clean_abs, noise_dist, outdir):
    """분절 단위 |r| 분포 — 심장이 잡음보다 높아야 2채널 전제가 선다."""
    keys = ["x_clean (46기록)"] + list(noise_dist)
    data = [clean_abs] + [noise_dist[n] for n in noise_dist]
    col = ["#4c72b0"] + ["#c44e52"] * len(noise_dist)
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    bp = ax.boxplot(data, showfliers=False, widths=.6, patch_artist=True,
                    tick_labels=keys)
    for p, c in zip(bp["boxes"], col):
        p.set_facecolor(c)
        p.set_alpha(.55)
        p.set_edgecolor("#333")
    for part in ("whiskers", "caps", "medians"):
        for ln in bp[part]:
            ln.set_color("#333")
    for y, lab in ((HI, f"{HI} — 이상이면 사실상 같은 신호"),
                   (LO, f"{LO} — 이하면 채널마다 다른 신호")):
        ax.axhline(y, ls="--", lw=1.1, color="#1f77b4")
        ax.text(len(keys) + .45, y, lab, fontsize=8, va="center", ha="right",
                color="#1f77b4")
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("분절 단위 |r| (10초 창)")
    ax.grid(alpha=.3, lw=.4, axis="y")
    ax.tick_params(labelsize=9)
    fig.suptitle(f"[s5] 두 채널 간 |r| — 심장 vs 잡음{NL}"
                 "심장이 잡음보다 높아야 2채널 접근의 전제가 선다", fontsize=12)
    fig.tight_layout()
    fig.savefig(f"{outdir}/figures/corr_dist.png", bbox_inches="tight", dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------- 본체
def main(config=None, record=None):
    cfg = load_cfg(config or os.path.join(ROOT, "configs", "default.yaml"))
    os.chdir(ROOT)                       # config 의 상대 경로 기준을 맞춘다
    outdir = HERE
    os.makedirs(f"{outdir}/figures", exist_ok=True)
    pd.set_option("display.width", 250)

    print("[s5] ① 기록별 리드 구성")
    ld, ldsum = leads(cfg, outdir)
    print(ldsum.to_string(index=False))
    print("")

    print("[s5] ② NSTDB 잡음의 채널 간 차이")
    nz, ndist = noise_channels(cfg, outdir)
    print(nz[["대상", "채널명", "전체_r", "abs_r_중앙", "abs_r_p05", "abs_r_p95",
              "진폭비_ch2/ch1"]].round(4).to_string(index=False))
    print("")
    print("  대역 전력비 (vlf 0.05-0.5 · lf 0.5-5 · qrs 5-15 · hf 15-40 Hz)")
    bcols = [f"{k}_ch{c}" for k in PSD_BANDS for c in (1, 2)]
    print(nz[["대상"] + bcols].round(4).to_string(index=False))
    print("")

    print("[s5] ③ x_clean 의 채널 간 차이 — 46기록")
    cl, cl_abs = clean_channels(cfg, outdir, ld)
    print("")
    print(f"  전체 {len(cl_abs):,}분절 · |r| 중앙 {np.median(cl_abs):.4f} · "
          f"평균 {cl_abs.mean():.4f} · p05 {np.percentile(cl_abs, 5):.4f} · "
          f"p95 {np.percentile(cl_abs, 95):.4f}")
    print("")

    # ---- 대역별 대조 — 전대역 한 숫자가 만드는 오해를 푼다
    bcols = ["전대역"] + list(PSD_BANDS)
    brow = [{"대상": "x_clean (심장)",
             **{k: float(cl[f"band_{k}"].mean()) for k in bcols}}]
    for _, r in nz.iterrows():
        brow.append({"대상": f"{r['대상']} (잡음)",
                     **{k: float(r[f"band_{k}"]) for k in bcols}})
    bd = pd.DataFrame(brow)
    for _, r in nz.iterrows():
        bd.loc[len(bd)] = {"대상": f"심장 − {r['대상']}",
                           **{k: float(cl[f"band_{k}"].mean() - r[f"band_{k}"])
                              for k in bcols}}
    bd.round(5).to_csv(f"{outdir}/bands.csv", index=False, encoding="utf-8-sig")
    print("")
    print("[s5] 대역별 두 채널 |r| — 양수면 심장이 더 공통 (아래 세 줄)")
    print(bd.round(4).to_string(index=False))

    # ---- 판정선 대조
    def verdict(v):
        return ("사실상 같은 신호 (>=0.9)" if v >= HI
                else "채널마다 다른 신호 (<=0.5)" if v <= LO else "중간 (0.5~0.9)")

    vr = [{"대상": "x_clean (심장)", "분절수": len(cl_abs),
           "abs_r_중앙": float(np.median(cl_abs)), "판정": verdict(np.median(cl_abs))}]
    for _, r in nz.iterrows():
        vr.append({"대상": f"{r['대상']} (잡음)", "분절수": int(r["분절수"]),
                   "abs_r_중앙": r["abs_r_중앙"], "판정": verdict(r["abs_r_중앙"])})
    vd = pd.DataFrame(vr)
    gap = float(np.median(cl_abs)) - float(nz["abs_r_중앙"].max())
    vd.loc[len(vd)] = {"대상": "심장 − 잡음(최대) 차", "분절수": np.nan,
                       "abs_r_중앙": gap,
                       "판정": "전제 성립 (심장 > 잡음)" if gap > 0
                               else "전제 불성립 (심장 <= 잡음)"}
    vd.round(5).to_csv(f"{outdir}/verdict.csv", index=False, encoding="utf-8-sig")
    print("[s5] 판정선 대조")
    print(vd.round(4).to_string(index=False))

    rec = record or ld[ld["split"] == "test"]["기록"].iloc[0]
    fig_overlay(cfg, rec, outdir)
    fig_dist(cl_abs, ndist, outdir)
    _note(outdir, ld, ldsum, nz, cl, cl_abs, vd, bd, rec)
    print(f"{NL}산출물 → {outdir}/")
    return ld, nz, cl, vd


def _note(outdir, ld, ldsum, nz, cl, cl_abs, vd, bd, rec):
    L = [
        "s5 — 2채널 활용 가능성 데이터 점검. 학습하지 않았다. 읽기만 했다.",
        "",
        "묻는 것",
        "  지금은 MLII 한 채널만 쓴다. 상한 측정(s4)에서 bw 의 |r| 이 0.65 에서 움직이지",
        "  않았고, 그것이 단일 채널 입력이 담은 정보의 한계로 보인다. 두 번째 리드를",
        "  넣으면 '심장은 두 채널에 공통, 잡음은 채널마다 다르다'는 새 판별 근거가",
        "  생길 수 있다. 그 전제가 데이터에서 성립하는지만 확인했다.",
        "",
        "재는 방법",
        "  분절 단위(10초, 겹치지 않음) Pearson r 을 낸다 - 모델이 보는 창과 같다.",
        "  부호 있는 r 과 |r| 을 둘 다 싣는다. MLII 와 V1 은 R 파 극성이 반대인 경우가",
        "  있어 r 이 음수로 나올 수 있는데, '두 채널에 같은 사건이 실렸는가'를 묻는",
        "  데는 |r| 이 맞고 극성은 r 이 갖고 있다.",
        "  대역은 03·04 와 같은 정의다 (src.metrics.PSD_BANDS).",
        "",
        f"판정선 - |r| >= {HI} 면 사실상 같은 신호(정보가 늘지 않는다),",
        f"         |r| <= {LO} 면 채널마다 다른 신호(활용 가치가 있다).",
        "         그리고 심장 상관 > 잡음 상관 이어야 2채널 접근의 전제가 선다.",
        "",
        "① 기록별 리드 구성 - 두 번째 리드의 분포",
        ldsum.to_string(index=False),
        "",
        "② NSTDB 잡음의 채널 간 차이",
        nz[["대상", "채널명", "전체_r", "abs_r_중앙", "abs_r_p05", "abs_r_p95",
            "진폭비_ch2/ch1"]].round(4).to_string(index=False),
        "",
        "③ x_clean 의 채널 간 차이",
        f"  46기록 {len(cl_abs):,}분절 · |r| 중앙 {np.median(cl_abs):.4f}",
        f"  기록별 값은 clean_channels.csv 에 있다.",
        "",
        "대역별 두 채널 |r| (양수면 심장이 더 공통)",
        "  전대역 한 숫자는 에너지가 큰 대역이 지배한다 - bw 는 에너지의 82%가 vlf 다.",
        bd.round(4).to_string(index=False),
        "",
        "판정선 대조 (전대역)",
        vd.round(4).to_string(index=False),
        "",
        f"그림 - figures/overlay_{rec}.png (두 채널 겹침), figures/corr_dist.png",
        "",
        "지금 파이프라인은 MLII 한 채널과 NSTDB 채널 1(noise_channel: 0)만 쓴다.",
        "이 점검은 그 설정을 바꾸지 않는다.",
        "",
        "값에 대한 해석과 판정은 붙이지 않는다. 판정선과의 대조만 표로 낸다.",
    ]
    with open(f"{outdir}/note.txt", "w", encoding="utf-8") as f:
        f.write(NL.join(L) + NL)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=None)
    p.add_argument("--record", default=None, help="겹침 그림에 쓸 기록. 기본은 test 첫 기록")
    a = p.parse_args()
    main(a.config, a.record)
