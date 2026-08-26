"""보조 실험 — 마스킹 순서 기준 검토.

**본 노선이 아니다.** 이미 산출한 결과만 다시 읽어 쓴다 — 모델 추론을 하지 않는다.

입력
  results/04_masked_denoising/<run>/<split>/exhaustive.csv   전수 256조합 × 지표 5종
  results/04_masked_denoising/<run>/<split>/single_mask.csv  단독 마스킹 8개
  results/03_bss/<run>/<split>/corr_matrix.csv               인코더 × 참조 |r|

    python experiments/masking_strategy/order.py --run K8_seed42 --split val

산출  experiments/masking_strategy/outputs/<run>/<split>/order/
  ① predictor_scatter.csv · figures/predictor_scatter.png
       단독 제거 효과(ΔSNR, M0 대비)와 03 지표들의 관계
  ② top20_combos.csv · encoder_frequency.csv · figures/top20_frequency.png
       상위 20개 조합의 구성과 인코더별 포함 빈도
  ③ enc4_split.csv · figures/enc4_split.png
       enc4 포함/제외 조합의 지표 분포
  ④ cumulative_clean_asc.csv · figures/cumulative_compare.png
       clean 상관 오름차순 누적 마스킹 (기존 잡음 상관 내림차순과 대조)

해석은 붙이지 않는다. 수치와 그림만.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.model.meae import enc_label                           # noqa: E402
from src.viz import plt                                        # noqa: E402

REFS = ["x_clean", "bw", "ma", "em"]
NOISE = ["bw", "ma", "em"]
METRICS = ["SSD", "MAD", "PRD", "CosSim", "SNR"]
LOWER_BETTER = {"SSD": True, "MAD": True, "PRD": True, "CosSim": False, "SNR": False}


def load_all(run, split):
    d4 = os.path.join("results", "04_masked_denoising", run, split)
    d3 = os.path.join("results", "03_bss", run, split)
    for p in (f"{d4}/exhaustive.csv", f"{d4}/single_mask.csv", f"{d3}/corr_matrix.csv"):
        if not os.path.exists(p):
            raise SystemExit(f"[순서검토] {p} 없음 — 03·04 를 먼저 돌려야 한다")
    # 마스크비트는 "00000000" 처럼 앞자리 0 이 있어 문자열로 읽어야 한다
    ex = pd.read_csv(f"{d4}/exhaustive.csv", encoding="utf-8-sig",
                     dtype={"마스크비트": str})
    ex["마스크비트"] = ex["마스크비트"].str.zfill(len(
        pd.read_csv(f"{d3}/corr_matrix.csv", encoding="utf-8-sig")))
    sg = pd.read_csv(f"{d4}/single_mask.csv", encoding="utf-8-sig")
    cm = pd.read_csv(f"{d3}/corr_matrix.csv", encoding="utf-8-sig").set_index("인코더")
    return ex, sg, cm


# ---------------------------------------------------------------- ① 예측력
def predictor_table(sg, cm):
    """단독 제거 효과와 03 지표들을 한 표에 모은다."""
    rows = []
    for _, r in sg.iterrows():
        enc = r["조합"]
        c = cm.loc[enc]
        noise_max = max(float(c[t]) for t in NOISE)
        rows.append({
            "인코더": enc,
            "ΔSNR_vs_M0": float(r["SNR_vs_M0"]),
            "ΔSNR_vs_noisy": float(r["SNR_vs_noisy"]),
            "clean_r": float(c["x_clean"]),
            "bw_r": float(c["bw"]), "ma_r": float(c["ma"]), "em_r": float(c["em"]),
            "잡음최대_r": noise_max,
            "clean빼기잡음": float(c["x_clean"]) - noise_max,
            "R피크_진폭비": float(r["R피크_진폭비_중앙"])})
    return pd.DataFrame(rows)


def fig_predictor(tab, out):
    cols = ["clean_r", "bw_r", "ma_r", "em_r", "잡음최대_r", "clean빼기잡음"]
    fig, ax = plt.subplots(2, 3, figsize=(15, 8))
    for a, c in zip(ax.ravel(), cols):
        a.scatter(tab[c], tab["ΔSNR_vs_M0"], s=55, color="#1f77b4")
        for _, r in tab.iterrows():
            a.annotate(r["인코더"], (r[c], r["ΔSNR_vs_M0"]),
                       xytext=(4, 3), textcoords="offset points", fontsize=7.5)
        x, y = tab[c].values, tab["ΔSNR_vs_M0"].values
        rp = np.corrcoef(x, y)[0, 1]
        rs = pd.Series(x).corr(pd.Series(y), method="spearman")
        a.set_title(f"{c}   피어슨 {rp:+.3f} · 스피어만 {rs:+.3f}", fontsize=9)
        a.set_xlabel(f"03 지표: {c}", fontsize=8)
        a.set_ylabel("단독 제거 ΔSNR (M0 대비, dB)", fontsize=8)
        a.axhline(0, color="#888", lw=.8, ls=":")
        a.grid(alpha=.3, lw=.4)
        a.tick_params(labelsize=7.5)
    fig.suptitle("① 단독 제거 효과와 03 지표의 관계 — 점 하나가 인코더 하나", fontsize=13)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- ② 상위 20
def top20(ex, K=8, metric="SNR_중앙", n=20):
    hi = not LOWER_BETTER[metric.split("_")[0]]
    t = ex.nlargest(n, metric) if hi else ex.nsmallest(n, metric)
    t = t.reset_index(drop=True)
    freq = []
    for k in range(K):
        inc = sum(1 for b in t["마스크비트"] if b[k] == "1")
        freq.append({"인코더": enc_label(k), "포함_횟수": inc,
                     "포함_비율": inc / len(t), "제외_횟수": len(t) - inc})
    return t, pd.DataFrame(freq)


def fig_top20(freq, out, metric):
    fig, ax = plt.subplots(figsize=(8, 4.4))
    x = np.arange(len(freq))
    ax.bar(x, freq["포함_비율"], color="#1f77b4")
    ax.axhline(0.5, color="#d62728", ls=":", lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels(freq["인코더"])
    ax.set_ylim(0, 1)
    ax.set_ylabel("상위 20개 조합에 포함된 비율")
    ax.grid(alpha=.3, lw=.4, axis="y")
    for i, v in enumerate(freq["포함_비율"]):
        ax.text(i, v, f"{v:.2f}", ha="center", va="bottom", fontsize=8)
    ax.set_title(f"② {metric} 상위 20개 조합의 인코더별 포함 빈도 "
                 f"(점선 = 0.5)", fontsize=11)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- ③ enc4 분할
def enc4_split(ex, k4=3):
    has = ex["마스크비트"].str[k4] == "1"
    rows = []
    for lab, g in (("enc4 포함", ex[has]), ("enc4 제외", ex[~has])):
        r = {"구분": lab, "조합수": len(g)}
        for m in METRICS:
            c = f"{m}_중앙"
            r[f"{m}_중앙값"] = float(g[c].median())
            r[f"{m}_최소"] = float(g[c].min())
            r[f"{m}_최대"] = float(g[c].max())
        r["R피크_진폭비_중앙값"] = float(g["R피크_진폭비_중앙"].median())
        rows.append(r)
    return pd.DataFrame(rows).round(4), has


def fig_enc4(ex, has, out):
    cols = [f"{m}_중앙" for m in METRICS] + ["R피크_진폭비_중앙"]
    fig, ax = plt.subplots(1, len(cols), figsize=(3.0 * len(cols), 4.2))
    for a, c in zip(np.atleast_1d(ax), cols):
        a.boxplot([ex.loc[~has, c].values, ex.loc[has, c].values],
                  labels=["제외", "포함"], widths=.55)
        a.set_title(c.replace("_중앙", ""), fontsize=9)
        a.grid(alpha=.3, lw=.4, axis="y")
        a.tick_params(labelsize=8)
        if c in ("SSD_중앙", "PRD_중앙"):
            a.set_yscale("log")
    fig.suptitle("③ enc4 포함 여부에 따른 전수 256조합의 지표 분포", fontsize=12)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- ④ 누적 곡선
def cumulative(ex, order, K=8):
    """order 순서로 하나씩 추가 마스킹했을 때의 지표. exhaustive 에서 찾아 쓴다."""
    bit2row = {r["마스크비트"]: r for _, r in ex.iterrows()}
    rows = []
    for m in range(K + 1):
        bits = "".join("1" if k in set(order[:m]) else "0" for k in range(K))
        r = dict(bit2row[bits])
        r["누적_개수"] = m
        r["추가된_인코더"] = enc_label(order[m - 1]) if m else "-"
        rows.append(r)
    return pd.DataFrame(rows)


def fig_cumulative(asc, desc, out):
    fig, ax = plt.subplots(1, len(METRICS), figsize=(3.2 * len(METRICS), 3.8))
    for a, m in zip(np.atleast_1d(ax), METRICS):
        c = f"{m}_중앙"
        a.plot(desc["누적_개수"], desc[c], "o-", ms=4, lw=1.2, color="#d62728",
               label="잡음 상관 내림차순")
        a.plot(asc["누적_개수"], asc[c], "s-", ms=4, lw=1.2, color="#1f77b4",
               label="clean 상관 오름차순")
        a.set_xlabel("누적 마스킹 개수", fontsize=8)
        a.set_title(m, fontsize=9)
        a.grid(alpha=.3, lw=.4)
        a.tick_params(labelsize=7.5)
        if m in ("SSD", "PRD"):
            a.set_yscale("log")
    np.atleast_1d(ax)[0].legend(fontsize=7.5)
    fig.suptitle("④ 누적 마스킹 순서 대조 — 같은 조합 집합, 추가 순서만 다르다",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def main(run="K8_seed42", split="val", metric="SNR_중앙", outdir=None):
    ex, sg, cm = load_all(run, split)
    outdir = outdir or os.path.join("experiments", "masking_strategy", "outputs",
                                    run, split, "order")
    figdir = os.path.join(outdir, "figures")
    os.makedirs(figdir, exist_ok=True)
    K = len(cm)
    pd.set_option("display.width", 280)
    print(f"[순서검토] {run} · {split} — 기존 산출물 재활용, 모델 추론 없음\n")

    # ① 예측력
    pred = predictor_table(sg, cm).round(4)
    pred.to_csv(f"{outdir}/predictor_scatter.csv", index=False, encoding="utf-8-sig")
    fig_predictor(pred, f"{figdir}/predictor_scatter.png")
    print("① 단독 제거 효과와 03 지표")
    print(pred.to_string(index=False), "\n")
    corr_rows = []
    for c in ["clean_r", "bw_r", "ma_r", "em_r", "잡음최대_r", "clean빼기잡음"]:
        corr_rows.append({"03지표": c,
                          "피어슨_vs_ΔSNR": float(np.corrcoef(pred[c], pred["ΔSNR_vs_M0"])[0, 1]),
                          "스피어만_vs_ΔSNR": float(pred[c].corr(pred["ΔSNR_vs_M0"],
                                                              method="spearman"))})
    ct = pd.DataFrame(corr_rows).round(4)
    ct.to_csv(f"{outdir}/predictor_corr.csv", index=False, encoding="utf-8-sig")
    print("①-보조 예측력 (인코더 8개 기준)")
    print(ct.to_string(index=False), "\n")

    # ② 상위 20
    t20, freq = top20(ex, K, metric)
    show = ["조합", "끈_인코더수", "마스크비트"] + [f"{m}_중앙" for m in METRICS] \
        + ["R피크_진폭비_중앙"]
    t20[show].round(4).to_csv(f"{outdir}/top20_combos.csv", index=False,
                              encoding="utf-8-sig")
    freq.round(4).to_csv(f"{outdir}/encoder_frequency.csv", index=False,
                         encoding="utf-8-sig")
    fig_top20(freq, f"{figdir}/top20_frequency.png", metric)
    print(f"② {metric} 상위 20개 조합")
    print(t20[show].round(4).to_string(index=False), "\n")
    print("②-보조 인코더별 포함 빈도")
    print(freq.round(3).to_string(index=False), "\n")

    # ③ enc4 분할
    sp, has = enc4_split(ex)
    sp.to_csv(f"{outdir}/enc4_split.csv", index=False, encoding="utf-8-sig")
    fig_enc4(ex, has, f"{figdir}/enc4_split.png")
    print("③ enc4 포함/제외 조합의 지표 분포")
    print(sp.to_string(index=False), "\n")

    # ④ 누적 곡선 두 가지
    clean_asc = list(cm["x_clean"].astype(float).sort_values().index)
    noise_desc = list(cm[NOISE].astype(float).max(axis=1).sort_values(
        ascending=False).index)
    o_asc = [int(s.replace("enc", "")) - 1 for s in clean_asc]
    o_desc = [int(s.replace("enc", "")) - 1 for s in noise_desc]
    asc, desc = cumulative(ex, o_asc, K), cumulative(ex, o_desc, K)
    keep = ["누적_개수", "추가된_인코더", "조합"] + [f"{m}_중앙" for m in METRICS] \
        + ["R피크_진폭비_중앙"]
    asc[keep].round(4).to_csv(f"{outdir}/cumulative_clean_asc.csv", index=False,
                              encoding="utf-8-sig")
    desc[keep].round(4).to_csv(f"{outdir}/cumulative_noise_desc.csv", index=False,
                               encoding="utf-8-sig")
    fig_cumulative(asc, desc, f"{figdir}/cumulative_compare.png")
    print("④ clean 상관 오름차순 누적 마스킹")
    print(f"   순서: {' → '.join(clean_asc)}")
    print(asc[keep].round(4).to_string(index=False), "\n")
    print("④-대조 잡음 상관 내림차순 (기존)")
    print(f"   순서: {' → '.join(noise_desc)}")
    print(desc[keep].round(4).to_string(index=False), "\n")

    with open(f"{outdir}/note.txt", "w", encoding="utf-8") as f:
        f.write(
            "보조 실험 — 마스킹 순서 기준 검토. 본 노선(04)이 아니다.\n"
            "이미 산출한 03·04 결과만 다시 읽어 쓴다. 모델 추론을 하지 않는다.\n\n"
            "1) predictor_scatter.csv / predictor_corr.csv\n"
            "   단독 제거 효과(SNR_vs_M0, dB)와 03 지표(clean |r|, 잡음별 |r|,\n"
            "   잡음최대 |r|, clean - 잡음최대)의 관계. 인코더 8개가 표본이다.\n"
            "   피어슨·스피어만 상관을 함께 싣는다.\n\n"
            f"2) top20_combos.csv / encoder_frequency.csv\n"
            f"   {metric} 기준 상위 20개 조합과, 그 안에서 각 인코더가 마스킹된 비율.\n\n"
            "3) enc4_split.csv\n"
            "   전수 256조합을 enc4 포함/제외로 갈라 지표 분포(중앙값·최소·최대)를 비교.\n\n"
            "4) cumulative_clean_asc.csv / cumulative_noise_desc.csv\n"
            "   같은 조합 집합을 서로 다른 순서로 누적 마스킹했을 때의 지표 변화.\n"
            "   asc  = clean 상관 오름차순 (clean 과 덜 닮은 인코더부터)\n"
            "   desc = 잡음 3종 최대 상관 내림차순 (04 에서 쓰던 순서)\n\n"
            "해석은 붙이지 않는다.\n")
    print(f"산출물 → {outdir}/")
    return pred, t20, freq, sp, asc, desc


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--run", default="K8_seed42")
    p.add_argument("--split", default="val")
    p.add_argument("--metric", default="SNR_중앙", help="상위 20개를 고를 지표 열")
    p.add_argument("--outdir", default=None)
    a = p.parse_args()
    main(a.run, a.split, a.metric, a.outdir)
