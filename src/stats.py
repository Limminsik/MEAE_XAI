"""S7 통계 (RESEARCH_DESIGN.md §10).

분절 쌍대 Wilcoxon 부호순위 + Holm 보정, α=.05, 효과크기 r = Z/√N.
표본이 크므로 정규근사 Z를 쓰고, 동점(차이 0)은 Wilcoxon 관례대로 제외한다.
"""
from typing import Dict, Sequence

import numpy as np
import pandas as pd
from scipy import stats


def wilcoxon_pair(a: Sequence[float], b: Sequence[float]) -> Dict[str, float]:
    """쌍대 비교 a vs b. 반환: n, 중앙차, W, Z, p, 효과크기 r."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    m = np.isfinite(a) & np.isfinite(b)
    d = a[m] - b[m]
    d = d[d != 0]                      # 동점 제외 (Wilcoxon 관례)
    n = len(d)
    if n < 10:
        return {"n": n, "median_diff": float(np.median(d)) if n else np.nan,
                "W": np.nan, "Z": np.nan, "p": np.nan, "r": np.nan}
    res = stats.wilcoxon(d, alternative="two-sided", method="approx")
    z = float(stats.norm.isf(res.pvalue / 2)) * np.sign(np.median(d))
    return {"n": n, "median_diff": float(np.median(d)), "W": float(res.statistic),
            "Z": z, "p": float(res.pvalue), "r": float(abs(z) / np.sqrt(n))}


def holm(pvals: Sequence[float], alpha: float = 0.05):
    """Holm–Bonferroni. 반환: (보정 p, 기각 여부)."""
    p = np.asarray(pvals, dtype=np.float64)
    ok = np.isfinite(p)
    order = np.argsort(np.where(ok, p, np.inf))
    m = int(ok.sum())
    adj = np.full(len(p), np.nan)
    running = 0.0
    for rank, i in enumerate(order[:m]):
        running = max(running, (m - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj, np.where(np.isfinite(adj), adj < alpha, False)


def paired_table(df: pd.DataFrame, value: str, cond: str, base: str,
                 by: str = "seg", alpha: float = 0.05) -> pd.DataFrame:
    """long 형식 표에서 base 조건 대비 나머지 조건 전부를 쌍대 검정하고 Holm 보정."""
    wide = df.pivot(index=by, columns=cond, values=value)
    rows = []
    for c in wide.columns:
        if c == base:
            continue
        r = wilcoxon_pair(wide[c].values, wide[base].values)
        rows.append({"조건": c, "기준": base, "지표": value, **r})
    out = pd.DataFrame(rows)
    if len(out):
        out["p_holm"], out["유의"] = holm(out.p.values, alpha)
    return out


def describe(x: Sequence[float]) -> Dict[str, float]:
    """중앙값[IQR] 보고용 (§8: 10초 창 SDNN은 불안정하므로 중앙값[IQR])."""
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if not len(x):
        return {"n": 0, "median": np.nan, "q1": np.nan, "q3": np.nan,
                "mean": np.nan, "sd": np.nan}
    return {"n": len(x), "median": float(np.median(x)),
            "q1": float(np.percentile(x, 25)), "q3": float(np.percentile(x, 75)),
            "mean": float(x.mean()), "sd": float(x.std(ddof=1)) if len(x) > 1 else 0.0}
