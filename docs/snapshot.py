"""docs/snapshot.py — 대시보드용 결과 스냅샷.

    python docs/snapshot.py

results/ 의 CSV 를 docs/data/snapshot.json 하나로 모은다. 대시보드는 먼저 저장소(상대
경로 → GitHub raw) 에서 CSV 를 직접 읽고, 그것이 막힌 환경(로컬 file://, 오프라인)에서만
이 스냅샷으로 대신한다. 결과를 다시 내고 push 하면 대시보드는 스냅샷 없이도 최신이다 —
스냅샷은 편의용이지 진실의 원천이 아니다.
"""
import glob
import io
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PATTERNS = ["results/04_masked_denoising/*/test/three_ways.csv",
            "results/04_masked_denoising/*/test/breakdown.csv",
            "results/06_ablation/*/test/sqi_summary.csv",
            "results/06_ablation/*/test/metric_summary.csv",
            "results/02_model/*/history.csv",
            "results/07_validation/*/*/sqi_summary.csv"]
snap = {}
for pat in PATTERNS:
    for p in sorted(glob.glob(os.path.join(ROOT, pat))):
        rel = os.path.relpath(p, ROOT).replace(os.sep, "/")
        snap[rel] = io.open(p, encoding="utf-8-sig").read()
figs = sorted(os.path.relpath(p, ROOT).replace(os.sep, "/")
              for p in glob.glob(os.path.join(ROOT, "results", "**", "*.png"), recursive=True))
out = os.path.join(ROOT, "docs", "data", "snapshot.json")
json.dump({"csv": snap, "figures": figs}, io.open(out, "w", encoding="utf-8"), ensure_ascii=False)
print(f"CSV {len(snap)}개 · 그림 {len(figs)}개 → {os.path.relpath(out, ROOT)}")
