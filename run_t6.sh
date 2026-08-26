#!/bin/sh
# 본 학습 6회 — K {8,4} x 시드 {42,202,2026}. config 동결값 사용.
# 산출물: _work/runs/K<k>_seed<seed>/  (가중치 · history.csv · checkpoints.json · plots/)
set -e
for K in 8 4; do
  for S in 42 202 2026; do
    echo "=== K=$K seed=$S 시작 $(date +%H:%M:%S) ==="
    PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe -m src.train --k $K --seed $S --plot-every 20
  done
done
echo "=== 전체 완료 $(date +%H:%M:%S) ==="
