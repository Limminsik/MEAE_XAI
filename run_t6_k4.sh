#!/bin/sh
set -e
for S in 42 202 2026; do
  echo "=== K=4 seed=$S 시작 $(date +%H:%M:%S) ==="
  PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe -m src.train --k 4 --seed $S --plot-every 20
done
echo "=== K=4 완료 $(date +%H:%M:%S) ==="
