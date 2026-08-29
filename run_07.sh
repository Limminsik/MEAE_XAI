#!/bin/sh
BASE="/c/Users/USER/AppData/Roaming/uv/python/cpython-3.9.21-windows-x86_64-none/python.exe"
export PYTHONIOENCODING=utf-8 PYTHONPATH="c:/dev/meae_xai/version5_supervised/.venv/Lib/site-packages"
for s in vitaldb mimic_iv galaxyppg; do
  "$BASE" 07_validation.py --run C16_seed42 --source $s \
      --outdir "results/07_validation/C16_seed42/$s" > _v07_$s.log 2>&1 \
      && echo "=== $s 완료" || echo "=== $s 실패"
done
