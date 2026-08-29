#!/bin/sh
BASE="/c/Users/USER/AppData/Roaming/uv/python/cpython-3.9.21-windows-x86_64-none/python.exe"
export PYTHONIOENCODING=utf-8 PYTHONPATH="c:/dev/meae_xai/version5_supervised/.venv/Lib/site-packages"
R="C16_seed42"
while [ ! -f results/02_model/$R/stage1.json ]; do sleep 30; done
echo "=== 02 완료"
"$BASE" 03_bss.py --run "$R" --split test --final --outdir "results/03_bss/$R/test" > _p03.log 2>&1 && echo "=== 03 완료"
"$BASE" 04_masked_denoising.py --run "$R" --split test --three-ways \
    --outdir "results/04_masked_denoising/$R/test" > _p04.log 2>&1 && echo "=== 04 완료"
"$BASE" 05_figure.py --run "$R" --split test \
    --outdir "results/05_figure/$R/test" > _p05.log 2>&1 && echo "=== 05 완료"
"$BASE" 06_ablation.py --run "$R" --split test \
    --outdir "results/06_ablation/$R/test" > _p06.log 2>&1 && echo "=== 06 완료"
for s in vitaldb mimic_iv galaxyppg; do
  "$BASE" 07_validation.py --run "$R" --source $s \
      --outdir "results/07_validation/$R/$s" > _p07_$s.log 2>&1 && echo "=== 07 $s 완료"
done
