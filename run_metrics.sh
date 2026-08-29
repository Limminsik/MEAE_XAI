#!/bin/sh
BASE="/c/Users/USER/AppData/Roaming/uv/python/cpython-3.9.21-windows-x86_64-none/python.exe"
export PYTHONIOENCODING=utf-8 PYTHONPATH="c:/dev/meae_xai/version5_supervised/.venv/Lib/site-packages"
R="C16_seed42"
"$BASE" 03_bss.py --run "$R" --split test --final --outdir "results/03_bss/$R/test" > _m03.log 2>&1 && echo "=== 03"
"$BASE" 04_masked_denoising.py --run "$R" --split test --three-ways --outdir "results/04_masked_denoising/$R/test" > _m04.log 2>&1 && echo "=== 04"
"$BASE" 06_ablation.py --run "$R" --split test --outdir "results/06_ablation/$R/test" > _m06.log 2>&1 && echo "=== 06"
