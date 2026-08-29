#!/bin/sh
BASE="/c/Users/USER/AppData/Roaming/uv/python/cpython-3.9.21-windows-x86_64-none/python.exe"
export PYTHONIOENCODING=utf-8 PYTHONPATH="c:/dev/meae_xai/version5_supervised/.venv/Lib/site-packages"
for r in C256 C64 C32; do
  d=experiments/02_model/s1_해상도/$r
  "$BASE" 06_ablation.py --run "$d/$r.pt" --split val --outdir "$d/ablation" > _abl_$r.log 2>&1
  echo "=== $r ablation 완료"
done
