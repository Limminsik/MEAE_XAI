#!/bin/sh
BASE="/c/Users/USER/AppData/Roaming/uv/python/cpython-3.9.21-windows-x86_64-none/python.exe"
export PYTHONIOENCODING=utf-8 PYTHONPATH="c:/dev/meae_xai/version5_supervised/.venv/Lib/site-packages"
for r in C16_res_w0 C16_res_w01 C16_res_w03 C16_res_w10; do
  d=experiments/02_model/s3_잔차/$r
  "$BASE" 04_masked_denoising.py --run "$d/$r.pt" --split val --three-ways \
      --outdir "$d/restore" > _fix_$r.log 2>&1
  echo "=== $r restore 재산출"
done
