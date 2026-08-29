#!/bin/sh
BASE="/c/Users/USER/AppData/Roaming/uv/python/cpython-3.9.21-windows-x86_64-none/python.exe"
export PYTHONIOENCODING=utf-8 PYTHONPATH="c:/dev/meae_xai/version5_supervised/.venv/Lib/site-packages"
for w in 0.1 0.3 1.0; do
  t=$(echo $w | tr -d '.')
  "$BASE" 02_model.py --k 4 --seed 42 --gamma 50 --group s3_잔차 --tag C16_res_w$t \
      --channels 32,32,64,64 --dilations 1,1,4,32 --skip-levels 0 --skip-weight $w \
      > _log_res_w$t.log 2>&1
  echo "=== 잔차 가중치 $w 완료"
done
