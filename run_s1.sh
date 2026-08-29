#!/bin/sh
BASE="/c/Users/USER/AppData/Roaming/uv/python/cpython-3.9.21-windows-x86_64-none/python.exe"
export PYTHONIOENCODING=utf-8 PYTHONPATH="c:/dev/meae_xai/version5_supervised/.venv/Lib/site-packages"
# 기준선 재현 (dilations 없음 = 원본과 동일한 경로)
"$BASE" 02_model.py --k 4 --seed 42 --gamma 50 --group s1_해상도 --tag C256 \
    --channels 32,32,64,64,128,128,256,256 > _log_C256.log 2>&1
echo "=== C256 완료"
"$BASE" 02_model.py --k 4 --seed 42 --gamma 50 --group s1_해상도 --tag C64 \
    --channels 32,32,64,64,128,128 --dilations 1,1,1,1,1,8 > _log_C64.log 2>&1
echo "=== C64 완료"
"$BASE" 02_model.py --k 4 --seed 42 --gamma 50 --group s1_해상도 --tag C32 \
    --channels 32,32,64,64,128 --dilations 1,1,1,2,16 > _log_C32.log 2>&1
echo "=== C32 완료"
