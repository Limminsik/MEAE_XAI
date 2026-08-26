"""01 — 데이터셋 구축 (RESEARCH_DESIGN.md §4). 완료·동결.

MIT-BIH Arrhythmia(참조 ECG)와 NSTDB(실측 잡음)를 내려받아 검증하고, 기록 단위로 분할한 뒤,
10초 비중첩 분절마다 bw·ma·em 세 잡음을 SNR 무작위로 주입해 실험용 데이터셋을 만든다.

    python 01_build.py                # 전체 (내려받기 → 분할 → 생성 → 스팟체크)
    python 01_build.py --skip-download

산출
  data/processed/{train,val,test}.npz   분절당 x_clean·x_noisy·bw·ma·em·rpeaks·meta
  data/processed/split.json             기록 단위 분할 고정
  results/00_data_spotcheck/            무작위 3개 분절 스팟체크 그림

구현은 `src/data/{download,split,build}.py` 에 있다 — S1은 동결이므로 그대로 호출만 한다.
"""
import argparse

from src.data import build, download, split
from src.data.build import load_cfg


def main(config="configs/default.yaml", skip_download=False):
    cfg = load_cfg(config)
    if not skip_download:
        print("[01] 원본 내려받기·검증")
        download.download_mitdb(cfg["paths"]["mitdb"])
        if not download.verify(cfg):
            raise SystemExit("[01] 원본 검증 실패 — 진행하지 않는다")
    print("[01] 기록 단위 분할")
    split.main(config)
    print("[01] 분절 생성 + 잡음 주입")
    build.main(config)
    print("[01] 완료 — data/processed/ · results/00_data_spotcheck/")
    return cfg


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--skip-download", action="store_true")
    a = p.parse_args()
    main(a.config, a.skip_download)
