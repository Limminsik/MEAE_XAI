# meae_xai

**ECG 잡음 성분 판별 및 선택적 복원** — 잡음이 섞인 ECG로 다중 인코더 오토인코더(MEAE)를
자기지도 학습시킨 뒤, 어떤 인코더가 실제 잡음에 대응하는지 정량 판별하고, 그 인코더의
**인코딩만** 마스킹하여 신호를 복원한다.

> 주장의 구조: ① 분해된다 → ② **판별된다**(핵심) → ③ **판별이 진짜다**(마스킹 전후 인과 검증)

## 문서 지도

| 문서 | 역할 |
|---|---|
| [RESEARCH_DESIGN.md](RESEARCH_DESIGN.md) | **실행 지시서.** 모든 구현은 이 문서를 따른다 |
| [decision_log.md](decision_log.md) | 설계 결정 대장 — 무엇을·왜·어떤 증거로·어디에 반영했는지 |
| [OPEN_ITEMS.md](OPEN_ITEMS.md) | 미결 사항 보드 — 차단 요인과 다음 행동 |
| [MANUSCRIPT_PENDING.md](MANUSCRIPT_PENDING.md) | 원고 골격과의 불일치 (확정 후 일괄 개정) |
| [results/data_notes.md](results/data_notes.md) | 실측 수치 기록 — 원고 방법·고찰에 인용 |
| [CLAUDE.md](CLAUDE.md) | 작업 규칙 |

## 환경

```bash
uv init
uv venv --python 3.9.21
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu129
uv pip install -r requirements.txt
```

Python 3.9.21 / torch 2.8.0+cu129 / RTX 5060 Ti. `neurokit2==0.2.10` 고정
(0.2.12는 `float | None` 문법이라 py3.9에서 import 실패).

## 실행

```bash
python -m src.data.download            # MIT-BIH 내려받기·검증 (이어받기 지원)
python -m src.data.split               # 기록 단위 분할 → data/processed/split.json
python -m src.data.build               # 8,280 분절 생성 → data/processed/segments/
python -m pytest tests/ -q             # 33개 테스트
python -m src.train --k 8 --seed 42    # 학습 (--lambda-mixing 으로 λ 스윕)
python -m src.pilot_report --run K8_seed42   # 파일럿 보고 4항목
```

데이터는 git에서 제외된다(`data/`, `checkpoints/`, `logs/`, `figures/`, `external/`).

## 현재 진행 상황

T1–T5 완료. **본 학습 진입 전 정리 단계(T5.5).**

| 태스크 | 상태 |
|---|---|
| T1 환경 | 완료 |
| T2 데이터 구축 | 완료 — 8,280분절, SNR 오차 최대 2.9e-07 dB |
| T3 선행 저장소 이식 | 완료 — 모델·손실 무수정 이식 |
| T4 지표 | 완료 — 비교군 없음(S5 재정의) |
| T5 파일럿 | 완료 — λ 2조건. **Gate G1 조건1 미충족**(bw·ma 미분리) |
| T6 본 학습 6회 | **대기** — λ 확정 필요 (OPEN-01) |

## 재현성 스냅샷 — `pilot-complete`

태그 `pilot-complete` 시점의 동결본이 [snapshots/pilot-complete/](snapshots/pilot-complete/)에 있다.

| 파일 | 내용 |
|---|---|
| `default.yaml` | config 동결본 |
| `requirements.freeze.txt` | `uv pip freeze` (87개 패키지) |
| `environment.json` | Python·torch·CUDA·cuDNN·GPU·OS |
| `lambda_comparison.csv` | **λ=1e-3 vs 1e-2 비교표** |
| `corr_matrix_lam1e-3.csv` / `corr_matrix_lam1e-2.csv` | 각 조건 최고 체크포인트의 인코더×참조 \|r\| 행렬 |
| `history_lam1e-3.csv` / `history_lam1e-2.csv` | 에폭별 전체 로그(4개 손실 항, 분리 품질, tops, 붕괴 지표) |

git에서 제외되는 산출물의 위치:

| 산출물 | 경로 |
|---|---|
| 파일럿 손실 곡선 | `figures/pilot/loss_terms.png` |
| 대응 행렬 히트맵 | `figures/pilot/corr_heatmap.png` |
| 성분 파형 적층 | `figures/pilot/components.png`, `logs/<run>/plots/ep*.png` |
| 경계 아티팩트 | `figures/pilot/boundary.png` |
| 데이터 스팟체크 | `figures/spotcheck/*.png` |
| 체크포인트 | `checkpoints/K8_seed42.pt`, `checkpoints/K8_seed42_lam0.01.pt` |

재현 절차: 태그 체크아웃 → `requirements.freeze.txt`로 환경 복원 →
`src.data.split` → `src.data.build`(시드 고정, 비트 단위 재현 확인됨) → `src.train`.

## 파일럿 핵심 결과 (2026-08-25)

| | λ=1e-3 | λ=1e-2 |
|---|---|---|
| sparse mixing 감소율 | 8.3% | **59.8%** |
| 검증 재구성 | 0.0318 | **0.0299** |
| clean 최대 \|r\| | 0.435 | **0.555** |
| 선택된 체크포인트 | 에폭 5 (붕괴) | 에폭 45 |
| bw·ma가 같은 인코더인 에폭 | 61/63 | **63/63** |

심장 vs 비심장 분리는 명확하나(신호 인코더 격차 0.542), **bw와 ma는 분리되지 않는다.**
판정은 선등록 기준(`results/data_notes.md` §18)에 따라 λ=1e-1 결과를 본 뒤 확정한다.

## 선행 연구

MIT 라이선스 코드를 무수정 이식했다 (`src/model/_vendor_*.py`, 원 저작권 고지 유지).

- Webster MB, Lee J. *Blind source separation via multi-encoder autoencoders.*
  Neurocomputing 2025. doi:10.1016/j.neucom.2025.131008 — 커밋 `d0c94a9d`
- Webster MB, Lee D, Lee J. *Heart rate extraction from noisy PPG via multi-encoder
  autoencoders.* Comput Biol Med 2025;199:111319 — 커밋 `91f1e0e2`

## 데이터

- MIT-BIH Arrhythmia Database (PhysioNet) — 참조 ECG 46기록 (MLII 보유)
- MIT-BIH Noise Stress Test Database — 순수 잡음 `bw`/`ma`/`em`, 채널 0
- GalaxyPPG (CC-BY 4.0) — S6 외부 시연용 Polar H10 ECG
