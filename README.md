# meae_xai — 연구 현황판

**의료 생체신호 품질 고도화를 위한 설명 가능한 디노이징.**
잡음 섞인 ECG로 다중 인코더 오토인코더(MEAE)를 자기지도 학습시켜 신호 성분을 분리하고,
각 성분이 어떤 잡음에 대응하는지 참조 신호로 정량 평가한 뒤,
잡음 성분의 **인코딩만 마스킹**하여 디노이징한다.

문서는 **이 파일**과 [RESEARCH_DESIGN.md](RESEARCH_DESIGN.md) 둘뿐이다.
[CLAUDE.md](CLAUDE.md)는 작업 규칙.

---

## 1. 현재 단계

| 단계 | 상태 |
|---|---|
| S1 데이터 구축 | ✅ 완료·동결 |
| S2 모델 이식 | ✅ 완료·동결 |
| **S3 학습** | 🔧 **여기** — 손실·선택기준 확정, 최종 실행 |
| S4 성분 분리 평가 | ⏸ 도구 완성, 실행 대기 |
| S5 마스킹·아블레이션 | ⏸ 설계 확정, 미실행 |
| S6 외부 적용 | ⏸ 설계 확정, 미실행 |

**다음**: S3 실행 완료 → 산출물 확인 → T6.9 리허설 → `pre-test-freeze` → **승인 후** S4(test 봉인 해제)
→ S5 → S6 → 통계·원고. **추가 구조·손실·K·시드 탐색 없음.**

---

## 2. 확정 사항

### 모델

| 항목 | 값 |
|---|---|
| 구조 | 선행 MEAE 무수정 이식 (`src/model/_vendor_meae.py`) |
| K (인코더 수) | **8** — 소스 수 비가정 설정, 선행 ECG 실험 기준선 |
| hidden | 64 (인코더당 인코딩 8채널) |
| channels | [32, 32, 64, 64, 128, 128, 256, 256] |
| 입력 | 3600 → **3840** 대칭 제로 패딩 (양쪽 120). 지표는 중앙 3600 크롭 후 계산 |
| 시드 | **42 단일**. 다중 시드 반복 없음 |

### 비용 함수 ① — 학습 손실

```
L = MSE(x̂, x_noisy) + λ_m·L_mix + λ_o·‖D(0)‖² + λ_z·Σ_k ‖z_k‖²/h

λ_m = 1e-2   sparse mixing        디코더 비대각 가중치 L1 (선행 Alternative 구현)
λ_o = 1e-2   zero reconstruction  전영 인코딩 → 출력 0
λ_z = 1e-3   인코딩 L2
```

**참조 신호(clean·bw·ma·em)를 일절 사용하지 않는다.** 입력도 목표도 `x_noisy` 하나다.

### 비용 함수 ② — 체크포인트 선택

```
x̂_k    = D(0,…,z_k,…,0)                    k번째 인코딩만 남긴 재구성, 중앙 3600 크롭
ρ_k(t) = median_{s∈V} |ρ(x̂_k, x_clean)|    |V| = 300 고정
S(t)   = max_k ρ_k(t)

1단계  C  = { t : L_recon^val(t) ≤ 1.5 × min_τ L_recon^val(τ) }
2단계  t* = argmax_{t∈C} S(t)
```

2에폭 간격 산출 · 학습 종료 후 전체 이력 일괄 판정 · 후보 구간 가중치 보관.
배율 민감도 {1.2, 1.5, 2.0}을 함께 산출한다. **사전 등록한 1.5를 유지한다** —
1.2가 고르는 에폭 88과 실물 대조 결과 역할 구조가 동일하고 지표 차이가 미미했다
(근거: `results/00_rehearsal/epoch_compare/`, RESEARCH_DESIGN §7).
`x_clean`은 이 선택에만 쓰이고 가중치 갱신에는 관여하지 않는다.

### 데이터 (S1, 동결)

| 항목 | 값 |
|---|---|
| 소스 | MIT-BIH Arrhythmia (MLII, 102·104 제외 46기록) + NSTDB `bw`·`ma`·`em` 채널 0 |
| 분절 | 10초 비중첩 **8,280개** (train 32 / val 5 / test 9 기록) |
| 주입 | 세 잡음 항상 주입, SNR 잡음별 독립 균등추첨 **[0, 12] dB** |
| 신호 전력 | 분산 기준 `var(x_clean)`, 잡음 전력 `mean(n²)` |
| 잡음 원본 | 시간 70/30 분할 (train 21.1분 / val·test 9.0분) |
| 분할 | 기록 단위, 층화 없음, `split_seed 42` 고정 |
| R-피크 | 104,748개 (`f` 심볼 포함) |

### 평가 지표

| 단계 | 지표 |
|---|---|
| **S4 성분 분리** | [S4-01] 상관 — 분절 내 Pearson → 분절 간 `ρ̄ = mean_s\|ρ\|`, `σ = std_s\|ρ\|`(ddof=1), 행별 최댓값 표시<br>[S4-02] 정규화 RMSE — 분절 내 표준화 → 분절 내 RMSE → 분절 간 평균±SD(ddof=1), 행별 최솟값 표시<br>[S4-03] MAD — 같은 표준화 → 분절 내 `max_i\|ã−r̃\|` → 분절 간 평균±SD(ddof=1), 단위 표준편차, 행별 최솟값 표시<br>r² 병기, 부호 분포·MAD 발생 시점 분포 별도 기록. p값·명명·해석 없음 |
| **S5 디노이징** | 마스킹 전후 SNR·RMSE·R-피크 F1·SDNN (clean 참조 기준) + 아블레이션 + 전수 2^K 지도 |
| **S6 외부** | 정량 성능 주장 없음. 적용 가능성 시연 |

**마스킹은 추론 시점 조작이다.** 가중치를 다시 학습하지 않고 인코딩만 0으로 치환한다.

---

## 3. 폴더 구조

```
meae_xai/
├── README.md  RESEARCH_DESIGN.md  CLAUDE.md
├── configs/default.yaml        모든 설정. 코드 하드코딩 금지
├── src/                        본 실험 코드
├── tests/                      33개
├── data/                       원본·분절 (git 제외)
│
├── results/               ★ 본 실험 산출물
│     01_train/<run>/          가중치 · history.csv · selection.json · console.log · pool/ · plots/
│     02_separation/           S4 — r·RMSE 표, 성분 그림, 충실도·스펙트럼
│     03_denoising/            S5 — 마스킹 M0–M5, 아블레이션, 전수 지도
│     04_external/             S6 — 외부 적용
│
├── experiments/           보조 기록 (본 노선 아님)
│     ssl/                     K 비교 (K=4·8·16 × 시드 3)
│     supervised_noise/        잡음 지도 방식 — 중단
│
└── _work/archive/         폐기된 실행·구버전
```

## 4. 실행

```bash
uv init && uv venv --python 3.9.21
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu129
uv pip install -r requirements.txt
```

```bash
python -m src.data.download     # MIT-BIH 내려받기·검증
python -m src.data.split        # 기록 단위 분할
python -m src.data.build        # 8,280 분절 생성
python -m pytest tests/ -q
python -m src.train --k 8 --seed 42          # S3
python -m src.s4_identify --run K8_seed42    # S4
```

Python 3.9.21 / torch 2.8.0+cu129 / RTX 5060 Ti. `neurokit2==0.2.10` 고정
(0.2.12는 `float | None` 문법이라 py3.9 import 실패).

## 5. 외부 데이터 (S6)

연구계획서 기반연구 ①에 명시된 데이터셋.

| 데이터 | 환경 | 경로 · 규격 |
|---|---|---|
| MIMIC-IV Waveform | 중환자실 | `D:/data/mimic_iv_waveform` · ECG 249.9 Hz, 유도 II |
| VitalDB | 수술실 | `D:/data/VitalDB` |
| NSRR (shhs·wsc·nfs) | 수면 | `D:/data/{shhs,wsc,nfs}` |
| GalaxyPPG (Polar H10 ECG) | 일상 웨어러블 | `data/GalaxyPPG` · 130 Hz |

## 6. 선행 연구

MIT 라이선스 코드를 무수정 이식했다 (`src/model/_vendor_*.py`, 원 저작권 고지 유지).

- Webster MB, Lee J. *Blind source separation via multi-encoder autoencoders.*
  Neurocomputing 2025. doi:10.1016/j.neucom.2025.131008 — 커밋 `d0c94a9d`
- Webster MB, Lee D, Lee J. *Heart rate extraction from noisy PPG via multi-encoder
  autoencoders.* Comput Biol Med 2025;199:111319 — 커밋 `91f1e0e2`

선행은 학습 후 어느 인코더가 원하는 소스를 만드는지 **수동 검사가 필요하다**고 명시한다.
또한 인코딩 크기를 "너무 작으면 좋은 재구성이 불가하고 너무 크면 인코더가 특화 대신
전체 특징 공간으로 일반화된다"고 명시한다 (재구성–분리 트레이드오프).

## 7. 보조 기록

본 노선에 채택하지 않은 실험. 결과 파일은 아래 위치에 보관한다.

| 실험 | 위치 |
|---|---|
| K 비교 (K=4·8·16 × 시드 3) | `experiments/ssl/outputs/` |
| 잡음 지도(supervised) 방식 | `experiments/supervised_noise/outputs/` |
| 차분 손실 γ · λ_z 제거 · hidden 128 | `_work/archive/` |
