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
| 01 데이터셋 구축 | ✅ 완료·동결 (모델 이식 포함) |
| 02 모델·학습 | ✅ 완료 — K=8 seed 42, **에폭 88** 확정 + 충실도 진단 |
| 03 분리·대응 분석 | ✅ 종료 — 지표 3종, **val · test** 산출 완료 |
| 04 마스킹 복원 평가 | ✅ 전수 256조합 val 산출 완료. **최적 조합 선정은 보류** |
| 05 외부 적용 | ✅ VitalDB · MIMIC-IV · GalaxyPPG 산출 완료 |
| S7 통계·원고 | ⏸ |

**다음**: 성분 분리 강화 방안 검토 (사용자). 그 뒤 통계·원고.
**추가 구조·손실·K·시드 탐색은 하지 않는다.**

보조 실험은 `experiments/masking_strategy/` 에 있다 — remove-some 대 keep-one 대조,
신호 공간 뺄셈, 마스킹 순서 기준 검토. 본 노선이 아니며 `results/` 에 섞지 않는다.

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
(근거는 RESEARCH_DESIGN §7. 대조 산출은 `_work/archive/` 에 보존).
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
| **S4 성분 분리** | 성분·참조를 **분절 내 표준화**(평균 0, SD 1)한 뒤 세 지표. 모두 분절 내 산출 → 900분절 평균±SD(ddof=1)<br>**\|r\|** 분절 내 Pearson 절댓값 · **RMSE** 표준화 신호 차이의 RMS · **MAD** 그 차이의 최댓값<br>표시는 **열별** — \|r\|은 상위 2, RMSE·MAD는 하위 2. 세 지표의 행별 지목 일치 여부 병기<br>p값·인코더 명명·해석 없음 |
| **S5 디노이징** | 마스킹 전후 SNR·RMSE·R-피크 F1·SDNN (clean 참조 기준) + 아블레이션 + 전수 2^K 지도 |
| **S6 외부** | 정량 성능 주장 없음. 적용 가능성 시연 |

**마스킹은 추론 시점 조작이다.** 가중치를 다시 학습하지 않고 인코딩만 0으로 치환한다.

---

## 3. 읽는 순서 — 어떤 `.py` 를 어떤 순서로 보는가

**실제로 쓰는 코드는 번호 붙은 다섯 개뿐이다. 번호가 곧 단계 순서다.**
각 파일 맨 위 독스트링에 그 단계의 명세(수식·규칙·산출물 목록)가 전부 들어 있다.

| # | 파일 | 무엇을 하나 | 무엇이 나오나 |
|---|---|---|---|
| **01** | [01_build.py](01_build.py) | 원본 내려받기·검증 → 기록 단위 분할 → 10초 분절 → bw·ma·em 주입 | `data/processed/*.npz` · `results/01_build/` |
| **02** | [02_model.py](02_model.py) | 구조 세팅 · 비용 함수 ①(학습 손실) · 비용 함수 ②(체크포인트 선택) · 학습<br>`--diagnose` 는 재구성 충실도 진단만 | `results/02_model/<run>/` (+`fidelity/`) |
| **03** | [03_bss.py](03_bss.py) | 성분 `x̂_k` 추출 → 참조 4종과 대응 분석 (지표 3종) | `results/03_bss/<run>/<split>/` |
| **04** | [04_masked_denoising.py](04_masked_denoising.py) | 전수 2^K 마스킹 복원 → clean 기준 채점 (지표 5종) | `results/04_masked_denoising/<run>/<split>/` |
| **05** | [05_validation.py](05_validation.py) | 외부 데이터 적용 시연 | `results/05_validation/<run>/<source>/` |

`src/` 는 위 다섯이 불러 쓰는 라이브러리다. 단계 흐름을 따라갈 때는 볼 필요가 없고,
계산의 세부를 확인할 때만 들어간다.

| 모듈 | 누가 쓰나 | 내용 |
|---|---|---|
| `src/data/` | 01 | 내려받기·분할·분절 생성·데이터셋 로더 (동결) |
| `src/model/` | 02~05 | MEAE 구조·손실·선행 이식본 (동결) |
| `src/core.py` | 02~05 | 체크포인트 로드 · 성분/재구성 추출 · 분절 내 표준화와 지표 · 표 렌더링 |
| `src/spectral.py` | 02 | PSD·대역별 보존율·기울기·보존 곡선 |
| `src/metrics.py` | 04 | 채점 지표 — 04의 5종 + R-피크·SNR·RMSE·SDNN |
| `src/viz.py` | 공용 | 한글 폰트 설정 (Malgun Gothic) |

**새 분석이 생기면 그 단계 스크립트 안에 넣는다.** 새 파일을 만들지 않고,
여러 단계가 함께 쓰는 것만 `src/core.py` 로 올린다.

---

## 4. 폴더 구조

```
meae_xai/
├── README.md  RESEARCH_DESIGN.md  CLAUDE.md
├── configs/default.yaml        모든 설정. 코드 하드코딩 금지
│
├── 01_build.py                 데이터셋 구축 — 로드·검증·분할·분절·잡음 주입
├── 02_model.py                 모델·학습 — 구조 세팅, 두 비용 함수, 체크포인트 선택
│                               `--diagnose` 로 재구성 충실도 진단
├── 03_bss.py                   성분 분리 + 참조 대응 분석 (지표 3종)
├── 04_masked_denoising.py      마스킹 복원 평가 (전수 2^K, 지표 5종)
├── 05_validation.py            외부 데이터 적용 시연
│
├── src/                        위 다섯이 공유하는 라이브러리
│   ├── core.py                 체크포인트 로드·성분 추출·표준화 지표·표 렌더링
│   ├── data/{download,split,build,dataset}.py
│   ├── model/{meae,losses,_vendor_*}.py
│   ├── metrics.py              S5 지표 5종 · R-피크 · SNR/RMSE/SDNN
│   └── spectral.py  viz.py
├── tests/                      27개
├── data/                       원본·분절 (git 제외)
│
├── results/               ★ 산출물 — 폴더 번호가 스크립트 번호와 같다
│     01_build/                 스팟체크 · 주입 부록
│     02_model/<run>/           가중치·history·selection·stage1·pool/·plots/
│                               + epoch_metrics/(에폭별 지표) + fidelity/(충실도 진단)
│     03_bss/<run>/<split>/     대응표 3종·일치표·그림 (val · test)
│     04_masked_denoising/<run>/<split>/   전수 256조합·기준선·단독·누적·R피크
│     05_validation/<run>/<source>/        외부 적용 (vitaldb · mimic_iv · galaxyppg)
│     05_validation/_check/     원 파형 점검 · 품질 조사
│
├── experiments/           보조 실험 (본 노선 아님)
│     masking_strategy/         remove-some 대 keep-one · 신호 공간 뺄셈 · 마스킹 순서
└── _work/archive/         과거 실행·구버전·구코드 (보존, 실행 대상 아님)
```

## 5. 실행

```bash
uv init && uv venv --python 3.9.21
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu129
uv pip install -r requirements.txt
```

```bash
python 01_build.py                                       # 데이터셋 구축 + 주입 부록
python -m pytest tests/ -q
python 02_model.py --k 8 --seed 42                       # 학습 (1단계 관문까지)
python 02_model.py --epoch-metrics --run K8_seed42       # 2단계 — 에폭 선정 + 표
python 02_model.py --diagnose     --run K8_seed42        # 재구성 충실도 진단
python 03_bss.py              --run K8_seed42 --split val    # (--split test)
python 04_masked_denoising.py --run K8_seed42 --split val
python 05_validation.py --run K8_seed42 --source vitaldb     # mimic_iv · galaxyppg
python 05_validation.py --survey --source vitaldb            # 분절 품질 점검
```

한글 콘솔 출력이 있으므로 `PYTHONIOENCODING=utf-8` 로 실행한다 (Windows cp949 오류).

Python 3.9.21 / torch 2.8.0+cu129 / RTX 5060 Ti. `neurokit2==0.2.10` 고정
(0.2.12는 `float | None` 문법이라 py3.9 import 실패).

## 6. 외부 데이터 (S6)

연구계획서 기반연구 ①에 명시된 데이터셋.

| 데이터 | 환경 | 경로 · 규격 |
|---|---|---|
| MIMIC-IV Waveform | 중환자실 | `D:/data/mimic_iv_waveform` · ECG 249.9 Hz, 유도 II |
| VitalDB | 수술실 | `D:/data/VitalDB` |
| NSRR (shhs·wsc·nfs) | 수면 | `D:/data/{shhs,wsc,nfs}` |
| GalaxyPPG (Polar H10 ECG) | 일상 웨어러블 | `data/GalaxyPPG` · 130 Hz |

## 7. 선행 연구

MIT 라이선스 코드를 무수정 이식했다 (`src/model/_vendor_*.py`, 원 저작권 고지 유지).

- Webster MB, Lee J. *Blind source separation via multi-encoder autoencoders.*
  Neurocomputing 2025. doi:10.1016/j.neucom.2025.131008 — 커밋 `d0c94a9d`
- Webster MB, Lee D, Lee J. *Heart rate extraction from noisy PPG via multi-encoder
  autoencoders.* Comput Biol Med 2025;199:111319 — 커밋 `91f1e0e2`

선행은 학습 후 어느 인코더가 원하는 소스를 만드는지 **수동 검사가 필요하다**고 명시한다.
또한 인코딩 크기를 "너무 작으면 좋은 재구성이 불가하고 너무 크면 인코더가 특화 대신
전체 특징 공간으로 일반화된다"고 명시한다 (재구성–분리 트레이드오프).

## 8. 보조 기록

본 노선에 채택하지 않은 실험. 결과 파일은 아래 위치에 보관한다.

| 실험 | 위치 |
|---|---|
| K 비교 (K=4·8·16 × 시드 3) | `_work/archive/experiments/ssl/outputs/` |
| 잡음 지도(supervised) 방식 | `_work/archive/experiments/supervised_noise/outputs/` |
| 차분 손실 γ · λ_z 제거 · hidden 128 | `_work/archive/` |
