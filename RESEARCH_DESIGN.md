# RESEARCH_DESIGN.md
# ECG 잡음 성분 판별 및 선택적 복원 — 실행 지시서

> **이 문서의 용도**: Claude Code가 이 문서만 읽고 실험 전체를 수행할 수 있도록 작성된 실행 지시서.
> 프로젝트 루트에 배치하고 `CLAUDE.md`에 다음을 추가한다:
> `모든 구현은 RESEARCH_DESIGN.md를 따른다. 설계와 다른 구현이 필요하면 진행하지 말고 사용자에게 먼저 보고한다.`

**연구 한 줄 요약**: 잡음이 섞인 ECG로 다중 인코더 오토인코더를 자기지도 학습시킨 뒤, 어떤 인코더가 실제 잡음(bw/ma/em)에 대응하는지 정량 판별하고, 그 인코더의 인코딩만 마스킹하여 신호를 복원한다.

---

## 0. 불변 원칙 (위반 금지 — 위반이 필요하면 사용자에게 보고)

1. **마스킹 대상은 인코더가 아니라 인코딩(인코더 출력 텐서)**이다. 인코더 가중치는 그대로 두고 순전파도 모두 수행하되, 디코더 입력 직전에 해당 인코더의 출력만 영벡터로 치환한다. 선행 MEAE는 완전 합성곱 구조이므로 인코딩은 (채널 × 시간) 특성맵이며 VAE식 잠재 벡터가 아니다. 문서·코드·원고 전반에서 용어는 **인코딩(encoding)**으로 통일한다.
2. **구조는 선행 MEAE를 차용**한다. 동일 입력을 받는 K개 완전 합성곱 인코더 → 인코딩을 채널 축으로 결합 → 단일 공유 디코더. 성분 x̂_k는 k번째 인코딩만 남기고 나머지 인코딩을 0으로 치환한 재구성이다. 임의 변경 금지 구현 전 §5A의 공개 저장소를 반드시 확인한다.
3. **학습은 자기지도**다. 입력 = 잡음 섞인 분절, 목표 = 동일 분절. **clean이나 잡음 정답을 손실에 절대 사용하지 않는다.** clean을 목표로 주면 인코더가 잡음을 표현할 이유가 없어져 판별 대상 자체가 사라진다.
4. **정답(clean, bw, ma, em)은 채점표이지 교과서가 아니다.** 판별 평가와 체크포인트 선택에만 쓰고, 가중치 갱신에는 관여하지 않는다.
5. **데이터 분할은 기록(환자) 단위**로만 한다. 분절 단위 분할은 누수다.
6. **S6에서 재학습·재판별 금지.** S4가 확정한 `configs/noise_encoders.yaml`을 그대로 적용한다.

---

## 1. 환경 설정

```bash
uv init                                   # 프로젝트 루트: meae_xai/
uv venv --python 3.9.21
source .venv/bin/activate                 # Windows: .venv\Scripts\activate
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu129
uv pip install -r requirements.txt        # 나머지 의존성은 requirements.txt에만 적는다
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```
- 실측 환경: Python 3.9.21 / torch 2.8.0+cu129 / cuDNN 91002 / **NVIDIA RTX 5060 Ti** / CUDA 12.9
- **`neurokit2==0.2.10` 고정.** 0.2.12는 `float | None` 문법이라 py3.9에서 import 실패한다. `pytest` 포함.
- 동결본: `snapshots/pilot-complete/{requirements.freeze.txt, environment.json}`
- Python 3.9와 휠 호환 문제 발생 시 **버전을 임의 변경하지 말고 보고**한다.

## 2. 저장소 구조

```
meae_xai/
├── RESEARCH_DESIGN.md / CLAUDE.md
├── configs/
│   ├── default.yaml              # 모든 설정. 코드 하드코딩 금지
│   └── noise_encoders.yaml       # S4 산출물 (S5·S6가 읽는 유일한 판별 결과)
├── data/raw/ data/processed/     # git 제외
├── src/
│   ├── data/{download.py, build.py, external.py}
│   ├── model/{meae.py, losses.py}
│   ├── train.py
│   ├── s4_identify.py  s5_restore.py  s6_external.py
│   ├── metrics.py  stats.py  viz.py            # 비교군 baselines.py 없음 (S5 재정의)
├── tests/  results/  figures/  checkpoints/  logs/
```

## 3. configs/default.yaml (현행값)

동결본: `snapshots/pilot-complete/default.yaml`. **코드 하드코딩 금지 (§13).**
`loss.lambda_mixing`은 파일럿 스윕 중이라 잠정값이다 → `OPEN_ITEMS.md` OPEN-01.

```yaml
# meae_xai — 전역 설정 (RESEARCH_DESIGN.md §3)
# 코드 하드코딩 금지. 모든 실험 파라미터는 이 파일에서만 바꾼다.

seeds: [42, 202, 2026]

paths:
  mitdb: "data/raw/mit-bih-arrhythmia-database-1.0.0"
  nstdb: "data/mit-bih-noise-stress-test-database-1.0.0"
  galaxyppg: "data/GalaxyPPG/GalaxyPPG/Dataset"
  mimic3: "data/raw/mimic3wdb"
  processed: "data/processed"

data:
  fs: 360
  seg_sec: 10                        # 3600 샘플
  mitdb_exclude: ["102", "104"]      # MLII 없음 → 46개 기록
  lead: "MLII"
  noise_channel: 0                   # NSTDB 잡음 레코드는 2채널(noise1/noise2).
                                     # nst 도구가 채널 번호끼리 짝지어 섞는 관례에 따라,
                                     # MLII가 0번 채널이므로 잡음도 0번(noise1)을 쓴다.
  noise_split_ratio: 0.7             # 잡음 레코드 앞 70% = 학습용, 뒤 30% = 검증·평가용
  noise_snr_range_db: [0, 12]        # 잡음별 독립 추첨 (확정)
  # 선행 인코더 깊이가 8(MaxPool 2배 x 8 = 256배)이라 입력 길이가 256의 배수여야 한다.
  # 3600 -> 3840 = 256 x 15. 선행 후속 저장소도 6000 -> 6144(=256x24)로 대칭 제로 패딩했다.
  pad_to: 3840                       # 모델 입력 길이
  pad_each: 120                      # 양쪽 대칭 제로 패딩 (3840-3600)/2
  # 모든 상관·지표는 패딩을 제외한 중앙 3600 구간을 크롭한 뒤 계산한다.
  # R-피크 인덱스는 크롭 좌표(0~3599) 기준을 유지한다.
  split:
    unit: "record"                   # 기록(환자) 단위 분할만. 분절 단위는 누수 (§0 원칙 5)
    counts: {train: 32, val: 5, test: 9}   # 46 x (0.7/0.1/0.2) = 32.2/4.6/9.2 -> 정수 배정
    stratify: false                  # 층화 없음. 완전 무작위
    split_seed: 42                   # 분할 전용 고정 시드. 학습 시드(seeds)와 무관하며
                                     # split.json은 1회 생성 후 모든 K/시드 실험이 공유한다

model:
  # 선행 mesa_ecg_bss 설정 차용 (T3 확인 완료).
  # K=6은 GroupNorm(num_groups=K)이 채널 32/64/128/256을 나누지 못해 실행 불가 -> {4, 8}로 확정.
  n_encoders: [4, 8]                 # 실험 변수. 8이 선행 기준선. 총 학습 2 x 3시드 = 6회
  channels: [32, 32, 64, 64, 128, 128, 256, 256]   # 인코더당 채널은 c // K 로 분배 (총 용량 K 불변)
  hidden: 64                         # 인코딩 채널은 hidden // K
  norm_type: group_norm
  use_weight_norm: true
  input_channels: 1

loss:
  # 재구성만 BCE -> MSE 교체 (정규화 없음, ECG는 음수). zero_recon도 같은 손실이라 함께 MSE가 된다.
  recon: mse
  sep_impl: alternative              # separation_loss.py의 두 구현 중 mesa_ecg_bss가 쓴 쪽
  sep_norm: L1
  lambda_mixing: 1.0e-3              # 선행 sep_lr (ECG 설정)
  lambda_zero_recon: 1.0e-2          # 선행 zero_lr
  lambda_z_l2: 1.0e-2                # 선행 코드에 하드코딩된 값

train:
  batch: 256
  lr: 1.0e-3
  weight_decay: 1.0e-6               # 선행 값
  lr_step_size: 50                   # StepLR, gamma 0.1 (선행 값)
  gradient_clip_val: 0.5             # 선행 값
  max_epoch: 100                     # 선행은 25지만 설계 §6 유지
  early_stop_patience: 10            # 검증 재구성 손실 기준
  # 체크포인트는 검증 '분리 품질' 최고 에폭 (설계 §6). 선행의 monitor: recon_loss/train 은 따르지 않는다.
  checkpoint_metric: val_separation_quality
  # 분리 품질 = (잡음별 최대 |r| 평균) - penalty x (잡음 간 최고 인코더 중복 수)
  # 중복 수 = 3 - (bw/ma/em의 최고 인코더 중 서로 다른 것의 개수)
  # 소급 검증: penalty 0.05~0.30 전 구간에서 같은 에폭을 고른다(중복=2인 붕괴 에폭만 걸러냄).
  separation_duplicate_penalty: 0.15
  log_loss_terms: true               # 4개 손실 항을 에폭별로 기록 (MSE 전환 후 항 간 균형 확인)

s4:
  margin_delta: 0.1                  # 1위–2위 격차 임계 (검증셋에서 확정)
  inactive_eps: 0.1                  # 비활성 판정 임계
  min_consistency: 0.8

external:
  mimic3: {fs_src: 125, lead: "II", n_segments: 500}
  galaxyppg: {source: "Polar H10 ECG", fs_src: 130, n_segments: 500}
```

---

## 4. S1 — 데이터 구축 (`src/data/build.py`)

**목적**: 이후 모든 단계의 정답을 포함한 분절 데이터셋 생성.

**소스와 역할**
| 소스 | 역할 | 사용 범위 |
|---|---|---|
| MIT-BIH Arrhythmia | 참조 ECG 공급 | MLII 보유 46개 기록 (102·104 제외) |
| NSTDB | 실측 순수 잡음 공급 | `bw`, `ma`, `em` 3개 레코드만. 118e/119e 혼합 레코드는 **사용 안 함** (em 단일·환자 2명이라 부적합) |

**처리**
1. 기록별 10초 비중첩 분절 (3600 샘플). 기록당 약 180개 → 총 약 8,280개.
2. R-피크: `wfdb.rdann(rec, 'atr')`에서 비트 심볼(N,L,R,V,A,F,j,E,/,a,J,S,e,Q,**f**)만 추출 → 분절 로컬 인덱스로 저장.
   (`f` = paced와 normal의 융합박. 초판 목록에서 누락돼 있었고, 이를 빼면 217번 기록의 실제 심박 260개가
   R-피크에서 사라진다. 실제 박동이므로 포함한다.)
3. **잡음 레코드 시간 분할**: 각 잡음 레코드(30분)의 앞 70% 구간은 train 분절 주입에만, 뒤 30%는 val·test 분절 주입에만 사용.
4. **혼합 — 세 잡음 모두 항상 주입, 강도만 무작위**:
```python
# 잡음별 독립 추첨. 파형은 그대로, 크기만 신호에 맞춤
for t in ["bw", "ma", "em"]:
    n_t   = 해당 분할 구간에서 무작위 시작점의 3600 샘플
    snr_t = uniform(*noise_snr_range_db)
    # 신호 전력은 분산(평균 제거) 기준. mean(x_clean**2)를 쓰면 MIT-BIH의 전극 오프셋(DC)이
    # 전력의 중앙 35.1%(최대 97.5%)를 차지해, 명목 SNR과 실제 SNR이 분절마다 0~10 dB 어긋난다.
    a_t   = sqrt( var(x_clean) / (mean(n_t**2) * 10**(snr_t/10)) )   # 잡음 전력은 mean(n**2) 유지
                                                                    # (기저선 오프셋은 제거 대상 잡음의 일부)
    comp[t] = a_t * n_t
x_noisy = x_clean + comp["bw"] + comp["ma"] + comp["em"]
```
5. **정규화·품질 필터 없음. 데이터 단계에서는 패딩도 하지 않는다** (모델 입력용 3840 패딩은 §5에서 별도 적용하며 저장 배열은 3600 그대로다). wfdb의 physical units(mV)로 읽기만 한다. (MIT-BIH는 게인 200 adu/mV, 기저선 1024가 전 기록 공통이라 별도 정규화가 불필요하며, 잡음 강도가 신호 전력 기준이므로 환자 간 진폭 차이는 자동 보정된다.)
6. 분할: 기록 단위 70/10/20. `data/processed/split.json`에 기록 목록 고정.
7. **생성 시드 고정** — 재실행 시 동일 데이터가 나와야 한다.

**출력 (npz, 분절당 1개 버전)**
```
x_clean(3600,)  x_noisy(3600,)  bw(3600,)  ma(3600,)  em(3600,)
rpeaks(list)  meta{record_id, seg_idx, snr_bw, snr_ma, snr_em}
```

**DoD**
- 주입 후 잡음별 실측 SNR = 추첨값 ±0.1 dB (**분산 기준으로 실측**: `10*log10(var(x_clean)/mean(comp**2))`. 주입과 검증의 SNR 정의가 다르면 테스트가 항상 실패한다)
- `x_noisy == x_clean + bw + ma + em` (부동소수 허용오차 내)
- train/val/test 기록 교집합 = 0
- 잡음 레코드 시간 구간이 train과 val·test에서 겹치지 않음
- 무작위 3개 분절의 5개 배열 그림을 `figures/spotcheck/`에 저장

**보고 지점**: 분절 수 표 + 스팟체크 그림 → 사용자 확인 후 S2.

---

## 5A. 선행 공개 저장소 — 확인 결과 및 활용 지침

**저장소 (계정명이 `webstah` → `mbwebster`로 변경됨. 논문에 적힌 옛 주소는 리다이렉트됨)**

| 저장소 | 논문 | 라이선스 | 우리 활용 |
|---|---|---|---|
| `github.com/mbwebster/self-supervised-bss-via-multi-encoder-ae` | Webster & Lee, *Neurocomputing* 2025, doi:10.1016/j.neucom.2025.131008 (arXiv:2309.07138) | MIT | **기반 저장소.** 손실 구현·구조 원본 |
| `github.com/mbwebster/meae-heart-rate-extraction-from-noisy-ppg` | Webster, Lee D, Lee J, *Comput Biol Med* 2025;199:111319 (arXiv:2504.09132) | MIT | 1D 생체신호 적용판. `generate_mesa_data.py`, 사전학습 모델(Google Drive) 공개 |

MIT 라이선스이므로 코드 활용·수정이 자유롭다. 단 원 저작권 고지를 유지하고 두 논문을 모두 인용한다.

**구조 (두 저장소 공통)**: `config/`(Hydra), `models/`, `utils/`, `experiments/`, `notebooks/`, `trainer.py`. 실행은 `python trainer.py experiment_config=<name>`.

**★ 선행 팀은 이미 MESA ECG로 MEAE를 학습했다.** 기반 저장소에 `experiment_config=mesa_ecg_bss` 설정과 ECG 디코더 가중치 시각화(`assets/ecg_w.png`)가 있다. **`mesa_ecg_bss` 설정을 우리 기본 출발점으로 삼고**, 우리 입력 규격(360 Hz, 3600 샘플)에 맞춰 조정한다. `mesa_ppg_bss`의 사전학습 모델도 구조 확인용으로 내려받아 대조할 것.

**손실 구현 위치**: `models/separation_loss.py`에 sparse mixing loss가 **두 가지 방식**으로 구현되어 있다. 어느 쪽을 쓸지 코드를 열어 확인하고 선택 근거를 보고한다.
- sparse mixing: 디코더의 **각 층 가중치 W(출력층 제외)** 를 인코딩 공간별 블록 B_{i,j}로 분할 → **비대각 블록(인코딩 혼합 담당)만 0으로 감쇠**
- zero reconstruction: 전영(全零) 인코딩 Z_zero를 디코더에 넣어 출력이 0이 되도록 강제 → 마스킹된 인코딩이 소스 추정에 기여하지 못하게 함

**★★ 반드시 확인할 것 — 입력 길이와 다운샘플링 깊이**
**확인 완료 (T3).** 선행 `EncoderBlock` 8개가 각각 `MaxPool1d(2, 2)`를 가진다 → **인코더 깊이 8,
다운샘플 배수 256(= 2⁸)**. 따라서 **필요 조건은 2의 거듭제곱이 아니라 256의 배수**다.
(초판은 "3600 = 2⁴×225이므로 stride-2가 4회까지만 가능"으로 적었으나 잘못된 전제였다.)

| 입력 길이 | 출력 길이 | 인코딩 길이 | 판정 |
|---|---|---|---|
| 3600 | 3584 | 14 | **불일치** |
| **3840** | 3840 | 15 | **채택** (= 256×15) |
| 4096 | 4096 | 16 | 가능하나 패딩이 두 배 |
| 12288 (선행 MESA) | 12288 | 48 | OK |

**★ 깊이 결정 규칙 (확정)**
**3600 → 3840 대칭 제로 패딩 (양쪽 120샘플).** **깊이 축소는 선택하지 않았다.**
원칙의 위계 때문이다 — 깊이 축소는 선행 구조 자체를 바꾸는 것이라 §0 원칙 2(구조 차용) 위반이며
수용 영역(receptive field)이 달라져 비교 근거가 무너진다. 패딩은 데이터를 구조에 맞추는 쪽이라
구조를 보존한다.

**4096이 아니라 3840인 이유**: 선행 후속 저장소가 같은 문제를 이렇게 처리했다 —
`signal_duration: 48  # 48s (6000 samples) + 144 padding is divisible by 2^8`, `padding: 144`
(6000 → 6144 = 256×24). **2의 거듭제곱이 아니라 256의 다음 배수**를 골랐다. 같은 규칙이면
3600 → 3840이고, 패딩 240샘플로 4096(496샘플)의 절반이라 경계 왜곡 구간이 작다.

패딩과 **반드시 함께** 구현할 두 가지.
1. **모든 상관·지표는 패딩 제외 중앙 3600 구간에서만 계산한다.** 성분 출력이 3840으로
   나오므로, S4의 참조 상관과 S5의 전 지표를 내기 전에 `crop()`으로 원 구간을 잘라낸다.
   빠뜨리면 양끝 0 구간이 상관을 희석해 판별 결과가 왜곡된다.
   R-피크 인덱스는 크롭 좌표(0–3599) 기준을 유지한다. `metrics.score()`는 3840 입력을
   `ValueError`로 막는다.
2. **경계 왜곡 확인** — 선행 논문도 그림에서 가장자리를 잘라냈을 만큼 경계 아티팩트가 있다.
   T5 파일럿에서 **양끝 120샘플**을 확인했고 아티팩트는 없었다
   (경계 |진폭| 평균 0.2501 vs 중앙 0.2554, 비 0.98배).

**★ 학습 불안정성 — 선행이 README에 명시한 경고**
"마지막 체크포인트가 최선이 아닐 수 있으며, 안정성을 개선했음에도 학습 중 어느 시점에서든 불안정으로 결과가 나빠질 수 있다. 가능한 한 많은 체크포인트를 평가해 최선을 찾으라"고 권고한다. 또한 학습 중 일정 간격으로 소스 예측 샘플 그림을 `plots/` 폴더에 저장해 이를 돕는다.
→ **우리 설계의 '검증셋 분리 품질 최고 에폭 선택'(§6)은 이 권고를 정답 잡음으로 자동화한 것**이며, 선행이 수동으로 하던 작업을 정량화한 우리 기여다. 원고 방법·고찰에 이 대비를 명시할 것.
→ 구현 시 선행처럼 **일정 간격 성분 예측 그림을 `logs/plots/`에 저장**한다(디버깅에 필수).

**★ 인코더 수 과대추정은 허용된다 — K 설계의 근거**
선행은 소스가 2개인 데이터에 **인코더 3개**를 써서, 여분 인코더가 있어도 2개만 소스를 담당하는 해로 수렴함을 보였다. 즉 **K를 실제 소스 수보다 크게 잡아도 무방**하며, 남는 인코더는 비활성이 된다.
→ 우리 K ∈ {4, 8} 설계와 §7의 **비활성 인코더 범주**가 이 성질에 정확히 대응한다. K 선택 근거로 원고에 인용 가능.

**★ 선행의 인코더 선택 방식 = 우리 novelty의 대조군**
후속 저장소 README: "학습 후 어느 인코더가 원하는 심혈관 관련 소스를 만드는지 **수동 검사(manual inspection)** 가 필요하다." → 우리의 참조 잡음 기반 자동·정량 판별(S4)이 대체하는 지점. 원고 서론의 novelty 문장 근거.

**T3 확인 항목 — 전부 완료.** 결과 상세는 `results/data_notes.md` §8–12 참조.
1. `mesa_ecg_bss` / `mesa_ppg_bss` config의 인코더 수, 채널 폭, 깊이, 손실 가중치 실제 값
2. `models/separation_loss.py`의 두 sparse mixing 구현 차이와 선택안
3. 인코더 깊이 → 3600 사용 가능 여부 (패딩 필요 여부)
4. 재구성 손실이 BCE인 지점 → 우리는 MSE로 교체 (정규화 없음·ECG 음수 때문). 교체 시 zero reconstruction 손실도 MSE로 통일
5. requirements.txt와 우리 환경(Python 3.9.21, cu129)의 충돌 여부

## 5. S2 — 모델 구현 (`src/model/`)

**구현 전 필수**: §5A의 확인 항목 5가지를 점검하고 **사용자에게 보고**한 뒤 착수한다. 처음부터 새로 구현하지 말고 `mesa_ecg_bss` 설정을 출발점으로 이식한다.

**인터페이스**
```python
class MEAE(nn.Module):
    def __init__(self, n_encoders, ...): ...
    def encode(self, x) -> List[Tensor]                 # 길이 K, 각 (B, C, T')
    def decode(self, zs: List[Tensor]) -> Tensor        # 채널 결합 → 단일 디코더
    def forward(self, x) -> Tuple[Tensor, List[Tensor]]
    def component(self, x, k) -> Tensor                 # k번째 인코딩만 유지, 나머지 0
    def masked_reconstruct(self, x, mask_idx) -> Tensor # mask_idx의 인코딩을 0으로 치환
```
입력 `(B,1,3840)` → 출력 `(B,1,3840)`. 원 분절 3600을 양쪽 120샘플씩 대칭 제로 패딩한 값이다.
선행 인코더 깊이가 **8**(다운샘플 배수 256)이라 입력이 **256의 배수**여야 하고, 3600을 그대로
넣으면 재구성이 3584로 어긋난다 (§5A 실측). **결정 완료 — 깊이 축소는 하지 않는다.**
`meae.pad()` / `meae.crop()`이 변환을 담당하며 **모든 상관·지표는 crop 후 중앙 3600에서 계산**한다.

**손실 (선행 4항 차용, 재구성만 교체)**
| 항 | 내용 | 가중치 | 비고 |
|---|---|---|---|
| 재구성 | `MSE(x̂, x_noisy)` | 1.0 | 선행은 BCE지만 우리는 정규화를 안 하고 ECG는 음수를 가지므로 **MSE로 교체** |
| sparse mixing | 디코더 가중치(출력층 제외)의 비대각 L1 | `lambda_mixing` **1e-3 (잠정 — OPEN-01)** | 구현은 `WeightSeparationLossAlternative` (선행 ECG 설정과 동일) |
| zero reconstruction | 전영 인코딩 입력 시 디코더 출력을 0으로 (MSE) | `lambda_zero_recon` 1e-2 | 선행 값 그대로 |
| 인코딩 L2 | 각 z의 **평균 제곱** `mean(z²)` — L2 노름이 아니다 | `lambda_z_l2` 1e-2 | 선행 코드에 하드코딩된 값 |

파일럿에서 λ_mixing 1e-3은 mixing 항을 8.3%밖에 줄이지 못했고(디코더 그래디언트 비 0.0002),
1e-2에서 59.8% 감소했다. 최종값은 λ=1e-1 결과 후 확정한다 (OPEN-01).

**K 실험 변수**: {4, 8}. 8이 선행 기준선. K=6은 디코더 `GroupNorm(num_groups=K)`이 채널 32/64/128/256을 나누지 못해 실행 불가 (T3 실측 확인).

## 6. S3 — 학습 (`src/train.py`)

**실행 순서**
1. **파일럿 1회**: K=8, seed 42. 성분 파형 적층 그림 1장 → **사용자 보고 후 진행 판단**.
2. **본 학습 6회**: K {4, 8} × 시드 {42, 202, 2026}. **전부 새로 실행한다** —
   파일럿은 λ_mixing 값이 달라 본 결과에 포함하지 않는다. 에폭당 4.5초 × 약 60에폭 × 6회 ≈ 30분.

**매 에폭 로깅 (3층위)**
| 층위 | 항목 | 잡아내는 실패 |
|---|---|---|
| 학습 상태 | 총 손실, 4개 항 개별, 검증 재구성 손실 | 발산·미수렴 |
| **분리 품질** | 인코더×잡음 상관, **(잡음별 최대상관 평균) − 0.15×(잡음 간 최고 인코더 중복 수)** | 재구성은 되는데 분리 실패 |
| 붕괴 감지 | 인코더별 성분 에너지 비율, 인코더 간 성분 상관 | 인코더 사망·성분 중복 |

**조기 종료**: 검증 재구성 손실, patience 10 (자기지도 손실만 사용).

**체크포인트 선택**: 검증셋 **분리 품질 최고 에폭**.

> **분리 품질 정의 (T5에서 수정)**
> `분리 품질 = mean_{t∈{bw,ma,em}} max_k |r(x̂_k, t)|  −  0.15 × (3 − |{argmax_k 인 인코더들}|)`
> 초판 정의(앞항만)는 **덜 학습된 모델을 최고로 뽑는 결함**이 있었다. 성분이 전부 완만한
> 곡선이면 bw·ma와 폭넓게 상관되어 앞항이 커지는데, 그때는 세 잡음의 최고 인코더가
> 한 곳으로 몰린다. 중복 페널티가 이 상태를 걸러낸다.
> **헝가리안 1:1 강제 배정은 쓰지 않는다** — 분리 실패를 강제로 가려버리기 때문이다.
> 중복 수와 최고 인코더 조합(`tops`)은 매 에폭 함께 기록한다.
> 페널티 0.15는 0.05–0.30 전 구간에서 동일한 에폭을 고르는 것을 소급 확인한 값이다.

이 선택은 선행 논문이 "에폭마다 소스가 달라져 수동으로 골랐다"고 밝힌 한계를 자동화한 것이다.
정답 잡음은 이 선택에만 쓰이고 손실에는 관여하지 않는다(§0 원칙 4).
갱신 시에만 덮어쓰기 저장 + 이력 CSV(`logs/<run>/history.csv`).

**같은 지표를 §7의 K 비교에도 그대로 쓴다** — 에폭도 K도 같은 잣대로 골라야 일관된다.

**붕괴 규칙 (경고만, 자동 중단 없음)**
- 인코더 사망: 성분 에너지 < 전체의 1%가 5에폭 지속 → 기록. K 비교 해석에 활용.
- 성분 중복: 인코더 쌍 |corr| > 0.7이 5에폭 지속 → sparse mixing 가중치 상향 검토.

**학습 중 성분 그림**: 일정 에폭 간격으로 성분 예측 파형을 `logs/plots/`에 저장 (선행 저장소 방식. 불안정성 디버깅에 필수).

**재현성**: Python/NumPy/PyTorch/CUDA 시드 + cudnn deterministic.
**DataLoader를 쓰지 않는다** — 분절 전량을 램에 올리고(train 5,760 × 3,600 float32 ≈ 83 MB)
시드 고정 인덱스 셔플만 한다. 워커 시드 문제가 원천적으로 사라진다.
학습셋에는 `x_clean`/`bw`/`ma`/`em`을 아예 적재하지 않아 §0 원칙 3을 코드 수준에서 강제한다.

---

## 7. S4 — 잡음 성분 판별 (`src/s4_identify.py`) ★핵심 단계

```python
for seg in test_set:
    x = meae.pad(seg.x_noisy, PAD_EACH)                  # 3600 -> 3840
    for k in range(K):
        xk = meae.crop(model.component(x, k), PAD_EACH)  # ★ 3840 -> 3600. 크롭 필수
        for ref in ["clean", "bw", "ma", "em"]:          #   참조는 3600이고, 패딩 0 구간이
            r[k][ref] = abs(pearsonr(xk, seg[ref]))      #   들어가면 상관이 희석된다
        top1, top2 = 상위 2개 참조                        # 필터링 없이 원신호 그대로
```
**대역 제한 후 상관을 재지 않는다.** bw=저주파/ma=고주파라는 사전지식을 주입하면 판별이 모델의 분리 능력이 아니라 필터 설계 결과가 된다. 부호는 인코더-디코더 가중치의 임의 부호에 따라 뒤집히므로 절대값을 쓴다.

**판정 규칙 (4분류)**
| 조건 | 판정 |
|---|---|
| 1위가 잡음 참조 & (1위−2위) ≥ δ | **잡음 인코더** |
| 1위가 clean & (1위−2위) ≥ δ | **신호 인코더** |
| (1위−2위) < δ | **혼재 인코더** |
| 모든 상관 < ε & 성분 에너지 미미 | **비활성 인코더** |

- δ, ε은 **검증셋에서만** 결정 (평가셋 사용은 누수).
- em은 본래 기저선 변동과 근전도를 포함하는 잡음이므로 잡음끼리 상관이 있을 수 있다. 그래서 절대 임계가 아니라 **상대 격차**로 판정한다.
- **혼재·비활성 인코더는 마스킹하지 않는다** (심박 훼손이 잡음 잔류보다 나쁘다).

**통계적 근거**
- 잡음 인코더별: 분절 단위 (해당 잡음 상관 − clean 상관)에 Wilcoxon 부호순위, Holm 보정
- 일관성 비율: 1위 참조가 유지된 분절 %

**시드 간 집계 — 주의**: 인코더 번호는 시드마다 의미가 다르다(순열 불변성). **번호가 아니라 역할("bw 인코더")로 정렬한 뒤** 집계할 것. 놓치면 평균이 뭉개진다.

**K 비교**: **§6의 분리 품질 정의를 그대로 쓴다** — 별도 지표를 두지 않는다.

```
분리 품질 = mean_{t∈{bw,ma,em}} max_k |r(x̂_k, t)|  −  0.15 × (3 − |서로 다른 최고 인코더 수|)
```

체크포인트 선택(§6)과 K 선택에 **같은 잣대**를 쓴다. 초판은 K 비교에만 "혼재 인코더 비율
페널티"라는 다른 정의를 뒀는데, 이름이 비슷한 두 지표가 공존하면 해석이 엇갈린다.
K별 대응 행렬과 성분 파형을 나란히 제시한다.

**Gate G1 (전부 충족해야 S5 진행)**

> ⚠️ **개정 예정 — `OPEN_ITEMS.md` OPEN-03 (판별 단위 확정 대기).**
> T5 파일럿에서 λ 1e-3·1e-2 모두 **bw와 ma가 같은 인코더를 공유**했고(1e-2에서는 63/63 에폭),
> 조건1을 충족하지 못한다. 선등록 기준(`results/data_notes.md` §18)에 따라 λ=1e-1 결과를 본 뒤
> 판별 단위를 **심장 / em / bw+ma군** 3분류로 조정할지 결정하며, 그때 아래 조항을 개정한다.
> **확정 전에는 조건1을 근거로 중단하거나 진행하지 않는다.**

1. bw·ma·em 각각에 잡음 인코더가 1개 이상 존재
2. 잡음 상관 > clean 상관 (Wilcoxon, Holm p < .05)
3. 1위 참조 일관성 ≥ 80%
4. 시드 3개 중 2개 이상에서 1~3 충족

미달 시: K 변경 재확인 → sparse mixing 가중치 조정 → **그래도 실패 시 사용자와 재설계 논의. 임의 진행 금지.**

**출력**: `stage1_matrix.csv`, `stage1_assignment.csv`, `k_comparison.csv`, `fig1_correspondence.png`, `fig_components_by_k.png`, **`configs/noise_encoders.yaml`**

---

## 8. S5 — 마스킹 전후 평가 (`src/s5_restore.py`) ★판별의 인과 검증

**S5는 비교군 성능 평가가 아니다.** S4가 "k번 인코더가 bw다"라고 특정한 판별이 실제로
맞는지를 **인과적으로 검증**하는 단계다. 우리 주장의 3단 구조에서 세 번째에 해당한다.

> ① 분해된다 → ② **판별된다**(S4, 핵심) → ③ **판별이 진짜다**(S5)
> ③의 내용: 잡음 인코더를 끄면 신호가 좋아지고, 신호 인코더를 끄면 나빠진다.

**마스킹**: `noise_encoders.yaml`의 잡음 인코더 **인코딩만** 0 치환. 혼재·비활성은 유지.

**비교 대상은 마스킹 전 vs 후 vs clean 참조**다. 대역통과·웨이블릿·DAE 비교군은 두지 않는다.
외부 기법과의 성능 경쟁은 이 연구의 주장이 아니며, 비교군을 두면 표의 초점이
"어느 기법이 더 좋은가"로 옮겨가 ③의 논리가 묻힌다.

**모든 지표는 clean 참조 기준** (S4는 잡음 참조, S5는 clean 참조 — 혼동 주의)

| 지표 | 정의 | 마스킹 전 | 마스킹 후 |
|---|---|---|---|
| SNR (dB) | `10log10(var(clean) / mean((est−clean)²))` | est = x_noisy | est = 복원 신호 |
| SNR 개선 | 후 − 전 (dB) | — | 양수여야 함 |
| RMSE | clean 대비 | 〃 | 〃 |
| R-피크 F1 | 검출 vs MIT-BIH 주석, 허용오차 150 ms | 〃 | 〃 |
| SDNN 오차 | \|est SDNN − 주석 SDNN\| (ms) | 〃 | 〃 |

SNR 정의는 §4-4 주입과 동일하게 **신호는 분산, 잡음(잔차)은 mean(·²)** 기준이다.
10초 창은 RR이 8~19개뿐이라 SDNN 분절별 값이 불안정 → **중앙값[IQR]로 보고**.

**지표 위계 (확정)** — 4개 모두 산출한다(채점 비용이 0). 다만 역할이 다르다.

| 지표 | 역할 | 근거 |
|---|---|---|
| **SNR 개선** | **주 지표** | clean 참조 기준의 직접 측정. 주입 SNR과 정의가 같아 해석이 일관 |
| RMSE | 교차 확인 | SNR과 같은 잔차에서 나오므로 단독 근거는 아니나 스케일 감각을 준다 |
| R-피크 F1 | **심박 훼손 경고등** | 상한이 0.981(검출기 vs 주석 불일치), 마스킹 전이 0.956 → 개선 여지 2.5%p뿐. **개선의 증명에 쓸 수 없다.** 원고에는 각주 + 상한 0.981 병기 |
| SDNN 오차 | **판정 보류** | T5 파일럿 데이터로 상한·분산을 실측한 뒤 채택/제거를 결정 |

**S5의 핵심 증거는 지표 수치가 아니라 아블레이션의 방향성 비대칭이다** —
잡음 인코더를 껐을 때만 개선되고 신호 인코더를 끄면 악화된다는 대비. 지표는 그 비대칭을
재는 자이지 주장 자체가 아니다.

**복원율(bSQI) 지표는 사용하지 않는다.** 임의 임계값에 좌우되고, clean 대비 직접 평가가
이미 있으므로 불필요.

### 주 실험 — 인코더 단독 마스킹 아블레이션

보조가 아니라 **S5의 주 실험**이다. 인코더를 하나씩만 끄고 K개 조건을 각각 채점한다.

| 끈 인코더의 S4 판정 | 기대 결과 | 이것이 뒷받침하는 것 |
|---|---|---|
| 잡음 인코더 (bw/ma/em) | 지표 **개선** | 그 인코더가 실제로 해당 잡음을 담고 있었다 |
| 신호 인코더 | 지표 **악화** | 심장 성분이 그 인코더에 있었다 |
| 혼재·비활성 인코더 | 변화 미미 | 판별이 "모르겠음"으로 분류한 것이 타당했다 |

이 방향성이 성립하면 **S4 판별의 독립적 증거**가 된다. S4는 상관이라는 연관 근거이고,
아블레이션은 개입(intervention) 근거이므로 논증의 성격이 다르다.

**혼재·비활성 인코더는 본 복원에서 마스킹하지 않는다** (심박 훼손이 잡음 잔류보다 나쁘다).
아블레이션에서는 판정과 무관하게 K개를 모두 단독으로 꺼서 대조군으로 삼는다.

**모델 선택**: 시드 3개 결과를 모두 보고(역할 정렬 후 평균±표준편차). K는 S4에서 가장 분리가 좋은 것.

**출력**: `table_masking.csv`(마스킹 전후×지표), `ablation.csv`, `stats_pairwise.csv`,
`fig_ablation.png`(인코더별 지표 변화), `fig_internal_example.png`

## 9. S6 — 외부 적용 시연 (`src/s6_external.py`)

**목적은 성능 검증이 아니라 적용 가능성 시연이다.** 정답이 없으므로 정량 성능을 주장하지 않는다.

| 데이터 | 성격 | 처리 |
|---|---|---|
| MIMIC-III Waveform (유도 II) | 정적 임상 (중환자실) | 125→360 Hz `resample_poly(x, 72, 25)` |
| GalaxyPPG의 **Polar H10 ECG** | 동적 일상 (활동 중 실제 motion artifact) | 130→360 Hz `resample_poly(x, 36, 13)` |

**GalaxyPPG 주의**: Galaxy Watch는 PPG 담당이고 **ECG는 Polar H10 흉부 스트랩**이다. 가슴 전극이라 MIT-BIH MLII와 도메인이 가깝다. ZIP 481 MB, CC-BY 4.0, 보조 코드 `github.com/Kaist-ICLab/GalaxyPPG-Supplementary-Code`. 단위가 mV인지 raw인지 확인 후 변환.

**절차**: 리샘플·단위 정합 → 성분 분해 → `noise_encoders.yaml` 그대로 마스킹 → 전후 제시. **재학습·재판별 금지.**

**모델**: 검증셋 분리 점수 최고인 K의 최고 시드 모델 1개.

**출력**: `fig2_external.png` — 데이터셋별 대표 사례 2~3개, 각 사례마다 [원 분절 → 성분 분해 → 복원 → 스펙트럼 전후].

---

## 10. S7 — 통계 및 산출물

| 대상 | 방법 |
|---|---|
| S4 판별 | Wilcoxon(잡음 상관 − clean 상관) + Holm, 일관성 %, 시드는 역할 정렬 후 평균±SD |
| S4 K 비교 | 기술 통계만 (검정 없음) |
| S5 마스킹 전후 | 분절 쌍대 Wilcoxon(전 vs 후) + Holm, α=.05, 효과크기 r=Z/√N. SDNN은 중앙값[IQR] |
| S5 아블레이션 | 인코더별로 마스킹 전 대비 쌍대 Wilcoxon + Holm. 판정 범주별 방향성 보고 |
| S6 | 통계 없음 |

**원고 매핑**
| 원고 위치 | 파일 |
|---|---|
| 표 1 (판별) | stage1_assignment.csv |
| 표 2 (마스킹 전후) | table_masking.csv + stats_pairwise.csv |
| 그림 1 (대응 행렬) | fig1_correspondence.png |
| 그림 2 (외부 시연) | fig2_external.png |
| 그림 3 (아블레이션, 주 실험) | fig_ablation.png + ablation.csv |
| 본문 서술 (K 비교) | k_comparison.csv |

## 11. 단위 테스트 (`tests/`)

- `test_injection`: 잡음별 실측 SNR = 추첨값 ±0.1 dB (분산 기준, §4-4와 동일 정의) / `x_noisy == x_clean + bw + ma + em`
- `test_noise_time_split`: train과 val·test의 잡음 원본 구간이 겹치지 않음
- `test_split_leakage`: 기록 교집합 0
- `test_shapes`: component·masked_reconstruct 출력 (B,1,3840), `crop()` 후 (B,1,3600)
- `test_mask_effect`: mask_idx에 해당하는 인코딩이 실제로 0으로 치환됨 (인코더 가중치는 불변)
- `test_metrics`: clean vs clean → RMSE 0, SNR 무한대.
  **F1 1.0은 동일한 피크 집합끼리 비교할 때만 성립**한다 — `score(clean, clean, 주석)`의 F1은
  검출기와 사람 주석이 어긋나 상한이 **0.981**이다 (test 300분절 실측, `data_notes` §13)
- `test_resample`: 125→360, 130→360 후 길이 정합

## 12. 태스크 순서

```
T1  환경 구축 → torch/cuda 확인 보고
T2  S1 데이터 구축 → 단위 테스트 → ★분절 수·스팟체크 보고
T3  §5A 저장소 확인 → ★5개 항목 보고 → mesa_ecg_bss 기반으로 S2 모델·손실 이식
T4  metrics 구현 + 단일 채점 진입점 → 단위 테스트 (비교군 없음)
T5  S3 파일럿 (K=8, seed42) → ★성분 파형 그림 보고
T5.1 λ_mixing 스윕 (1e-3 / 1e-2 / 1e-1) → ★선등록 기준 판정
T5.5 정리 — 설계서 감사, decision_log, OPEN_ITEMS, 재현성 스냅샷
T6  S3 본 학습 6회 (K {4,8} × 시드 3개)
T7  S4 판별 → ★Gate G1 판정 보고 (미달 시 중단)
T8  S5 마스킹 전후 평가 + 아블레이션(주 실험)
T9  S6 외부 시연
T10 S7 통계 → 원고 ■ 채우기용 수치 정리 보고
```
★ = 사용자 확인 없이 다음 단계로 진행하지 않는다.

## 13. 코딩 규약

- 설정은 configs에만. 하드코딩 금지. 결과 파일 메타에 시드·설정 해시 기록.
- 결과는 CSV/parquet 저장 후 표·그림 생성.
- 그림 300 dpi PNG, 한글 폰트 지정(Malgun Gothic / NanumGothic 폴백).
- 원본 데이터는 절대 수정하지 않는다. 커밋 단위 = 태스크.
