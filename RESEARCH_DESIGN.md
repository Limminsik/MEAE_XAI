# RESEARCH_DESIGN.md
# ECG 잡음 성분 판별 및 선택적 복원 — 실행 지시서

**한 줄 요약**: 잡음이 섞인 ECG로 다중 인코더 오토인코더를 자기지도 학습시킨 뒤, 어떤 인코더가
실제 잡음(bw/ma/em)에 대응하는지 정량 판별하고, 그 인코더의 **인코딩만 마스킹**하여 신호를 복원한다.

**주장의 구조**: ① 분해된다 → ② **판별된다**(S4, 핵심 기여) → ③ 판별이 진짜다(S5, 인과 검증)

---

## 0. 불변 원칙 (위반 금지 — 위반이 필요하면 사용자에게 보고)

1. **마스킹 대상은 인코더가 아니라 인코딩(인코더 출력 텐서)**이다. 인코더 가중치는 그대로 두고
   순전파도 K개 모두 수행하되, 디코더 입력 직전에 해당 인코딩만 영텐서로 치환한다.
   선행 MEAE는 완전 합성곱 구조이므로 인코딩은 (채널 × 시간) 특성맵이며 VAE식 잠재 벡터가 아니다.
   **재학습이 아니라 추론 시점 조작이다.** 용어는 문서·코드·원고 전반에서 **인코딩**으로 통일한다.
2. **구조는 선행 MEAE를 차용**한다. 동일 입력을 받는 K개 완전 합성곱 인코더 → 인코딩 채널 결합 →
   단일 공유 디코더. 성분 x̂_k는 k번째 인코딩만 남기고 나머지를 0으로 치환한 재구성이다. 임의 변경 금지.
3. **학습은 자기지도**다. 입력 = 잡음 섞인 분절, 목표 = 동일 분절.
   **clean·bw·ma·em을 손실에 절대 사용하지 않는다.**
4. **참조(clean, bw, ma, em)는 채점표이지 교과서가 아니다.** 판별 평가와 체크포인트 선택에만 쓰고,
   가중치 갱신에는 관여하지 않는다.
5. **데이터 분할은 기록(환자) 단위**로만 한다. 분절 단위 분할은 누수다.
6. **S6에서 재학습·재판별 금지.** S4가 확정한 `configs/noise_encoders.yaml`을 그대로 적용한다.
7. **test는 S4 착수 시점까지 봉인**한다. 그 이전 모든 판단은 val에서만 한다.

---

## 1. 환경

```bash
uv init && uv venv --python 3.9.21
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu129
uv pip install -r requirements.txt
```
실측 환경: Python 3.9.21 / torch 2.8.0+cu129 / cuDNN 91002 / NVIDIA RTX 5060 Ti / CUDA 12.9.
`neurokit2==0.2.10` 고정(0.2.12는 `float | None` 문법이라 py3.9 import 실패).
동결본 `snapshots/pilot-complete/{requirements.freeze.txt, environment.json}`.
휠 호환 문제 발생 시 **버전을 임의 변경하지 말고 보고**한다.

## 2. 저장소 구조

```
meae_xai/
├── README.md  RESEARCH_DESIGN.md  CLAUDE.md
├── configs/default.yaml            모든 설정. 코드 하드코딩 금지
├── data/raw/  data/processed/      git 제외
├── src/
│   ├── data/{download,split,build,dataset}.py
│   ├── model/{meae,losses,_vendor_*}.py
│   ├── train.py  fidelity.py  spectral.py
│   ├── s4_identify.py  s5_restore.py  rehearsal.py
│   └── metrics.py  stats.py  viz.py
├── tests/
├── results/                        ★ 본 실험 산출물
│     00_rehearsal/  01_train/<run>/  02_separation/  03_denoising/  04_external/
├── experiments/{ssl,supervised_noise}/outputs/   보조 기록 (본 노선 아님)
└── _work/archive/                  폐기된 실행·구버전
```

실행 이름 = `K<인코더수>_seed<시드>` + 오버라이드 접미사(`_lz0`, `_h128`). 접미사가 없으면 config 그대로다.

## 3. configs/default.yaml

설정의 단일 원천은 `configs/default.yaml`이다. 이 문서는 값을 복제하지 않는다.
값을 바꿀 때는 config를 고치고 이 문서의 해당 절을 함께 갱신한다.

---

## 4. S1 — 데이터 구축 (`src/data/build.py`) — 완료·동결

**소스**

| 소스 | 역할 | 사용 범위 |
|---|---|---|
| MIT-BIH Arrhythmia | 참조 ECG | MLII 보유 46개 기록 (102·104 제외) |
| NSTDB | 실측 순수 잡음 | `bw`·`ma`·`em` 레코드의 **채널 0**만. 118e/119e 혼합 레코드는 사용 안 함 |

채널 0(noise1)을 쓰는 근거: nst 도구가 채널 번호끼리 짝지어 섞는 관례를 따른다.

**처리**
1. 기록별 10초 비중첩 분절(3600 샘플) → 총 **8,280개**.
2. R-피크: `wfdb.rdann(rec,'atr')`의 비트 심볼(N,L,R,V,A,F,j,E,/,a,J,S,e,Q,**f**)만 추출.
   `f`(paced–normal 융합박)를 빼면 217번 기록의 실제 심박 260개가 사라진다. 총 104,748개.
3. **잡음 레코드 시간 분할**: 각 잡음 레코드(30분)의 앞 70%는 train 주입 전용, 뒤 30%는 val·test 전용.
4. **혼합 — 세 잡음 모두 항상 주입, 강도만 무작위**:

```python
for t in ["bw", "ma", "em"]:
    n_t   = 해당 분할 구간에서 무작위 시작점의 3600 샘플
    snr_t = uniform(*noise_snr_range_db)                                  # [0, 12] dB
    # 신호 전력은 분산(평균 제거) 기준. mean(x_clean**2)를 쓰면 MIT-BIH 전극 오프셋(DC)이
    # 전력의 중앙 35.1%(최대 97.5%)를 차지해 명목 SNR과 실제 SNR이 0~10 dB 어긋난다.
    a_t   = sqrt(var(x_clean) / (mean(n_t**2) * 10**(snr_t/10)))          # 잡음은 mean(n**2)
    comp[t] = a_t * n_t
x_noisy = x_clean + comp["bw"] + comp["ma"] + comp["em"]
```

5. **정규화·품질 필터 없음. 데이터 단계에서 패딩도 하지 않는다** (3840 패딩은 §5에서 적용, 저장 배열은 3600).
   wfdb의 physical units(mV)로 읽기만 한다.
6. 분할은 기록 단위 32/5/9, 층화 없음. `data/processed/split.json`에 고정.
7. 생성 시드 고정 — 재실행 시 동일 데이터.

**출력(npz)**: `x_clean, x_noisy, bw, ma, em` 각 (3600,) · `rpeaks` · `meta{record_id, seg_idx, snr_bw, snr_ma, snr_em}`

**DoD**: 잡음별 실측 SNR = 추첨값 ±0.1 dB(분산 기준) / `x_noisy == x_clean+bw+ma+em` /
기록 교집합 0 / 잡음 시간 구간 비중첩 / 스팟체크 그림.

---

## 5. S2 — 모델 (`src/model/`) — 완료·동결

**출처**: 선행 `mesa_ecg_bss` 설정을 무수정 이식(`_vendor_*.py`, MIT 라이선스, 원 고지 유지).

```python
class MEAE(nn.Module):
    def encode(self, x) -> List[Tensor]                  # 길이 K, 각 (B, C, T')
    def decode(self, zs) -> Tensor                       # 채널 결합 → 단일 디코더
    def forward(self, x) -> Tuple[Tensor, List[Tensor]]
    def component(self, x, k) -> Tensor                  # k번째 인코딩만 유지
    def masked_reconstruct(self, x, mask_idx) -> Tensor  # mask_idx의 인코딩을 0으로
```

**입력 길이 3840**. 선행 `EncoderBlock` 8개가 각각 `MaxPool1d(2,2)` → 깊이 8, 다운샘플 배수 **256**.
필요 조건은 2의 거듭제곱이 아니라 **256의 배수**다.

| 입력 | 출력 | 인코딩 길이 | 판정 |
|---|---|---|---|
| 3600 | 3584 | 14 | 불일치 |
| **3840** | 3840 | 15 | **채택** (256×15) |
| 4096 | 4096 | 16 | 가능하나 패딩 두 배 |

깊이 축소를 하지 않는 이유: 선행 구조 변경은 §0 원칙 2 위반이고 수용 영역이 달라져 비교 근거가 무너진다.
4096이 아닌 이유: 선행 후속 저장소가 6000 → 6144(=256×24, padding 144)로 **256의 다음 배수**를 택했다.
같은 규칙이면 3600 → 3840, 패딩 240샘플로 4096(496샘플)의 절반이다.

패딩과 반드시 함께 지킬 것:
1. **모든 상관·지표는 `crop()` 후 중앙 3600에서만 계산**한다. R-피크 인덱스는 0–3599 기준을 유지하고,
   `metrics.score()`는 3840 입력을 `ValueError`로 막는다.
2. 경계 왜곡 없음을 확인했다 — 양끝 120샘플 |진폭| 평균 0.2501 vs 중앙 0.2554 (비 0.98).

**K = 8 확정.** 소스 수를 가정하지 않는 설정이며 선행 ECG 실험의 기준선이다.
K는 디코더 `GroupNorm(num_groups=K)`이 채널 32/64/128/256을 나눠야 하므로 **32의 약수**만 가능하다
(K ∈ {1,2,4,8,16}. K=6·7은 실행 불가). 선행은 소스 2개 데이터에 인코더 3개를 써서 여분 인코더가
비활성이 되는 해로 수렴함을 보였다 — **K 과대추정은 허용된다.**

**표시 규칙**: 내부 인덱스는 0-based, 사람이 보는 이름은 1-based (`enc_label(k) = f"enc{k+1}"`).

**시드는 42 단일.** 다중 시드 반복은 수행하지 않는다.

---

## 6. 비용 함수 ① — 학습 손실 (`src/model/losses.py`)

```
L = MSE(x̂, x_noisy) + λ_m·L_mix + λ_o·‖D(0)‖² + λ_z·Σ_k mean(z_k²)
```

| 항 | 내용 | 가중치 |
|---|---|---|
| 재구성 | `MSE(x̂, x_noisy)` — 선행은 BCE지만 정규화를 안 하고 ECG는 음수라 MSE로 교체 | 1.0 |
| sparse mixing | 디코더 각 층 가중치(출력층 제외)를 인코딩별 블록으로 나눠 **비대각 블록만 L1 감쇠**. 구현은 선행 `WeightSeparationLossAlternative` | λ_m = **1e-2** |
| zero reconstruction | 전영 인코딩을 디코더에 넣어 출력이 0이 되도록 강제 — 마스킹된 인코딩이 추정에 기여하지 못하게 한다 | λ_o = **1e-2** |
| 인코딩 L2 | 각 z의 **평균 제곱** `mean(z²)`. L2 노름이 아니다. 선행이 밝힌 목적은 그래디언트 폭주 방지 | λ_z = **1e-3** |

λ_m을 1e-3에서 1e-2로 올린 근거: 1e-3은 mixing 항을 8.3%밖에 줄이지 못했고(디코더 그래디언트 비 0.0002),
1e-2에서 59.8% 감소했다.

**참조 신호는 이 함수에 일절 등장하지 않는다.** 학습셋에는 `x_clean`/`bw`/`ma`/`em`을 아예 적재하지
않아 §0 원칙 3을 코드 수준에서 강제한다.

---

## 7. 비용 함수 ② — 체크포인트 선택 (`src/train.py`)

```
x̂_k    = D(0,…,z_k,…,0)                     k번째 인코딩만 남긴 재구성, crop 후 중앙 3600
ρ_k(t) = median_{s∈V} |ρ(x̂_k, x_clean)|     V = val의 앞 val_n(300) 분절, 전 에폭 고정
S(t)   = max_k ρ_k(t)

1단계  C  = { t : L_recon^val(t) ≤ recon_ratio × min_τ L_recon^val(τ) }     recon_ratio = 1.5
2단계  t* = argmax_{t∈C} S(t)
```

- `eval_every`(2) 에폭 간격으로 산출하고, **학습 종료 후 전체 이력을 일괄 판정**한다.
- 후보 구간의 가중치를 보관한다 — 지표를 다시 손보더라도 **재학습 없이 재선택**할 수 있게.
- 배율 민감도 **{1.2, 1.5, 2.0}** 를 함께 산출해 `selection.json`에 남긴다.
- 1단계가 없으면 미수렴 시점이 선택된다. 성분이 전부 완만한 곡선인 단계에서는 참조 상관이
  부풀려지기 때문이다.
- `x_clean`은 이 선택에만 쓰이고 가중치 갱신에는 관여하지 않는다(§0 원칙 4).

이 선택은 선행이 "마지막 체크포인트가 최선이 아닐 수 있으니 가능한 한 많은 체크포인트를 평가하라"고
권고하며 **수동으로** 하던 작업을 정량화한 것이다.

### 배율 민감도 기록 (K8_seed42, val 900분절)

**사전 등록한 배율 1.5를 유지한다.** 1.2가 고르는 에폭 88과 실물을 대조한 결과를 근거로 남긴다.
가중치는 `pool/`에 보관돼 있어 재학습 없이 대조했다 (`src/compare_epochs.py`,
산출물 `results/00_rehearsal/epoch_compare/`).

| | 에폭 48 (배율 1.5·2.0) | 에폭 88 (배율 1.2) |
|---|---|---|
| clean 최대 인코더 | enc4 | enc4 |
| bw·ma·em 최대 인코더 | enc3 / enc3 / enc3 | enc3 / enc3 / enc3 |
| clean \|r\| | 0.641 | 0.651 |
| clean 기여 % | 42.95 | 43.73 |
| 보존율 전대역 | 0.912 | 0.874 |
| 15–25 / 25–40 Hz | 0.585 / 0.172 | 0.538 / 0.212 |
| R-피크 진폭비 | 0.921 | 0.891 |
| 잔차–clean | 0.230 | 0.255 |
| 기울기차 | −2.879 | −2.532 |

역할 구조(어느 인코더가 어느 참조와 1위인가)가 동일하고 지표 차이가 미미하므로,
배율은 결과를 보고 옮기지 않는다.

**조기 종료**: `L_recon^val`, patience 30. **DataLoader를 쓰지 않는다** — 분절 전량을 램에 올리고
(train 5,760 × 3,600 float32 ≈ 83 MB) 시드 고정 인덱스 셔플만 한다. 워커 시드 문제가 사라진다.
**재현성**: Python/NumPy/PyTorch/CUDA 시드 + cudnn deterministic.

**출력**: `results/01_train/<run>/` — 선택 가중치 · `history.csv` · `selection.json` · `console.log` ·
`pool/`(후보 구간 가중치) · `plots/`(에폭 간격 성분 파형)

---

## 7.5 재구성 충실도 진단 (`src/fidelity.py`, `src/spectral.py`)

**관문이 아니라 서술 지표다.** 선행이 명시한 재구성–분리 트레이드오프
("인코딩이 너무 작으면 좋은 재구성이 불가하고, 너무 크면 인코더가 특화 대신 전체 특징 공간으로
일반화된다")를 어디쯤에서 취했는지 보고하기 위한 산출물이며, 합격/불합격 수치를 두지 않는다.

| 지표 | 정의 |
|---|---|
| 대역별 보존율 | `P(x̂)/P(x_noisy)` — 5–15 / 15–25 / 25–40 / **40–60(59–61 Hz 노치 제외)** / 60–90 Hz |
| log-log 기울기 · keep 곡선 | 주파수축 전체의 보존 형태 |
| 디노이징 지수 | `RMSE(x̂, x_noisy)` vs `RMSE(x̂, clean)` |
| 잔차 상관 | `|r(x_noisy − x̂, ref)|` — 못 담은 것의 정체 |
| R-피크 진폭비 | R-피크 ±30샘플 첨두간 진폭의 재구성/입력 비 |

**출력**: `results/02_separation/fidelity/`, `results/02_separation/spectral/`

---

## 8. S4 — 인코더–참조 대응 분석 (`src/s4_identify.py`) ★핵심 단계

**라벨을 붙이지 않는다.** 인코더를 잡음/신호/혼재/비활성으로 분류하고 임계 δ·ε로 판정하는 방식은
폐기했다. 이유 둘.
1. **참조끼리 상관돼 있어 "어느 잡음 전담"이라는 라벨이 성립하지 않는다.** 주입 성분 자체의 |r|이
   bw–ma **0.318**, bw–em 0.221, ma–em 0.164다(심장–잡음은 0.065–0.083).
2. **임계가 결과를 보고 움직일 유혹을 만든다.**

대신 **대응 관계를 표와 그림으로 그대로 보고**하고, 독자가 수치를 보고 판단한다.

```python
for seg in split:
    x = meae.pad(seg.x_noisy, PAD_EACH)                  # 3600 → 3840
    for k in range(K):
        xk = meae.crop(model.component(x, k), PAD_EACH)  # ★ 크롭 필수. 패딩 0 구간이
        for ref in ["clean", "bw", "ma", "em"]:          #   들어가면 상관이 희석된다
            r[k][ref] = abs(pearsonr(xk, seg[ref]))
```

**대역 제한 후 상관을 재지 않는다.** bw=저주파/ma=고주파라는 사전지식을 주입하면 판별이 모델의
분리 능력이 아니라 필터 설계 결과가 된다. 부호는 인코더–디코더 가중치의 임의 부호에 따라 뒤집히므로
절대값을 쓴다.

### 평가 지표 (확정) — z-정규화 후 3종

선행 연구는 실제 소스 파형을 알 수 없어 심박수 같은 파생 지표로 **간접** 평가했다.
본 연구는 주입한 잡음 성분을 개별 보존하므로 **성분과 참조 파형을 직접 대조하는 정량 평가**가
가능하다. 이것이 대응표를 실을 수 있는 근거다.

**성분과 참조를 각각 z-정규화한 뒤 지표를 산출한다.**

| 지표 | 정의 | 성격 |
|---|---|---|
| **① 상관 ρ** (+r²) | 피어슨 상관. 부호는 인코더–디코더 가중치의 임의 부호이므로 절대값 | 형태 유사도 |
| **② RMSE_norm** | 표준화 후 두 신호 차이의 RMS. **`= √(2(1−ρ))` 로 ρ의 함수다**(각주 명시). 해석 편의를 위해 병기하며 독립 근거가 아니다 | ①과 같은 정보의 다른 표현 |
| **③ MAD** | 같은 차이 신호의 `max_i\|ã[i] − r̃[i]\|`. 정규화 후에도 ρ와 독립인 유일한 지표 | 국소 최대 편차 (보완적) |

SSD·PRD도 ②와 마찬가지로 ρ의 함수이므로 싣지 않는다.

**원값 RMSE는 폐기한다.** 성분의 절대 크기가 임의 스케일이라 순위가 뒤집힌다 —
에폭 48 실측에서 enc3이 bw ρ̄ 0.602로 1위인데 원값 RMSE는 0.4936으로 8개 중 최하위였다.

#### [S4-01] 상관계수 산출 명세 (확정)

**입력** — 성분 `x̂_k = D(0,…,z_k,…,0)` k=1…K(K=8) · 참조 4종
(`x_clean`, `α_bw·n_bw`, `α_ma·n_ma`, `α_em·n_em` — npz에 보존된 실제 주입 성분) ·
패딩 제외 **중앙 N=3600 표본**(crop 후 계산 필수) · 분절 val 전체 **S=900**.

**1단계 — 분절 내 Pearson**

```
ρ_kr^(s) = Σ_i (x̂_k^(s)[i] − mean(x̂_k^(s)))(r^(s)[i] − mean(r^(s)))
           ──────────────────────────────────────────────────────────
           √Σ(x̂_k^(s)[i] − mean(x̂_k^(s)))² · √Σ(r^(s)[i] − mean(r^(s)))²
```

평균·표준편차는 **해당 분절 안에서** 구한다. 분절을 이어붙여 일괄 계산하지 않는다 —
분절마다 다른 잡음 구간이 주입되었고, 진폭 큰 분절이 결과를 지배한다.

**2단계 — 분절 간 집계**

```
ρ̄_kr = mean_s |ρ_kr^(s)|
σ_kr  = std_s  |ρ_kr^(s)|      (ddof=1)
```

절댓값을 쓰는 이유: 인코딩과 디코더 가중치가 동시에 부호 반전되어도 재구성이 불변하므로
성분이 참조와 반대 위상으로 수렴할 수 있다. **부호 반전은 무관이 아니라 반대 위상의 일치다.**
원 부호 `sign(ρ_kr^(s))`의 분절별 분포(양수 비율)는 별도 CSV에 기록한다.

**표 렌더링** — 행별 최댓값을 표시한다. 잡음 3종을 "잡음 최대"로 요약하지 말고 **4열 그대로** 싣는다.

**하지 않을 것**
- 표본 수(N=3600) 기반 **p값 산출 금지** — 시간적 자기상관으로 독립 관측 가정이 성립하지 않는다.
- 인코더에 참조 이름을 붙이는 **명명 금지** (예: "enc3 = bw 인코더").
- 값에 대한 **해석·판정 금지.** 수치와 표만 산출한다.

순열 귀무분포, 코히런스, 대역별 분해, 무관 수준 기준선, 중앙값[IQR] 집계는 모두 제외한다.

#### [S4-02] 정규화 RMSE 산출 명세 (확정)

구조는 S4-01과 같다 — 한 분절에서 한 성분을 참조 4종과 각각 비교하고, 900분절 반복 후 평균±SD.

**1단계 — 분절 내 표준화**

```
ã = (x̂_k^(s) − mean(x̂_k^(s))) / std(x̂_k^(s))
r̃ = (r^(s)   − mean(r^(s)))   / std(r^(s))
```

성분과 참조 모두 해당 분절 내에서 평균 0, 표준편차 1로 표준화한다. 성분 진폭이 비선형
디코더의 임의 출력이고 참조 4종의 RMS도 서로 다르므로(clean 0.204, 잡음 0.113~0.119 mV)
표준화 없이는 크기 차이가 값을 지배한다.

**2단계 — 분절 내 RMSE**

```
RMSE_kr^(s) = sqrt( mean_i (ã[i] − r̃[i])² )
```

**3단계 — 분절 간 집계**

```
RMSE̅_kr = mean_s RMSE_kr^(s)
σ_kr     = std_s  RMSE_kr^(s)      (ddof=1)
```

구간은 패딩 제외 중앙 N=3600, 분절은 val 900.
산출물 `rmse_norm_matrix.csv` (K×4, `RMSE̅ ± σ`) · `rmse_norm_persegment.csv`.
**표 렌더링 시 행별 최솟값을 표시한다**(낮을수록 유사). 값에 대한 해석·명명은 붙이지 않는다.

**MAD**도 같은 표준화·같은 차이 신호를 쓰되 RMS 대신 `max_i|ã[i] − r̃[i]|`를 취한다.
두 지표 모두 부호 정렬을 하지 않으므로 반대 위상은 값이 커진다.

**그림 규칙**: 성분 파형 그림은 전 성분을 **공통 y축**으로 그린다(성분마다 축을 따로 잡으면
크기가 작은 성분이 크게 보여 비교가 안 된다). y축 단위를 명기한다 — 성분·참조는 z-정규화
후 단위 없음, 입력 `x_noisy`만 원값 mV로 축을 따로 둔다.

### 산출물

| 산출물 | 파일 |
|---|---|
| ① 인코더 × 참조 대응표 (ρ̄ ± σ) | `corr_matrix.csv`, `fig1_correspondence.png` |
| ①-보조 분절별 원값 (부호 포함) · 부호 양수 비율 | `corr_persegment.csv`, `corr_sign.csv` |
| ② 정규화 RMSE 대응표 (RMSE̅ ± σ) | `rmse_norm_matrix.csv`, `rmse_norm_persegment.csv` |
| ①②-보조 통합표 (ρ̄·σ·r²·RMSE_norm·MAD·energy_ratio) | `stage1_matrix.csv` + `stage1_matrix_note.txt` |
| ② 참조 간 상관·스펙트럼 | `reference_correlation.csv`, `reference_spectrum.csv` |
| ③ 기여 분해 (다중 회귀, 서술용) | `stage1_contribution.csv` |
| ④ 성분 파형 적층 그림 · 분절별 \|r\| 히스토그램 | `fig_components_by_k.png`, `hist_enc*.png` |
| ⑤ 인코더별 Wilcoxon + Holm, 효과크기 r=Z/√N | `stage1_stats.csv` |
| ⑥ 마스킹 규칙 | **`configs/noise_encoders.yaml`** (S5·S6가 읽는 유일한 산출물) |

②는 **분리 한계의 독립 근거**다. "모델이 bw와 ma를 못 나눴다"가 아니라 "참조끼리 0.32 상관이라
나눌 수 없다"를 보이는 표이며 ①을 읽는 기준선이다. 중심주파수 실측: bw 0.85 · em 2.49 · ma 5.49 ·
clean 13.30 Hz.

③ 기여 분해는 참조를 K개 성분에 회귀해 설명 분산을 성분별로 쪼갠다.
```
ref ≈ Σ_k β_k · x̂_k          (분절마다, 평균 제거 후)
몫_k = β_k · cov(x̂_k, ref) / var(ref),     Σ_k 몫_k = R²
```
상관은 모양의 닮음만 말한다. 기여 분해는 "그 참조 에너지의 몇 %를 그 성분이 설명하나"에 답한다.
몫은 억제 시 음수가 될 수 있다. **선정 기준이 아니라 서술용이다.**

④에서 분절별 히스토그램을 싣는 이유: 평균값 하나로는 산포가 감춰진다. 상관의 산포는 주입 SNR로
설명된다 — 실측에서 bw 상관이 SNR 0–3 dB에서 0.805, 9–12 dB에서 0.461이었다(스피어만 −0.50).
clean 상관은 SNR과 무관하다.

**보고 지점 ★**: S4 결과 보고 후 사용자 확인 없이 S5로 넘어가지 않는다.

---

## 9. S5 — 마스킹 전후 평가 (`src/s5_restore.py`) ★판별의 인과 검증

**성능 경쟁이 아니다.** S4의 대응 판별이 실제로 맞는지를 개입(intervention)으로 검증한다.
대역통과·웨이블릿·DAE 비교군은 두지 않는다 — 표의 초점이 "어느 기법이 더 좋은가"로 옮겨가면
③의 논리가 묻힌다. **비교 대상은 마스킹 전 vs 후 vs clean 참조**다.

### 마스킹 조건 사다리 M0–M5

| 조건 | 마스킹 대상 | 목적 |
|---|---|---|
| **M0** | 없음 | 기준선. `x_noisy` 자체도 함께 싣는다 |
| **M1** | 인코더 1개씩 단독 (K개 조건) | **주 실험 — 아블레이션** |
| **M2** | 주력 잡음 인코더만 | 최소 개입 |
| **M3** | M2 + 잔여 잡음 우세 인코더 | 중간 개입 |
| **M4** | clean 1위 인코더 제외 전부 | 최대 개입 |
| **M5** | clean 인코더만 | **음성 대조.** 지표가 악화돼야 한다 |

**M2·M3 선정 규칙** (기여 분해 기준, **val에서 정의하고 test에 그대로 적용**)
```
잡음 기여 합_k = share_k(bw) + share_k(ma) + share_k(em)      [%]
M2 = { k : 잡음 기여 합_k ≥ 10% }
M3 = M2 ∪ { k : 잡음 기여 합_k > clean 기여_k }
```
임계 10%는 결과를 보고 옮기지 않는다.

**표 2 = M0 대비 M2·M3·M4의 SNR 계단.** 각 조건에 **전수 2^K 지도에서의 백분위**를 병기해
그 조건이 가능한 모든 조합 중 어디쯤인지 함께 보인다(K=8이면 256 조합). 규칙 선택의 자의성을
지형 전체로 대체하는 장치다.

### 지표 — 모두 clean 참조 기준

(S4는 잡음 참조, S5는 clean 참조 — 혼동 주의)

| 지표 | 정의 | 역할 |
|---|---|---|
| **SNR (dB)** | `10log10(var(clean) / mean((est−clean)²))`, 개선 = 후 − 전 | **주 지표.** 주입 SNR과 정의가 같아 해석이 일관 |
| RMSE | clean 대비 | 교차 확인 |
| R-피크 F1 | 검출 vs MIT-BIH 주석, 허용오차 150 ms | **심박 훼손 경고등.** 상한 0.981(검출기 vs 주석 불일치), 마스킹 전 0.956 → 여지 2.5%p뿐 |
| SDNN 오차 | \|est SDNN − 주석 SDNN\| (ms) | 10초 창은 RR이 8~19개뿐 → **중앙값[IQR]로 보고** |

SNR 정의는 §4 주입과 동일하게 신호는 분산, 잔차는 mean(·²) 기준이다.
복원율(bSQI)은 임의 임계에 좌우되므로 사용하지 않는다.

### 주 실험 — 단독 마스킹 아블레이션 (M1)

| 끈 인코더의 S4 대응 | 기대 방향 | 뒷받침하는 것 |
|---|---|---|
| 잡음 참조와 상관·기여가 높음 | 개선 | 그 인코더가 실제로 잡음을 담고 있었다 |
| clean 최대 상관 | 악화 | 심장 성분이 그 인코더에 있었다 |
| 어느 참조와도 낮음 | 변화 미미 | 기여가 미미하다는 판독이 맞았다 |

**S5의 핵심 증거는 지표 수치가 아니라 이 방향성 비대칭이다.** 지표는 비대칭을 재는 자다.
S4는 연관(상관) 근거이고 아블레이션은 개입 근거이므로 논증의 성격이 다르다.

**출력**: `results/03_denoising/` — `table_masking.csv`(표 2), `ablation.csv`(M1),
`mask_map.csv`(전수 지도 + 백분위), `stats_pairwise.csv`,
`fig_ablation.png`, `fig_mask_map.png`, `fig_internal_example.png`

---

## 10. S6 — 외부 적용 시연 (`src/s6_external.py`)

**목적은 성능 검증이 아니라 적용 가능성 시연이다.** 정답이 없으므로 정량 성능을 주장하지 않는다.
연구계획서 기반연구 ①에 명시된 데이터셋을 쓴다.

| 데이터 | 환경 | 처리 |
|---|---|---|
| **MIMIC-IV Waveform** (유도 II) | 중환자실 | 249.9 → 360 Hz 리샘플 |
| **VitalDB** | 수술실 | 원 표본율 확인 후 360 Hz 리샘플 |
| **NSRR** (shhs·wsc·nfs) | 수면 | 원 표본율 확인 후 360 Hz 리샘플 |
| **GalaxyPPG의 Polar H10 ECG** | 일상 웨어러블 (실제 motion artifact) | 130 → 360 Hz `resample_poly(x, 36, 13)` |

GalaxyPPG 주의: Galaxy Watch는 PPG 담당이고 **ECG는 Polar H10 흉부 스트랩**이다. 가슴 전극이라
MIT-BIH MLII와 도메인이 가깝다. CC-BY 4.0. 단위가 mV인지 raw인지 확인 후 변환.

**절차**: 리샘플·단위 정합 → 성분 분해 → `noise_encoders.yaml` 그대로 마스킹 → 전후 제시.
**재학습·재판별 금지**(§0 원칙 6).

**출력**: `results/04_external/fig2_external.png` — 데이터셋별 대표 사례 2~3개,
각 사례마다 [원 분절 → 성분 분해 → 복원 → 스펙트럼 전후]

---

## 11. S7 — 통계 및 원고 매핑

| 대상 | 방법 |
|---|---|
| S4 대응 분석 | 인코더별 Wilcoxon(잡음 상관 − clean 상관) + Holm, α=.05, 효과크기 r=Z/√N |
| S5 마스킹 전후 | 분절 쌍대 Wilcoxon(전 vs 후) + Holm, α=.05, 효과크기 r=Z/√N. SDNN은 중앙값[IQR] |
| S5 아블레이션 | 인코더별로 마스킹 전 대비 쌍대 Wilcoxon + Holm |
| S6 | 통계 없음 |

| 원고 위치 | 파일 |
|---|---|
| 표 1 (인코더 × 참조 대응, \|r\| + RMSE) | `stage1_matrix.csv` + `stage1_stats.csv` |
| 표 1-보조 (참조 간 상관) | `reference_correlation.csv` |
| 표 1-보조 (기여 분해) | `stage1_contribution.csv` |
| 표 2 (M0 대비 SNR 계단 + 전수 지도 백분위) | `table_masking.csv` + `mask_map.csv` + `stats_pairwise.csv` |
| 그림 1 (대응 행렬) | `fig1_correspondence.png` |
| 그림 2 (외부 시연) | `fig2_external.png` |
| 그림 3 (아블레이션, 주 실험) | `fig_ablation.png` + `ablation.csv` |

---

## 12. 단위 테스트 (`tests/`)

- `test_injection`: 잡음별 실측 SNR = 추첨값 ±0.1 dB(분산 기준) / `x_noisy == x_clean+bw+ma+em`
- `test_noise_time_split`: train과 val·test의 잡음 원본 구간 비중첩
- `test_split_leakage`: 기록 교집합 0
- `test_shapes`: component·masked_reconstruct 출력 (B,1,3840), `crop()` 후 (B,1,3600)
- `test_mask_effect`: mask_idx의 인코딩이 실제로 0으로 치환됨 (인코더 가중치는 불변)
- `test_metrics`: `rpeak_prf(ref, ref) == 1.0` (동일 피크 집합) / `score(clean, clean, 주석)`의
  F1 > 0.9 — 상한이 **0.981**이라 1.0이 아니다(검출기 vs 사람 주석 불일치, test 300분절 실측)
- `test_resample`: 외부 표본율 → 360 Hz 후 길이 정합

---

## 13. 태스크 순서

```
S1  데이터 구축                                  ✅ 완료·동결
S2  모델 이식                                    ✅ 완료·동결
S3  학습 (K=8, seed 42) → 비용 함수 ②로 체크포인트 선택
    → ★선택 에폭·S·후보 집합 크기 보고
T6.9 val 전체 리허설 (test 봉인 해제 전 필수 관문) → results/00_rehearsal/
    → ★보고 후 `pre-test-freeze` 태그
T7-0 ★사용자 승인 — test 봉인 해제. 승인 전 해제 금지
S4  인코더–참조 대응 분석 → ★보고
S5  마스킹 전후 평가 + 아블레이션
S6  외부 적용 시연
S7  통계 → 원고 수치 정리
```

★ = 사용자 확인 없이 다음 단계로 진행하지 않는다.
**추가 구조·손실·K·시드 탐색은 하지 않는다.** 모델 탐색은 종료됐다.

---

## 14. 코딩 규약

- 설정은 `configs/`에만. 하드코딩 금지. 결과 파일 메타에 시드·설정 해시 기록.
- 결과는 CSV 저장 후 표·그림 생성. 그림은 dpi 지정 PNG, 한글 폰트(Malgun Gothic / NanumGothic 폴백).
- 원본 데이터는 절대 수정하지 않는다. 커밋 단위 = 태스크.
- 성분·인코더 표시는 **1부터** (`enc_label`). 내부 인덱스만 0-based.
- 폐기된 실행·구버전은 `_work/archive/`로 옮긴다. 지우지 않는다.
- 본 노선이 아닌 실험은 `experiments/<이름>/outputs/`에 둔다. `results/`에 섞지 않는다.

## 15. 선행 저장소

| 저장소 | 논문 | 라이선스 |
|---|---|---|
| `github.com/mbwebster/self-supervised-bss-via-multi-encoder-ae` | Webster & Lee, *Neurocomputing* 2025, doi:10.1016/j.neucom.2025.131008 | MIT |
| `github.com/mbwebster/meae-heart-rate-extraction-from-noisy-ppg` | Webster, Lee D, Lee J, *Comput Biol Med* 2025;199:111319 | MIT |

원 저작권 고지를 유지하고 두 논문을 모두 인용한다. 이식 커밋: `d0c94a9d`, `91f1e0e2`.

후속 저장소 README: "학습 후 어느 인코더가 원하는 소스를 만드는지 **수동 검사(manual inspection)** 가
필요하다." → 우리의 참조 기반 자동·정량 판별(S4)이 대체하는 지점.
