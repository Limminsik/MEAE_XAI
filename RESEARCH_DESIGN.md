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
│
├── 01_build.py   02_model.py   03_bss.py   04_masked_denoising.py   05_validation.py
│      ↑ 실제로 쓰는 코드는 이 다섯뿐이다. 번호가 단계 순서다.
│
├── src/                            다섯이 공유하는 라이브러리
│   ├── core.py                     체크포인트 로드·성분 추출·표준화 지표·표 렌더링
│   ├── data/{download,split,build,dataset}.py      §4
│   ├── model/{meae,losses,_vendor_*}.py            §5
│   ├── metrics.py                  §9 지표 5종 · R-피크 · SNR/RMSE/SDNN
│   └── spectral.py  stats.py  viz.py
├── tests/   data/raw/  data/processed/   (data 는 git 제외)
│
├── results/       ★ 폴더 번호 = 스크립트 번호
│     01_build/  02_model/<run>/(+fidelity/)  03_bss/<run>/<split>/
│     04_masked_denoising/<run>/<split>/      05_validation/<run>/<source>/
│
└── _work/archive/    과거 실행·구버전·구코드·보조 실험 (보존, 실행 대상 아님)
```

실행 이름 = `K<인코더수>_seed<시드>` + 오버라이드 접미사(`_lz0`, `_h128`). 접미사가 없으면 config 그대로다.
**번호 붙은 스크립트만이 실제로 쓰는 코드다.** 새 분석이 생기면 그 단계 스크립트 안에 넣고,
여러 단계가 함께 쓰는 것만 `src/core.py` 로 올린다.

## 3. configs/default.yaml

설정의 단일 원천은 `configs/default.yaml`이다. 이 문서는 값을 복제하지 않는다.
값을 바꿀 때는 config를 고치고 이 문서의 해당 절을 함께 갱신한다.

---

## 4. 01 — 데이터셋 구축 (`01_build.py`) — 완료·동결

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

## 5. 02 — 모델 (`02_model.py`, `src/model/`) — 완료·동결

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

## 6. 02 — 비용 함수 ① 학습 손실 (`src/model/losses.py`)

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

## 7. 02 — 비용 함수 ② 체크포인트 선택 (`02_model.py`)

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
산출물 `results/01_train/K8_seed42/epoch_compare/`).

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

**출력**: `results/02_model/<run>/` — 선택 가중치 · `history.csv` · `selection.json` · `console.log` ·
`pool/`(후보 구간 가중치) · `plots/`(에폭 간격 성분 파형)

---

## 7.5 02 — 재구성 충실도 진단 (`02_model.py --diagnose`)

**관문이 아니라 서술 지표다.** 선행이 명시한 재구성–분리 트레이드오프
("인코딩이 너무 작으면 좋은 재구성이 불가하고, 너무 크면 인코더가 특화 대신 전체 특징 공간으로
일반화된다")를 어디쯤에서 취했는지 보고하기 위한 산출물이며, 합격/불합격 수치를 두지 않는다.

| 지표 | 정의 |
|---|---|
| 전대역 보존율 | `P(x̂)/P(x_noisy)` 의 분절 중앙값 |
| 대역별 보존율 | 5–15 / 15–25 / 25–40 / **40–60·60–90(59–61 Hz 노치 제외)** Hz |
| 꺾임 지점 | 보존율이 0.7 / 0.5 아래로 처음 내려가는 주파수 |
| log-log PSD 기울기 | 입력·재구성과 그 차이 (10–60 Hz 회귀) |
| R-피크 진폭비 | R-피크 ±30샘플 첨두간 진폭의 재구성/입력 비 |

대역을 나누는 이유: 저주파 포락선만 재현해도 전대역 보존율은 높게 나온다.

**디노이징 지수(`RMSE(x̂,x_noisy)` vs `RMSE(x̂,clean)`)와 잔차 상관은 산출하지 않는다.**

**출력**: `results/02_model/<run>/fidelity/` — `fidelity.csv` · `fidelity_note.txt` ·
`keep_curve.npz` · `figures/{zoom, spectrum, keep_curve}.png`

```bash
python -m src.fidelity --run K8_seed42 --split val --n 900
```

---

## 8. 03 — 분리·참조 대응 분석 (`03_bss.py`) ★핵심 단계

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

**표 렌더링** — 아래 §8-표시규칙을 따른다. 잡음 3종을 "잡음 최대"로 요약하지 말고 **4열 그대로** 싣는다.

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
표시는 §8-표시규칙을 따른다(낮을수록 유사). 값에 대한 해석·명명은 붙이지 않는다.

#### [S4-03] MAD 산출 명세 (확정)

구조는 S4-01·S4-02와 같다.

**1단계 — 분절 내 표준화** (S4-02와 동일) `ã = (x̂_k − mean)/std`, `r̃ = (r − mean)/std`

**2단계 — 분절 내 MAD**

```
MAD_kr^(s) = max_i | ã[i] − r̃[i] |
```

**3단계 — 분절 간 집계**

```
MAD̅_kr = mean_s MAD_kr^(s)
σ_kr    = std_s  MAD_kr^(s)      (ddof=1)
```

구간은 패딩 제외 중앙 N=3600, 분절은 val 900, **단위는 표준편차**.
산출물 `mad_matrix.csv` (K×4, `MAD̅ ± σ`) · `mad_persegment.csv`(분절별 값 + argmax 표본·초) ·
`figures/mad_argmax_hist.png` — **최대 편차 발생 시점의 분포**(분절별 argmax 위치 히스토그램).
MAD가 어느 구간에서 발생하는지 확인하기 위한 것이다.

표시는 §8-표시규칙을 따른다.
**각주**: 값이 낮을수록 유사하며, 참조 파형의 첨도에 영향받으므로 **열 내 비교에 적합**하다.
값에 대한 해석·명명은 붙이지 않는다.

S4-02·S4-03 모두 부호 정렬을 하지 않으므로 반대 위상은 값이 커진다.

#### §8-표시규칙 — 표에서 무엇을 굵게 하는가

**표시는 열별이다.** 한 참조를 어느 인코더가 가장 잘 잡는지를 **열 안에서** 비교한다.
행별 표시는 쓰지 않는다.

| 지표 | 굵게 표시할 칸 |
|---|---|
| \|r\| | **열별 상위 2개** (높을수록 유사) |
| RMSE_norm | **열별 하위 2개** (낮을수록 유사) |
| MAD | **열별 하위 2개** (낮을수록 유사) |

CSV·콘솔에서는 굵은 글씨를 쓸 수 없으므로 `[1]` = 열 1위, `[2]` = 열 2위로 표기한다.
히트맵에서는 지표 세 줄을 각각 굵게 처리하고 `●`(1위)·`○`(2위)를 붙인다 —
지표마다 표시 대상이 다르므로 칸 하나에 여러 지표가 동시에 표시될 수 있다.

**세 지표의 행별 지목 일치 여부**를 따로 표기한다 — 한 인코더에 대해 \|r\| 최대 참조,
RMSE_norm 최소 참조, MAD 최소 참조가 같은지. 산출물 `metric_agreement.csv`.

#### §8-마무리 — S4 종료 산출물

모델은 **K=8 · seed 42**, 체크포인트 규칙(`val_recon ≤ 1.5 × 최소` 관문 안에서 clean 최대 상관
에폭 = 에폭 48)으로 확정된 것을 쓴다. **추가 학습·탐색 없음.**

| 구분 | 산출물 |
|---|---|
| 표 (8행×4열, 각 칸 세 지표) | `corr_matrix.csv` · `rmse_norm_matrix.csv` · `mad_matrix.csv` · `stage1_matrix.csv`(통합) |
| 지표 일치 여부 | `metric_agreement.csv` |
| 대응 히트맵 | `figures/fig1_correspondence.png` |
| **겹침 격자 8×4** | `figures/fig_overlay_grid.png` |
| 기초 성분 파형 (적층) | `figures/components_bw_{high,mid,low}.png` |
| 분절별 원값 CSV (부호 포함) | `corr_persegment.csv` · `rmse_norm_persegment.csv` · `mad_persegment.csv` |
| MAD 최대 편차 발생 시점 분포 | `figures/mad_argmax_hist.png` |
| 참조 간 상관·스펙트럼 · 기여 분해 | `reference_correlation.csv` · `reference_spectrum.csv` · `stage1_contribution.csv` |

**겹침 격자 그림**: 기초 성분 파형(적층)을 먼저 도출한 뒤, 성분과 참조를 표준화해 **같은 패널에
겹쳐 그린** 8×4 격자를 싣는다. 패널마다 그 분절의 세 지표를 병기하고 전 패널이 공통 y축을 쓴다.

**대표 분절 선정 — 사전 규칙**: 주입 SNR 3종의 평균이 val 중앙값에 가장 가까운 분절.
모델 출력·지표와 무관한 데이터 속성만으로 정해, 결과를 보고 고르는 일을 막는다.

**하지 않을 것**: 인코더 명명 · 값 해석 · 순열 검정 · 코히런스 · 대역 분해 ·
SSD·PRD(RMSE_norm과 순위 동일) · N 기반 p값.

**산출 위치**: val 산출물은 그 실행 폴더 안 `results/02_model/<run>/metric/` 에 둔다
(`--split test` 로 실행하면 봉인 해제 후 `results/02_separation/` 에 같은 코드로 생성된다).

**이것으로 S4를 종료한다.** 마스킹·디노이징(S5)과 외부 적용(S6)은 별도 단계로 이후 착수한다.

**그림 규칙**: 성분 파형 그림은 전 성분을 **공통 y축**으로 그린다(성분마다 축을 따로 잡으면
크기가 작은 성분이 크게 보여 비교가 안 된다). y축 단위를 명기한다 — 성분·참조는 z-정규화
후 단위 없음, 입력 `x_noisy`만 원값 mV로 축을 따로 둔다.

### 산출물

| 산출물 | 파일 |
|---|---|
| ① 인코더 × 참조 대응표 (ρ̄ ± σ) | `corr_matrix.csv`, `fig1_correspondence.png` |
| ①-보조 분절별 원값 (부호 포함) · 부호 양수 비율 | `corr_persegment.csv`, `corr_sign.csv` |
| ② 정규화 RMSE 대응표 (RMSE̅ ± σ) | `rmse_norm_matrix.csv`, `rmse_norm_persegment.csv` |
| ③ MAD 대응표 (MAD̅ ± σ) · 최대 편차 시점 분포 | `mad_matrix.csv`, `mad_persegment.csv`, `fig mad_argmax_hist.png` |
| 세 지표 지목 일치 여부 | `metric_agreement.csv` |
| 성분 × 참조 겹침 격자 8×4 (대표 분절) | `figures/fig_overlay_grid.png` |
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

## 9. 04 — 마스킹 복원 평가 (`04_masked_denoising.py`) ★판별의 인과 검증

**성능 경쟁이 아니다.** 03의 대응 판별이 실제로 맞는지를 개입(intervention)으로 검증한다.
대역통과·웨이블릿·DAE 비교군은 두지 않는다 — 표의 초점이 "어느 기법이 더 좋은가"로 옮겨가면
③의 논리가 묻힌다. 모델은 03에서 확정한 **K=8 seed 42 체크포인트**, **추론만** 한다.

### 전수 조합

**2^K = 256개 마스킹 조합 × 해당 split 전체 분절.** 조건 사다리(M0–M5)를 미리 정해 변호하는
대신 지형 전체를 산출한다. **최적 조합 선정은 하지 않는다** — 전수 결과를 먼저 보고,
지표 5종이 각각 어느 조합을 지목하는지 표로 제시한 뒤 일치 여부를 확인하고
선정 기준을 따로 논의해 결정한다.

### 비교 구조 — 기준은 `x_clean`, **mV 원단위**(표준화하지 않는다)

| | 상태 |
|---|---|
| **ⓐ** | `x_noisy` — 처리 전 |
| **ⓑ** | M0 복원 (마스킹 없음) — 재구성만 거친 상태 |
| **ⓒ** | 각 마스킹 조합 — 최종 |

```
ⓒ − ⓐ = 전체 개선량        ⓒ − ⓑ = 마스킹 순효과
```

세 상태를 각각 clean과 비교한다. ⓑ를 따로 두는 이유: 개선량 중 어디까지가 재구성 자체의
몫이고 어디부터가 마스킹의 몫인지 갈라야 하기 때문이다.

### 지표 5종 (DeepFilter·MECG-E 표준 세트)

| 지표 | 정의 | 방향 |
|---|---|---|
| **SSD** | `Σ_i (est − clean)²` | 낮을수록 유사 |
| **MAD** | `max_i \|est − clean\|` | 낮을수록 유사 |
| **PRD** | `100·√(Σ(est−clean)² / Σclean²)` [%] | 낮을수록 유사 |
| **CosSim** | 코사인 유사도 | 높을수록 유사 |
| **ΔSNR** | `10log10(var(clean) / mean(잔차²))` | 높을수록 유사 |

MAD는 03의 MAD와 이름만 같다 — **여기는 표준화하지 않은 mV 원단위**다.
분절별로 산출하고 중앙값·평균±SD(ddof=1)로 집계한다.

### 부수 산출

- **단독 마스킹 8개** — 인코더를 하나씩만 껐을 때의 개별 효과.
- **누적 곡선** — 03의 잡음 유사도(잡음 3종 최대 ρ̄) 높은 순으로 하나씩 추가 마스킹.
- **R-피크 진폭비(복원/clean)** — 형태 훼손 감시. 1.0이면 clean과 같은 첨두간 진폭.

### 산출물 `results/04_masked_denoising/<run>/<split>/`

| 파일 | 내용 |
|---|---|
| `exhaustive.csv` | 256조합 × 지표 5종 (중앙·평균·SD) + ⓐ·ⓑ 대비 델타 + R피크 진폭비 |
| `baseline.csv` | ⓐ `x_noisy` · ⓑ M0 |
| `best_by_metric.csv` | 지표별 지목 조합과 **일치 여부** |
| `single_mask.csv` · `cumulative.csv` (+`_order`) · `rpeak_ratio.csv` | 부수 산출 |
| `persegment_top.csv` | M0·단독 8개의 분절별 원값 |
| `meta.json` | 실행·에폭·분절 수·누적 순서 |
| `figures/` | `exhaustive_scatter` · `single_mask` · `cumulative` · `rpeak_ratio` |

**값에 대한 해석·명명은 붙이지 않는다.**

---

## 10. 05 — 외부 적용 시연 (`05_validation.py`)

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

## 11. 통계 및 원고 매핑

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
01  데이터셋 구축                    ✅ 완료·동결 (모델 이식 포함)
02  학습 (K=8, seed 42) + 충실도 진단  ✅ 완료 — 에폭 48 확정
03  분리·참조 대응 분석               ✅ 종료 — 지표 3종 확정, val 산출 완료
04  마스킹 복원 평가                  🔧 전수 256조합 val 산출 완료
    → ★지표 5종의 지목 조합 비교 보고 → **선정 기준 논의·확정** (승인 전 선정 금지)
T7-0 ★사용자 승인 — test 봉인 해제. 승인 전 해제 금지
05  외부 적용 시연
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
- 본 노선이 아닌 실험은 `_work/archive/`에 둔다. `results/`에 섞지 않는다.

## 15. 선행 저장소

| 저장소 | 논문 | 라이선스 |
|---|---|---|
| `github.com/mbwebster/self-supervised-bss-via-multi-encoder-ae` | Webster & Lee, *Neurocomputing* 2025, doi:10.1016/j.neucom.2025.131008 | MIT |
| `github.com/mbwebster/meae-heart-rate-extraction-from-noisy-ppg` | Webster, Lee D, Lee J, *Comput Biol Med* 2025;199:111319 | MIT |

원 저작권 고지를 유지하고 두 논문을 모두 인용한다. 이식 커밋: `d0c94a9d`, `91f1e0e2`.

후속 저장소 README: "학습 후 어느 인코더가 원하는 소스를 만드는지 **수동 검사(manual inspection)** 가
필요하다." → 우리의 참조 기반 자동·정량 판별(S4)이 대체하는 지점.
