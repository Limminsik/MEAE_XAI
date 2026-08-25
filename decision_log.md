# decision_log.md — 설계 결정 대장

원고 방법·고찰 집필과 심사 답변에 바로 인용하기 위한 대장. **한 줄 = 한 결정.**
근거의 상세 수치는 `results/data_notes.md`의 해당 절을 참조한다.
설계서 본문(`RESEARCH_DESIGN.md`)과 어긋나는 항목이 생기면 이 대장이 아니라 설계서를 고친다.

날짜는 전부 2026-08-25 (T1–T5 세션).

| # | 결정 | 근거·증거 | 반영 위치 |
|---|---|---|---|
| D01 | Python 3.9.21 + uv + torch cu129 고정, `neurokit2==0.2.10` 핀 | 최신 neurokit2(0.2.12)는 `float \| None` 문법이라 py3.9에서 import 실패 | `requirements.txt`, 설계 §1 |
| D02 | NSTDB 잡음은 **채널 0(noise1)** 사용 | 잡음도 2채널 홀터로 녹음돼 noise1/noise2를 가짐. `nst` 도구가 채널 번호끼리 짝지어 섞는 관례를 따름 (118e00−118 역산에서 ch0↔noise1, ch1↔noise2가 **동일 지연 531200**에 수렴) | `configs: data.noise_channel: 0`, notes §2 |
| D03 | 기록 단위 분할 **32/5/9**, 층화 없음, `split_seed: 42` 고정 | 46 × (0.7/0.1/0.2) = 32.2/4.6/9.2로 정수가 안 떨어짐. split.json 1회 생성 후 모든 K·시드가 공유하며 학습 시드는 분할에 무관 | `src/data/split.py`, `data/processed/split.json`, notes §1 |
| D04 | 주입 SNR의 신호 전력을 `mean(x²)` → **`var(x)`** 로 변경 | MIT-BIH 전극 오프셋(DC)이 `mean(x²)`의 **중앙 35.1%·최대 97.5%** 를 차지해 명목 SNR과 실제 SNR이 분절마다 0–10 dB 어긋남 | 설계 §4-4, `src/data/build.py`, notes §2 |
| D05 | 실측 SNR 검증도 동일하게 분산 기준 | 주입과 검증의 정의가 다르면 테스트가 항상 실패 | 설계 §4 DoD·§11, `tests/test_s1.py` |
| D06 | 비트 심볼에 **`f`(paced+normal 융합박) 추가** | 제외하면 217번 기록의 실제 심박 260개가 R-피크에서 누락 | 설계 §4-2, `src/data/build.py`, notes §7 |
| D07 | 잡음 강도 범위 **[0, 12] dB 확정** | 주입 후 RMS가 NSTDB 원본 범위 안(배율 중앙 0.23–0.96×). 단 ma는 47.3%에서 증폭되며 95%분위 3.5배 | `configs: data.noise_snr_range_db`, notes §3 |
| D08 | `nstdbgen`(nst 도구) 재사용 **기각** | nst는 잡음 1종만·구간의 43%만 주입하고 SNR이 6단계 고정이며 잡음 원본 시간 분할을 통제할 수 없음 | notes §5-3, 설계 §4 |
| D09 | MIT-BIH 자체 잡음 구간을 **학습에 쓰지 않음** | 전체의 1.68%(139분절)뿐이고 clean·bw/ma/em 참조가 없어 S4 채점이 불가능 | notes §5-2 |
| D10 | 선행 저장소 2개를 **무수정 이식**(모델·손실 파일만) | MIT 라이선스. 커밋 `d0c94a9d`(기반), `91f1e0e2`(후속) | `src/model/_vendor_*.py`, notes §T3 |
| D11 | sparse mixing은 **`WeightSeparationLossAlternative`** 채택 | ECG 실험 설정(`mesa_ecg_bss`)이 쓴 구현. PPG 논문은 `WeightSeparationLoss`+`sep_lr 0.05`로 짝이 다름 | `configs: loss.sep_impl`, `src/model/losses.py`, notes §8 |
| D12 | 입력을 **3840으로 대칭 제로 패딩**(양쪽 120) | 선행 인코더 깊이가 **8**(MaxPool 2배×8 = 256배)이라 입력이 256의 배수여야 함. 3600→3584로 어긋남. 후속 저장소도 6000→6144(=256×24)로 패딩 | 설계 §5·§5A, `configs: data.pad_to/pad_each`, `src/model/meae.py`, notes §9 |
| D13 | 4096이 아니라 3840 | 필요 조건은 2의 거듭제곱이 아니라 **256의 배수**. 3840이 패딩 240으로 절반이라 경계 왜곡 구간이 작음 | 설계 §5A, notes §9 |
| D14 | 모든 상관·지표는 **crop 후 중앙 3600**에서 계산 | 패딩 0 구간이 상관을 희석. `metrics.score()`가 3840 입력을 `ValueError`로 차단 | `src/model/meae.py`, `src/metrics.py`, `tests/test_metrics.py` |
| D15 | **K = {4, 8}** (6 제외) | K=6은 디코더 `GroupNorm(num_groups=K)`가 채널 32/64/128/256을 나누지 못해 `ValueError`. 채널 변경은 §0 원칙 2 위반 | 설계 §3·§5, `src/model/meae.py`, notes §10 |
| D16 | 재구성 손실 **BCE → MSE**, zero-recon도 함께 MSE | 정규화를 하지 않고 ECG는 음수를 가짐. 선행은 min-max [0,1] + BCEWithLogits | 설계 §5, `src/model/losses.py`, notes §11 |
| D17 | 학습 설정은 선행을 따르지 않음 (`max_epoch 100` + 조기종료, 체크포인트는 분리 품질) | 선행은 `max_epochs 25`, `monitor: recon_loss/train`. 선행이 수동으로 하던 체크포인트 고르기를 자동화한 것이 우리 기여 | 설계 §6, `configs: train.*`, notes §11 |
| D18 | **S5를 "비교군 성능 평가" → "마스킹 전후 = 판별의 인과 검증"으로 재정의.** DAE·대역통과·웨이블릿 전부 삭제 | 주장의 구조가 ①분해→②판별→③판별이 진짜다이며, ③의 대조는 외부 기법이 아니라 같은 모델에서 마스킹만 바꾼 조건 | 설계 §8·§10·§12, notes §14 |
| D19 | 인코더 단독 마스킹 아블레이션을 **S5의 주 실험으로 승격** | S4는 상관(연관) 근거, 아블레이션은 개입(intervention) 근거로 논증 성격이 다름 | 설계 §8, notes §14 |
| D20 | **지표 위계**: SNR 개선=주, RMSE=교차확인, F1=경고등, SDNN=판정 보류 | R-피크 F1 상한이 **0.981**(검출기 vs 사람 주석), 마스킹 전 0.956 → 개선 여지 2.5%p뿐이라 개선의 증명에 쓸 수 없음 | 설계 §8, notes §13 |
| D21 | **체크포인트 지표 교체**: `분리 품질 = 잡음별 max\|r\| 평균 − 0.15×(최고 인코더 중복 수)` | 초판 정의는 붕괴 상태(에폭 5, 네 참조 모두 같은 인코더 1위, 재구성에 R-피크 없음)를 최고로 뽑음. 페널티는 0.05–0.30 전 구간에서 같은 에폭 선택 | 설계 §6, `configs: train.separation_duplicate_penalty`, `src/train.py`, notes §16 |
| D22 | **헝가리안 1:1 강제 배정 기각** | 참조를 서로 다른 인코더에 강제 배정하면 분리 실패 시에도 그럴듯한 점수가 나와 **실패를 가림** | 설계 §6, notes §16 |
| D23 | **λ_mixing 1e-3 → 1e-2 상향** (파일럿 검증) | λ=1e-3에서 mixing 원값이 8.3%만 감소(재구성은 91.6% 감소), 디코더 그래디언트 비 **0.0002**. 1e-2에서 59.8% 감소·재구성 손해 없음·clean\|r\| 0.435→0.555 | `--lambda-mixing` 오버라이드, notes §17 · **본 설정 반영은 미확정(OPEN-01)** |
| D24 | **판정 기준 선등록** — λ 1e-2·1e-1 모두에서 bw·ma 최고 인코더가 같으면 결과로 확정하고 판별 단위를 심장/잡음군으로 조정 검토 | 결과를 보고 기준을 옮기면 선택 편향. 실행 **전에** 기록 | notes §18 |

## 아직 결정되지 않은 것

`OPEN_ITEMS.md` 참조. 특히 λ 최종값(OPEN-01), δ 결정 절차(OPEN-02), 판별 단위(OPEN-03)는
설계서 §6·§7과 직결되므로 확정 전까지 해당 조항을 수정하지 않는다.
