# 실험 SV1 — 잡음 지도(supervised) 방식 병렬 테스트

**기존 자기지도 파이프라인과 완전히 분리된 방향 탐색 실험이다.**
`src/`·`configs/`·`RESEARCH_DESIGN.md`·`outputs/`를 일절 건드리지 않는다.
여기서 나온 결과는 기존 노선을 대체하지 않으며, 대체 여부는 사용자가 결정한다.

## 기존 노선과 무엇이 다른가

| | 기존 (자기지도) | **SV1 (잡음 지도)** |
|---|---|---|
| 손실에 정답 사용 | **금지** (§0 원칙 3) | **bw·ma·em 3종을 성분 목표로 사용** |
| 인코더 역할 | 학습이 스스로 정함 | enc1·2·3을 bw·ma·em에 **고정 할당** |
| sparse mixing | 사용 | **제외** (정답으로 분업이 강제되므로) |
| clean 사용 | 채점만 | **손실에 쓰지 않음** (여전히 채점만) |

**핵심 질문**: 잡음 3종만 지도했을 때 **지도받지 않은 enc4가 심장 성분을 담당하게 되는가.**

clean은 손실에 넣지 않는다. 심장 성분은 "남은 것"으로만 주어지며, enc4가 그것을 가져가는지가
이 실험이 보려는 전부다.

## 구조 (기존과 동일)

`src/model/_vendor_meae.py` 그대로 · K=4 · hidden 64(인코더당 16) · base_ch 32 ·
3840 대칭 제로 패딩 · 지표는 crop 후 중앙 3600.

## 손실

```
L = MSE(x̂, x_noisy)                                  재구성 (유지)
  + λ_sup · Σ_{t∈{bw,ma,em}} MSE(x̂_t, α_t·n_t)        잡음 3종 지도 (신설)
  + λ_zero · MSE(decode(0), 0)                        zero reconstruction (유지)
  + λ_z · Σ_k mean(z_k²)                              인코딩 L2 (유지)
```

`α_t·n_t`는 S1에서 저장한 성분 배열 그대로다(주입 시 배율이 이미 적용됨).
성분 목표는 3600이므로 3840으로 제로 패딩해 맞춘다.

## 비용

성분 손실 때문에 **디코더를 인코더 수만큼 더 통과**한다. 기존은 스텝당 디코더 2회
(재구성·zero), SV1은 6회(재구성·zero·성분 4개). 스텝당 소요를 로그에 기록한다.

## 실행

```bash
python -m experiments.supervised_noise.train_sv --seed 42
```

## 산출물

| 위치 | 내용 |
|---|---|
| `_work/runs/SV1_K4_seed<seed>/` | 학습 산출물 — 가중치 3개 · `history.csv` · `checkpoints.json` · `console.log` |
| **`experiments/supervised_noise/outputs/`** | **분석 결과 — 이 실험의 표와 그림 전부** |

```
outputs/
  fidelity.csv        재구성 충실도 판정 (기존 자기지도 K4와 나란히)
  diagnostics.csv     잔차 상관 · R-피크 진폭비
  band_keep.csv       대역별 보존율 · 스펙트럼 기울기
  separation.csv      인코더별 분리 요약
  figures/
    components_compare.png            자기지도 vs SV1 나란히 (4성분 + 참조 4종)
    by_component_SV1_K4_seed42/       성분 하나당 그림 한 장 (분절 3개 × 파형·스펙트럼)
    by_component_K4_seed42/           같은 형식, 기존 자기지도
    *_zoom.png · *_spectrum.png       재구성 충실도 그림
    keep_curve.png                    주파수별 보존율 곡선
```

## 성분별 그림 다시 뽑기

```bash
python -m experiments.supervised_noise.fig_components --run SV1_K4_seed42
python -m experiments.supervised_noise.fig_components --run K4_seed42   # 대조군
```
