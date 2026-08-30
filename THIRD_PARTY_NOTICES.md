# 제3자 저작물 고지 — Third-Party Notices

이 저장소는 아래 저작물의 소스를 포함한다. 각 저작물의 라이선스가 그대로 적용된다.

---

## 1. self-supervised-bss-via-multi-encoder-ae

- **출처** https://github.com/mbwebster/self-supervised-bss-via-multi-encoder-ae
- **커밋** `d0c94a9d5dec8dd5d54baebdb9963b79860cb200` (2025-12-13)
- **라이선스** MIT — Copyright (c) 2023 Matthew B. Webster

이 저장소에 포함된 파일과 개작 여부:

| 이 저장소의 파일 | 원본 파일 | 개작 |
|---|---|---|
| `src/model/_vendor_meae.py` | `models/cnn_multi_enc_ae_1d.py` | **개작함** (아래) |
| `src/model/_vendor_separation_loss.py` | `models/separation_loss.py` | 무수정 |

`_vendor_meae.py` 의 개작 지점은 세 곳이며, 파일 머리말에 근거와 함께 적어 두었다.

1. `EncoderBlock` · `DecoderBlock` 에 `dilation` 인자 추가 (padding 을 dilation 배로 조정)
2. `ConvolutionalEncoder` · `ConvolutionalDecoder` · `ConvolutionalAutoencoder` 가
   블록별 dilation 목록을 받아 전달
3. 선택적 잔차 연결(`skip_levels` · `skip_weight`)과 인코더별 잔차를 돌려주는 `encode_all`

기본 인자(`dilations=None`, `skip_levels=None`)에서는 원본과 동일하게 동작하며,
이는 회귀 테스트로 확인한다 —
`tests/test_model.py::test_dilation_default_matches_original`.

### 인용

> Webster MB, Lee J. Blind source separation via multi-encoder autoencoders.
> *Neurocomputing* 2025. doi:10.1016/j.neucom.2025.131008 (arXiv:2309.07138)

> Webster MB, Lee D, Lee J. Heart rate extraction from noisy PPG via
> multi-encoder autoencoders. *Comput Biol Med* 2025;199:111319 (arXiv:2504.09132)

---

## 2. 데이터셋

데이터는 이 저장소에 포함하지 않는다. 각 제공처의 이용 조건을 따른다.

| 데이터셋 | 용도 | 출처 |
|---|---|---|
| MIT-BIH Arrhythmia Database | 학습·평가의 clean ECG | PhysioNet (ODC-BY 1.0) |
| MIT-BIH Noise Stress Test Database | 주입 잡음 bw · ma · em | PhysioNet (ODC-BY 1.0) |
| VitalDB | 외부 적용 (07) | https://vitaldb.net — 이용 약관 |
| MIMIC-IV-ECG | 외부 적용 (07) | PhysioNet — 자격 승인 필요 |
| GalaxyPPG | 외부 적용 (07) | 데이터셋 배포처의 조건 |

MIMIC-IV 계열은 PhysioNet 자격 심사와 데이터 사용 협약(DUA)을 거쳐야 접근할 수 있다.
내려받는 방법은 `README.md` 「실행」 절에 있다.

---

## 3. 의존 패키지

`requirements.txt` 의 패키지는 각자의 라이선스(BSD-3-Clause · MIT · Apache-2.0 등)를
따르며, 이 저장소는 그 소스를 포함하지 않는다.
