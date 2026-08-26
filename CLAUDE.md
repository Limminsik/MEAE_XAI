# meae_xai

모든 구현은 RESEARCH_DESIGN.md를 따른다. 설계와 다른 구현이 필요하면 진행하지 말고 사용자에게 먼저 보고한다.

- §0 불변 원칙(마스킹 대상은 인코딩, 자기지도 학습, 정답은 채점표, 기록 단위 분할)은 위반 금지.
- 태스크 순서는 RESEARCH_DESIGN §13을 따르며 ★ 표시 지점에서는 사용자 확인 없이 다음 단계로 진행하지 않는다.
- 설정은 configs/default.yaml 에만. 코드 하드코딩 금지.
- 환경: uv + Python 3.9.21 + requirements.txt, torch는 cu129 인덱스로 설치.

## 문서 규칙

문서는 **두 개뿐**이다. 새로 만들지 않는다.
- `README.md` — 연구 현황판. 현황·산출물 지도·실험 대장·핵심 수치·결정 이력·미결·원고 대기.
  단계를 마치거나 결정이 확정되면 **여기를 갱신**한다.
- `RESEARCH_DESIGN.md` — 설계서. 구현은 이 문서를 따르며, 어긋나면 문서를 고친다.

## 코드 규칙

**실제로 쓰는 코드는 번호 붙은 다섯 개뿐이다.** 번호가 단계 순서다.

| 스크립트 | 역할 |
|---|---|
| `01_build.py` | 데이터셋 로드·검증·분할·분절·잡음 주입 |
| `02_model.py` | 모델·학습·두 비용 함수. `--diagnose` 로 재구성 충실도 진단 |
| `03_bss.py` | 성분 분리 + 참조 대응 분석 (지표 3종) |
| `04_masked_denoising.py` | 마스킹 복원 평가 (전수 2^K, 지표 5종) |
| `05_validation.py` | 외부 데이터 적용 (VitalDB · MIMIC-IV · GalaxyPPG) |

- 새 분석이 생기면 **그 단계 스크립트 안에** 넣는다. 새 파일을 만들지 않는다.
- 여러 단계가 함께 쓰는 것만 `src/core.py` 로 올린다.
- `src/` 의 나머지는 라이브러리다 — `data/`(S1 동결) · `model/`(S2 동결) ·
  `core.py`(공용) · `metrics.py` · `spectral.py` · `viz.py`.
- 한글 콘솔 출력이 있으므로 `PYTHONIOENCODING=utf-8` 로 실행한다 (Windows cp949 오류).

## 산출물 규칙

- **`results/` 의 폴더 번호는 스크립트 번호와 같다.** 앞으로 나오는 자료만 여기 남긴다.

```
results/01_build/                          분절 스팟체크
results/02_model/<run>/                    가중치·history·selection·stage1·pool/·plots/
                      epoch_metrics/       에폭별 지표와 2단계 선정
                      fidelity/            재구성 충실도 진단
results/03_bss/<run>/<split>/              대응표 3종·일치표·그림 (val · test)
results/04_masked_denoising/<run>/<split>/ 전수 지도·기준선·단독·누적·R피크
results/05_validation/<run>/<source>/      외부 적용 (vitaldb·mimic_iv·galaxyppg)
results/05_validation/_check/              원 파형 점검·품질 조사
```

- 실행 이름은 `K<인코더수>_seed<시드>` + 오버라이드 접미사(`_lz0`, `_h128`).
  접미사가 없으면 config 그대로라는 뜻이다.
- 본 노선이 아닌 보조 실험은 `experiments/<이름>/`. `results/`에 섞지 않는다.
- 폐기된 실행·구버전·구코드는 `_work/archive/`로 옮긴다. 지우지 않는다.
- 성분·인코더 표시는 **1부터** (`enc_label`). 내부 인덱스만 0-based.

## 해석 규칙

**산출물만 보고한다. 해석·판독·원고 문구는 붙이지 않는다.** 표와 그림, 수치를 그대로 제시하고
그것이 무엇을 뜻하는지는 사용자가 결정한다.
