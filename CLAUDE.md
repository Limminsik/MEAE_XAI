# meae_xai

모든 구현은 RESEARCH_DESIGN.md를 따른다. 설계와 다른 구현이 필요하면 진행하지 말고 사용자에게 먼저 보고한다.

- §0 불변 원칙(마스킹 대상은 인코딩, 자기지도 학습, 정답은 채점표, 기록 단위 분할)은 위반 금지.
- 태스크 순서는 §12를 따르며 ★ 표시 지점에서는 사용자 확인 없이 다음 단계로 진행하지 않는다.
- 설정은 configs/default.yaml 에만. 코드 하드코딩 금지.
- 환경: uv + Python 3.9.21 + requirements.txt, torch는 cu129 인덱스로 설치.

## 문서 규칙

문서는 **두 개뿐**이다. 새로 만들지 않는다.
- `README.md` — 연구 현황판. 현황·산출물 지도·실험 대장·핵심 수치·결정 이력·미결·원고 대기.
  단계를 마치거나 결정이 확정되면 **여기를 갱신**한다.
- `RESEARCH_DESIGN.md` — 설계서. 구현은 이 문서를 따르며, 어긋나면 문서를 고친다.

## 산출물 규칙

- **본 실험 산출물은 `results/` 에만** 둔다.
  `01_train/<run>/`(가중치·history.csv·selection.json·console.log·pool/·plots/) ·
  `02_separation/`(S4) · `03_denoising/`(S5) · `04_external/`(S6) · `00_rehearsal/`(봉인 전 val 리허설)
- 본 노선이 아닌 보조 실험은 `experiments/<이름>/outputs/`. `results/`에 섞지 않는다.
- 폐기된 실행·구버전은 `_work/archive/`로 옮긴다. 지우지 않는다.
- 실행 이름은 `K<인코더수>_seed<시드>` + 오버라이드 접미사(`_lz0`, `_h128`).
  접미사가 없으면 config 그대로라는 뜻이다.
- 성분·인코더 표시는 **1부터** (`enc_label`). 내부 인덱스만 0-based.
- 콘솔 출력에 한글이 있으면 `PYTHONIOENCODING=utf-8` 로 실행한다 (Windows cp949 인코딩 오류).

## 해석 규칙

**산출물만 보고한다. 해석·판독·원고 문구는 붙이지 않는다.** 표와 그림, 수치를 그대로 제시하고
그것이 무엇을 뜻하는지는 사용자가 결정한다.
