# docs/ — 연구 대시보드 (GitHub Pages)

`index.html` 하나가 저장소의 `results/` 를 직접 읽어 표·그림을 그린다. 실험을 다시
내고 push 하면 대시보드도 바뀐다 — 손으로 옮겨 적는 값이 없다.

| 파일 | 역할 | 누가 고치나 |
|---|---|---|
| `index.html` | 페이지 (CSS·JS 인라인) | 구조를 바꿀 때만 |
| `data/site.json` | 제목·버전·run 목록·그림 목록·원고 링크·참고문헌·확정/미결/이력 | **진행할 때마다** |
| `data/snapshot.json` | results CSV 스냅샷 — 저장소를 못 읽는 환경(오프라인) 대비 | `python docs/snapshot.py` |

읽는 순서: `../results/…`(같은 저장소에서 서빙될 때) → GitHub raw → 스냅샷.

## 켜는 법

1. `git add docs && git commit && git push origin <브랜치>`
2. GitHub 저장소 → Settings → Pages → Source: **Deploy from a branch**, Branch: `<브랜치>` / `/docs`
3. `https://<owner>.github.io/MEAE_XAI/` — `site.json` 의 `repo.branch` 를 실제 브랜치 이름으로 맞춘다

로컬에서 볼 때: `python -m http.server -d docs 8000` → `http://localhost:8000` (file:// 로 직접 열면 브라우저가 fetch 를 막는다)

## 새 run 을 올릴 때

`site.json` 의 `runs` 에 항목 하나 추가 — `id`(results 폴더 이름) · `label` · `kind`(meae/deepfilter/descod) · `row`(three_ways.csv 의 방식 이름) · `status`. 표·그래프·검산은 자동이다.
