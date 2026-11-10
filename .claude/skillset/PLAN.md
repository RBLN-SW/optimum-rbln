# optimum-rbln Claude 스킬셋 구축 계획

> 작성일 2026-10-21. 이 문서는 "어떻게 만들 것인가"의 방법론이며, 스킬셋 자체가 아니다.
> 스킬셋이 완성되면 이 문서는 `.claude/skillset/` 아래 provenance 기록으로만 남는다.

## 0. 핵심 결정 (요약)

| 질문 | 결정 |
|---|---|
| 룰을 몇 개 만들 것인가 | 개수를 미리 정하지 않는다. "서로 다른 PR 2건 이상에서 반복됨 + 코드로 검증됨" 기준으로 수렴시킨다. 상한은 `CLAUDE.md` 300줄(모든 세션이 매번 읽는 비용). 기계적으로 검사 가능한 룰은 산문이 아니라 `tests/`로 내린다. |
| 룰의 출처 | 모든 룰에 provenance(PR 번호·코멘트 인용·인터뷰 응답)를 붙인다. 출처 없는 룰은 `CLAUDE.md`에 들어가지 않는다. |
| 구조 | `rbln_sdk_doc`의 2층 구조(항상 켜진 `CLAUDE.md` + 필요 시 로드되는 `.claude/skills/*`)를 그대로 쓰고, 3층으로 **자동 검사**(`tests/test_conventions.py`)를 둔다. |
| 모델 배분 | 수집·추출은 Haiku, 병합·모순 탐지는 Sonnet, 코드 대조·최종 룰 작성·검사 설계는 Opus/Fable. 상위 모델은 "원문 전체"를 읽지 않고 하위 모델의 구조화된 추출물만 읽는다. |
| 인터뷰 | 무작위 100문답이 아니라, 마이닝 결과에서 **애매하거나 모순되거나 코드와 어긋난 룰만** 질문으로 만든다. 응답자별 담당 영역 라우팅으로 1인당 30~40문항, 20분 이내. 모든 문항에 "모른다 / 기억나지 않는다" 선택지를 둔다. |

## 1. 현재 상태 (2026-10-21 측정)

| 항목 | 값 | 비고 |
|---|---|---|
| 저장소 공개 여부 | PUBLIC | `CLAUDE.md`·스킬 본문이 그대로 배포된다(§11) |
| `CLAUDE.md` / `.claude/` | 없음 | `CONTRIBUTING.md`는 포크·ruff 절차만 다룸. `CODEOWNERS`도 없음 |
| PR 전체 | 804건 | 본문 200자 이상 756건, 본문 합계 약 2.5M자(≈ 70만 토큰) |
| 인라인 리뷰 코멘트 | 1,037건 | 최상위 709 + 답글 328, 약 164K자(≈ 5만 토큰), 2025-02 ~ 2026-10 |
| 대화(issue/PR) 코멘트 | 375건 | 약 527K자 |
| 이슈 | 17건 | 룰 출처로는 비중 낮음 |
| 리뷰어 분포 | jongho 338 · seinpark 199 · thkim 192 · kblee 100 · dkhong 61 · Copilot 43 · eunji 41 · junamsong 31 | 상위 1인 비중 33% → 개인 취향 과적합 주의 |
| 코멘트 집중 경로 | `decoderonly` 161 · `diffusers/autoencoders` 55 · `qwen3_5` 53 · `configuration_utils.py` 50 · `cosmos` 47 | 룰 밀도가 높은 영역 = 스킬 우선순위 |
| 코드 규모 | `RBLN*Config` 클래스 158개, `configuration_*.py` 74개, transformers 모델 디렉터리 55개, `_rbln_submodules` 선언 23곳 | |
| PR 제목 prefix | model 298 · other 228 · dependency 83 · refactor 51 · doc 38 · release 25 · performance 15 | `other`가 2위 → 분류 룰 자체가 약함 |

수집된 원본은 세션 스크래치패드 `corpus/`에 있다(`review_comments.jsonl`, `prs.jsonl`, `issue_comments.jsonl`). 재수집 명령은 §3.1에 있다.

## 2. 산출물 구조

```
optimum-rbln/
├── CLAUDE.md                         항상 켜짐. ≤300줄. 역할 기반 서술, 출처 id 각주.
├── .claude/
│   ├── skills/
│   │   ├── add-model/                새 모델 추가 절차 (configuration_/modeling_/__init__ 등록/테스트/docstring)
│   │   ├── rbln-config-design/       rbln_config 설계·리뷰 (graph-affecting vs runtime, submodule 전파, 우선순위)
│   │   ├── decoderonly-change/       decoderonly 공유 코드 수정 시 영향 범위·회귀 체크
│   │   ├── pr-review/                팀 리뷰 관행 체크리스트 (리뷰 코멘트 마이닝 산출물)
│   │   └── transformers-bump/        transformers 버전 상향 절차 (구조 변경 diff → 영향 모델 목록)
│   └── skillset/
│       ├── PLAN.md                   이 문서
│       ├── rules.yaml                룰 DB (id, 문장, 상태, 증거, 검사 방법, 담당)
│       ├── interview/                문항 생성 스크립트·응답 집계
│       └── corpus/                   수집 스크립트 (원본 데이터는 git-ignore)
└── tests/test_conventions.py         mechanical 룰을 테스트로 고정
```

스킬 5개는 **후보**다. §9.5.2와 같은 기준(실제 PR 2건 이상에서 패턴 반복, 다단계 절차, 다른 PR 독자에게는 소음)을 통과한 것만 만든다. 선제 생성은 하지 않는다.

## 3. 룰 수집 파이프라인

```
A 수집(API, LLM 0)  →  B 추출(Haiku, map)  →  C 병합(Sonnet, reduce)
       →  D 코드 대조(Opus/Fable)  →  E 인터뷰(사람)  →  F 승격(rules.yaml → CLAUDE.md/skill/test)
```

### 3.1 A — 수집

이미 돌려 본 명령이다. 결과는 jsonl 한 줄 = 한 레코드이고, 모든 레코드가 PR 번호를 갖는다(provenance의 기본 키).

```bash
# 인라인 리뷰 코멘트 (약 1,037건)
for p in $(seq 1 16); do gh api "repos/rebellions-sw/optimum-rbln/pulls/comments?per_page=100&page=$p&sort=created&direction=asc" \
  | jq -c '.[] | {pr: (.pull_request_url|split("/")|last|tonumber), id, user: .user.login, path, created: .created_at, in_reply_to: .in_reply_to_id, body}'; done > review_comments.jsonl
# PR 본문 + 메타 (804건)
gh pr list --repo rebellions-sw/optimum-rbln --state all --limit 2000 --json number,title,body,author,mergedAt,baseRefName,labels,files \
  | jq -c '.[] | {number, title, author: .author.login, mergedAt, base: .baseRefName, files: [.files[].path], body}' > prs.jsonl
# 대화 코멘트 (375건)
for p in $(seq 1 12); do gh api "repos/rebellions-sw/optimum-rbln/issues/comments?per_page=100&page=$p&sort=created&direction=asc" \
  | jq -c '.[] | {issue: (.issue_url|split("/")|last|tonumber), user: .user.login, created: .created_at, body}'; done > issue_comments.jsonl
```

추가 소스:

- **Slack** — Slack MCP는 현재 인증이 안 되어 있어 `/mcp`로 먼저 연결해야 한다. 연결 후 `optimum`, `rbln_config`, `submodule`, `decoderonly` 등으로 검색해 스레드 단위로 같은 jsonl 형식에 추가한다(`source: slack`, 키는 스레드 permalink).
- **Notion** — API 연결이 없으므로 페이지를 마크다운으로 내보내 `corpus/notion/*.md`에 둔다. 이번에 받은 요약 페이지(MultiModal refactoring, submodule 논의)는 이미 §6의 인터뷰 문항 소재로 썼다.
- **코드 자체** — D 단계에서 읽는다. 룰 추출 소스가 아니라 검증 소스다.

### 3.2 B — 추출 (Haiku, map)

원문을 읽는 유일한 단계이고, 가장 싼 모델만 쓴다. 입력 청크는 소스별 ≤ 30K 토큰(PR 본문은 25~30개 청크, 리뷰 코멘트는 2~3개 청크면 충분). 출력은 고정 스키마 JSON:

```json
{"candidate": "rbln_config에는 컴파일 그래프를 바꾸는 값만 둔다",
 "evidence": [{"pr": 123, "user": "…", "quote": "원문 그대로 ≤200자"}],
 "area": "rbln_config", "kind": "mechanical|judgment", "polarity": "do|dont|debate"}
```

Haiku 단계의 규칙 세 가지. (1) 인용은 **원문 그대로**만 허용하고 의역은 금지한다. 상위 단계가 `grep`으로 원문을 되찾을 수 있어야 한다. (2) 한 PR에서만 나온 것도 버리지 않고 `low`로 남긴다. 병합 단계에서 다른 소스와 합쳐질 수 있다. (3) "typo", "변수명 틀림" 같은 1회성 내용 수정은 추출하지 않는다.

비용 감각: 전체 코퍼스 약 90만 토큰을 Haiku로 한 번 훑는 데 수 달러 수준이다. 반복해도 된다.

### 3.3 C — 병합 (Sonnet, reduce)

B의 JSON 전체(수백 건)만 읽고 원문은 읽지 않는다. 같은 ask를 하나로 합치고, 서로 다른 PR 수로 confidence를 매기고, **서로 충돌하는 후보 쌍**을 따로 뽑는다. 충돌 쌍과 `low` 후보가 E 단계 인터뷰 문항의 1차 소재다.

리뷰어 편향 보정: confidence는 "서로 다른 리뷰어 수"도 함께 센다. 한 사람이 10번 말한 룰과 세 사람이 각각 한 번씩 말한 룰은 후자가 더 팀 룰에 가깝다.

### 3.4 D — 코드 대조 (Opus/Fable)

여기서 구전 룰이 검증된 룰로 바뀐다. 각 후보에 대해:

1. 룰을 `grep`/AST 질의로 번역한다. 예: "rbln_config에는 graph를 바꾸는 값만" → `RBLN*Config.__init__` 인자 중 `RUNTIME_KEYWORDS`(`create_runtimes`, `device`, `device_map`, `activate_profiler`, `timeout`)·`non_save_attributes`에 없으면서 `input_info`/`compile` 경로에서 읽히지 않는 필드를 찾는다.
2. 준수 수와 위반 수를 센다. 위반 목록이 곧 사용자가 말한 "안 지켜진 코드들"이다.
3. 결과로 룰 상태를 나눈다.
   - 준수 ≫ 위반 → `verified`. 위반 목록은 기술부채 이슈로.
   - 준수 ≈ 위반 → 룰이 틀렸거나 조건이 빠진 것. 인터뷰로.
   - 번역 불가(판단형) → `judgment`로 표시하고 `CLAUDE.md` 산문으로만 간다.
4. 기계적으로 번역된 룰은 그 질의를 그대로 `tests/test_conventions.py`의 테스트로 만든다. **(2단계에서 배운 것)** 대조는 반드시 기본 브랜치 tip에서 하고 `grounding.measured_on`에 ref를 적는다. 머지되지 않은 PR 본문에서 나온 후보는 룰 근거가 아니다. 위반이 남아 있는 동안은 알려진 위반 목록을 allowlist로 들고 있는 ratchet 테스트로 두고, 새 위반만 막는다.

### 3.5 E — 인터뷰 (사람)

§6에서 따로 다룬다. 핵심은 "모든 룰을 묻지 않는다"이다. C·D를 통과해 이미 확실한 룰은 묻지 않고 통보만 한다. 묻는 것은 세 종류다. 충돌 쌍, 코드와 어긋난 룰, 그리고 Notion 논의처럼 결정이 보류된 설계 질문.

### 3.6 F — 승격

`rules.yaml`의 상태 전이:

```
candidate ─(C: ≥2 PR)─▶ supported ─(D: 코드 일치)─▶ verified ─▶ CLAUDE.md / skill / test
    │                        │
    │                        └─(D: 코드 불일치)─▶ interview ─▶ verified | aspirational | rejected
    └─(1 PR, 원칙 서술)─▶ low ─▶ interview
```

`aspirational`은 "팀은 동의하지만 코드가 아직 안 따르는" 룰이다. `CLAUDE.md`에는 들어가되 "현재 N곳 위반, 이슈 #…" 각주를 달고, 위반이 0이 되면 `verified`로 바꾼다.

## 4. 룰 스키마 (`rules.yaml`)

```yaml
- id: CFG-001
  statement: rbln_config의 필드는 컴파일 결과(graph, input_info, compiled_model 수)를 바꾸는 값만 담는다. 런타임 동작만 바꾸는 값은 RUNTIME_KEYWORDS로 분리한다.
  area: rbln_config
  kind: mechanical
  status: candidate          # candidate|supported|verified|aspirational|rejected|interview
  confidence: medium
  evidence:
    - {source: review_comment, pr: 000, user: "…", quote: "…"}
    - {source: notion, ref: "Submodule 논의 2.", quote: "rbln-config runtime config (create_runtimes, device, …)"}
  check: tests/test_conventions.py::test_config_fields_affect_compile
  violations_known: []       # D 단계가 채움
  interview: null            # 문항 id
  owner: null                # 인터뷰에서 "내가 정했다"고 답한 사람. 공개 저장소에는 쓰지 않음.
```

`statement`는 `rbln_sdk_doc` CLAUDE.md §1.1처럼 역할 기반으로 쓴다(클래스 이름 대신 "컴파일 설정 객체"). 구체 이름은 `evidence`와 `check`에만 둔다. 그래야 리팩터링 때 룰 문장이 안 바뀐다.

## 5. 룰은 몇 개가 적당한가

고정 수치 대신 두 가지 신호로 결정한다.

- **수렴 신호.** 소스를 하나씩 추가할 때(리뷰 코멘트 → PR 본문 → 대화 코멘트 → Slack) 새로 생기는 룰 수가 줄어들면 포화다. `rbln_sdk_doc` §9.8.3과 같은 곡선이다. 포화 뒤에 나오는 것은 룰이 아니라 1회성 결정이다.
- **예산 신호.** `CLAUDE.md` 300줄. 한 줄짜리 룰이면 50~60개, 예시가 붙으면 30~40개 선이다. 그 이상은 스킬로 내리거나 테스트로 내린다. 테스트로 내려간 룰은 `CLAUDE.md`에 한 줄 포인터만 남는다.

경험적으로 `rbln_sdk_doc`는 7개 카테고리를 돌아 CLAUDE.md 하나와 스킬 7개로 수렴했다. optimum-rbln은 영역이 좁고(config/모델 추가/decoderonly/diffusers/테스트) 리뷰 코멘트가 1천 건이므로, **1차 마이닝에서 후보 40~70개, 인터뷰·코드 대조 후 verified 25~40개, 스킬 3~5개**가 현실적인 기대치다. 이 숫자는 §10 파일럿 결과로 보정한다.

## 6. 인터뷰 설계

### 6.1 형식

- 매체: 공유 리뷰 페이지(Artifact, 공유 DB). 룰마다 응답 버튼·메모·댓글이 있고 다른 사람의 응답이 막대로 바로 보인다. 응답은 `answers` 컬렉션(문서 id `<룰id>__<사용자id>`), 댓글은 `notes` 컬렉션, 팀원이 직접 제안한 새 룰은 `proposals` 컬렉션("룰 제안" 탭, 카드 id `NEW-<문서id>`)에 쌓이며 `ArtifactData`로 읽어 `rules.yaml`에 집계한다. 소스는 `review/app/`(페이지, 데이터), 데이터 생성기는 `corpus/scripts/make_review_data.py`. 마크다운 시트(`review/*.md`)는 같은 내용의 오프라인 사본이다. 룰 한 줄(≤70자) + 조건 한 줄 + 원문 인용 2개(PR 링크) + 코드 대조 결과 + 응답 체크박스. **룰은 Claude와 작성자가 정하는 것이 아니라 팀이 리뷰해서 정한다**는 전제이므로, 시트는 사람이 30초에 한 룰을 판단할 수 있는 길이를 넘지 않는다.
- 라우팅: `git log --format=%an -- <dir>` 로 영역별 상위 기여자를 뽑아, 각자 담당 영역 문항 + 공통 문항만 보낸다. 1인당 30~40문항, 20분 이내.
- 응답 선택지(모든 확인형 문항 공통):
  - 맞다
  - 틀리다 → 한 줄 이유
  - 조건부로 맞다 → 조건
  - **모른다**
  - **기억나지 않는다** (과거에 정했지만 지금은 모름)
  - 추가: "이 룰을 어기는 코드를 아는가" 자유기술

"모른다"와 "기억나지 않는다"를 나누는 이유: 전자는 룰이 공유 지식이 아니라는 뜻(→ 문서화 필요), 후자는 룰이 있었지만 유실됐다는 뜻(→ 원 결정자를 찾거나 재결정). 집계가 다르다.

### 6.2 문항 유형

| 유형 | 소재 | 예 |
|---|---|---|
| 확인형 | C·D에서 `supported`인 룰 | "rbln_config 필드는 컴파일 결과를 바꾸는 값만 담는다. 맞는가?" |
| 양자택일형 | C에서 나온 충돌 쌍(44개 룰, 같은 질문을 공유하는 15개 묶음은 한 번만 답함) + 미해결 논쟁 27개 | "TP size가 상위에서 전파된 값과 submodule에 명시된 값이 다르면, (a) 명시 우선 (b) 전파 우선 (c) 에러" |
| 코드 지목형 | D에서 나온 위반 사례 | 스니펫을 보여주고 "이 코드는 룰 CFG-001을 위반하는가?" |
| 서술형 | Notion에서 보류된 설계 질문 | "submodule의 통제 가능 깊이를 1로 제한하는 전제는 아직 유효한가?" |

### 6.3 Notion 논의에서 바로 나오는 문항 예시

이번에 받은 Notion 페이지만으로도 다음 문항이 나온다. 모두 "결정됐다고 믿지만 코드에서 확인되지 않은" 종류다.

1. `rbln_config` 필드와 런타임 옵션(`create_runtimes`, `device`, `device_map`, `activate_profiler`, `timeout`)의 분리 기준은 "컴파일 그래프를 바꾸는가"인가?
2. submodule에 명시된 디바이스 수(`num_devices`, 구 `tensor_parallel_size`는 deprecated alias)는 상위에서 전파된 값보다 우선하는가? (코드의 `initialize_submodule_config`는 submodule이 alias로 지정했을 때 상속된 `num_devices`를 떨어뜨리는 분기를 갖고 있다. 이것이 그 결정의 구현인가, 아니면 alias 호환용 임시 처리인가?)
3. `batch_size`는 부모가 자식을 조정하는 방식인가, 자식이 독립 지정 가능한가? 잘못 지정 시 에러인가 무시인가?
4. 통제 가능한 submodule 깊이는 1이라는 전제가 아직 유효한가? (`qwen3_vl`은 `_rbln_submodules`를 두 클래스에 선언한다.)
5. 필드로 가진 모든 `PreTrainedModel`은 submodule이어야 하는가, 최적화 사유로 예외를 둘 수 있는가? 예외를 두는 기준은?
6. 본모듈이 TP 4, submodule이 TP 미지원일 때 `device=[0,1,2,3]`은 어떻게 해석되는가?
7. `export` 인자는 제거 대상인가? (`from_pretrained`에 아직 남아 있다.) 제거 시 deprecation 기간은?
8. `compile_cfgs`의 fusion 제거는 완료된 것으로 보인다(`configuration_utils.py`에 `fusion` 없음). Notion의 "done" 항목처럼 완료된 결정은 룰이 아니라 기록으로만 남기면 되는가?
9. `RBLNPytorchRuntime`은 아직 13개 파일에서 쓰인다. 제거 방향이 유효하다면 새 코드에서 사용 금지인가, 아니면 방향이 철회되었는가?
10. `modeling_generic.py`는 아직 존재하고 11곳에서 참조된다. 모델별 디렉터리로 쪼개는 방향은 확정인가? 새 모델은 어디에 둬야 하는가?
11. Seq2Seq를 decoderonly처럼 configurable 구조로 바꾸는 것은 진행 중인가? 새 Seq2Seq 모델은 어느 구조를 따라야 하는가?
12. PR 제목 type에서 `other`가 228건이다. `fix`, `test`, `ci`를 공식 type으로 추가할 것인가?

### 6.4 집계 규칙

- 응답자 중 "모른다/기억나지 않는다"를 뺀 사람의 80% 이상이 "맞다" → `verified`(코드도 일치할 때) 또는 `aspirational`(코드 불일치).
- 갈리면 → 룰이 아니라 **결정 대기 항목**. `rules.yaml`에 `status: debate`로 두고 ADR 한 장으로 넘긴다. CLAUDE.md에는 "결정 전까지 X 방식을 따른다" 식의 임시 룰만 쓴다.
- "모른다" 비율이 50%를 넘는 룰은 맞더라도 `CLAUDE.md`만으로는 부족하다. 공개 문서(`rbln_sdk_doc`)에도 써야 한다.

## 7. 저가 모델 활용 원칙

| 단계 | 모델 | 읽는 것 | 내는 것 |
|---|---|---|---|
| B 추출 | Haiku | 원문 청크 ≤30K 토큰 | 스키마 JSON, 원문 인용 |
| C 병합 | Sonnet | B의 JSON 전체 | 클러스터, 충돌 쌍, confidence |
| D 코드 대조 | Opus/Fable | C의 클러스터 + 코드 | 상태, 위반 목록, 검사 질의 |
| F 작성 | Opus/Fable | rules.yaml | CLAUDE.md, SKILL.md, 테스트 |

검증 장치 두 가지. (1) Haiku 추출물의 5%를 무작위로 뽑아 상위 모델이 원문과 대조한다. 인용 불일치율이 2%를 넘으면 청크를 줄이거나 프롬프트를 고친다. (2) `CLAUDE.md`에 들어가는 모든 룰은 `evidence.quote`가 코퍼스에서 `grep`으로 찾아져야 한다. 찾아지지 않으면 그 룰은 들어가지 않는다.

실행 방식은 두 가지다. 수동으로 Agent 도구에 `model: haiku`를 줘서 청크별로 돌리거나, Claude Code의 Workflow 도구로 map-reduce를 스크립트화한다. 후자가 재현성이 좋지만 사용자가 세션에서 "use a workflow"로 명시적으로 켜야 한다.

## 8. 검증 루프 (skill-creator)

스킬은 쓰고 끝이 아니라 측정한다. 스킬마다 2~3개의 현실적인 테스트 프롬프트를 만들고, 스킬 있음/없음으로 서브에이전트를 돌려 비교한다.

| 스킬 후보 | 테스트 프롬프트 예 | 기계적 assertion 예 |
|---|---|---|
| `add-model` | "Qwen3-VL과 같은 구조로 새 VLM `foo_vl`을 추가해줘" | `configuration_foo_vl.py`·`modeling_foo_vl.py` 생성, `__init__.py`·`auto` 매핑 등록, `_rbln_submodules` 선언, `tests/test_transformers.py`에 케이스 추가, docstring 검증 스크립트 통과 |
| `rbln-config-design` | "이 PR의 rbln_config 변경을 리뷰해줘" | 런타임 전용 값이 config 필드로 들어갔는지 지적, submodule 전파 누락 지적 |
| `decoderonly-change` | "decoderonly attention에 X를 추가해줘" | 영향받는 모델 목록 산출, `test_llm.py` 회귀 케이스 추가 |

측정 결과는 `skill-creator`의 뷰어로 사람이 보고, 피드백을 받아 스킬을 고친다. 마지막에 description 최적화 루프를 돌려 트리거 정확도를 맞춘다. 스킬 없음 대비 개선이 없는 스킬은 만들지 않는다.

## 9. 단계와 산출물

| 단계 | 기간 | 산출물 | 완료 기준 |
|---|---|---|---|
| 0 코퍼스·파일럿 | 완료(오늘) | `corpus/*.jsonl`, §10 파일럿 | 파이프라인이 실제 룰을 뽑는지 확인 |
| 1 전체 마이닝 | 완료(2026-10-21) | `rules.yaml` v0 — 273 룰(supported 116 · candidate 115 · interview 42), `mining_report.md`, `mining/{procedures,decisions,debates}_v0.md` | 리뷰 코멘트·PR 본문·대화 코멘트 반영. **Slack 미반영**(인증 필요). 수렴 곡선은 `mining_report.md` §2 |
| 2 코드 대조 | mechanical+supported 41개 완료(2026-10-22) | `rules.yaml` 갱신(verified 25 · aspirational 8 · interview 2 · rejected 5 · unverifiable 1), `mining/grounding_report.md`, `mining/violations.md`, 테스트 제안 33건 | **남은 것:** judgment+supported 75개는 코드 대조 대상이 아니라 인터뷰·CLAUDE.md 산문 대상. 1단계 코퍼스에 미머지 PR이 섞여 있어 `mergedAt` 필터로 재추출 필요 |
| 3 인터뷰 | 리뷰 페이지 게시(2026-10-23, https://claude.ai/artifact/2LTDRgqHMaqRub2NoNcUpZ , 공유 설정 필요), 응답 대기 | `review/README.md` + 영역별 12개 시트(룰 158개, 룰당 7~9줄: 룰·조건·근거 인용 2개·코드 대조·질문·응답) + `appendix_candidates.md`(후보 115개 한 줄씩). 생성기 `corpus/scripts/make_review.py` | 응답률 80%, 모든 `interview` 룰이 다른 상태로 이동. 100문답 대신 영역별 10~26문항 |
| 4 v1 작성 | 2~3일 | `CLAUDE.md`, 스킬 2~3개, `tests/test_conventions.py` | `CLAUDE.md` ≤300줄, 모든 룰에 출처 id, 테스트 CI 통과 |
| 5 측정 | 2~3일 | 스킬별 with/without 벤치마크 | 스킬마다 개선 확인, description 최적화 |
| 6 운영 | 상시 | 리뷰 코멘트 → 룰 피드백 | 같은 리뷰 코멘트가 두 PR 연속으로 나오지 않음 |

## 10. 파일럿 결과 (리뷰 코멘트 1,037건, Haiku 1회)

B 단계만 Haiku 에이전트 하나로 리뷰 코멘트 전량에 돌렸다. 소요 약 10분, 38만 토큰. 산출물은 [pilot/rules_from_review_comments.md](pilot/rules_from_review_comments.md)(사람용)와 같은 이름의 `.json`(C 단계 입력용)이다.

| 항목 | 값 |
|---|---|
| 클러스터 | 42개 (high 37 · medium 4 · low 1; high = 서로 다른 PR 3건 이상) |
| 종류 | judgment 31 · mechanical 11 |
| 영역 | style/other 15 · rbln_config 6 · docstring 5 · modeling_base/lifecycle 5 · api/naming 4 · dependency/version 4 · submodule 1 · diffusers 1 · testing 1 |
| 미해결 논쟁 | 19건 |
| 인용 검증 | 무작위 25건 모두 원문과 byte 단위 일치 (§7의 5% 표본 검사 기준 통과) |
| 리뷰어 편향 | jongho가 인용된 클러스터 35/42 → §11의 보정이 필요함을 확인 |

**검증된 것.** 사용자가 예로 든 "rbln_config에는 graph를 바꾸는 config만" 룰은 리뷰 코멘트에서 독립적으로 나왔다(PR 67·475·702, high). 함께 나온 rbln_config 계열 룰 6개는 다음과 같다.

- rbln_config에는 컴파일 그래프를 바꾸는 값만 두고, device처럼 런타임에서만 의미가 있는 값은 runtime 옵션으로 분리한다.
- submodule 설정의 우선순위(submodule에 명시한 값 > 상위 설정 > 기본값)를 한 곳에서 정의한다. (PR 267·433·445 — Notion의 TP 우선순위 논의와 같은 결정)
- 설정 클래스에는 실제 동작에 반영되는 옵션만 노출하고, 무시되는 옵션은 제거하거나 명시적으로 거부한다.
- 기본값과 크기는 하드코딩하지 않고 모델 config·전처리기·기존 rbln_config 속성에서 유도한다.
- 검증 함수는 검증만 하고, 기본값 설정·패딩은 지정된 훅에서 한다.
- 파생 가능한 값은 멤버로 중복 저장하지 않고 property로 계산한다.

**인터뷰로 가야 하는 것.** 19건의 논쟁 중 rbln_config와 직접 닿는 것은 다섯이다. 미지원 dtype을 예외로 막을지 fp32로 자동 승격할지(PR 673), top-level `batch_size`를 submodule로 전파할지(PR 15·445), 충돌 설정을 예외로 막을지 경고 후 사용자 값을 쓸지, 비생성 모델에서 `batch_size`의 의미, dict로 넘긴 rbln_config에 runtime 옵션이 섞였을 때의 처리 범위. 이 다섯은 §6.3의 문항 1·3·6과 합쳐 양자택일형으로 만든다.

**방법론에 대한 교정 두 가지.**

1. high 클러스터 37개 중 31개가 judgment다. 즉 리뷰 코멘트에서 나오는 룰은 대부분 테스트로 내릴 수 없고 `CLAUDE.md` 산문으로 가야 한다. §5의 "테스트로 내려 줄을 아낀다"는 11개에만 적용된다. 300줄 예산 안에서 judgment 룰 30개를 담으려면 룰당 2~3줄이 상한이다.
2. style/other가 15개로 가장 크다("개발 흔적 남기지 않기", "이유 없는 변경에 주석", "미사용 코드 제거" 등). 이것들은 optimum-rbln 고유 룰이 아니라 일반 코드 리뷰 규범이므로, `CLAUDE.md` 본문보다 `pr-review` 스킬의 체크리스트가 맞는 자리다. 리뷰 코멘트 소스는 `pr-review` 스킬을 만들기에는 충분하고, `add-model`·`decoderonly-change` 같은 절차형 스킬의 재료로는 부족하다. 절차형 스킬의 재료는 PR 본문(모델 추가 PR 298건)과 코드에서 나온다. 1단계 전체 마이닝에서 PR 본문은 "절차 추출"을 목표로 별도 프롬프트를 쓴다.

§5의 기대치를 보정하면, 1차 마이닝 후보 60~90개(리뷰 코멘트 42 + PR 본문·대화·Slack에서 추가), verified 30~45개, 그중 `CLAUDE.md` 본문 약 25개, `pr-review` 스킬 체크리스트 약 15개, 테스트 10개 안팎이다.

**1단계 실행 후 재보정(2026-10-21).** 실제 1차 룰은 273개(supported 116)로 기대치의 3배였다. PR 본문이 리뷰 코멘트와 거의 겹치지 않는 종류의 룰(설계 근거·절차·호환성 정책)을 냈기 때문이다. 자세한 집계와 수렴 곡선은 [mining_report.md](mining_report.md). `CLAUDE.md` 300줄 예산은 그대로이므로, D 단계(코드 대조)가 supported 116개 중 verified로 올릴 것과 스킬·테스트로 내릴 것을 가르는 실질적 필터가 된다.

## 10.1 쌍 재판정 (2026-10-23)

리뷰어 지적(CFG-007과 CFG-028은 중복)을 계기로 C-2가 '모순'으로 묶은 16쌍을 근거와 dev 코드로 다시 봤다. 진짜 선택이 필요한 쌍은 2개(LIFE-004/005, SUB-001/CFG-014)뿐이었다. 8쌍은 한 룰의 본칙과 예외라 병합했고, 5쌍은 대상이 달라 분리했으며, 3개 룰은 미머지 PR 근거이거나 대상이 사라져 폐기했다. 결정과 이유는 `mining/pair_resolution.json`.

**원인과 교훈.** C-2 프롬프트가 "모순 5~15개"라는 기대 개수를 제시했고, 모델이 그 개수를 채우려고 적용 범위가 다른 보완 룰을 모순으로 몰았다. 이후 단계의 프롬프트에는 찾을 대상의 **개수를 제시하지 않는다.** 모순 후보는 "같은 상황에서 두 룰이 서로 다른 행동을 요구하는가"로만 판정하고, 상황이 다르면 한 룰의 본칙·예외로 합친다.

## 11. 리스크

- **리뷰어 1인 편향.** 상위 리뷰어가 전체 코멘트의 33%다. confidence에 "서로 다른 리뷰어 수"를 넣고, 인터뷰에서 그 리뷰어를 제외한 동의율도 본다.
- **낡은 룰.** 코멘트는 2025-02부터다. 그 사이 refactor가 많았다(`compile_cfgs` fusion 제거, submodule config 전파 등). D 단계의 코드 대조가 이 문제를 걸러내며, 룰마다 최신 증거 날짜를 기록한다.
- **공개 저장소.** `CLAUDE.md`와 `.claude/skills/`는 저장소와 함께 배포된다. 개인 이름, 내부 링크(Notion, Slack), 내부 저장소 이름은 `rules.yaml`의 `evidence`까지만 두고 `CLAUDE.md`·스킬 본문에는 쓰지 않는다. PR 번호는 같은 저장소 것이므로 괜찮다.
- **Slack 미연결.** 현재 세션에서 Slack MCP는 인증이 필요하다. 1단계 전에 `/mcp`로 연결한다.
- **룰과 결정의 혼동.** "이렇게 하기로 했다"(결정)와 "이렇게 해야 한다"(룰)는 다르다. 결정은 ADR로, 룰은 `CLAUDE.md`로 간다. 인터뷰에서 갈린 항목을 룰로 적지 않는다.
