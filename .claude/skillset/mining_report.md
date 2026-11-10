# 1단계 전체 마이닝 결과 (rules.yaml v0)

작성일 2026-10-21. 파이프라인: 수집(API) → 추출(Haiku ×22 청크) → 영역 병합(Sonnet ×12) → 교차 중복 제거(Sonnet ×1) → 검증·적용(스크립트). 원본은 읽지 않고 상위 단계로 올라갈수록 구조화된 추출물만 읽었다.

## 1. 집계

| 단계 | 입력 | 출력 |
|---|---|---|
| B 추출 (Haiku) | PR 본문 746건(2.0M자, 21청크) + 대화 코멘트 196건(1청크) + 리뷰 코멘트 클러스터 42 | 후보 626, 절차 76, 결정 87, 논쟁 43. 인용 미검증 2건 |
| C-1 영역 병합 (Sonnet) | 후보 626 (12영역) | 348 룰, 드롭 8 |
| C-2 교차 중복 제거 (Sonnet) | 348 | 폴드 44, 룰 아님 31, 모순 16쌍 |
| 최종 | — | **273 룰** (supported 116 · candidate 115 · interview 42) |

영역별 (supported / candidate / interview):

| 영역 | 합계 | supported | candidate | interview |
|---|---|---|---|---|
| pr_process | 41 | 20 | 15 | 6 |
| testing | 32 | 21 | 9 | 2 |
| rbln_config | 31 | 12 | 11 | 8 |
| modeling_base/lifecycle | 29 | 9 | 10 | 10 |
| dependency/version | 28 | 10 | 13 | 5 |
| diffusers | 20 | 5 | 14 | 1 |
| style/other | 20 | 10 | 7 | 3 |
| attention/kvcache | 19 | 4 | 13 | 2 |
| api/naming | 16 | 9 | 5 | 2 |
| multimodal/submodule | 14 | 8 | 4 | 2 |
| docstring/docs | 14 | 5 | 8 | 1 |
| decoderonly | 9 | 3 | 6 | 0 |

kind: mechanical 84 · judgment 189. confidence: high 78 · medium 78 · low 117.

## 2. 수렴 곡선

소스를 리뷰 코멘트 → PR 본문 → 대화 코멘트 순으로 더했을 때 각 소스에서 **처음** 나온 룰 수:

| 소스 | 신규 룰 |
|---|---|
| 리뷰 코멘트 (1,037건) | 42 |
| PR 본문 (746건) | 221 |
| 대화 코멘트 (196건) | 10 |

해석: PR 본문은 리뷰 코멘트와 종류가 다른 룰(설계 근거, 절차, 호환성 정책)을 내므로 아직 포화가 아니다. 대화 코멘트는 10개뿐이라 이 종류의 소스는 포화에 가깝다. Slack은 대화 코멘트와 같은 종류이므로 기대 신규 룰은 10개 안팎이며, 다만 Notion에 있던 보류 결정의 **결론**을 찾는 용도로는 여전히 가치가 있다.

## 3. 계획 §5 기대치 대비

- 기대: 1차 후보 60~90, verified 30~45. 실제: 1차 룰 273, supported 116. 기대치가 3배 낮았다. 원인은 PR 본문을 과소평가한 것이다. 리뷰 코멘트는 '하지 말 것'을, PR 본문은 '이렇게 한다'를 담고 있어 겹치는 부분이 적었다.
- `CLAUDE.md` 300줄 예산에 supported 116개 전부를 넣을 수는 없다. 다음 단계(D 코드 대조)에서 supported 중 **코드가 실제로 따르는 것**만 verified로 올리고, 영역별 스킬(`rbln-config-design`, `pr-review`, `add-model`)로 내릴 것을 가른다. mechanical+supported 41개는 테스트 후보다.
- pr_process가 41개로 가장 크다. 이 영역은 코드 룰이 아니라 PR 작성·CI·브랜치 운영 룰이므로 `CLAUDE.md` 본문이 아니라 `pr-review` 스킬과 PR 템플릿 보강의 재료다.

## 4. 인터뷰로 가는 룰 (status: interview)

42개. 서로 반대 방향의 증거가 있거나(예: 미지원 dtype을 예외로 막을지 fp32로 승격할지), 같은 주제에 두 룰이 공존한다(예: import 상단 고정 vs 지연 import 허용). 전체 목록은 `rules.yaml`에서 `status: interview`로 찾고, 질문 문장은 각 룰의 `open_question`에 있다. 모순 쌍은 `contradicts` 필드로 서로를 가리킨다.

## 5. 주의할 점

- **단일 리뷰어 근거.** supported인데 근거 사용자가 1명뿐인 룰이 30개다. 인터뷰 집계에서 그 리뷰어를 제외한 동의율을 따로 본다.
- **제목·경로만 근거.** PR 제목이나 파일 경로만 인용한 룰이 3개다(`note: title/path-only evidence`, confidence low). D 단계에서 코드가 뒷받침하지 않으면 버린다.
- **룰 아님으로 버린 31개**는 `mining/dropped_not_a_rule.yaml`에 사유와 함께 남겼다. 교차 병합의 판단이 틀렸을 수 있으니 한 번 훑어볼 가치가 있다.
- **브랜치 운영 룰은 낡았다.** PR 727(2026-09) 본문대로 원격에는 `dev` 하나만 있고 `main`은 삭제됐다(`git ls-remote --heads` 확인, 기본 브랜치 dev). 이 워크트리의 `origin/main`은 낡은 로컬 추적 ref였다. 따라서 dev→main 릴리스·hotfix 흐름을 전제한 PRC-005/023/024/025는 2단계에서 현재 릴리스 방식으로 재검토한다.
- **ID는 영구 provenance 키**다. 병합으로 사라진 44개 id는 살아남은 룰의 `folded_ids`에 남아 있다. 번호를 다시 매기지 않는다.

## 6. 코드 대조 우선순위 (D 단계 입력)

mechanical이면서 supported인 룰부터 코드 질의로 번역한다. 상위 15개:

| id | PRs | users | 룰 |
|---|---|---|---|
| DEP-001 | 6 | 5 | deprecated 인자·API는 deprecate 데코레이터로 표시하고 제거 예정 버전을 명시하며, 해당 버전에 도달하면 제거한다. 이전 메이저 버전 전용 기능도 같은 방식으로 deprecate  |
| PRC-008 | 6 | 5 | 선행 PR이나 다른 저장소의 PR에 의존하는 PR은 base 브랜치를 선행 PR 브랜치로 두고, 본문에 선행·관련 PR 번호와 링크, 머지 순서, 순서가 어긋났을 때의 대응(닫기 등)을 명시한다. |
| LIFE-002 | 6 | 4 | 모델 forward의 입력 인자·순서·반환 타입은 원본 HF 모델 구현과 일치시키고(예: token_type_ids 누락 금지), 사용자가 지정한 입력 이름이 forward 시그니처에 없으면 에러를 |
| STY-018 | 6 | 3 | 머지 전에 PR 변경 코드 전체에 `ruff format`과 `ruff check`를 통과시키고 타입 검사와 독스트링 검사도 통과시킨다. |
| API-003 | 6 | 2 | API·설정 구조·공통 런타임 인터페이스·모델 출력 타입을 제거·교체·변경할 때는 라이브러리 내부 호출부, 하위 클래스와 멀티모달 래퍼, vLLM 등 소비 측 코드, README·문서 예제, mod |
| CFG-010 | 5 | 4 | 설정 키나 장치 지원을 폐지하거나 이름을 바꿀 때는 deprecate 데코레이터로 경고 후 옛 키를 새 키에 재매핑해 기존 동작을 유지하고(생성자 인자와 setattr 경로 포함), 구현 제거는 이 |
| PRC-003 | 5 | 4 | PR 본문은 템플릿의 절(변경 유형, 변경 개요, Motivation, Verification, 관련 이슈)을 갖추고, Type of Change에서 해당 유형을 하나 체크하며(Other는 내용을  |
| DEP-012 | 4 | 4 | PR CI가 쓰는 컴파일러 버전은 .github/version.yaml 한 곳에서 고정하고 모든 CI가 그 값을 읽게 하며, 컴파일러 업데이트가 필요하면 그 파일에 반영하고 pytest에서 쓰는 배 |
| PRC-001 | 4 | 4 | PR 제목은 `type(optional scope): description` 형식의 conventional commit을 따르고, type은 PR 템플릿이 정의한 목록(model, performan |
| DOC-011 | 4 | 2 | Google 스타일 docstring에서 파라미터 선언(이름과 괄호 속 타입)과 Returns 설명은 각각 한 줄에 쓰고, 파라미터 이름은 첫 `:` 앞에서 끝나게 하며, 기본값은 "Defaults |
| PRC-005 | 4 | 2 | 릴리스 PR은 dev를 main으로 병합하는 PR로 만들고, base를 main으로 두며, 제목을 `release: vX.Y.Z` 형식으로 쓰고 Type of Change에서 Release를 체크한 |
| STY-011 | 4 | 2 | 함수 시그니처에는 실제로 전달되는 객체와 일치하는 타입 힌트를 명시하고, Python 3.10 이상의 네이티브 문법(`X | None`, `list`)을 쓰며 typing의 Optional·List |
| STY-013 | 4 | 2 | 로그와 경고는 built-in logging/warnings 대신 optimum-rbln의 `utils.logging`을 사용한다. |
| DEP-033 | 4 | 1 | 의존성 봇이 만든 업데이트 PR은 자동 병합하지 않고 검토 후 사람이 직접 병합한다. |
| STY-004 | 4 | 1 | 사용하지 않는 코드(미사용 변수·import·메서드·클래스, 부모와 동일한 override, 레거시 코드, 효과 없는 컴파일 옵션, 오래되었거나 중복된 CI 워크플로)는 남기지 않고 제거한다. |

## 7. 파일

- `rules.yaml` — 룰 DB v0 (이 보고서의 대상)
- `mining/stats.json` — 집계 원본
- `mining/dedupe.json` — 교차 병합 결정 (폴드·재배치·드롭·모순)
- `mining/dropped_not_a_rule.yaml` — 버린 후보
- `mining/procedures_v0.md`, `decisions_v0.md`, `debates_v0.md` — 절차·결정·논쟁 (보조 병합)
- `corpus/scripts/` — 재실행용 스크립트와 프롬프트. 원본 데이터는 git-ignore.
