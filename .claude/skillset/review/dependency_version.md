# 의존성 · 버전 — 룰 13개

결정 필요 0 · 코드 일치 6 · 코드 불일치 1 · 판단형 4 · 폐기 제안 2. 순서는 결정이 필요한 것부터. 각 룰의 **응답** 줄에서 하나를 `[x]`로 표시하고, 틀리거나 조건부면 **메모**에 한 줄 적어 주세요.

## 코드 불일치(의도는 분명)

### DEP-001 · 코드 불일치(의도는 분명)

**룰** 공개 kwarg와 메서드의 deprecation은 데코레이터와 version으로 표시한다.
**조건** `deprecate_kwarg`/`deprecate_method`를 쓰고, 불가하면 경고에 제거 예정 버전을 명시하며 도달 시 릴리스 전에 제거하거나 별도 PR로 연기한다.
**근거** [#495](https://github.com/rebellions-sw/optimum-rbln/pull/495) (jongho): “This decorator was present in the dev branch (lines 314 and 986) and should be preserved to maintain the deprecation warning for…”
　　　[#481](https://github.com/rebellions-sw/optimum-rbln/pull/481) (dkhong): “Don't we have to guide the last version where this method will be supported?”
　　　외 PR 4건
**코드 대조** 코드 불일치(의도는 분명) — 준수 7 / 위반 8. dev tip(74e97fa, 버전 0.12.1.dev)에서 데코레이터 사용 7곳은 모두 제거 버전 0.13.0 > 현재로 정상이지만, 데코레이터 없이 버전도 없는 ad-hoc deprecation 경고가 8곳(53%) 남아 있다(worktree HEAD v0.11.1 기준 3 대 7…
**응답** [ ] 맞다 　[ ] 틀리다 　[ ] 조건부로 맞다 　[ ] 필요 없는 룰이다 　[ ] 모른다 　[ ] 기억나지 않는다 　· 메모:

## 코드 일치

### DEP-012 · 코드 일치

**룰** CI가 설치하는 rebel-compiler 버전은 `.github/version.yaml` 한 곳에서만 정한다.
**조건** CI 스크립트는 이 값을 읽거나 `REBEL_COMPILER_VERSION`으로만 바꾸고 갱신은 봇 PR이나 이 파일 수정으로만 한다.
**근거** [#54](https://github.com/rebellions-sw/optimum-rbln/pull/54) (jongho): “컴파일러 업데이트 필요하면 그것도 version.yml에 반영해주세요”
　　　[#310](https://github.com/rebellions-sw/optimum-rbln/pull/310) (kblee): “need to check deploy compiler version which uses in pytest”
　　　외 PR 2건
**코드 대조** 코드 일치 — 준수 3 / 위반 0
**응답** [ ] 맞다 　[ ] 틀리다 　[ ] 조건부로 맞다 　[ ] 필요 없는 룰이다 　[ ] 모른다 　[ ] 기억나지 않는다 　· 메모:

### DEP-033 · 코드 일치

**룰** 의존성 갱신 봇이 만든 `dependency:` PR은 auto-merge를 켜지 않는다.
**조건** `[pytest-full]` 결과를 확인한 팀원이 직접 병합한다.
**근거** [#306](https://github.com/rebellions-sw/optimum-rbln/pull/306) (thkim): “**Automerge**: Disabled by config. Please merge this manually once you are satisfied.”
　　　[#372](https://github.com/rebellions-sw/optimum-rbln/pull/372) (thkim): “Please merge this manually once you are satisfied.”
　　　외 PR 2건
**코드 대조** 코드 일치 — 준수 42 / 위반 0
**응답** [ ] 맞다 　[ ] 틀리다 　[ ] 조건부로 맞다 　[ ] 필요 없는 룰이다 　[ ] 모른다 　[ ] 기억나지 않는다 　· 메모:

### DEP-015 · 코드 일치

**룰** 외부 라이브러리가 deprecated로 표시한 속성, 호출 방식, import 경로는 권장 대체 방식으로 옮긴다.
**근거** [#186](https://github.com/rebellions-sw/optimum-rbln/pull/186) (dkhong): “Because importing 'ControlNetOutput' from 'diffusers.models.controlnet' is deprecated after 'diffusers==0.34.0', it is changed li…”
　　　[#346](https://github.com/rebellions-sw/optimum-rbln/pull/346) (thkim): “Since passing the github_token directly as an argument to GitHub is now deprecated, we are modifying this.”
　　　외 PR 1건
**코드 대조** 코드 일치 — 준수 37 / 위반 0
**응답** [ ] 맞다 　[ ] 틀리다 　[ ] 조건부로 맞다 　[ ] 필요 없는 룰이다 　[ ] 모른다 　[ ] 기억나지 않는다 　· 메모:

### DEP-019 · 코드 일치

**룰** 컴파일러 기능이 있는지는 버전 번호가 아니라 그 함수·인자가 실제로 있는지로 확인한다.
**조건** 옛 컴파일러에도 같은 결과를 내는 방법이 있으면 그걸로 대신하고, 없으면 RuntimeError로 업그레이드를 안내한다.
**근거** [#597](https://github.com/rebellions-sw/optimum-rbln/pull/597) (jongho): “This avoids version-string comparison, so it works against both new and pre-rename compilers”
　　　[#708](https://github.com/rebellions-sw/optimum-rbln/pull/708) (jongho): “The helpers look the API up with 'getattr' and, when it is absent, take a per-model sum that is byte-for-byte what the code does…”
　　　외 PR 1건
**코드 대조** 코드 일치 — 준수 None / 위반 None
**응답** [ ] 맞다 　[ ] 틀리다 　[ ] 조건부로 맞다 　[ ] 필요 없는 룰이다 　[ ] 모른다 　[ ] 기억나지 않는다 　· 메모:

### DEP-008 · 코드 일치

**룰** 모델 구현에 필수인 외부 패키지는 `pyproject.toml` 기본 dependencies에 버전과 함께 선언한다.
**조건** 특정 파이프라인 전용 비필수 패키지는 `is_*_available()` 가드 아래에서 import하고 미설치 시 대체 클래스로 실패를 늦춘다.
**근거** [#533](https://github.com/rebellions-sw/optimum-rbln/pull/533) (ChanyongJeon): “**Dependencies** ('pyproject.toml'): 'einops', 'trimesh', 'scikit-image', 'omegaconf'”
　　　[#168](https://github.com/rebellions-sw/optimum-rbln/pull/168) (jongho): “cosmos guardrail is not a necessary dependency”
**코드 대조** 코드 일치 — 준수 10 / 위반 3
**응답** [ ] 맞다 　[ ] 틀리다 　[ ] 조건부로 맞다 　[ ] 필요 없는 룰이다 　[ ] 모른다 　[ ] 기억나지 않는다 　· 메모:

### DEP-011 · 코드 일치

**룰** `torch.load`는 항상 `weights_only`를 명시한다.
**조건** 라이브러리가 직접 저장한 파일은 `False`, 사용자·외부 체크포인트(LoRA 등)는 `True`다.
**근거** [#106](https://github.com/rebellions-sw/optimum-rbln/pull/106) (junamsong): “partial(torch.load, weights_only=True, map_location=torch.device("cpu"))”
　　　[#15](https://github.com/rebellions-sw/optimum-rbln/pull/15) (jongho): “pytorch 2.6부터 torch.load의 default behavior가 바뀌어서 아마 에러가 날 거 같습니다”
**코드 대조** 코드 일치 — 준수 25 / 위반 2
**응답** [ ] 맞다 　[ ] 틀리다 　[ ] 조건부로 맞다 　[ ] 필요 없는 룰이다 　[ ] 모른다 　[ ] 기억나지 않는다 　· 메모:

## 판단형

### DEP-002 · 판단형

**룰** 의존 라이브러리 버전에 따라 달라지는 동작은 검증한 버전을 명시한다.
**조건** 차이가 나면 버전 분기나 주석으로 남긴다.
**근거** [#134](https://github.com/rebellions-sw/optimum-rbln/pull/134) (seinpark): “From 'transformers==4.51.0', they don't use 'return_dict=False' to input argument.”
　　　[#97](https://github.com/rebellions-sw/optimum-rbln/pull/97) (junamsong): “we've already verified that models other than Whisper are functioning correctly in transformers 4.50.3.”
　　　외 PR 2건
**응답** [ ] 맞다 　[ ] 틀리다 　[ ] 조건부로 맞다 　[ ] 필요 없는 룰이다 　[ ] 모른다 　[ ] 기억나지 않는다 　· 메모:

### DEP-007 · 판단형

**룰** 의존성 버전은 model-zoo 등으로 검증한 뒤에 상향한다.
**조건** 상향 전 기능 점검과 wheel 가용성 확인을 하고 상한은 최신 출시에 맞춰 갱신하며 불필요해진 의존성은 제거한다.
**근거** [#8](https://github.com/rebellions-sw/optimum-rbln/pull/8) (jongho): “it would be nice to go with model-zoo validation. I'll revert torch version”
　　　[#671](https://github.com/rebellions-sw/optimum-rbln/pull/671) (thkim): “Dependency readiness (verified against the indexes)”
　　　외 PR 1건
**응답** [ ] 맞다 　[ ] 틀리다 　[ ] 조건부로 맞다 　[ ] 필요 없는 룰이다 　[ ] 모른다 　[ ] 기억나지 않는다 　· 메모:

### DEP-016 · 판단형

**룰** torch 메이저/마이너 bump PR은 `[pytest-full]`로 전체 컴파일 테스트를 돌린다.
**조건** 혼합 dtype 행렬곱 실패는 attention 확률·마스크 등을 query/value dtype으로 명시 `.to(dtype)` 캐스팅해 고친다.
**근거** [#774](https://github.com/rebellions-sw/optimum-rbln/pull/774) (thkim): “torch 2.13 (bumped in #769) no longer accepts mixed-dtype matmul, so Pixtral compilation fails.”
　　　[#773](https://github.com/rebellions-sw/optimum-rbln/pull/773) (thkim): “torch 2.13 (commit 23a918cd0d68) rejects mixed-dtype matmul”
**코드 대조** unverifiable — 준수 10 / 위반 0
**응답** [ ] 맞다 　[ ] 틀리다 　[ ] 조건부로 맞다 　[ ] 필요 없는 룰이다 　[ ] 모른다 　[ ] 기억나지 않는다 　· 메모:

### DEP-017 · 판단형

**룰** torch 버전 상향은 같은 torch를 검증한 컴파일러 빌드와 함께 또는 먼저 반영한다.
**조건** torch에 의존하는 패키지(vllm-rbln, torch-rbln)도 같은 PR에서 맞춘다.
**근거** [#769](https://github.com/rebellions-sw/optimum-rbln/pull/769) (thkim): “a compiler build validated on torch 2.13 should land alongside (or before) this PR.”
　　　[#525](https://github.com/rebellions-sw/optimum-rbln/pull/525) (kblee): “update torch version & dependent with cpu”
**응답** [ ] 맞다 　[ ] 틀리다 　[ ] 조건부로 맞다 　[ ] 필요 없는 룰이다 　[ ] 모른다 　[ ] 기억나지 않는다 　· 메모:

## 폐기 제안

### DEP-005 · 폐기 제안

**룰** 지원하는 transformers 버전은 범위 선언과 단일 정확 버전(`==`) 중 하나의 방식으로 통일한다.
**근거** [#564](https://github.com/rebellions-sw/optimum-rbln/pull/564) (jongho): “so we need to operate under both rather than force-pin v4 forever.”
　　　[#301](https://github.com/rebellions-sw/optimum-rbln/pull/301) (thkim): “| project.dependencies | minor | '==4.51.3' -> '==4.53.0' |”
**응답** [ ] 맞다 　[ ] 틀리다 　[ ] 조건부로 맞다 　[ ] 필요 없는 룰이다 　[ ] 모른다 　[ ] 기억나지 않는다 　· 메모:

### DEP-009 · 폐기 제안

**룰** 모델 구현의 무거운 선택적 의존성은 함수 안이 아니라 모듈 최상단에서 import한다.
**근거** [#533](https://github.com/rebellions-sw/optimum-rbln/pull/533) (ChanyongJeon): “top-level 'einops' / 'trimesh' / 'scikit-image'”
**응답** [ ] 맞다 　[ ] 틀리다 　[ ] 조건부로 맞다 　[ ] 필요 없는 룰이다 　[ ] 모른다 　[ ] 기억나지 않는다 　· 메모:
