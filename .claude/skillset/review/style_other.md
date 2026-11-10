# 코드 스타일 · 기타 — 룰 13개

결정 필요 1 · 코드 일치 6 · 코드 불일치 1 · 판단형 5 · 폐기 제안 0. 순서는 결정이 필요한 것부터. 각 룰의 **응답** 줄에서 하나를 `[x]`로 표시하고, 틀리거나 조건부면 **메모**에 한 줄 적어 주세요.

## 결정 필요

### STY-003 · 결정 필요

**룰** 같은 로직, 상수, 결정 지점은 한 곳에만 두고 복제된 것은 공용 헬퍼나 단일 상수로 모은다.
**근거** [#228](https://github.com/rebellions-sw/optimum-rbln/pull/228) (jongho): “set_compile_cfgs is the one and only point of determining npu.”
　　　[#673](https://github.com/rebellions-sw/optimum-rbln/pull/673) (jongho): “the same function is on multiple sites.”
　　　외 PR 1건
**질문** 파이프라인 공통 검증 코드는 mixin에 두는가, utils 모듈에 두는가?
**응답** [ ] 맞다 　[ ] 틀리다 　[ ] 조건부로 맞다 　[ ] 필요 없는 룰이다 　[ ] 모른다 　[ ] 기억나지 않는다 　· 메모:

## 코드 불일치(의도는 분명)

### STY-005 · 코드 불일치(의도는 분명)

**룰** `src/optimum/rbln` 아래 새 파일은 Rebellions 저작권 한 줄과 Apache-2.0 블록을 붙인다.
**조건** HF 코드를 옮긴 파일은 원 저작권 헤더를 유지하고 tests/는 헤더를 붙이지 않는 것이 관행이다.
**근거** [#471](https://github.com/rebellions-sw/optimum-rbln/pull/471) (jongho): “# Copyright 2026 Rebellions Inc. All rights reserved.”
　　　[#798](https://github.com/rebellions-sw/optimum-rbln/pull/798) (seinpark): “# Copyright 2026 Rebellions Inc. All rights reserved.”
　　　외 PR 1건
**코드 대조** 코드 불일치(의도는 분명) — 준수 290 / 위반 47. dev 최신 src의 Rebellions 소유 파일 337개 중 290개(86%)만 표준 헤더와 맞는 연도를 갖고, 헤더 없음 11·형식 불일치 20·연도 불일치 16건이 있다.
**응답** [ ] 맞다 　[ ] 틀리다 　[ ] 조건부로 맞다 　[ ] 필요 없는 룰이다 　[ ] 모른다 　[ ] 기억나지 않는다 　· 메모:

## 코드 일치

### STY-018 · 코드 일치

**룰** 머지 전에 CI 품질 게이트(ruff, mypy, `check_inits.py`, docstring 검사)를 통과시킨다.
**조건** base가 바뀌었으면(특히 lint 규칙 변경 PR 직후) 최신 base에서 CI를 다시 돌린 뒤 머지한다.
**근거** [#808](https://github.com/rebellions-sw/optimum-rbln/pull/808) (jongho): “'scripts/check_inits.py' and the mkdocstrings docstring check”
　　　[#481](https://github.com/rebellions-sw/optimum-rbln/pull/481) (seinpark): “Resolve below warnings when building sdk docs”
　　　외 PR 4건
**코드 대조** 코드 일치 — 준수 627 / 위반 4
**응답** [ ] 맞다 　[ ] 틀리다 　[ ] 조건부로 맞다 　[ ] 필요 없는 룰이다 　[ ] 모른다 　[ ] 기억나지 않는다 　· 메모:

### STY-004 · 코드 일치

**룰** 미사용 import와 변수는 ruff가, 부모와 동일한 override와 미참조 private 헬퍼는 리뷰가 잡는다.
**조건** docstring만 덧붙이는 pass-through override는 문서화 목적이므로 허용한다.
**근거** [#271](https://github.com/rebellions-sw/optimum-rbln/pull/271) (jongho): “there are much old codes that are not needed anymore.”
　　　[#761](https://github.com/rebellions-sw/optimum-rbln/pull/761) (jongho): “was it just dead code?”
　　　외 PR 2건
**코드 대조** 코드 일치 — 준수 2154 / 위반 2
**응답** [ ] 맞다 　[ ] 틀리다 　[ ] 조건부로 맞다 　[ ] 필요 없는 룰이다 　[ ] 모른다 　[ ] 기억나지 않는다 　· 메모:

### STY-011 · 코드 일치

**룰** annotation은 `X | None`, `list[...]` 등 3.10+ 문법으로 쓴다.
**조건** `None` 기본값은 `| None`을 명시하며, 문자열 forward-ref 인자는 예외로 통일 방식이 미정이다.
**근거** [#267](https://github.com/rebellions-sw/optimum-rbln/pull/267) (thkim): “Type 'language_projection' is 'REBEL.runtime' rather than 'RBLNModel'.”
　　　[#736](https://github.com/rebellions-sw/optimum-rbln/pull/736) (jongho): “240+ implicit-Optional parameters ('x: Tensor = None') are annotated 'Tensor | None'”
　　　외 PR 2건
**코드 대조** 코드 일치 — 준수 1705 / 위반 4
**응답** [ ] 맞다 　[ ] 틀리다 　[ ] 조건부로 맞다 　[ ] 필요 없는 룰이다 　[ ] 모른다 　[ ] 기억나지 않는다 　· 메모:

### STY-013 · 코드 일치

**룰** 로거는 항상 `utils.logging`의 `get_logger`로 만들고 `logging.getLogger`를 쓰지 않는다.
**조건** 레벨 상수 참조는 허용하며 사용자 경고도 `logger.warning`/`warning_once`로 낸다.
**근거** [#15](https://github.com/rebellions-sw/optimum-rbln/pull/15) (jongho): “built-in logging 대신 optimum-rbln 의 logging을 사용해야 합니다.”
　　　[#38](https://github.com/rebellions-sw/optimum-rbln/pull/38) (jongho): “native logging 대신 library에 있는 utils.logging 사용해주세요”
　　　외 PR 2건
**코드 대조** 코드 일치 — 준수 56 / 위반 6
**응답** [ ] 맞다 　[ ] 틀리다 　[ ] 조건부로 맞다 　[ ] 필요 없는 룰이다 　[ ] 모른다 　[ ] 기억나지 않는다 　· 메모:

### STY-010 · 코드 일치

**룰** annotation 전용 import를 `TYPE_CHECKING`으로 옮길 때는 annotation을 문자열로 감싼다.
**조건** 같은 이름을 모듈 수준 런타임 import와 중복하지 않으며 ruff는 TC004/TC005만 켠다.
**근거** [#733](https://github.com/rebellions-sw/optimum-rbln/pull/733) (jongho): “Twelve modules imported the same names both at runtime and again inside a 'TYPE_CHECKING' block”
　　　[#556](https://github.com/rebellions-sw/optimum-rbln/pull/556) (seinpark): “move the import under 'TYPE_CHECKING' and quote the annotation strings.”
　　　외 PR 1건
**코드 대조** 코드 일치 — 준수 103 / 위반 1
**응답** [ ] 맞다 　[ ] 틀리다 　[ ] 조건부로 맞다 　[ ] 필요 없는 룰이다 　[ ] 모른다 　[ ] 기억나지 않는다 　· 메모:

### STY-012 · 코드 일치

**룰** export·등록 목록(`__all__`, re-export, 모델/설정 등록 목록)은 정렬 순서를 유지한다.
**조건** 로그 레벨, 단위, 버전 호환표처럼 순서가 의미인 상수 표는 예외다.
**근거** [#631](https://github.com/rebellions-sw/optimum-rbln/pull/631) (jongho): “please keep the list sorted”
　　　[#50](https://github.com/rebellions-sw/optimum-rbln/pull/50) (kblee): “사소한거지만 e,f,g 순서...?”
　　　외 PR 1건
**코드 대조** 코드 일치 — 준수 264 / 위반 0
**응답** [ ] 맞다 　[ ] 틀리다 　[ ] 조건부로 맞다 　[ ] 필요 없는 룰이다 　[ ] 모른다 　[ ] 기억나지 않는다 　· 메모:

## 판단형

### STY-001 · 판단형

**룰** 의도가 바로 드러나지 않는 추가, 삭제, 순서 변경, dtype 변경에는 그 이유를 주석이나 PR 설명에 남긴다.
**근거** [#135](https://github.com/rebellions-sw/optimum-rbln/pull/135) (jongho): “Could you add a comment that describes why we remove inputs_embeds”
　　　[#84](https://github.com/rebellions-sw/optimum-rbln/pull/84) (kblee): “How was this number determined?”
　　　외 PR 4건
**응답** [ ] 맞다 　[ ] 틀리다 　[ ] 조건부로 맞다 　[ ] 필요 없는 룰이다 　[ ] 모른다 　[ ] 기억나지 않는다 　· 메모:

### STY-007 · 판단형

**룰** 오류 메시지는 무엇이 잘못됐는지, 어떤 조건인지, 사용자가 무엇을 해야 하는지를 구체적으로 적는다.
**조건** 실제로 일어나지 않는 동작은 설명하지 않고 같은 목적의 기존 가드 메시지 양식을 따른다.
**근거** [#345](https://github.com/rebellions-sw/optimum-rbln/pull/345) (jongho): “Error message should be more concrete.”
　　　[#809](https://github.com/rebellions-sw/optimum-rbln/pull/809) (dkhong): “The previous error did not clarify whether its patch count was before or after the spatial merge.”
　　　외 PR 3건
**응답** [ ] 맞다 　[ ] 틀리다 　[ ] 조건부로 맞다 　[ ] 필요 없는 룰이다 　[ ] 모른다 　[ ] 기억나지 않는다 　· 메모:

### STY-009 · 판단형

**룰** import는 파일 맨 위에 둔다. 함수 안 import는 순환 import를 피하거나 무거운 선택 의존성을 미룰 때만 쓴다.
**조건** ruff의 함수 안 import 금지 규칙(PLC0415)은 이 예외 때문에 꺼 두었으므로 리뷰에서 이유를 확인한다.
**근거** [#733](https://github.com/rebellions-sw/optimum-rbln/pull/733) (jongho): “would forbid the lazy imports that keep 'import optimum.rbln' light.”
　　　[#267](https://github.com/rebellions-sw/optimum-rbln/pull/267) (kblee): “architecting -> lazy import for submodule config”
　　　외 PR 3건
**응답** [ ] 맞다 　[ ] 틀리다 　[ ] 조건부로 맞다 　[ ] 필요 없는 룰이다 　[ ] 모른다 　[ ] 기억나지 않는다 　· 메모:

### STY-002 · 판단형

**룰** 디버깅 코드, 개발 중 로그와 주석, 예제 속 내부 검증 코드 같은 개발 흔적은 PR에 남기지 않는다.
**근거** [#136](https://github.com/rebellions-sw/optimum-rbln/pull/136) (jongho): “Example script should not include internal validation code.”
　　　[#249](https://github.com/rebellions-sw/optimum-rbln/pull/249) (jongho): “No debugging code for examples”
　　　외 PR 1건
**응답** [ ] 맞다 　[ ] 틀리다 　[ ] 조건부로 맞다 　[ ] 필요 없는 룰이다 　[ ] 모른다 　[ ] 기억나지 않는다 　· 메모:

### STY-014 · 판단형

**룰** 상위 계층에서 이미 처리하는 검사나 기본값 처리를 중복해서 추가하지 않는다.
**조건** 과한 방어 코드를 피한다.
**근거** [#356](https://github.com/rebellions-sw/optimum-rbln/pull/356) (jongho): “return empty dict when no compile_cfgs, so that we do not need any code changes here”
　　　[#376](https://github.com/rebellions-sw/optimum-rbln/pull/376) (thkim): “So adding a rbln_config = {} if rbln_config is None else rbln_config would be redundant.”
　　　외 PR 1건
**응답** [ ] 맞다 　[ ] 틀리다 　[ ] 조건부로 맞다 　[ ] 필요 없는 룰이다 　[ ] 모른다 　[ ] 기억나지 않는다 　· 메모:
