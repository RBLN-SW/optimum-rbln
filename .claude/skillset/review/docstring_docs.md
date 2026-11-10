# docstring · 문서 — 룰 6개

결정 필요 1 · 코드 일치 2 · 코드 불일치 0 · 판단형 3 · 폐기 제안 0. 순서는 결정이 필요한 것부터. 각 룰의 **응답** 줄에서 하나를 `[x]`로 표시하고, 틀리거나 조건부면 **메모**에 한 줄 적어 주세요.

## 결정 필요

### DOC-004 · 결정 필요

**룰** docstring의 가변 키워드 인자는 `kwargs`로 표기하고 전달 대상을 적는다.
**조건** `rbln_` 접두사 인자는 `rbln_config`로, 나머지는 HuggingFace 라이브러리로 전달된다고 적는다.
**근거** [#277](https://github.com/rebellions-sw/optimum-rbln/pull/277) (seinpark): “Arguments with the prefix 'rbln_' are passed to rbln_config, while the remaining arguments are passed to the HuggingFace library.”
　　　[#367](https://github.com/rebellions-sw/optimum-rbln/pull/367) (seinpark): “Is it okay if there is no mention of 'kwargs'?”
　　　외 PR 1건
**질문** 사용되지 않는 인자(`kwargs` 등)를 docstring에 적는가, 생략하는가?
**응답** [ ] 맞다 　[ ] 틀리다 　[ ] 조건부로 맞다 　[ ] 필요 없는 룰이다 　[ ] 모른다 　[ ] 기억나지 않는다 　· 메모:

## 코드 일치

### DOC-011 · 코드 일치

**룰** 공개 API docstring은 Google 스타일만 쓰고 numpy 스타일 `name : type`은 금지한다.
**조건** `name (type): 설명`은 한 줄에 쓰고 Returns는 `type: 설명` 한 행으로 쓰며 기본값은 'Defaults to X.'로 쓴다.
**근거** [#181](https://github.com/rebellions-sw/optimum-rbln/pull/181) (jongho): “batch_size (Optional[int]): The batch size for inference. Defaults to 1.”
　　　[#779](https://github.com/rebellions-sw/optimum-rbln/pull/779) (junamsong): “Put the 'get_load_overrides' Returns description on one line”
　　　외 PR 2건
**코드 대조** 코드 일치 — 준수 1138 / 위반 7
**응답** [ ] 맞다 　[ ] 틀리다 　[ ] 조건부로 맞다 　[ ] 필요 없는 룰이다 　[ ] 모른다 　[ ] 기억나지 않는다 　· 메모:

### DOC-012 · 코드 일치

**룰** docstring과 사용자 노출 메시지의 SDK 문서 링크는 `https://docs.rbln.ai/` 절대 URL로 쓴다.
**조건** 대상 경로는 문서 빌드에 존재해야 하며 상대 경로는 `scripts/validate_docstrings.py`에서 깨지므로 금지한다.
**근거** [#732](https://github.com/rebellions-sw/optimum-rbln/pull/732) (thkim): “The relative form only resolves inside the docs site, so 'validate_docstrings.py'”
　　　[#550](https://github.com/rebellions-sw/optimum-rbln/pull/550) (junamsong): “Linkchecker on 'rbln_sdk_doc' then reported 404s from the generated T5 API pages.”
**코드 대조** 코드 일치 — 준수 42 / 위반 2
**응답** [ ] 맞다 　[ ] 틀리다 　[ ] 조건부로 맞다 　[ ] 필요 없는 룰이다 　[ ] 모른다 　[ ] 기억나지 않는다 　· 메모:

## 판단형

### DOC-003 · 판단형

**룰** docstring의 인자 목록, 기본값, 텐서 shape, 수치 정의, 예시 클래스 이름은 실제 코드에 맞춘다.
**근거** [#363](https://github.com/rebellions-sw/optimum-rbln/pull/363) (dkhong): “pixel_values ('torch.FloatTensor' of shape '(batch_size, channels, height, width)'):”
　　　[#787](https://github.com/rebellions-sw/optimum-rbln/pull/787) (jswoo): “The vision model compares 'max_seq_len' against 'grid_h * grid_w'”
　　　외 PR 3건
**응답** [ ] 맞다 　[ ] 틀리다 　[ ] 조건부로 맞다 　[ ] 필요 없는 룰이다 　[ ] 모른다 　[ ] 기억나지 않는다 　· 메모:

### DOC-001 · 판단형

**룰** 새로 추가하는 모델, 설정, 파이프라인 클래스와 함수, 오버라이드한 classmethod에는 docstring을 둔다.
**조건** 용도와 추가 이유, 주의사항을 적고 가능하면 사용 예시를 포함한다.
**근거** [#308](https://github.com/rebellions-sw/optimum-rbln/pull/308) (junamsong): “docs: add docstrings for overridden classmethod 'from_pretrained' (exaone, midm)”
　　　[#177](https://github.com/rebellions-sw/optimum-rbln/pull/177) (jongho): “Add class docstring. if available, please add example code, too.”
　　　외 PR 2건
**응답** [ ] 맞다 　[ ] 틀리다 　[ ] 조건부로 맞다 　[ ] 필요 없는 룰이다 　[ ] 모른다 　[ ] 기억나지 않는다 　· 메모:

### DOC-002 · 판단형

**룰** 주석은 코드가 말하지 않는 이유만 짧게 적는다.
**조건** 코드 풀이, PR 이력, 마이그레이션 맥락, 개발 메모는 남기지 않고 내부 구현 설명은 docstring 대신 `#` 주석으로 쓴다.
**근거** [#735](https://github.com/rebellions-sw/optimum-rbln/pull/735) (seinpark): “Please minimize unnecessary comments that arise during the development.”
　　　[#631](https://github.com/rebellions-sw/optimum-rbln/pull/631) (thkim): “Some of the comments seem unnecessarily long.”
　　　외 PR 1건
**응답** [ ] 맞다 　[ ] 틀리다 　[ ] 조건부로 맞다 　[ ] 필요 없는 룰이다 　[ ] 모른다 　[ ] 기억나지 않는다 　· 메모:
