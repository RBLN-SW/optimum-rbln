# 모델 베이스 · 라이프사이클 — 룰 17개

결정 필요 4 · 코드 일치 3 · 코드 불일치 0 · 판단형 9 · 폐기 제안 1. 순서는 결정이 필요한 것부터. 각 룰의 **응답** 줄에서 하나를 `[x]`로 표시하고, 틀리거나 조건부면 **메모**에 한 줄 적어 주세요.

## 결정 필요

### LIFE-002 · 결정 필요

**룰** 컴파일 입력 순서는 HF forward 시그니처 순서로 정렬하고 시그니처에 없는 입력 이름은 ValueError로 거부한다.
**조건** RBLN forward는 HF forward의 데이터 입력을 모두 받거나 `**kwargs`로 흡수한다.
**근거** [#11](https://github.com/rebellions-sw/optimum-rbln/pull/11) (kblee): “make compile model input order with original hf model class forward()”
　　　[#51](https://github.com/rebellions-sw/optimum-rbln/pull/51) (sunwook): “which was not properly updated when the hotfix for”
　　　외 PR 4건
**코드 대조** 결정 필요 — 준수 48 / 위반 19. 입력 집합 일치(MISSING은 Blip2QFormer 1건)와 사용자 지정 입력 이름 검증 raise(modeling_generic.py 143/151행, compile 입력 순서도 HF 시그니처 순서로 정렬), token_type_ids 유/무 테스트(BERT 계열 vs Moder…
**질문** 'HF와 순서 일치'는 컴파일 그래프 입력 순서만인가, 사용자 forward의 위치 인자 순서까지인가?
**응답** [ ] 맞다 　[ ] 틀리다 　[ ] 조건부로 맞다 　[ ] 필요 없는 룰이다 　[ ] 모른다 　[ ] 기억나지 않는다 　· 메모:

### LIFE-001 · 결정 필요

**룰** 모델 forward의 반환은 HF 규약을 따른다.
**조건** `return_dict`가 `None`이면 config의 `use_return_dict`를 따르고, 반환 형식을 바꾸기 전에 내부 호출자를 감사한다.
**근거** [#631](https://github.com/rebellions-sw/optimum-rbln/pull/631) (thkim): “Is the fact that 'return_dict=True' is hardcoded consistent with how Transformers work?”
　　　[#715](https://github.com/rebellions-sw/optimum-rbln/pull/715) (seinpark): “return_dict = return_dict if return_dict is not None else self.config.use_return_dict”
　　　외 PR 2건
**질문** `return_dict`를 항상 `True`로 고정하는 구현이 HF 동작에 맞는 기준인가?
**응답** [ ] 맞다 　[ ] 틀리다 　[ ] 조건부로 맞다 　[ ] 필요 없는 룰이다 　[ ] 모른다 　[ ] 기억나지 않는다 　· 메모:

### LIFE-015 · 결정 필요

**룰** `save_pretrained`는 저장 경로를 바꾸면 로드 경로도 바꿔 저장·로드 레이아웃의 대칭을 유지한다.
**조건** 기존 저장 디렉터리를 통째로 삭제하지 않고 실패 시 기존 파일이 보존되는 방식을 속도보다 우선한다.
**근거** [#215](https://github.com/rebellions-sw/optimum-rbln/pull/215) (jongho): “Save HF config when save_pretrained is called, instead of copying original file.”
　　　[#14](https://github.com/rebellions-sw/optimum-rbln/pull/14) (jongho): “대신 temp dir에 copy한 다음, rename하는 식으로 변경합니다.”
　　　외 PR 2건
**질문** 저장은 임시 디렉터리 완성 후 rename으로 교체하는가, 대상 파일만 덮어쓰는가?
**응답** [ ] 맞다 　[ ] 틀리다 　[ ] 조건부로 맞다 　[ ] 필요 없는 룰이다 　[ ] 모른다 　[ ] 기억나지 않는다 　· 메모:

### LIFE-010 · 결정 필요

**룰** RBLN 모델 클래스는 원본 가중치를 CPU/GPU에 미리 적재하지 않는다.
**조건** 래퍼는 원본 텐서 뷰를 공유하고, 로드 후에는 체크포인트 mmap 뷰가 남지 않도록 파라미터를 복제한다.
**근거** [#168](https://github.com/rebellions-sw/optimum-rbln/pull/168) (jongho): “other existing instances of RBLN-classes do not need to load the weight of the existing model into cpu/gpu”
　　　[#714](https://github.com/rebellions-sw/optimum-rbln/pull/714) (thkim): “The 'nn.Linear(1, 1)' shells were never called;”
　　　외 PR 1건
**질문** 원본 텐서 뷰 공유 규칙과 mmap 뷰 방지용 파라미터 복제 규칙은 언제 각각 적용하는가?
**응답** [ ] 맞다 　[ ] 틀리다 　[ ] 조건부로 맞다 　[ ] 필요 없는 룰이다 　[ ] 모른다 　[ ] 기억나지 않는다 　· 메모:

## 코드 일치

### LIFE-024 · 코드 일치

**룰** torch.export로 추적되는 코드에서는 텐서 값에 따라 갈리는 파이썬 분기를 쓰지 않는다.
**조건** .item() 값의 범위 가정은 torch._check로 알리고, 값에 따른 계산 분기(예: fp16 넘침 막기)는 dtype 같은 정적 조건으로 바꾼다.
**근거** [#303](https://github.com/rebellions-sw/optimum-rbln/pull/303) (joonsoo): “But now the models are now traced with torch.export, so the custom linear op is not necessary anymore.”
　　　[#711](https://github.com/rebellions-sw/optimum-rbln/pull/711) (thkim): “HF clamps fp16 activations with a data-dependent 'isinf(...).any()' guard that cannot be traced.”
　　　외 PR 3건
**코드 대조** 코드 일치 — 준수 None / 위반 None
**응답** [ ] 맞다 　[ ] 틀리다 　[ ] 조건부로 맞다 　[ ] 필요 없는 룰이다 　[ ] 모른다 　[ ] 기억나지 않는다 　· 메모:

### LIFE-007 · 코드 일치

**룰** 전역 상태(모듈 함수, import한 클래스 메서드)를 바꿀 때는 try/finally로 감싸 컴파일 실패 시에도 원복한다.
**조건** 원본은 try 이전에 캡처하며, 컴파일 대상 인스턴스의 서브모듈 forward 교체(MethodType)는 대상이 아니다.
**근거** [#118](https://github.com/rebellions-sw/optimum-rbln/pull/118) (jongho): “If monkey patching is used, you must revert the original method after you're done using it.”
　　　[#168](https://github.com/rebellions-sw/optimum-rbln/pull/168) (jongho): “recover logic should be work even if a compilation is failed.”
　　　외 PR 1건
**코드 대조** 코드 일치 — 준수 3 / 위반 0
**응답** [ ] 맞다 　[ ] 틀리다 　[ ] 조건부로 맞다 　[ ] 필요 없는 룰이다 　[ ] 모른다 　[ ] 기억나지 않는다 　· 메모:

### LIFE-020 · 코드 일치

**룰** 베이스 메서드를 오버라이드할 때는 인자 목록을 베이스와 똑같이 둔다.
**조건** 모델마다 필요한 값은 개별 인자로 받지 않고 model_config·rbln_config에서 꺼낸다. 베이스에 인자를 더하면 모든 오버라이드에 함께 더한다.
**근거** [#160](https://github.com/rebellions-sw/optimum-rbln/pull/160) (thkim): “Modified the get_input_info method to accept model_config and rbln_config as parameters instead of individual variables.”
　　　[#766](https://github.com/rebellions-sw/optimum-rbln/pull/766) (thkim): “All 'get_compiled_model' overrides accept the new keyword (unused where irrelevant) so the contract is uniform.”
**코드 대조** 코드 일치 — 준수 118 / 위반 1
**응답** [ ] 맞다 　[ ] 틀리다 　[ ] 조건부로 맞다 　[ ] 필요 없는 룰이다 　[ ] 모른다 　[ ] 기억나지 않는다 　· 메모:

## 판단형

### LIFE-003 · 판단형

**룰** 부모 클래스가 제공하는 구현은 override나 복제 없이 상속하고 공통 동작은 mixin으로 공유한다.
**조건** 클래스마다 메서드를 감싸지 않는다.
**근거** [#722](https://github.com/rebellions-sw/optimum-rbln/pull/722) (thkim): “How about use 'Mixin' pattern for all MoE classes, instead of wrapping 'get_pytorch_model' of every class?”
　　　[#444](https://github.com/rebellions-sw/optimum-rbln/pull/444) (thkim): “Avoid overriding unless forward() behavior must change (or the architecture truly diverges).”
　　　외 PR 4건
**응답** [ ] 맞다 　[ ] 틀리다 　[ ] 조건부로 맞다 　[ ] 필요 없는 룰이다 　[ ] 모른다 　[ ] 기억나지 않는다 　· 메모:

### LIFE-004 · 판단형

**룰** 공통 기능 구현은 베이스 클래스에 두고 클래스 고유 특성은 클래스 또는 config 속성으로 선언한다.
**조건** 지원을 선언하지 않은 클래스에 기능이 요청되면 컴파일 전에 거부한다.
**근거** [#359](https://github.com/rebellions-sw/optimum-rbln/pull/359) (thkim): “Passing rbln_config should allow removing the following three parameters”
　　　[#356](https://github.com/rebellions-sw/optimum-rbln/pull/356) (jongho): “How about making 'allow-no-compile-cfgs' as a class instance? Because it depends on the characteristics of Class itself.”
　　　외 PR 4건
**응답** [ ] 맞다 　[ ] 틀리다 　[ ] 조건부로 맞다 　[ ] 필요 없는 룰이다 　[ ] 모른다 　[ ] 기억나지 않는다 　· 메모:

### LIFE-008 · 판단형

**룰** 미지원 입력, 컴파일 상한 초과, 런타임 생성 실패는 즉시 명시적 예외로 알린다.
**조건** 메시지에는 입력 길이와 상한 같은 원인과 컴파일만 하려면 `create_runtimes=False`를 쓰라는 안내를 담는다.
**근거** [#736](https://github.com/rebellions-sw/optimum-rbln/pull/736) (jongho): “now raise a clear 'ValueError'/'FileNotFoundError' instead of a 'TypeError'/'AttributeError'”
　　　[#706](https://github.com/rebellions-sw/optimum-rbln/pull/706) (thkim): “Unmappable layouts raise instead of silently permuting wrong.”
　　　외 PR 4건
**응답** [ ] 맞다 　[ ] 틀리다 　[ ] 조건부로 맞다 　[ ] 필요 없는 룰이다 　[ ] 모른다 　[ ] 기억나지 않는다 　· 메모:

### LIFE-021 · 판단형

**룰** 새 모델은 `models/<모델>/` 아래 configuration, modeling, architecture로 추가한다.
**조건** 상위 `__init__`에 등록하고 `get_pytorch_model`, `_update_rbln_config`, `_create_runtimes`를 기존 관례대로 구현한다.
**근거** [#158](https://github.com/rebellions-sw/optimum-rbln/pull/158) (thkim): “ColQwen is not a class from the transformers library, so we need to enable the model with the [custom class]”
　　　[#533](https://github.com/rebellions-sw/optimum-rbln/pull/533) (ChanyongJeon): “align 'get_pytorch_model' / '_update_rbln_config' / '_create_runtimes' with RBLN conventions”
　　　외 PR 4건
**응답** [ ] 맞다 　[ ] 틀리다 　[ ] 조건부로 맞다 　[ ] 필요 없는 룰이다 　[ ] 모른다 　[ ] 기억나지 않는다 　· 메모:

### LIFE-011 · 판단형

**룰** HF 모듈을 그대로 쓰는 부분에는 dtype 변환을 더하지 않는다.
**조건** RBLN이 직접 쓴 커널에서 fp16 오버플로가 나는 항(예: A_log.exp())만 fp32로 계산하고 커널 경계에서 그래프 dtype으로 되돌린다.
**근거** [#711](https://github.com/rebellions-sw/optimum-rbln/pull/711) (thkim): “Code that reuses HF 'nn.Module's as-is (RMSNorm, MLP, q/k norm, projectors) already carries HF's casts and is not touched.”
　　　[#673](https://github.com/rebellions-sw/optimum-rbln/pull/673) (dkhong): “I removed '_run()' and explicit dtype casting to follow HF implementations.”
　　　외 PR 2건
**응답** [ ] 맞다 　[ ] 틀리다 　[ ] 조건부로 맞다 　[ ] 필요 없는 룰이다 　[ ] 모른다 　[ ] 기억나지 않는다 　· 메모:

### LIFE-006 · 판단형

**룰** 파생 가능한 값은 멤버 변수로 중복 저장하지 않고 property로 계산한다.
**조건** 같은 상태를 여러 곳에서 갱신하지 않으며, 필요하면 캐시한다.
**근거** [#735](https://github.com/rebellions-sw/optimum-rbln/pull/735) (jongho): “please use property + cache(if needed) instead to prevent inconsist values”
　　　[#219](https://github.com/rebellions-sw/optimum-rbln/pull/219) (jongho): “how about using '@property' instead of multiple assigning??”
　　　외 PR 1건
**응답** [ ] 맞다 　[ ] 틀리다 　[ ] 조건부로 맞다 　[ ] 필요 없는 룰이다 　[ ] 모른다 　[ ] 기억나지 않는다 　· 메모:

### LIFE-009 · 판단형

**룰** RBLN 접두사 모델 클래스는 기존 HF 클래스를 교체할 수 있는 인스턴스로 설계한다.
**조건** 래퍼 클래스의 저장과 로드는 별도 함수 없이 파이프라인의 `save_pretrained`/`from_pretrained`에 포함한다.
**근거** [#168](https://github.com/rebellions-sw/optimum-rbln/pull/168) (jongho): “other pipeline components are saved and loaded together during the pipeline's from_pretrained / save_pretrained”
　　　[#100](https://github.com/rebellions-sw/optimum-rbln/pull/100) (seinpark): “So, 'Optimum-rbln' have required removing/modifying the function including the '.parameters()' one by one.”
　　　외 PR 1건
**응답** [ ] 맞다 　[ ] 틀리다 　[ ] 조건부로 맞다 　[ ] 필요 없는 룰이다 　[ ] 모른다 　[ ] 기억나지 않는다 　· 메모:

### LIFE-012 · 판단형

**룰** 모델 로드 dtype의 기본값은 auto(체크포인트 dtype)이고 사용자가 준 값을 우선한다.
**조건** 인자 이름은 dtype이 정식이고, torch_dtype은 경고를 내는 deprecated alias로만 받는다(configuration_utils.py).
**근거** [#259](https://github.com/rebellions-sw/optimum-rbln/pull/259) (jongho): “change the dtype of placeholders that must follow the dtype of model parameters”
　　　[#567](https://github.com/rebellions-sw/optimum-rbln/pull/567) (junamsong): “Default 'dtype' to '"auto"' in 'RBLNModel.get_pytorch_model' so HF”
　　　외 PR 1건
**응답** [ ] 맞다 　[ ] 틀리다 　[ ] 조건부로 맞다 　[ ] 필요 없는 룰이다 　[ ] 모른다 　[ ] 기억나지 않는다 　· 메모:

### LIFE-019 · 판단형

**룰** 가중치와 파일 로드 경로는 로컬 디렉터리뿐 아니라 Hugging Face Hub에서도 동작해야 한다.
**조건** 매 로드마다 저장소 전체 목록을 조회하지 않고 필요한 파일만 해석해 캐시를 검증한다.
**근거** [#683](https://github.com/rebellions-sw/optimum-rbln/pull/683) (jongho): “Both call sites listed the repository on every load, even when the files were already in the local cache.”
　　　[#130](https://github.com/rebellions-sw/optimum-rbln/pull/130) (jongho): “so that it can also load from huggingface hub.”
**응답** [ ] 맞다 　[ ] 틀리다 　[ ] 조건부로 맞다 　[ ] 필요 없는 룰이다 　[ ] 모른다 　[ ] 기억나지 않는다 　· 메모:

## 폐기 제안

### LIFE-005 · 폐기 제안

**룰** 기능을 추가할 때 공유 레이어와 베이스 클래스는 수정하지 않고 모델별 서브클래스나 신규 클래스로 확장한다.
**근거** [#804](https://github.com/rebellions-sw/optimum-rbln/pull/804) (seinpark): “Shared code ('modeling.py', 'modeling_base.py', 'configuration_utils.py', 'modeling_diffusers.py', the existing classes in 'vae.p…”
　　　[#815](https://github.com/rebellions-sw/optimum-rbln/pull/815) (yskim): “without modifying shared layer classes”
**응답** [ ] 맞다 　[ ] 틀리다 　[ ] 조건부로 맞다 　[ ] 필요 없는 룰이다 　[ ] 모른다 　[ ] 기억나지 않는다 　· 메모:
