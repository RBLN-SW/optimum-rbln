# rbln_config (설정 객체) — 룰 18개

결정 필요 4 · 코드 일치 1 · 코드 불일치 1 · 판단형 11 · 폐기 제안 1. 순서는 결정이 필요한 것부터. 각 룰의 **응답** 줄에서 하나를 `[x]`로 표시하고, 틀리거나 조건부면 **메모**에 한 줄 적어 주세요.

## 결정 필요

### CFG-004 · 결정 필요

**룰** 설정 클래스에는 실제 동작에 반영되는 옵션만 노출한다.
**조건** 무시되는 옵션은 제거하거나 명시적으로 거부한다. 폐지 절차는 DEP-001·CFG-010을 따른다.
**근거** [#735](https://github.com/rebellions-sw/optimum-rbln/pull/735) (dkhong): “users may don't know why it works when they set 'use_inputs_embeds=False' explicitly.”
　　　[#694](https://github.com/rebellions-sw/optimum-rbln/pull/694) (dkhong): “users can set 'use_slicing', but it never takes effect.”
　　　외 PR 2건
**질문** 필수 옵션은 기본값을 고정하는가, 사용자 선택을 허용하고 잘못된 값에 예외를 내는가?
**응답** [ ] 맞다 　[ ] 틀리다 　[ ] 조건부로 맞다 　[ ] 필요 없는 룰이다 　[ ] 모른다 　[ ] 기억나지 않는다 　· 메모:

### CFG-005 · 결정 필요

**룰** 컴파일 시점 값의 원천은 산출물의 `rbln_config.json`으로 둔다.
**조건** 그래프를 바꾸지 않는 런타임 전용 옵션은 컴파일 값과 분리해 정해진 경로(runtime kwargs 등)로 노출한다.
**근거** [#67](https://github.com/rebellions-sw/optimum-rbln/pull/67) (seinpark): “'passed_rbln_config._runtime_options' in 'RBLNAutoConfig' will raise the error.”
　　　[#88](https://github.com/rebellions-sw/optimum-rbln/pull/88) (jongho): “By specifying decoder_batch_sizes=[1,2,3,4] in rbln_config”
　　　외 PR 2건
**질문** device 같은 런타임 전용 옵션은 로드할 때 어디로 넘기는가?
**응답** [ ] 맞다 　[ ] 틀리다 　[ ] 조건부로 맞다 　[ ] 필요 없는 룰이다 　[ ] 모른다 　[ ] 기억나지 않는다 　· 메모:

### CFG-014 · 결정 필요

**룰** 배치 크기, 최대 시퀀스 길이 같은 형상 인자는 서브모듈이 아닌 상위 설정에서 지정한다.
**조건** 서브모듈 설정만 지정한 호출에서도 부모 기본값이 서브모듈 값과 충돌하지 않아야 한다.
**근거** [#650](https://github.com/rebellions-sw/optimum-rbln/pull/650) (thkim): “A caller that configures only the submodule never sets the top-level key, so the parent silently becomes 1 and conflicts with the…”
　　　[#777](https://github.com/rebellions-sw/optimum-rbln/pull/777) (seinpark): “'batch_size' and 'max_seq_len' already arrive from pipeline-level arguments that also reach the transformer”
**질문** 서브모듈의 명시값이 상위값보다 우선하는가, 형상 인자는 서브모듈에서 지정을 금지하는가?
**응답** [ ] 맞다 　[ ] 틀리다 　[ ] 조건부로 맞다 　[ ] 필요 없는 룰이다 　[ ] 모른다 　[ ] 기억나지 않는다 　· 메모:

### CFG-022 · 결정 필요

**룰** 같은 설정 키를 rbln_config와 rbln_ 인자로 함께 주면 모든 경로에서 같은 쪽이 이긴다.
**조건** 지금은 네 조합 중 '컴파일된 모델 로드 + dict rbln_config' 하나만 rbln_config가 이기고 나머지는 rbln_ 인자가 이긴다.
**근거** [#445](https://github.com/rebellions-sw/optimum-rbln/pull/445) (seinpark): “makes 'rbln_config' and 'rbln_' prefixed arguments mutually exclusive by raising a ValueError when both are provided simultaneous…”
　　　[#2](https://github.com/rebellions-sw/optimum-rbln/pull/2) (thkim): “Users can now specify the prefill_chunk_size by passing either 'prefill_chunk_size' in rbln_config or 'rbln_prefill_chunk_size' w…”
**코드 대조** 폐기 제안 — 준수 0 / 위반 6. dev에서도 동시 지정 시 ValueError를 내는 경로는 없다(#445는 2026-01-23 CLOSED 미병합). 대신 공식 계약은 우선순위다: 클래스 docstring "Method 4: Combining a config object with override parameters…
**질문** 같은 키를 rbln_config와 rbln_ 인자에 함께 주면 어느 쪽이 이겨야 하는가?
**응답** [ ] 맞다 　[ ] 틀리다 　[ ] 조건부로 맞다 　[ ] 필요 없는 룰이다 　[ ] 모른다 　[ ] 기억나지 않는다 　· 메모:

## 코드 불일치(의도는 분명)

### CFG-010 · 코드 불일치(의도는 분명)

**룰** 설정 키를 폐지하거나 개명할 때는 즉시 제거하지 않고 경고 후 새 키로 재매핑한다.
**조건** 생성자로 넘긴 옛 키뿐 아니라, 이미 만든 설정 객체에 대입한 옛 키(cfg.옛키 = 값)도 경고 후 새 키로 옮긴다.
**근거** [#760](https://github.com/rebellions-sw/optimum-rbln/pull/760) (jongho): “a decorator on '__init__' alone would never see it”
　　　[#666](https://github.com/rebellions-sw/optimum-rbln/pull/666) (seinpark): “is renamed to **'cache_metas'** (the list now holds non-KV”
　　　외 PR 3건
**코드 대조** 코드 불일치(의도는 분명) — 준수 10 / 위반 6. dev에서 헬퍼 사용 사이트가 6→10으로 늘었다(#666 kvcache_metas→cache_metas deprecate_kwarg, Cosmos 설정 2곳의 fps deprecate_kwarg, qwen3_5 _mask_padded_keys_and_queries deprecate_…
**응답** [ ] 맞다 　[ ] 틀리다 　[ ] 조건부로 맞다 　[ ] 필요 없는 룰이다 　[ ] 모른다 　[ ] 기억나지 않는다 　· 메모:

## 코드 일치

### CFG-018 · 코드 일치

**룰** 런타임 전용 값과 컴파일 시점 전용 추정 입력은 `subclass_non_save_attributes`에 등록한다.
**조건** `rbln_config.json`에 직렬화되지 않게 하며, 하위 클래스가 재정의하면 상위 항목을 포함한다.
**근거** [#750](https://github.com/rebellions-sw/optimum-rbln/pull/750) (seinpark): “a 'subclass_non_save_attributes' entry, mirroring Qwen3-VL — runtime-only, never serialized”
　　　[#623](https://github.com/rebellions-sw/optimum-rbln/pull/623) (jongho): “'subclass_non_save_attributes' is a plain class attribute that is shadowed (not merged) when a subclass redefines it.”
　　　외 PR 1건
**코드 대조** 코드 일치 — 준수 10 / 위반 0
**응답** [ ] 맞다 　[ ] 틀리다 　[ ] 조건부로 맞다 　[ ] 필요 없는 룰이다 　[ ] 모른다 　[ ] 기억나지 않는다 　· 메모:

## 판단형

### CFG-001 · 판단형

**룰** 지원하지 않는 설정 값은 경고나 대체 없이 명시적 예외로 실패시킨다.
**조건** 값을 생략(None)한 경우만 경고 후 기본값으로 채우고, 0 같은 falsy 값은 바꾸지 않는다. 미지원 dtype의 fp32 승격 여부는 DBT-009에서 따로 묻는다.
**근거** [#646](https://github.com/rebellions-sw/optimum-rbln/pull/646) (thkim): “instead of silently chunking into wrong outputs.”
　　　[#114](https://github.com/rebellions-sw/optimum-rbln/pull/114) (jongho): “No implicit change, Raise error instead.”
　　　외 PR 4건
**응답** [ ] 맞다 　[ ] 틀리다 　[ ] 조건부로 맞다 　[ ] 필요 없는 룰이다 　[ ] 모른다 　[ ] 기억나지 않는다 　· 메모:

### CFG-009 · 판단형

**룰** 설정 포맷, 옵션, dtype, 산출물 구조를 바꿔도 이전에 컴파일된 아티팩트가 그대로 로드되게 한다.
**조건** 로드 동작은 아티팩트가 기록한 입력 정의를 따르며, 레거시 경로는 DEPRECATED 표지와 제거 예정 버전을 명시한다.
**근거** [#803](https://github.com/rebellions-sw/optimum-rbln/pull/803) (seinpark): “The conv cache layout changes, so **artifacts compiled before this must be recompiled.**”
　　　[#694](https://github.com/rebellions-sw/optimum-rbln/pull/694) (dkhong): “Please check whether removing 'fps' from 'RBLNCosmosTransformer3DModelConfig' breaks backward compatibility.”
　　　외 PR 4건
**응답** [ ] 맞다 　[ ] 틀리다 　[ ] 조건부로 맞다 　[ ] 필요 없는 룰이다 　[ ] 모른다 　[ ] 기억나지 않는다 　· 메모:

### CFG-007 · 판단형

**룰** 기본값은 하드코딩하지 않고 모델 config나 전처리기에서 유도한다.
**조건** 단 기기마다 달라지는 설정(예: prefill_chunk_size)에서는 유도하지 않고, 그런 값은 기기와 무관한 고정 상수(예: MAX_GDN_CHUNK_SIZE)로 둔다.
**근거** [#560](https://github.com/rebellions-sw/optimum-rbln/pull/560) (thkim): “Instead of hard-coding it as 'batch_size + 1', why not use this attribute?”
　　　[#718](https://github.com/rebellions-sw/optimum-rbln/pull/718) (seinpark): “Since 'gdn_chunk_size' previously defaulted to 'prefill_chunk_size', the GDN chunk size silently diverged across devices as well.”
　　　외 PR 3건
**응답** [ ] 맞다 　[ ] 틀리다 　[ ] 조건부로 맞다 　[ ] 필요 없는 룰이다 　[ ] 모른다 　[ ] 기억나지 않는다 　· 메모:

### CFG-031 · 판단형

**룰** 새 컴파일·커널 경로는 꺼진 상태에서 기존 경로와 결과가 똑같도록 추가한다.
**조건** 켜는 방식은 사용자 옵션(기본값 False), 입력 모양, 검증된 타깃에서의 자동 선택 중 하나다. 꺼진 상태나 다른 타깃의 코드와 결과는 바꾸지 않는다.
**근거** [#706](https://github.com/rebellions-sw/optimum-rbln/pull/706) (thkim): “Everything is a no-op when the gate is off (CA targets, old configs, batch=1).”
　　　[#815](https://github.com/rebellions-sw/optimum-rbln/pull/815) (yskim): “The option defaults to 'False', preserving the existing compilation path.”
　　　외 PR 3건
**응답** [ ] 맞다 　[ ] 틀리다 　[ ] 조건부로 맞다 　[ ] 필요 없는 룰이다 　[ ] 모른다 　[ ] 기억나지 않는다 　· 메모:

### CFG-008 · 판단형

**룰** 검증 함수는 검증만 하고, 기본값 설정과 패딩은 `set_default_values` 훅 한 곳에서만 한다.
**조건** 모델별 설정 생성 경로나 소비처에 같은 기본값을 중복 정의하지 않는다.
**근거** [#81](https://github.com/rebellions-sw/optimum-rbln/pull/81) (jongho): “validate 함수에서는 validation만 하고, padding은 안하는게 좋을 것 같습니다.”
　　　[#717](https://github.com/rebellions-sw/optimum-rbln/pull/717) (junamsong): “Every consumer independently re-deriving the same explicit value on one tier is the signature of a defaults bug”
　　　외 PR 2건
**응답** [ ] 맞다 　[ ] 틀리다 　[ ] 조건부로 맞다 　[ ] 필요 없는 룰이다 　[ ] 모른다 　[ ] 기억나지 않는다 　· 메모:

### CFG-015 · 판단형

**룰** 서브모듈에 고정해야 하는 제약은 `force_kwargs`로 명시적으로 강제한다.
**조건** 부모의 해석된 값이나 `None`일 수 있는 값은 `force_kwargs`가 아닌 기본값으로 전달한다.
**근거** [#682](https://github.com/rebellions-sw/optimum-rbln/pull/682) (seinpark): “Read 'rbln_config.batch_size' in the vision '_update_rbln_config' instead of”
　　　[#495](https://github.com/rebellions-sw/optimum-rbln/pull/495) (jongho): “This allows users to create configs with 'batch_size != 1' for vision models, breaking a hard requirement.”
　　　외 PR 2건
**응답** [ ] 맞다 　[ ] 틀리다 　[ ] 조건부로 맞다 　[ ] 필요 없는 룰이다 　[ ] 모른다 　[ ] 기억나지 않는다 　· 메모:

### CFG-002 · 판단형

**룰** 컴파일러가 지원하지 않는 기능과 런타임 생성 불가 설정은 컴파일 시작 전에 한 곳에서 거부한다.
**조건** 생성자나 검증 함수에서 한 메시지로 거부한다.
**근거** [#195](https://github.com/rebellions-sw/optimum-rbln/pull/195) (jongho): “If the compile function is passed compile settings that prevent it from generating runtimes on this machine, and create_runtimes=…”
　　　[#803](https://github.com/rebellions-sw/optimum-rbln/pull/803) (seinpark): “is rejected by 'validate_gdn_custom_kernel', in one place and with”
**응답** [ ] 맞다 　[ ] 틀리다 　[ ] 조건부로 맞다 　[ ] 필요 없는 룰이다 　[ ] 모른다 　[ ] 기억나지 않는다 　· 메모:

### CFG-012 · 판단형

**룰** 같은 인자를 받는 형제 설정 클래스와 모델 클래스는 함께 점검해 일관되게 변경한다.
**근거** [#682](https://github.com/rebellions-sw/optimum-rbln/pull/682) (thkim): “If it needs to be included, 'RBLNExaone4_5_ModelConfig' should probably be included as well.”
　　　[#383](https://github.com/rebellions-sw/optimum-rbln/pull/383) (kblee): “Should we modify this here as well?”
**응답** [ ] 맞다 　[ ] 틀리다 　[ ] 조건부로 맞다 　[ ] 필요 없는 룰이다 　[ ] 모른다 　[ ] 기억나지 않는다 　· 메모:

### CFG-013 · 판단형

**룰** 연결된 파이프라인의 설정은 전역, 파이프라인, 서브모듈 계층으로 정의하고 명시적 우선순위를 따른다.
**조건** 같은 이름의 값은 호출자 인자, 서브모듈 설정, 파이프라인 설정 순으로 우선한다.
**근거** [#770](https://github.com/rebellions-sw/optimum-rbln/pull/770) (dkhong): “So even within a single pipeline, different arguments follow different policies.”
　　　[#39](https://github.com/rebellions-sw/optimum-rbln/pull/39) (dkhong): “3 > 4 > 1 > 2 순으로 중요도가 높다고 판단하여 중요도가 높은 값을 우선하여 사용하도록 구현하였습니다.”
**응답** [ ] 맞다 　[ ] 틀리다 　[ ] 조건부로 맞다 　[ ] 필요 없는 룰이다 　[ ] 모른다 　[ ] 기억나지 않는다 　· 메모:

### CFG-026 · 판단형

**룰** `rbln_config`에 저장된 출력 옵션은 HF와 같게 다루고 forward 인자와 다르면 ValueError를 낸다.
**조건** 추론 시에도 설정 값을 따른다.
**근거** [#166](https://github.com/rebellions-sw/optimum-rbln/pull/166) (seinpark): “If 'output_hidden_states' in forward kwargs is not equal to 'output_hidden_states' in 'rbln_config', it always raise ValueError.”
　　　[#174](https://github.com/rebellions-sw/optimum-rbln/pull/174) (thkim): “Make the saving of rbln_config for output_attention, output_hiddens_states behave like huggingface and follow it at inference tim…”
**응답** [ ] 맞다 　[ ] 틀리다 　[ ] 조건부로 맞다 　[ ] 필요 없는 룰이다 　[ ] 모른다 　[ ] 기억나지 않는다 　· 메모:

### CFG-027 · 판단형

**룰** 하드웨어 계열별 제약값은 전역 상수가 아니라 계열 단위 limits 객체로 정의한다.
**조건** 알 수 없는 장치가 다른 계열의 한계를 조용히 물려받지 않게 한다.
**근거** [#729](https://github.com/rebellions-sw/optimum-rbln/pull/729) (junamsong): “unknown name → 'ValueError' rather than the wider limits.”
　　　[#721](https://github.com/rebellions-sw/optimum-rbln/pull/721) (junamsong): “bounds the same two axes per NPU family”
**응답** [ ] 맞다 　[ ] 틀리다 　[ ] 조건부로 맞다 　[ ] 필요 없는 룰이다 　[ ] 모른다 　[ ] 기억나지 않는다 　· 메모:

## 폐기 제안

### CFG-019 · 폐기 제안

**룰** 서브모듈을 가진 설정 클래스는 `submodules`를 선언하고 공통 헬퍼로 서브모듈을 초기화한다.
**조건** 부모의 런타임 옵션을 `initialize_submodule_config`로 전파하며, 클래스를 모르는 dict는 dict로 남겨 모델 클래스가 나중에 해석한다.
**근거** [#495](https://github.com/rebellions-sw/optimum-rbln/pull/495) (jongho): “The new approach uses Pydantic's '@model_validator' to handle this conversion automatically and consistently.”
　　　[#473](https://github.com/rebellions-sw/optimum-rbln/pull/473) (seinpark): “Add missing submodule definitions”
**코드 대조** 폐기 제안 — 준수 0 / 위반 36. dev(74e97fa4)에서도 'submodule_config_classes'·'model_validator'는 src/tests 어디에도 없고(0건), 서브모듈을 가진 36개 클래스(main 32) 모두 매핑이 없으며 36개 전부가 __init__에서 'self.initialize_…
**응답** [ ] 맞다 　[ ] 틀리다 　[ ] 조건부로 맞다 　[ ] 필요 없는 룰이다 　[ ] 모른다 　[ ] 기억나지 않는다 　· 메모:
