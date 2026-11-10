# 멀티모달 · 서브모듈 — 룰 9개

결정 필요 1 · 코드 일치 1 · 코드 불일치 2 · 판단형 4 · 폐기 제안 1. 순서는 결정이 필요한 것부터. 각 룰의 **응답** 줄에서 하나를 `[x]`로 표시하고, 틀리거나 조건부면 **메모**에 한 줄 적어 주세요.

## 결정 필요

### SUB-001 · 결정 필요

**룰** 서브모듈이 명시한 설정값은 상위 설정에서 상속한 값보다 우선하며 이 우선순위는 한 곳에서 정의한다.
**조건** 명시하지 않은 경우에만 상위 값(batch_size, 런타임 옵션 등)을 기본값으로 상속한다.
**근거** [#267](https://github.com/rebellions-sw/optimum-rbln/pull/267) (thkim): “Shouldn't the submodule_config specified by the user take precedence over the default given by the parent?”
　　　[#445](https://github.com/rebellions-sw/optimum-rbln/pull/445) (jongho): “since we don't propagate compile options to its submodule.”
　　　외 PR 2건
**질문** 컴파일 옵션 중 서브모듈로 전파할 키(예: 최상위 `batch_size`)는 무엇인가?
**응답** [ ] 맞다 　[ ] 틀리다 　[ ] 조건부로 맞다 　[ ] 필요 없는 룰이다 　[ ] 모른다 　[ ] 기억나지 않는다 　· 메모:

## 코드 불일치(의도는 분명)

### SUB-005 · 코드 불일치(의도는 분명)

**룰** 서브모듈의 NPU 런타임을 띄우지 않을 때는 모델별 플래그 대신 그 서브모듈의 create_runtimes=False를 쓴다.
**조건** 서브모듈은 그대로 컴파일·저장되고 목록 순서와 속성(예: self.visual)도 남는다. 그 자리는 호출하면 에러를 내는 UnavailableRuntime이 된다.
**근거** [#760](https://github.com/rebellions-sw/optimum-rbln/pull/760) (jongho): “so 'rbln_submodules' keeps its positional layout and 'self.visual' is always present.”
　　　[#777](https://github.com/rebellions-sw/optimum-rbln/pull/777) (seinpark): “It is a mandatory submodule all the same, so the pipeline compiles the smallest legal graph for it and creates no runtime”
　　　외 PR 1건
**코드 대조** 코드 불일치(의도는 분명) — 준수 2 / 위반 4. 공통 방법(서브모듈 create_runtimes=False → UnavailableRuntime)은 modeling_base.py:352-358에 있고 Cosmos(#777)가 쓴다.
**응답** [ ] 맞다 　[ ] 틀리다 　[ ] 조건부로 맞다 　[ ] 필요 없는 룰이다 　[ ] 모른다 　[ ] 기억나지 않는다 　· 메모:

### SUB-008 · 코드 불일치(의도는 분명)

**룰** dtype 경계는 HF 원본을 따라 fp32 또는 `rbln_config.dtype` 중 HF가 쓰는 쪽을 쓴다.
**조건** HF가 fp32로 두는 계열(Qwen-VL, EXAONE-4.5)은 fp32, `x.dtype`으로 캐스트하는 계열(Gemma4)만 `rbln_config.dtype`을 쓴다.
**근거** [#764](https://github.com/rebellions-sw/optimum-rbln/pull/764) (thkim): “so the tables are never rounded to the activation dtype.”
　　　[#711](https://github.com/rebellions-sw/optimum-rbln/pull/711) (thkim): “Each family's architecture file now carries its own 'apply_rotary_pos_emb_vision' mirroring the HF one, as HF does per model.”
**코드 대조** 코드 불일치(의도는 분명) — 준수 7 / 위반 4. main(1/5)에서 크게 개선됐다: #711(2026-10-13 병합)로 5개 계열(qwen2_vl, qwen2_5_vl, qwen3_vl, qwen3_5, exaone4_5) 모두 아키텍처 파일에 "Mirrors HF 'apply_rotary_pos_emb_vision': rota…
**응답** [ ] 맞다 　[ ] 틀리다 　[ ] 조건부로 맞다 　[ ] 필요 없는 룰이다 　[ ] 모른다 　[ ] 기억나지 않는다 　· 메모:

## 코드 일치

### SUB-002 · 코드 일치

**룰** 서브모듈 설정 조정은 모델마다 새 함수를 만들지 않고 정해진 공통 훅을 오버라이드한다.
**조건** 부모가 자식을: _update_submodule_rbln_config, 자식이 스스로: _update_submodule_config, 파이프라인 안: update_rbln_config_using_pipe.
**근거** [#135](https://github.com/rebellions-sw/optimum-rbln/pull/135) (jongho): “Instead of implementing update_using_parent_ something in each submodule,”
　　　[#816](https://github.com/rebellions-sw/optimum-rbln/pull/816) (seinpark): “Adds 'update_rbln_config_using_pipe' so that it can be a pipeline submodule.”
**코드 대조** 코드 일치 — 준수 28 / 위반 0
**응답** [ ] 맞다 　[ ] 틀리다 　[ ] 조건부로 맞다 　[ ] 필요 없는 룰이다 　[ ] 모른다 　[ ] 기억나지 않는다 　· 메모:

## 판단형

### SUB-004 · 판단형

**룰** 파이프라인과 멀티모달 모델 내부의 비전 모델은 별도 RBLN 모델 클래스로 감싸 서브모듈로 컴파일한다.
**조건** 여러 상위 모델이 공유하는 서브모듈의 `rbln_config`는 특정 모델 이름에 묶지 않고 일반화한다.
**근거** [#167](https://github.com/rebellions-sw/optimum-rbln/pull/167) (seinpark): “Converted siglipvision to Optimum-rbln Model: Optimized the SigLipVisionModel using the Optimum-rbln framework for deployment eff…”
　　　[#134](https://github.com/rebellions-sw/optimum-rbln/pull/134) (thkim): “add SigLipVisionModel for submodule”
　　　외 PR 1건
**응답** [ ] 맞다 　[ ] 틀리다 　[ ] 조건부로 맞다 　[ ] 필요 없는 룰이다 　[ ] 모른다 　[ ] 기억나지 않는다 　· 메모:

### SUB-003 · 판단형

**룰** 이미 지원하는 체크포인트나 같은 아키텍처의 신규 모델은 기존 클래스를 재사용한다.
**조건** 구성 요소가 다른 부분만 새로 구현하고 나머지는 테스트만 추가한다.
**근거** [#817](https://github.com/rebellions-sw/optimum-rbln/pull/817) (seinpark): “'RBLNIdefics3ForConditionalGeneration' already supports it.”
　　　[#735](https://github.com/rebellions-sw/optimum-rbln/pull/735) (dkhong): “Reuse 'RBLNQwen3ForCausalLM'”
**응답** [ ] 맞다 　[ ] 틀리다 　[ ] 조건부로 맞다 　[ ] 필요 없는 룰이다 　[ ] 모른다 　[ ] 기억나지 않는다 　· 메모:

### SUB-010 · 판단형

**룰** VLM 구현은 설정 접근과 전처리를 검증된 같은 계열 모델에 맞춘다.
**조건** composite VLM의 텍스트 차원(hidden size, 레이어 수)은 텍스트 설정을 통해 읽는다.
**근거** [#701](https://github.com/rebellions-sw/optimum-rbln/pull/701) (seinpark): “instead of flat 'self.config.*' access”
　　　[#651](https://github.com/rebellions-sw/optimum-rbln/pull/651) (seinpark): “mirroring Qwen3-VL's '_preprocess_prefill'.”
**응답** [ ] 맞다 　[ ] 틀리다 　[ ] 조건부로 맞다 　[ ] 필요 없는 룰이다 　[ ] 모른다 　[ ] 기억나지 않는다 　· 메모:

### SUB-015 · 판단형

**룰** 서브모듈 아티팩트는 부모 모델의 서브폴더 안에 중첩해 저장한다.
**조건** 서브모듈의 로드 경로와 generation config 탐색은 부모 디렉터리와 서브모듈 서브폴더를 기준으로 한다.
**근거** [#93](https://github.com/rebellions-sw/optimum-rbln/pull/93) (jongho): “if submodule can generate, then submodule should find generation config from its subfolder, not predecessor's.”
　　　[#703](https://github.com/rebellions-sw/optimum-rbln/pull/703) (seinpark): “The lookup base is each **parent's own directory** instead of the root”
**응답** [ ] 맞다 　[ ] 틀리다 　[ ] 조건부로 맞다 　[ ] 필요 없는 룰이다 　[ ] 모른다 　[ ] 기억나지 않는다 　· 메모:

## 폐기 제안

### SUB-011 · 폐기 제안

**룰** 생산 측과 소비 측이 공유하는 값(예: 이미지 토큰 수)은 같은 계산 경로에서 다시 유도한다.
**조건** 근거 PR 2건이 모두 머지되지 않아 폐기를 제안한다.
**근거** [#705](https://github.com/rebellions-sw/optimum-rbln/pull/705) (thkim): “counting reuses each family's '_preprocess_prefill' mechanism verbatim, so producer and consumer can never disagree”
　　　[#678](https://github.com/rebellions-sw/optimum-rbln/pull/678) (junamsong): “a disagreement is logged rather than swallowed, so a genuine tower bug stays visible.”
**응답** [ ] 맞다 　[ ] 틀리다 　[ ] 조건부로 맞다 　[ ] 필요 없는 룰이다 　[ ] 모른다 　[ ] 기억나지 않는다 　· 메모:
