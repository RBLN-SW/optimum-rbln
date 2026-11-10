# diffusers — 룰 6개

결정 필요 1 · 코드 일치 0 · 코드 불일치 0 · 판단형 5 · 폐기 제안 0. 순서는 결정이 필요한 것부터. 각 룰의 **응답** 줄에서 하나를 `[x]`로 표시하고, 틀리거나 조건부면 **메모**에 한 줄 적어 주세요.

## 결정 필요

### DIF-004 · 결정 필요

**룰** 컴파일 시점 옵션은 그래프 형상을 직접 정하는 값(서브모듈 batch size 등)으로 명시한다.
**조건** guidance scale처럼 그래프를 간접적으로 바꾸는 값은 컴파일 옵션으로 쓰지 않는다.
**근거** [#15](https://github.com/rebellions-sw/optimum-rbln/pull/15) (jongho): “guidance scale 대신 명시적으로 submodule의 batch size를 지정하도록 바꿔주시면 좋겠습니다~”
　　　[#104](https://github.com/rebellions-sw/optimum-rbln/pull/104) (seinpark): “How do we handle the cases where the 'guidance scale' has different values”
**질문** 컴파일 때 지정한 guidance scale과 runtime 값이 다르면 에러인가, 허용하는가?
**응답** [ ] 맞다 　[ ] 틀리다 　[ ] 조건부로 맞다 　[ ] 필요 없는 룰이다 　[ ] 모른다 　[ ] 기억나지 않는다 　· 메모:

## 판단형

### DIF-001 · 판단형

**룰** diffusers 모듈의 파일 구성, 하위 패키지 배치, 설정 파일 이름은 upstream diffusers 구조를 따른다.
**조건** 선택 서브모듈 목록 속성 이름도 upstream 네이밍을 따르고 같은 범주 모듈은 같은 하위 패키지에 둔다.
**근거** [#726](https://github.com/rebellions-sw/optimum-rbln/pull/726) (seinpark): “matching the upstream diffusers layout and our own 'models/{autoencoders,transformers,unets}/' convention.”
　　　[#187](https://github.com/rebellions-sw/optimum-rbln/pull/187) (dkhong): “So, in my opinion, it would be more natural that follow the implementation of 'diffusers'.”
　　　외 PR 2건
**응답** [ ] 맞다 　[ ] 틀리다 　[ ] 조건부로 맞다 　[ ] 필요 없는 룰이다 　[ ] 모른다 　[ ] 기억나지 않는다 　· 메모:

### DIF-002 · 판단형

**룰** 파이프라인 호출 래퍼는 호출자가 준 형상 인자를 덮어쓰지 않고 생략된 경우에만 컴파일된 형상으로 채운다.
**조건** 컴파일된 batch size와 런타임 인자가 맞지 않으면 명시적 에러를 낸다.
**근거** [#58](https://github.com/rebellions-sw/optimum-rbln/pull/58) (dkhong): “compiled batch size와 arguments의 정합성이 맞지 않는 경우 이에 대한 에러를 발생하도록 하였습니다.”
　　　[#54](https://github.com/rebellions-sw/optimum-rbln/pull/54) (dkhong): “compiled batch size와 arguments의 정합성이 맞지 않는 경우 이에 대한 에러를 발생하도록 하였습니다.”
　　　외 PR 1건
**응답** [ ] 맞다 　[ ] 틀리다 　[ ] 조건부로 맞다 　[ ] 필요 없는 룰이다 　[ ] 모른다 　[ ] 기억나지 않는다 　· 메모:

### DIF-007 · 판단형

**룰** 모든 diffusers 모델과 파이프라인은 `rbln_config` 설정 클래스를 갖도록 구성한다.
**조건** API 형식은 다른 모델의 설정 객체와 동일하게 맞춘다.
**근거** [#202](https://github.com/rebellions-sw/optimum-rbln/pull/202) (seinpark): “sync Cosmos model API with other model API format”
　　　[#129](https://github.com/rebellions-sw/optimum-rbln/pull/129) (seinpark): “Appling 'rbln_config' to Stable Video Diffusion”
**응답** [ ] 맞다 　[ ] 틀리다 　[ ] 조건부로 맞다 　[ ] 필요 없는 룰이다 　[ ] 모른다 　[ ] 기억나지 않는다 　· 메모:

### DIF-010 · 판단형

**룰** 새 변종을 추가해도 기존 모델의 컴파일 그래프와 입력 순서는 바이트 단위로 동일하게 유지한다.
**조건** 변종 분기는 사용자 옵션이 아닌 설정 값으로 내부 결정하며, 그래프가 다르면 별도 래퍼 클래스로 분리한다.
**근거** [#694](https://github.com/rebellions-sw/optimum-rbln/pull/694) (seinpark): “separate 'CosmosTransferTransformerWrapper' so the Predict graph stays byte-identical”
　　　[#725](https://github.com/rebellions-sw/optimum-rbln/pull/725) (seinpark): “Predict graphs and their input order stay byte-identical”
**응답** [ ] 맞다 　[ ] 틀리다 　[ ] 조건부로 맞다 　[ ] 필요 없는 룰이다 　[ ] 모른다 　[ ] 기억나지 않는다 　· 메모:

### DIF-023 · 판단형

**룰** 여러 구성요소로 된 diffusers 파이프라인은 컴파일과 로딩을 파이프라인 단위로 지원한다.
**조건** 연결된 파이프라인은 새 구현 없이 단일 파이프라인용 기존 구현을 재사용한다.
**근거** [#39](https://github.com/rebellions-sw/optimum-rbln/pull/39) (dkhong): “reuse existing implementations for supporting a single pipeline structure”
　　　[#15](https://github.com/rebellions-sw/optimum-rbln/pull/15) (dkhong): “Support compiling & loading connected pipelines”
**응답** [ ] 맞다 　[ ] 틀리다 　[ ] 조건부로 맞다 　[ ] 필요 없는 룰이다 　[ ] 모른다 　[ ] 기억나지 않는다 　· 메모:
