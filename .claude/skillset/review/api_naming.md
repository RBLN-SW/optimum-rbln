# 공개 API · 이름 — 룰 10개

결정 필요 0 · 코드 일치 4 · 코드 불일치 0 · 판단형 5 · 폐기 제안 1. 순서는 결정이 필요한 것부터. 각 룰의 **응답** 줄에서 하나를 `[x]`로 표시하고, 틀리거나 조건부면 **메모**에 한 줄 적어 주세요.

## 코드 일치

### API-003 · 코드 일치

**룰** 공개 API, 설정 키, 출력 타입을 제거·변경하는 PR은 같은 PR에서 내부 호출부와 문서를 고친다.
**조건** 하위 클래스, README, examples에서 제거된 이름 참조를 0으로 만들고 vLLM·model-zoo 영향은 PR 본문에 명시한다.
**근거** [#614](https://github.com/rebellions-sw/optimum-rbln/pull/614) (junamsong): “a stale '.load()' call that survived all '0.11.0a*' builds and only raised at the '0.11.0' tag.”
　　　[#104](https://github.com/rebellions-sw/optimum-rbln/pull/104) (jongho): “the current main implementation does not work with the example code specified in model_zoo or the user docs, which used to work.”
　　　외 PR 4건
**코드 대조** 코드 일치 — 준수 162 / 위반 0
**응답** [ ] 맞다 　[ ] 틀리다 　[ ] 조건부로 맞다 　[ ] 필요 없는 룰이다 　[ ] 모른다 　[ ] 기억나지 않는다 　· 메모:

### API-014 · 코드 일치

**룰** 공개 객체는 정의 모듈의 `__all__`에만 등록하고 패키지 `__init__.py`는 lazy 템플릿으로 둔다.
**조건** `scripts/check_inits.py --fix`로 TYPE_CHECKING 블록을 맞추며, 사용자 정의 클래스만 `register`를 쓴다.
**근거** [#734](https://github.com/rebellions-sw/optimum-rbln/pull/734) (jongho): “the module that defines a public object lists it in '__all__', and the package '__init__.py' is a fixed template.”
　　　[#533](https://github.com/rebellions-sw/optimum-rbln/pull/533) (ChanyongJeon): “Register both models in 'optimum.rbln' transformers model registry ('__init__.py' exports).”
　　　외 PR 1건
**코드 대조** 코드 일치 — 준수 305 / 위반 3
**응답** [ ] 맞다 　[ ] 틀리다 　[ ] 조건부로 맞다 　[ ] 필요 없는 룰이다 　[ ] 모른다 　[ ] 기억나지 않는다 　· 메모:

### API-015 · 코드 일치

**룰** 내부 모듈을 옮길 때 옛 경로에 호환용 파일(shim)을 남기지 않고, 모델 클래스는 그 모델의 디렉터리에만 둔다.
**조건** 사용자는 from optimum.rbln import로만 쓰므로 최상위 export만 유지한다. 여러 모델 클래스를 모은 공용 alias 파일은 만들지 않는다.
**근거** [#157](https://github.com/rebellions-sw/optimum-rbln/pull/157) (jongho): “create a directory structure for each model and put the alias class in there (delete the alias file)”
　　　[#17](https://github.com/rebellions-sw/optimum-rbln/pull/17) (seinpark): “클래스를 구현할 수 있는 위치가 중복되어 오히려 구현에 혼동을 줄 수 있음”
　　　외 PR 1건
**코드 대조** 코드 일치 — 준수 125 / 위반 1
**응답** [ ] 맞다 　[ ] 틀리다 　[ ] 조건부로 맞다 　[ ] 필요 없는 룰이다 　[ ] 모른다 　[ ] 기억나지 않는다 　· 메모:

### API-011 · 코드 일치

**룰** 공개 모델 클래스는 `RBLN` + 대응 HF 클래스명으로, 설정은 그 뒤에 `Config`를 붙여 짓는다.
**조건** 자동 클래스가 이 이름으로 클래스를 찾으므로 이름 규칙이 곧 로딩 계약이다.
**근거** [#15](https://github.com/rebellions-sw/optimum-rbln/pull/15) (dkhong): “Add 'RBLNVQModel', 'RBLNPriorTransformer', and 'RBLNCLIPVisionModelWithProjection' to support the pipeline”
　　　[#533](https://github.com/rebellions-sw/optimum-rbln/pull/533) (ChanyongJeon): “consistent with other 'RBLN*For*' model integrations”
**코드 대조** 코드 일치 — 준수 132 / 위반 1
**응답** [ ] 맞다 　[ ] 틀리다 　[ ] 조건부로 맞다 　[ ] 필요 없는 룰이다 　[ ] 모른다 　[ ] 기억나지 않는다 　· 메모:

## 판단형

### API-007 · 판단형

**룰** 함수, 메서드, 속성 이름은 실제 동작과 역할을 정확히 나타내야 한다.
**조건** 모호한 약어나 HF 용어와 어긋나는 이름은 바꾸고 여러 역할을 한 메서드에 묶지 않는다.
**근거** [#248](https://github.com/rebellions-sw/optimum-rbln/pull/248) (junamsong): “rather than the hardware vendor itself.”
　　　[#433](https://github.com/rebellions-sw/optimum-rbln/pull/433) (seinpark): “I think '_update_submodule_rbln_config' and '_update_submodule_config' is too confusing to understand.”
　　　외 PR 2건
**응답** [ ] 맞다 　[ ] 틀리다 　[ ] 조건부로 맞다 　[ ] 필요 없는 룰이다 　[ ] 모른다 　[ ] 기억나지 않는다 　· 메모:

### API-008 · 판단형

**룰** 모델 클래스 이름, 연산 순서, 메서드 구현 순서와 래퍼 메서드 시그니처는 upstream 원본을 따른다.
**조건** upstream은 transformers, vLLM 등이다.
**근거** [#168](https://github.com/rebellions-sw/optimum-rbln/pull/168) (jongho): “the interface of RBLN safety checker seems to be different from the original cosmos safety checker class.”
　　　[#797](https://github.com/rebellions-sw/optimum-rbln/pull/797) (seinpark): “class GraniteDecoderLayer(DecoderOnlyLayer):”
　　　외 PR 2건
**응답** [ ] 맞다 　[ ] 틀리다 　[ ] 조건부로 맞다 　[ ] 필요 없는 룰이다 　[ ] 모른다 　[ ] 기억나지 않는다 　· 메모:

### API-006 · 판단형

**룰** 함수 인자는 `*args`/`**kwargs`나 위치 인덱싱에 기대지 않고 명시적 인자로 받는다.
**조건** 호출은 key=value 형식으로 쓴다.
**근거** [#226](https://github.com/rebellions-sw/optimum-rbln/pull/226) (thkim): “How about use explicit arg rather than implicit args & kwargs?”
　　　[#557](https://github.com/rebellions-sw/optimum-rbln/pull/557) (jongho): “How about using 'key=value' scheme for overall code?”
　　　외 PR 1건
**응답** [ ] 맞다 　[ ] 틀리다 　[ ] 조건부로 맞다 　[ ] 필요 없는 룰이다 　[ ] 모른다 　[ ] 기억나지 않는다 　· 메모:

### API-009 · 판단형

**룰** 설정 필드 이름과 상속 구조는 공통 기반 설정 클래스의 이름과 맞춘다.
**조건** dict로 전달되는 하위 모듈 설정 이름은 config 속성 이름과 일치시킨다.
**근거** [#260](https://github.com/rebellions-sw/optimum-rbln/pull/260) (thkim): “How about using image_size instead of max_image_size?”
　　　[#98](https://github.com/rebellions-sw/optimum-rbln/pull/98) (jongho): “I'm concerned that the name of the submodule config passed as a dict (model.text_model) is different from the name of the attribu…”
　　　외 PR 1건
**응답** [ ] 맞다 　[ ] 틀리다 　[ ] 조건부로 맞다 　[ ] 필요 없는 룰이다 　[ ] 모른다 　[ ] 기억나지 않는다 　· 메모:

### API-002 · 판단형

**룰** 폐기된 API는 폐기 버전에서 한 릴리스 동안 오류를 낸 뒤 다음 릴리스에서 제거한다.
**조건** 차단 시점은 프리릴리스·포스트릴리스 접미사와 무관하게 기본 버전 X.Y.Z로 비교한다.
**근거** [#645](https://github.com/rebellions-sw/optimum-rbln/pull/645) (thkim): “raise an error for one release at the deprecated version, then remove in the following release”
　　　[#614](https://github.com/rebellions-sw/optimum-rbln/pull/614) (junamsong): “so any release on the same 'X.Y.Z' line”
**응답** [ ] 맞다 　[ ] 틀리다 　[ ] 조건부로 맞다 　[ ] 필요 없는 룰이다 　[ ] 모른다 　[ ] 기억나지 않는다 　· 메모:

## 폐기 제안

### API-019 · 폐기 제안

**룰** 설정 입력 경로 사이에 암묵적 우선순위를 두어 한쪽이 다른 쪽을 덮어쓰게 하지 않는다.
**근거** [#445](https://github.com/rebellions-sw/optimum-rbln/pull/445) (seinpark): “This behavior could be confusing for users.”
**응답** [ ] 맞다 　[ ] 틀리다 　[ ] 조건부로 맞다 　[ ] 필요 없는 룰이다 　[ ] 모른다 　[ ] 기억나지 않는다 　· 메모:
