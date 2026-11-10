# optimum-rbln 코드 리뷰 코멘트에서 추출한 암묵적 코딩 규칙

입력: `corpus/review_comments.jsonl` (인라인 리뷰 코멘트 1,037건, PR 184개, 작성자 17명(Copilot 봇 포함)). 전체 본문을 읽고 같은 요구(ask)끼리 묶었다. 인용문은 원문 그대로이며 스크립트가 원문 부분 문자열인지 검증했다.

- 신뢰도: `high` = 3개 이상 PR, `medium` = 2개 PR, `low` = 1개 PR이지만 일반 원칙으로 말한 경우.
- Copilot 봇 코멘트는 근거로 쓰지 않았다(사람 리뷰어가 되풀이한 경우만 고려).
- 리뷰어 집계는 각 클러스터에서 인용문(최대 3개)으로 등장한 횟수다. 전체 코멘트 수가 많은 리뷰어가 유리하다.

## 클러스터 목록

| # | area | kind | confidence | rule | #PRs |
|:---:|:---|:---:|:---:|:---|:---:|
| 1 | rbln_config | judgment | high | 지원하지 않는 설정이나 잘못된 입력은 경고만 남기고 다른 값으로 대체하거나 조용히 무시하지 않고, 명시적 예외(ValueError/NotImplementedError 등)로 실패시킨다. | 11 |
| 2 | style/other | judgment | high | 의도가 바로 드러나지 않는 추가·삭제·순서 변경·dtype 변경 같은 수정에는 이유를 주석이나 PR 설명에 남긴다. | 10 |
| 3 | style/other | judgment | high | 디버깅용 코드, 개발 중 로그와 주석, 예제 스크립트 속 내부 검증 코드 같은 개발 흔적은 PR에 남기지 않는다. | 9 |
| 4 | docstring/docs | mechanical | high | 새로 추가하는 모델·설정·파이프라인 클래스에는 클래스 docstring을 둔다(무엇을 위한 클래스인지, 가능하면 사용 예시 포함). | 8 |
| 5 | dependency/version | mechanical | high | deprecated 인자·API는 deprecate 데코레이터로 표시하고, 제거 예정 버전을 명시하며, 경고는 한 번만 출력하고, 해당 버전이 되면 제거한다. | 8 |
| 6 | docstring/docs | judgment | high | 주석은 코드가 말하지 않는 이유만 짧게 적고, 코드를 그대로 풀어 설명하거나 PR 이력·마이그레이션 맥락을 적지 않는다. 내부 구현 설명은 docstring(""") 대신 # 주석을 쓴다. | 8 |
| 7 | modeling_base/lifecycle | judgment | high | forward는 HF 규약에 맞춰 return_dict 인자를 받아 처리하고, dict/tuple 반환 형식은 transformers와 같은 방식을 따른다. | 8 |
| 8 | modeling_base/lifecycle | judgment | high | 부모 클래스가 이미 제공하는 구현은 override하거나 복제하지 않고 상속해서 쓰며, 실제로 다른 부분만 바꾼다. | 7 |
| 9 | style/other | judgment | high | 같은 로직·상수·결정 지점은 한 곳에만 둔다. 여러 곳에 복제된 함수나 값(NPU 판별, 어텐션 한도, 파이프라인 공통 검증 등)은 공용 헬퍼나 한 곳의 상수로 모으고, 값의 결정은 한 번만 수행한다. | 7 |
| 10 | style/other | mechanical | high | 사용하지 않는 코드(미사용 변수·import·메서드·클래스, 부모와 동일한 override)는 제거한다. | 7 |
| 11 | style/other | mechanical | high | 새로 추가하는 소스 파일에는 기존 파일과 같은 형식의 Rebellions 저작권 헤더를 현재 연도로 붙인다. | 6 |
| 12 | api/naming | judgment | high | 함수 인자는 *args/**kwargs나 위치 인덱싱에 기대지 않고 명시적 인자로 받으며, 호출은 key=value 형식으로 쓴다. | 6 |
| 13 | style/other | judgment | high | PR의 목적과 무관한 수정(다른 모델의 정리, 공통 로직 리팩터링 등)은 별도 PR로 나눈다. | 6 |
| 14 | style/other | judgment | high | 오류 메시지는 무엇이 잘못됐는지, 어떤 조건인지, 사용자가 무엇을 해야 하는지(재컴파일, 다른 패키지 사용 등)를 구체적으로 적는다. | 6 |
| 15 | api/naming | judgment | high | 함수·속성 이름은 실제 동작을 정확히 나타내야 하며, 모호한 약어(mod 등)나 HF 용어(model_type 등)와 의미가 어긋나면 이름을 바꾼다. | 5 |
| 16 | api/naming | judgment | high | 모델 클래스 이름, 연산 순서, 메서드 구현 순서는 upstream(transformers, vLLM 등)의 명명과 순서를 따른다. | 5 |
| 17 | testing | judgment | high | 테스트는 CI 워크플로우가 실제로 실행하는 파일에 두고, 기대값은 실제 모델 출력으로 검증하며, 실패를 가리는 우회 줄을 남기지 않는다. 중복 테스트는 하나로 합친다. | 5 |
| 18 | style/other | judgment | high | decode 등 추론 경로에서는 매 스텝 텐서를 새로 할당하거나 host와 동기화하거나 불필요하게 복사하지 않는다. 마스크는 미리 만들어 재사용하고 broadcast 연산을 쓴다. | 5 |
| 19 | style/other | mechanical | high | import는 파일 상단에 두고 지연 import를 쓰지 않으며, 불필요한 import와 그에 대한 lint 억제 주석(noqa)은 제거한다. | 4 |
| 20 | multimodal/submodule | judgment | high | submodule 설정의 우선순위(사용자가 submodule에 명시한 값 > 상위(전역) 설정 > 기본값)를 한 곳에서 정의하고, 충돌은 강제되는 키에 한해 예외로 처리한다. | 4 |
| 21 | style/other | mechanical | high | 함수 시그니처에는 타입 힌트를 명시하고, 힌트는 실제로 전달되는 객체와 일치시키며, Optional 표기는 파일 안에서 하나로 통일한다. | 4 |
| 22 | rbln_config | judgment | high | rbln_config에는 컴파일 그래프를 바꾸는 값만 두고, device처럼 런타임에서만 의미가 있는 값은 runtime 옵션으로 분리한다. 설정을 dict로 넘기든 설정 객체로 넘기든 두 종류가 섞이지 않아야 한다. | 4 |
| 23 | style/other | mechanical | high | __init__ 목록, export 목록, 키 목록 같은 나열형 목록은 정렬된 순서를 유지한다. | 4 |
| 24 | docstring/docs | judgment | high | docstring의 인자 목록, 기본값, 텐서 shape, 예시 클래스 이름은 실제 시그니처와 구현에 맞춘다. | 4 |
| 25 | style/other | mechanical | high | 로그와 경고는 built-in logging/warnings 대신 optimum-rbln의 utils.logging을 사용한다. | 4 |
| 26 | modeling_base/lifecycle | judgment | high | 파생 가능한 값은 멤버 변수로 중복 저장하지 않고 property로 계산하며, 같은 상태를 여러 곳에서 갱신하지 않는다. | 4 |
| 27 | style/other | judgment | high | 상위 계층에서 이미 처리하는 검사나 기본값 처리를 중복해서 추가하지 않으며, 과한 방어 코드(over-engineering)를 피한다. | 4 |
| 28 | dependency/version | judgment | high | transformers 등 의존 라이브러리 버전에 따라 달라지는 동작은 검증한 버전을 명시하고, 차이가 나면 분기나 주석으로 남긴다. | 4 |
| 29 | dependency/version | judgment | high | 의존성 버전은 검증된 값으로 고정하고(검증 후 상향), 필수 기능이 아닌 패키지는 기본 의존성에 넣지 않으며, 릴리스 설정 파일은 production 값으로 맞춘다. | 4 |
| 30 | docstring/docs | judgment | high | docstring과 Args 항목에서 **kwargs는 kwargs:로 표기하고, rbln_ 접두사 인자가 어디로 전달되는지 적는다. | 3 |
| 31 | rbln_config | judgment | high | 설정 클래스에는 실제 동작에 반영되는 옵션만 노출하고, 무시되거나 지원하지 않는 옵션은 제거하거나 명시적으로 거부한다. | 3 |
| 32 | modeling_base/lifecycle | judgment | high | 클래스 고유 특성(지원 여부, attention 종류, 플래그 등)은 호출 인자로 흘리지 않고 클래스 속성이나 config 속성으로 선언한다. | 3 |
| 33 | rbln_config | judgment | high | 기본값과 크기 같은 값은 코드에 직접 하드코딩하지 않고 모델 config, 전처리기, 기존 rbln_config 속성에서 유도한다. | 3 |
| 34 | docstring/docs | judgment | high | 근거가 바로 보이지 않는 상수나 버퍼 값은 산출 근거(계산식, 경험적 값, 보수적 여유분 등)를 코드 주석이나 PR 설명에 남긴다. | 3 |
| 35 | api/naming | judgment | high | 설정 필드 이름과 상속 구조는 공통 기반 설정 클래스(예: 이미지 설정의 image_size)와 맞춘다. | 3 |
| 36 | rbln_config | judgment | high | 검증 함수는 검증만 수행하고, 기본값 설정·패딩·config 변경은 지정된 훅(예: set_default_values)이나 생성 시점에서 한다. | 3 |
| 37 | modeling_base/lifecycle | mechanical | high | monkey patch로 전역 함수나 메서드를 바꿔 쓴 경우 실행 뒤 원래대로 복원하며, 컴파일이 실패해도 복원되도록 try/finally로 감싼다. | 3 |
| 38 | diffusers | judgment | medium | 컴파일 시점 옵션은 그래프를 간접적으로 바꾸는 값(예: guidance scale)이 아니라 그래프 형상을 직접 정하는 값(예: submodule batch size)으로 명시한다. | 2 |
| 39 | dependency/version | mechanical | medium | torch.load는 weights_only를 명시적으로 지정한다(PyTorch 2.6부터 기본값이 바뀌었다). | 2 |
| 40 | rbln_config | judgment | medium | 설정 포맷이나 산출물 구조가 바뀌면 기존 산출물 로드 실패를 명확한 메시지로 알리고, 하위호환 영향을 확인한다. | 2 |
| 41 | style/other | mechanical | medium | CI 워크플로우에서 시크릿은 secrets 컨텍스트로 전달하고, 내부 엔드포인트를 평문으로 적지 않는다. | 2 |
| 42 | style/other | judgment | low | 범용 유틸리티는 utils/에 두고, 특정 모델 전용 로직은 해당 모델 디렉터리에 둔다. | 1 |

## 클러스터 상세

### 1. C01 · rbln_config · high

**규칙**: 지원하지 않는 설정이나 잘못된 입력은 경고만 남기고 다른 값으로 대체하거나 조용히 무시하지 않고, 명시적 예외(ValueError/NotImplementedError 등)로 실패시킨다.

- kind: `judgment`
- 관련 PR(11): #88, #96, #114, #118, #136, #177, #259, #267, #631, #673, #729
- 관련 코멘트 수: 13

인용:
- PR #114 · rebel-jongho: `No implicit change, Raise error instead.`
- PR #729 · rebel-junamsong: `An unrecognised `npu` silently gets ATOM's wider limits.`
- PR #118 · rebel-jongho: `exception should be raised(`main`) instead of skipping (`this branch`).`

**열린 질문**: dtype 등 미지원 입력을 예외 대신 fp32로 자동 승격(fallback)해도 되는지, 아니면 예외로 막아야 하는지 기준이 필요한가?

### 2. C23 · style/other · high

**규칙**: 의도가 바로 드러나지 않는 추가·삭제·순서 변경·dtype 변경 같은 수정에는 이유를 주석이나 PR 설명에 남긴다.

- kind: `judgment`
- 관련 PR(10): #12, #15, #50, #98, #135, #194, #383, #436, #631, #673
- 관련 코멘트 수: 10

인용:
- PR #135 · rebel-jongho: `Could you add a comment that describes why we remove inputs_embeds`
- PR #50 · rebel-jongho: `이건 왜 변경되나요?`
- PR #383 · rebel-jongho: `Why re-ordering is needed?`

**열린 질문**: 없음

### 3. C12 · style/other · high

**규칙**: 디버깅용 코드, 개발 중 로그와 주석, 예제 스크립트 속 내부 검증 코드 같은 개발 흔적은 PR에 남기지 않는다.

- kind: `judgment`
- 관련 PR(9): #15, #75, #109, #136, #162, #249, #260, #356, #416
- 관련 코멘트 수: 10

인용:
- PR #249 · rebel-jongho: `No debugging code for examples`
- PR #260 · rebel-thkim: `delete debugging scripts.`
- PR #136 · rebel-jongho: `Example script should not include internal validation code.`

**열린 질문**: 없음

### 4. C15 · docstring/docs · high

**규칙**: 새로 추가하는 모델·설정·파이프라인 클래스에는 클래스 docstring을 둔다(무엇을 위한 클래스인지, 가능하면 사용 예시 포함).

- kind: `mechanical`
- 관련 PR(8): #98, #106, #168, #177, #218, #249, #363, #495
- 관련 코멘트 수: 10

인용:
- PR #218 · rebel-jongho: `Please add class docstring`
- PR #177 · rebel-jongho: `Add class docstring. if available, please add example code, too.`
- PR #363 · rebel-jongho: `No docstring for this config?`

**열린 질문**: 없음

### 5. C25 · dependency/version · high

**규칙**: deprecated 인자·API는 deprecate 데코레이터로 표시하고, 제거 예정 버전을 명시하며, 경고는 한 번만 출력하고, 해당 버전이 되면 제거한다.

- kind: `mechanical`
- 관련 PR(8): #228, #389, #473, #481, #599, #631, #682, #796
- 관련 코멘트 수: 9

인용:
- PR #481 · rebel-dkhong: `Don't we have to guide the last version where this method will be supported?`
- PR #682 · rebel-thkim: `I think we'll need to add the `deprecate_kwargs` decorator.`
- PR #631 · rebel-jongho: `already exceeded version 0.11`

**열린 질문**: 없음

### 6. C18 · docstring/docs · high

**규칙**: 주석은 코드가 말하지 않는 이유만 짧게 적고, 코드를 그대로 풀어 설명하거나 PR 이력·마이그레이션 맥락을 적지 않는다. 내부 구현 설명은 docstring(""") 대신 # 주석을 쓴다.

- kind: `judgment`
- 관련 PR(8): #160, #563, #630, #631, #673, #729, #735, #743
- 관련 코멘트 수: 8

인용:
- PR #631 · rebel-thkim: `Some of the comments seem unnecessarily long.`
- PR #735 · rebel-seinpark: `Please minimize unnecessary comments that arise during the development.`
- PR #630 · rebel-jongho: `it explains the PR's migration context, not`

**열린 질문**: 없음

### 7. C30 · modeling_base/lifecycle · high

**규칙**: forward는 HF 규약에 맞춰 return_dict 인자를 받아 처리하고, dict/tuple 반환 형식은 transformers와 같은 방식을 따른다.

- kind: `judgment`
- 관련 PR(8): #38, #98, #114, #133, #134, #219, #226, #631
- 관련 코멘트 수: 8

인용:
- PR #38 · rebel-jongho: `Please implement `return_dict``
- PR #114 · rebel-jongho: `Please handle `return_dict``
- PR #631 · rebel-thkim: `Is the fact that `return_dict=True` is hardcoded consistent with how Transformers work?`

**열린 질문**: return_dict를 항상 True로 고정하는 구현이 HF 동작과 맞는지 기준을 확인해야 하는가?

### 8. C14 · modeling_base/lifecycle · high

**규칙**: 부모 클래스가 이미 제공하는 구현은 override하거나 복제하지 않고 상속해서 쓰며, 실제로 다른 부분만 바꾼다.

- kind: `judgment`
- 관련 PR(7): #112, #118, #158, #219, #253, #326, #631
- 관련 코멘트 수: 10

인용:
- PR #112 · rebel-jongho: `I think this is not necessary since this class already inherits DecoderOnlyModel`
- PR #253 · rebel-thkim: `How about fix the `RBLNTransformerEncoder`'s method instead of overriding it?`
- PR #118 · rebel-jongho: `Llama is already implemented in optimum-rbln, what is the purpose of this class?`

**열린 질문**: 없음

### 9. C02 · style/other · high

**규칙**: 같은 로직·상수·결정 지점은 한 곳에만 둔다. 여러 곳에 복제된 함수나 값(NPU 판별, 어텐션 한도, 파이프라인 공통 검증 등)은 공용 헬퍼나 한 곳의 상수로 모으고, 값의 결정은 한 번만 수행한다.

- kind: `judgment`
- 관련 PR(7): #41, #54, #64, #133, #228, #673, #729
- 관련 코멘트 수: 8

인용:
- PR #673 · rebel-jongho: `the same function is on multiple sites.`
- PR #228 · rebel-jongho: `set_compile_cfgs is the one and only point of determining npu.`
- PR #54 · rebel-jongho: `비슷한 코드가 곳곳에 있는 것 같은데 합칠 수 있나요?`

**열린 질문**: 파이프라인 간 공통 검증 코드를 mixin으로 둘지 utils 모듈로 둘지 위치를 정해야 하는가?

### 10. C13 · style/other · high

**규칙**: 사용하지 않는 코드(미사용 변수·import·메서드·클래스, 부모와 동일한 override)는 제거한다.

- kind: `mechanical`
- 관련 PR(7): #80, #96, #158, #177, #631, #760, #761
- 관련 코멘트 수: 8

인용:
- PR #96 · rebel-jongho: `nit: unused codes`
- PR #177 · rebel-jongho: `can be deleted.`
- PR #761 · rebel-jongho: `was it just dead code?`

**열린 질문**: 없음

### 11. C27 · style/other · high

**규칙**: 새로 추가하는 소스 파일에는 기존 파일과 같은 형식의 Rebellions 저작권 헤더를 현재 연도로 붙인다.

- kind: `mechanical`
- 관련 PR(6): #38, #471, #500, #631, #734, #798
- 관련 코멘트 수: 8

인용:
- PR #38 · rebel-jongho: `다른 파일들 라이센스 주석 참고해서 똑같이 맞춰주세요`
- PR #471 · rebel-jongho: `# Copyright 2026 Rebellions Inc. All rights reserved.`
- PR #798 · rebel-seinpark: `# Copyright 2026 Rebellions Inc. All rights reserved.`

**열린 질문**: 없음

### 12. C09 · api/naming · high

**규칙**: 함수 인자는 *args/**kwargs나 위치 인덱싱에 기대지 않고 명시적 인자로 받으며, 호출은 key=value 형식으로 쓴다.

- kind: `judgment`
- 관련 PR(6): #133, #209, #219, #226, #256, #557
- 관련 코멘트 수: 7

인용:
- PR #226 · rebel-thkim: `How about use explicit arg rather than implicit args & kwargs?`
- PR #557 · rebel-jongho: `How about using `key=value` scheme for overall code?`
- PR #256 · rebel-jongho: `Use explicit keywords`

**열린 질문**: 없음

### 13. C17 · style/other · high

**규칙**: PR의 목적과 무관한 수정(다른 모델의 정리, 공통 로직 리팩터링 등)은 별도 PR로 나눈다.

- kind: `judgment`
- 관련 PR(6): #15, #38, #133, #162, #444, #673
- 관련 코멘트 수: 6

인용:
- PR #162 · rebel-thkim: `This is not relevant to this PR and I think it's better to create a separate PR for it.`
- PR #673 · rebel-thkim: `Isn't this also unrelated to FP32? If necessary, it seems better to create a separate PR.`
- PR #38 · rebel-seinpark: `I will create a new PR for handing this.`

**열린 질문**: 없음

### 14. C24 · style/other · high

**규칙**: 오류 메시지는 무엇이 잘못됐는지, 어떤 조건인지, 사용자가 무엇을 해야 하는지(재컴파일, 다른 패키지 사용 등)를 구체적으로 적는다.

- kind: `judgment`
- 관련 PR(6): #56, #88, #256, #259, #345, #648
- 관련 코멘트 수: 6

인용:
- PR #345 · rebel-jongho: `Error message should be more concrete.`
- PR #256 · rebel-jongho: `Add an kind error message that handles if there is no key `_class_name``
- PR #648 · rebel-eunji: `sounds a bit awkward. How about we take this opportunity to rephrase it?`

**열린 질문**: 없음

### 15. C21 · api/naming · high

**규칙**: 함수·속성 이름은 실제 동작을 정확히 나타내야 하며, 모호한 약어(mod 등)나 HF 용어(model_type 등)와 의미가 어긋나면 이름을 바꾼다.

- kind: `judgment`
- 관련 PR(5): #162, #248, #433, #437, #444
- 관련 코멘트 수: 7

인용:
- PR #433 · rebel-seinpark: `I think `_update_submodule_rbln_config` and `_update_submodule_config` is too confusing to understand.`
- PR #162 · rebel-junamsong: `The abbreviation mod is ambiguous and could be misinterpreted`
- PR #248 · rebel-junamsong: `rather than the hardware vendor itself.`

**열린 질문**: 없음

### 16. C20 · api/naming · high

**규칙**: 모델 클래스 이름, 연산 순서, 메서드 구현 순서는 upstream(transformers, vLLM 등)의 명명과 순서를 따른다.

- kind: `judgment`
- 관련 PR(5): #162, #629, #797, #801, #802
- 관련 코멘트 수: 6

인용:
- PR #802 · rebel-seinpark: `Please follow the upstream order`
- PR #797 · rebel-seinpark: `class GraniteDecoderLayer(DecoderOnlyLayer):`
- PR #162 · rebel-jongho: `I think `cache_implementation` would be good enough.`

**열린 질문**: 없음

### 17. C26 · testing · high

**규칙**: 테스트는 CI 워크플로우가 실제로 실행하는 파일에 두고, 기대값은 실제 모델 출력으로 검증하며, 실패를 가리는 우회 줄을 남기지 않는다. 중복 테스트는 하나로 합친다.

- kind: `judgment`
- 관련 PR(5): #98, #117, #180, #475, #722
- 관련 코멘트 수: 6

인용:
- PR #722 · rebel-jongho: `please remove this line so that we can recognize if this test goes wrong`
- PR #475 · rebel-jongho: `the tests added to test_transformers and test_llm seem to perform the same function.`
- PR #180 · rebel-jongho: `are outputs of pytest, not llm's outputs.`

**열린 질문**: 없음

### 18. C35 · style/other · high

**규칙**: decode 등 추론 경로에서는 매 스텝 텐서를 새로 할당하거나 host와 동기화하거나 불필요하게 복사하지 않는다. 마스크는 미리 만들어 재사용하고 broadcast 연산을 쓴다.

- kind: `judgment`
- 관련 PR(5): #563, #604, #631, #673, #714
- 관련 코멘트 수: 6

인용:
- PR #631 · rebel-jongho: `or build them once in `__init__` and reuse them`
- PR #563 · rebel-thkim: `attn_mask 4d is too slow. You must use 2d attn_mask as input and use brocating ops`
- PR #604 · rebel-eunji: `tensor.sum().item()보다는 이게 저비용이지 않을까요?`

**열린 질문**: 없음

### 19. C11 · style/other · high

**규칙**: import는 파일 상단에 두고 지연 import를 쓰지 않으며, 불필요한 import와 그에 대한 lint 억제 주석(noqa)은 제거한다.

- kind: `mechanical`
- 관련 PR(4): #15, #38, #445, #653
- 관련 코멘트 수: 7

인용:
- PR #445 · rebel-jongho: `could you move this `import` to the top`
- PR #653 · rebel-jongho: `why lazy import`
- PR #15 · rebel-jongho: `relative import로 수정해주세요`

**열린 질문**: 없음

### 20. C07 · multimodal/submodule · high

**규칙**: submodule 설정의 우선순위(사용자가 submodule에 명시한 값 > 상위(전역) 설정 > 기본값)를 한 곳에서 정의하고, 충돌은 강제되는 키에 한해 예외로 처리한다.

- kind: `judgment`
- 관련 PR(4): #15, #267, #433, #445
- 관련 코멘트 수: 6

인용:
- PR #267 · rebel-thkim: `Shouldn't the submodule_config specified by the user take precedence over the default given by the parent?`
- PR #433 · rebel-jongho: `Can you explain those priority order`
- PR #445 · rebel-jongho: `since we don't propagate compile options to its submodule.`

**열린 질문**: top-level 컴파일 옵션(예: batch_size)을 submodule로 전파할지 정책을 확정해야 하는가?

### 21. C31 · style/other · high

**규칙**: 함수 시그니처에는 타입 힌트를 명시하고, 힌트는 실제로 전달되는 객체와 일치시키며, Optional 표기는 파일 안에서 하나로 통일한다.

- kind: `mechanical`
- 관련 PR(4): #118, #162, #267, #735
- 관련 코멘트 수: 6

인용:
- PR #118 · rebel-jongho: `Could you add type hints?`
- PR #735 · rebel-jongho: `please use type hint in consistently ( No `Optional` )`
- PR #267 · rebel-thkim: `Type `language_projection` is `REBEL.runtime` rather than `RBLNModel`.`

**열린 질문**: 없음

### 22. C06 · rbln_config · high

**규칙**: rbln_config에는 컴파일 그래프를 바꾸는 값만 두고, device처럼 런타임에서만 의미가 있는 값은 runtime 옵션으로 분리한다. 설정을 dict로 넘기든 설정 객체로 넘기든 두 종류가 섞이지 않아야 한다.

- kind: `judgment`
- 관련 PR(4): #67, #475, #495, #702
- 관련 코멘트 수: 5

인용:
- PR #67 · rebel-seinpark: ``passed_rbln_config._runtime_options` in `RBLNAutoConfig` will raise the error.`
- PR #475 · rebel-seinpark: `we cannot guarantee all the rbln_config is converted to rbln_ prefix.`
- PR #702 · rebel-seinpark: `a passed config object is a carrier of runtime overrides`

**열린 질문**: runtime 전용 옵션(device 등)을 rbln_config 객체에 담을지, from_pretrained 인자로만 받을지 정책을 확정해야 하는가?

### 23. C16 · style/other · high

**규칙**: __init__ 목록, export 목록, 키 목록 같은 나열형 목록은 정렬된 순서를 유지한다.

- kind: `mechanical`
- 관련 PR(4): #38, #50, #256, #631
- 관련 코멘트 수: 5

인용:
- PR #50 · rebel-kblee: `사소한거지만 e,f,g 순서...?`
- PR #256 · rebel-jongho: `nit: sort`
- PR #631 · rebel-jongho: `please keep the list sorted`

**열린 질문**: 없음

### 24. C19 · docstring/docs · high

**규칙**: docstring의 인자 목록, 기본값, 텐서 shape, 예시 클래스 이름은 실제 시그니처와 구현에 맞춘다.

- kind: `judgment`
- 관련 PR(4): #38, #308, #360, #363
- 관련 코멘트 수: 5

인용:
- PR #38 · rebel-dkhong: `These 2 variables are not in the arguments.`
- PR #363 · rebel-dkhong: `pixel_values (`torch.FloatTensor` of shape `(batch_size, channels, height, width)`):`
- PR #308 · rebel-junamsong: `(e.g., `RBLNMidmLMHeadModelConfig` for Mi:dm models)`

**열린 질문**: 없음

### 25. C28 · style/other · high

**규칙**: 로그와 경고는 built-in logging/warnings 대신 optimum-rbln의 utils.logging을 사용한다.

- kind: `mechanical`
- 관련 PR(4): #2, #15, #38, #96
- 관련 코멘트 수: 5

인용:
- PR #2 · rebel-jongho: `built-in warning 대신 utils.logging.py 를 사용해야 합니다.`
- PR #15 · rebel-jongho: `built-in logging 대신 optimum-rbln 의 logging을 사용해야 합니다.`
- PR #38 · rebel-jongho: `native logging 대신 library에 있는 utils.logging 사용해주세요`

**열린 질문**: 없음

### 26. C38 · modeling_base/lifecycle · high

**규칙**: 파생 가능한 값은 멤버 변수로 중복 저장하지 않고 property로 계산하며, 같은 상태를 여러 곳에서 갱신하지 않는다.

- kind: `judgment`
- 관련 PR(4): #75, #81, #219, #735
- 관련 코멘트 수: 5

인용:
- PR #219 · rebel-jongho: `how about using `@property` instead of multiple assigning??`
- PR #735 · rebel-jongho: `please use property + cache(if needed) instead to prevent inconsist values`
- PR #81 · rebel-jongho: `멤버 변수로 가질 필요 없지 않을까요?`

**열린 질문**: 없음

### 27. C32 · style/other · high

**규칙**: 상위 계층에서 이미 처리하는 검사나 기본값 처리를 중복해서 추가하지 않으며, 과한 방어 코드(over-engineering)를 피한다.

- kind: `judgment`
- 관련 PR(4): #356, #376, #389, #437
- 관련 코멘트 수: 4

인용:
- PR #389 · rebel-jongho: `I think it's over-engineering.`
- PR #376 · rebel-thkim: `So adding a rbln_config = {} if rbln_config is None else rbln_config would be redundant.`
- PR #356 · rebel-jongho: `return empty dict when no compile_cfgs, so that we do not need any code changes here`

**열린 질문**: 입력 정규화(dtype 등)를 방어적으로 둘지 최소한으로 둘지 기준이 필요한가?

### 28. C33 · dependency/version · high

**규칙**: transformers 등 의존 라이브러리 버전에 따라 달라지는 동작은 검증한 버전을 명시하고, 차이가 나면 분기나 주석으로 남긴다.

- kind: `judgment`
- 관련 PR(4): #97, #134, #162, #389
- 관련 코멘트 수: 4

인용:
- PR #134 · rebel-seinpark: `From `transformers==4.51.0`, they don't use `return_dict=False` to input argument.`
- PR #162 · rebel-thkim: `From transformers 4.51.0 onwards, the previous method gets an error`
- PR #97 · rebel-junamsong: `we've already verified that models other than Whisper are functioning correctly in transformers 4.50.3.`

**열린 질문**: 없음

### 29. C34 · dependency/version · high

**규칙**: 의존성 버전은 검증된 값으로 고정하고(검증 후 상향), 필수 기능이 아닌 패키지는 기본 의존성에 넣지 않으며, 릴리스 설정 파일은 production 값으로 맞춘다.

- kind: `judgment`
- 관련 PR(4): #8, #38, #168, #549
- 관련 코멘트 수: 4

인용:
- PR #8 · rebel-jongho: `it would be nice to go with model-zoo validation. I'll revert torch version`
- PR #168 · rebel-jongho: `cosmos guardrail is not a necessary dependency`
- PR #549 · rebel-junamsong: `Could you please update the compiler configuration to production?`

**열린 질문**: 없음

### 30. C36 · docstring/docs · high

**규칙**: docstring과 Args 항목에서 **kwargs는 kwargs:로 표기하고, rbln_ 접두사 인자가 어디로 전달되는지 적는다.

- kind: `judgment`
- 관련 PR(3): #270, #277, #367
- 관련 코멘트 수: 7

인용:
- PR #270 · rebel-seinpark: `**kwargs -> kwargs`
- PR #367 · rebel-seinpark: `Is it okay if there is no mention of `kwargs`?`
- PR #277 · rebel-seinpark: `Arguments with the prefix `rbln_` are passed to rbln_config, while the remaining arguments are passed to the HuggingFace library.`

**열린 질문**: 사용되지 않는 인자(kwargs 등)를 docstring에 적을지 생략할지 정책을 정해야 하는가?

### 31. C37 · rbln_config · high

**규칙**: 설정 클래스에는 실제 동작에 반영되는 옵션만 노출하고, 무시되거나 지원하지 않는 옵션은 제거하거나 명시적으로 거부한다.

- kind: `judgment`
- 관련 PR(3): #114, #694, #735
- 관련 코멘트 수: 5

인용:
- PR #694 · rebel-dkhong: `users can set `use_slicing`, but it never takes effect.`
- PR #114 · rebel-kblee: `Are these configurable parameters?`
- PR #735 · rebel-dkhong: `users may don't know why it works when they set `use_inputs_embeds=False` explicitly.`

**열린 질문**: 모델이 반드시 필요로 하는 옵션은 기본값을 고정할지, 사용자 선택을 허용하고 잘못된 값에 예외를 낼지 정해야 하는가?

### 32. C39 · modeling_base/lifecycle · high

**규칙**: 클래스 고유 특성(지원 여부, attention 종류, 플래그 등)은 호출 인자로 흘리지 않고 클래스 속성이나 config 속성으로 선언한다.

- kind: `judgment`
- 관련 PR(3): #267, #356, #359
- 관련 코멘트 수: 5

인용:
- PR #356 · rebel-jongho: `How about making `allow-no-compile-cfgs` as a class instance? Because it depends on the characteristics of Class itself.`
- PR #359 · rebel-thkim: `Passing rbln_config should allow removing the following three parameters`
- PR #267 · rebel-thkim: `is there a downside to defining _tp_support in config?`

**열린 질문**: 없음

### 33. C03 · rbln_config · high

**규칙**: 기본값과 크기 같은 값은 코드에 직접 하드코딩하지 않고 모델 config, 전처리기, 기존 rbln_config 속성에서 유도한다.

- kind: `judgment`
- 관련 PR(3): #478, #549, #560
- 관련 코멘트 수: 4

인용:
- PR #478 · rebel-jongho: `we can derive default value from preprocessors. no hard-code if possible.`
- PR #549 · rebel-junamsong: `Is hardcoding `cls_name == "BertModel"` the only way here?`
- PR #560 · rebel-thkim: `Instead of hard-coding it as `batch_size + 1`, why not use this attribute?`

**열린 질문**: 없음

### 34. C04 · docstring/docs · high

**규칙**: 근거가 바로 보이지 않는 상수나 버퍼 값은 산출 근거(계산식, 경험적 값, 보수적 여유분 등)를 코드 주석이나 PR 설명에 남긴다.

- kind: `judgment`
- 관련 PR(3): #53, #84, #96
- 관련 코멘트 수: 4

인용:
- PR #84 · rebel-kblee: `How was this number determined?`
- PR #96 · rebel-jongho: `How were these values determined?`
- PR #53 · rebel-jongho: `보수적인 계산을 위한 버퍼`

**열린 질문**: 없음

### 35. C22 · api/naming · high

**규칙**: 설정 필드 이름과 상속 구조는 공통 기반 설정 클래스(예: 이미지 설정의 image_size)와 맞춘다.

- kind: `judgment`
- 관련 PR(3): #98, #260, #478
- 관련 코멘트 수: 4

인용:
- PR #260 · rebel-thkim: `How about using image_size instead of max_image_size?`
- PR #478 · rebel-jongho: `why do we use `max` image size as its name?`
- PR #98 · rebel-jongho: `I'm concerned that the name of the submodule config passed as a dict (model.text_model) is different from the name of the attribute in config (text_model).`

**열린 질문**: 없음

### 36. C05 · rbln_config · high

**규칙**: 검증 함수는 검증만 수행하고, 기본값 설정·패딩·config 변경은 지정된 훅(예: set_default_values)이나 생성 시점에서 한다.

- kind: `judgment`
- 관련 PR(3): #66, #81, #654
- 관련 코멘트 수: 3

인용:
- PR #66 · rebel-jongho: `이 함수가 default value도 정하고, validation check도 하고 해서 좀 역할이 큰 느낌이네요`
- PR #81 · rebel-jongho: `validate 함수에서는 validation만 하고, padding은 안하는게 좋을 것 같습니다.`
- PR #654 · rebel-jongho: `setting default values -> `set_default_values` (already exist)`

**열린 질문**: 없음

### 37. C10 · modeling_base/lifecycle · high

**규칙**: monkey patch로 전역 함수나 메서드를 바꿔 쓴 경우 실행 뒤 원래대로 복원하며, 컴파일이 실패해도 복원되도록 try/finally로 감싼다.

- kind: `mechanical`
- 관련 PR(3): #114, #118, #168
- 관련 코멘트 수: 3

인용:
- PR #114 · rebel-jongho: `for all monkey patching(s), Please revert patches after the execution.`
- PR #118 · rebel-jongho: `If monkey patching is used, you must revert the original method after you're done using it.`
- PR #168 · rebel-jongho: `recover logic should be work even if a compilation is failed.`

**열린 질문**: 없음

### 38. C08 · diffusers · medium

**규칙**: 컴파일 시점 옵션은 그래프를 간접적으로 바꾸는 값(예: guidance scale)이 아니라 그래프 형상을 직접 정하는 값(예: submodule batch size)으로 명시한다.

- kind: `judgment`
- 관련 PR(2): #15, #104
- 관련 코멘트 수: 3

인용:
- PR #15 · rebel-jongho: `guidance scale 대신 명시적으로 submodule의 batch size를 지정하도록 바꿔주시면 좋겠습니다~`
- PR #104 · rebel-seinpark: `How do we handle the cases where the `guidance scale` has different values`
- PR #104 · rebel-jongho: `So runtime raises error.`

**열린 질문**: 컴파일 때 지정한 guidance scale과 runtime 값이 다를 때 어떻게 처리할지(에러 or 허용) 정해야 하는가?

### 39. C29 · dependency/version · medium

**규칙**: torch.load는 weights_only를 명시적으로 지정한다(PyTorch 2.6부터 기본값이 바뀌었다).

- kind: `mechanical`
- 관련 PR(2): #15, #106
- 관련 코멘트 수: 3

인용:
- PR #15 · rebel-jongho: `pytorch 2.6부터 torch.load의 default behavior가 바뀌어서 아마 에러가 날 거 같습니다`
- PR #106 · rebel-junamsong: `partial(torch.load, weights_only=True, map_location=torch.device("cpu"))`

**열린 질문**: 없음

### 40. C40 · rbln_config · medium

**규칙**: 설정 포맷이나 산출물 구조가 바뀌면 기존 산출물 로드 실패를 명확한 메시지로 알리고, 하위호환 영향을 확인한다.

- kind: `judgment`
- 관련 PR(2): #88, #694
- 관련 코멘트 수: 3

인용:
- PR #88 · rebel-jonghewk: `Just to confirm—after this PR, will existing compiled models fail with an error?`
- PR #694 · rebel-dkhong: `Please check whether removing `fps` from `RBLNCosmosTransformer3DModelConfig` breaks backward compatibility.`
- PR #88 · rebel-jongho: `please recompile the model using the latest optimum-rbln`

**열린 질문**: 없음

### 41. C41 · style/other · medium

**규칙**: CI 워크플로우에서 시크릿은 secrets 컨텍스트로 전달하고, 내부 엔드포인트를 평문으로 적지 않는다.

- kind: `mechanical`
- 관련 PR(2): #46, #630
- 관련 코멘트 수: 3

인용:
- PR #46 · rebel-jswoo: `이렇게 넘겨주면 한번에 secret variable들이 넘어갑니다`
- PR #630 · rebel-jongho: `please hide the internal endpoint. no plain text.`
- PR #46 · rebel-jswoo: `이부분 없애도 아래에서 secrets.XXXX로 받아올수 있을것 같습니다.`

**열린 질문**: 없음

### 42. C42 · style/other · low

**규칙**: 범용 유틸리티는 utils/에 두고, 특정 모델 전용 로직은 해당 모델 디렉터리에 둔다.

- kind: `judgment`
- 관련 PR(1): #760
- 관련 코멘트 수: 2

인용:
- PR #760 · rebel-jongho: ``rbln/utils/` refers to general-purpose tools.`
- PR #760 · rebel-seinpark: `How about move into `optimum-rbln/src/optimum/rbln/utils/deprecation.py`?`

**열린 질문**: 없음

## 미해결 논쟁

### D01. 리뷰 지적을 같은 PR에서 고칠지, 후속 PR로 미룰지

- PR: #114, #118, #133
- PR 114 작성자: 여러 리뷰 지적을 '다음 PR에서 반영'하겠다고 답변
- PR 118 리뷰어: 그런 패턴은 권장되지 않으니 릴리스 전까지 이번 PR에서 고치자고 요구
- PR 133 작성자: 처음에는 별도 PR로 미루려 했으나 다른 코드를 확인한 뒤 이번 PR에서 반영

### D02. CI 테스트에 공식 체크포인트를 쓸지, tiny 모델을 쓸지

- PR: #38
- 리뷰어(jongho): tiny 모델이 안 되면 TEST_LEVEL=FULL로 PR CI에서 제외하거나, 가중치를 업로드하는 방안 제시
- 작성자(seinpark): diffusers 테스트용 tiny 모델을 찾아 추가(PR 52)하는 방향으로 해결

### D03. output_hidden_states / output_attentions 옵션을 어디서 결정할지(하드코딩 vs 파이프라인별 설정)

- PR: #38, #174, #631
- PR 38 작성자: 당분간 _VisionEncoder에서 output_hidden_states=True를 유지하고, 필요해지면 파이프라인별 관리로 전환
- PR 174 작성자: 사용자가 None을 넘기면 컴파일 시점 rbln_config 값을 따르는 것이 의도라고 설명
- PR 631: 미지원 옵션은 config 단계에서 NotImplementedError로 거부하고 실제 지원은 후속 작업으로 분리

### D04. top-level 컴파일 옵션(batch_size 등)을 submodule로 전파할지

- PR: #15, #445
- PR 445 작성자: batch_size가 language_model에 전달되지 않는 것은 버그로 보고 수정
- PR 445 리뷰어: submodule로 컴파일 옵션을 전파하지 않는 것이 의도된 동작이며 전파 정책은 transformers 쪽에서 논의 필요
- PR 15 리뷰어: 전역 설정이 connected pipeline으로 넘어가는지 확인 요청

### D05. 충돌하는 설정 값을 예외로 막을지, 경고 후 사용자 값을 쓸지

- PR: #267
- PR 267 작성자: 강제되는 키만 예외, 나머지는 경고 없이 사용자 값을 적용하는 혼합안 제시
- PR 267 리뷰어(kblee): 예외 처리가 낫다는 입장

### D06. 미지원 dtype 입력을 예외로 막을지, fp32로 자동 승격할지

- PR: #673
- 리뷰어(jongho): 기존에는 non-fp32에서 실패했는데 새 코드는 조용히 fallback하므로 의도인지 질문
- 작성자(dkhong): 미지원 클래스는 예외 대신 fp32로 지원하는 편이 낫다고 판단(VAE force_upcast 등 파이프라인 혼합 dtype 고려)

### D07. config 값 검증 범위: 잘못된 값도 필드별로 검사할지

- PR: #437
- 리뷰어(kblee): 0 이하 같은 잘못된 값에 예외를 추가하자고 제안
- 리뷰어(jongho): 다른 필드도 잘못된 값을 검사하지 않으므로 이 필드만 추가할 필요가 없다는 입장

### D08. 입력 정규화(dtype 등)를 방어적으로 둘지 최소화할지

- PR: #389
- 작성자(kblee): _dtype이 항상 str이라는 보장이 없어 방어적 normalize 유지
- 리뷰어(jongho): 과한 설계(over-engineering)이며 단순 대입으로 충분하다는 입장

### D09. 파이프라인 간 공통 검증 코드를 어디에 둘지(mixin vs utils)

- PR: #54
- 리뷰어(jongho): 비슷한 코드가 여러 곳에 있으니 합칠 수 있는지 질문
- 작성자(dkhong): mixin 또는 pipelines 하위 utils 파일 중 어디에 둘지 리뷰어 의견 요청

### D10. 사용하지 않는 인자를 docstring에 적을지 생략할지

- PR: #367
- 작성자(dkhong): UNet과 같이 미사용 인자는 생략해 사용자 혼란을 줄이는 방향을 택함
- 대안(작성자): 일관성을 위해 kwargs 전체를 항상 기재하는 방법도 가능하며 정책 결정이 필요

### D11. 혼동되는 함수 이름을 바꿔야 하는지

- PR: #433
- 리뷰어(jongho): 혼동될 만한 함수가 없으니 이름 변경은 불필요
- 작성자(seinpark): 두 이름이 혼동되어 _update_submodule_config_from_model로 변경

### D12. generation_config에 transformers 버전 정보를 모든 모델에 저장할지

- PR: #97
- 리뷰어(kblee): 모델 전반에 적용하는 것이 맞다는 입장
- 작성자(junamsong): Whisper 외 모델은 transformers 4.50.3에서 검증되었고, 일반화는 불필요한 오버헤드를 준다는 입장

### D13. Qwen3.5 구현을 Qwen3-VL 클래스 상속으로 묶을지, 의도적으로 분리 유지할지

- PR: #631
- 리뷰어(jongho): 메서드 복사가 많아 상속이나 공통 base를 쓰도록 제안하고, 분리가 의도라면 명시 요청
- 작성자(seinpark): 두 tower는 qwen3_vl 변경 영향을 막기 위해 의도적으로 분리했다는 입장

### D14. KV 캐시 텐서 이름을 고유하게 할지, 이름을 외부(vLLM)에서 넘길지

- PR: #437
- 리뷰어(eunji): key와 value 텐서 이름이 중복이므로 고유 이름을 쓰자는 입장
- 리뷰어(yskim): mark_static_address(tensor, name)을 유지하고 vLLM이 이름을 명시적으로 넘기는 방식 제안
- 리뷰어(jongho): kv 튜플 가정이 많아 이름 체계 변경은 다음 PR로 미루자는 입장

### D15. 입력 이름(model_input_names) 순서를 고정할지

- PR: #383
- 리뷰어(jongho): key-value 매핑만으로 충분하지 않은지 질문
- 작성자(kblee): 컴파일 설정과 forward 인자 불일치를 막기 위해 순서를 고정하자는 입장
- 리뷰어(jongho): 그 불일치는 사용자 실수이므로 그 경우까지 지원할 필요 없다는 입장

### D16. getattr 기본값으로 fallback할지, 명시적으로 속성을 읽을지

- PR: #162
- 리뷰어(junamsong): 기존의 명시적 방식이 더 안전하고 의도가 분명하다는 입장
- 작성자(thkim): transformers 4.51 이후 config의 head_dim이 None으로 들어와 기존 방식이 실패한다는 근거 제시

### D17. 비생성(non-generation) 모델에서 batch_size 속성의 의미

- PR: #245
- 작성자(thkim): 생성 모델에서는 디코더 최대 배치를 뜻하지만, can_generate가 false이면 의미가 모호하며 대안은 아직 없음

### D18. Cosmos 안전 검사기(safety checker)를 원본 파이프라인 구조대로 둘지, submodule로 옮길지

- PR: #118, #168, #191
- 리뷰어(jongho): 원본 checker 없이 파이프라인을 초기화하고 자체 checker를 붙이는 방식을 제안
- 작성자(seinpark): 그 방식도 가능하나 rbln_config를 checker에 전달하는 경로가 필요하다는 입장
- 작성자(seinpark, PR 191): 원본 파이프라인과 달리 _submodules에 넣는 것이 괜찮은지 질문

### D19. dict로 넘긴 rbln_config(runtime 옵션 포함) 처리 범위

- PR: #67, #475, #495
- 리뷰어(jongho): dict 형태 rbln_config가 들어오는 경우를 지원하는지 질문
- 작성자(seinpark): dict는 rbln_ 접두사로 바꿔 from_pretrained에 넘기며, 설정 객체를 load로 읽는 경로는 접두사 변환을 보장하지 못해 후속 수정 필요

## 통계

- 전체 클러스터: 42
- 신뢰도별: high 37, low 1, medium 4
- 영역별:
  - style/other: 15
  - rbln_config: 6
  - docstring/docs: 5
  - modeling_base/lifecycle: 5
  - api/naming: 4
  - dependency/version: 4
  - multimodal/submodule: 1
  - diffusers: 1
  - testing: 1
- 규칙 근거가 가장 많은 리뷰어 5명(인용 기준):
  - rebel-jongho: 35개 클러스터
  - rebel-thkim: 15개 클러스터
  - rebel-seinpark: 10개 클러스터
  - rebel-junamsong: 7개 클러스터
  - rebel-dkhong: 4개 클러스터
- 미해결 논쟁: 19건
