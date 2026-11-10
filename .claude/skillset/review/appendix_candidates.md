# 부록 — 근거가 PR 1건뿐인 후보 116개

리뷰 대상이 아닙니다. 팀이 "이건 실제 룰이다"라고 지목하면 본 시트로 올립니다. 한 줄씩만 적습니다.

## rbln_config (설정 객체)

- `CFG-025` 모델 및 파이프라인 설정 객체의 인자는 HF 원본이 받는 인자와 컴파일러가 지원하는 입력 구조(예: 여러 input_info)를 그대로 반영한다. — [#146](https://github.com/rebellions-sw/optimum-rbln/pull/146)
- `ATT-029` 슬라이딩 윈도우 등 어텐션 구성은 모델마다 따로 정의하지 않고 컴파일 설정 객체의 공통 필드로 표현한다. — [#162](https://github.com/rebellions-sw/optimum-rbln/pull/162)
- `CFG-006` 로드 시 rbln_config 객체 계약은 엄격하게 유지하고, 부분 설정 객체는 호출 지점에서 dict로 변환해 검증된 병합 경로로 넘기며, 직렬화되지 않는 로드 동작 플래그는 호출하는 쪽이 함께… — [#702](https://github.com/rebellions-sw/optimum-rbln/pull/702)
- `CFG-016` 서브모듈의 디바이스 수가 1로 줄어드는 경우 디바이스 목록과 device_map 등 모든 배치 정보를 함께 축소하며, 이미 지정된 디바이스를 기본값(0번)으로 바꾸지 않는다. — [#776](https://github.com/rebellions-sw/optimum-rbln/pull/776)
- `CFG-020` 서브클래스 인스턴스를 포함하는 중첩 설정 객체는 serialize_as_any=True로 직렬화하고, 직렬화 후 역직렬화해도 같은 값으로 복원되게 한다. — [#495](https://github.com/rebellions-sw/optimum-rbln/pull/495)
- `CFG-021` 중첩된 설정 dict를 갱신할 때는 dict.update() 대신 내부 값을 보존하는 중첩 병합 유틸리티(nested_update)를 쓴다. — [#473](https://github.com/rebellions-sw/optimum-rbln/pull/473)
- `CFG-023` 컴파일 설정 객체는 dict가 아닌 클래스 인스턴스로 표현해 모델별 속성을 명확히 하고 모델과 무관한 속성이 설정되지 않게 한다. — [#67](https://github.com/rebellions-sw/optimum-rbln/pull/67)
- `CFG-024` 양자화 설정은 여러 양자화 방식을 수용할 수 있도록 코드에서 해석 가능한 설정 클래스로 표현한다. — [#130](https://github.com/rebellions-sw/optimum-rbln/pull/130)
- `CFG-029` 하드웨어 종류를 조회해 기본값을 정하는 코드는 장치가 연결되지 않은 호스트에서도 예외 없이 기본값으로 폴백한다. — [#634](https://github.com/rebellions-sw/optimum-rbln/pull/634)
- `CFG-033` 설정 검증을 우회하는 환경 변수 스위치를 두지 않고 한계 값 자체를 근거에 맞게 조정한다. — [#642](https://github.com/rebellions-sw/optimum-rbln/pull/642)
- `CFG-034` 메모리 예산 같은 비율 인자는 외부 프레임워크(vLLM)의 관례에 맞춰 0 초과 1 이하의 실수 비율을 기본 표현으로 삼고 문자열 단위 표기는 CLI 편의용으로만 허용한다. — [#637](https://github.com/rebellions-sw/optimum-rbln/pull/637)

## 멀티모달 · 서브모듈

- `ATT-028` ViT 계열 비전 모듈의 attention은 수동 matmul-softmax-matmul 대신 F.scaled_dot_product_attention을 사용해 컴파일러 SDPA 경로를 타게 하되,… — [#548](https://github.com/rebellions-sw/optimum-rbln/pull/548)
- `SUB-013` 배치 정렬이 필요한 멀티모달 forward는 샘플 소유 관계에 따라 이미지 입력을 재배열하고(배치 차원 index_select 금지), 정렬되지 않은 다중 배치 직접 호출은 KV 캐시를 어긋나게… — [#707](https://github.com/rebellions-sw/optimum-rbln/pull/707)
- `SUB-014` 한 프롬프트에 포함된 여러 이미지를 이미지 수만큼의 batch_size로 처리하지 않는다. — [#370](https://github.com/rebellions-sw/optimum-rbln/pull/370)
- `SUB-020` CLI와 설정 인자에서 서브모듈 설정은 점 표기(dot-notation)로 지정한다. — [#181](https://github.com/rebellions-sw/optimum-rbln/pull/181)

## 모델 베이스 · 라이프사이클

- `DIF-019` 컴파일된 서브모델을 로드할 때는 호스트 메모리 최적화 옵션(optimize_host_memory)을 사용한다. — [#206](https://github.com/rebellions-sw/optimum-rbln/pull/206)
- `DEC-015` 래퍼는 원본 Hugging Face 모듈 전체를 보관하거나 _original_mod를 참조하지 않고, 필요한 하위 모듈과 속성만 __init__에서 명시적으로 등록하여 nn.Module 트리와 s… — [#444](https://github.com/rebellions-sw/optimum-rbln/pull/444)
- `LIFE-016` 컴파일 전에 공통으로 수행하는 전처리(예: 비연속 텐서의 contiguous 정규화)는 모든 표준 컴파일 경로가 거치는 단일 진입점에 두고, compile_from_torch를 직접 호출하는 곳에… — [#600](https://github.com/rebellions-sw/optimum-rbln/pull/600)
- `LIFE-017` Auto 클래스의 해석은 호출자가 넘긴 원격 코드 신뢰 옵션 값을 따르고, 기본 동작에서는 베이스 라이브러리가 구현한 모델 타입을 내장 클래스로 해석한다. — [#813](https://github.com/rebellions-sw/optimum-rbln/pull/813)
- `LIFE-018` 설정 클래스를 식별할 때는 model_type 대신 architectures 값을 기준으로 삼는다. — [#679](https://github.com/rebellions-sw/optimum-rbln/pull/679)
- `LIFE-022` 컴파일되어 NPU에서 실행되는 모델의 forward 메서드에는 @torch.no_grad 데코레이터를 붙이지 않는다. — [#533](https://github.com/rebellions-sw/optimum-rbln/pull/533)
- `LIFE-023` NPU에서 실행하기 어려운 후처리(예: 메시 추출)는 CPU 대체 경로로 구현하고 나머지 연산만 NPU에서 실행한다. — [#533](https://github.com/rebellions-sw/optimum-rbln/pull/533)
- `LIFE-027` 양자화 체크포인트 로딩 시 레이어 파라미터는 고정 키를 쓰고 체크포인트 키는 공통 패턴으로 정규화한 뒤 가중치를 로드한다. — [#257](https://github.com/rebellions-sw/optimum-rbln/pull/257)
- `LIFE-028` 모델 경로에 있는 컴파일 산출물은 컴파일 설정 객체에 명시된 것만 로드한다. — [#140](https://github.com/rebellions-sw/optimum-rbln/pull/140)
- `LIFE-030` 원본 모델을 수정해 컴파일하는 방식(예: VLM)으로 지원되는 모델은 'from_pretrained'뿐 아니라 'from_model'도 함께 지원한다. — [#337](https://github.com/rebellions-sw/optimum-rbln/pull/337)

## decoderonly

- `DEC-001` 임베딩 레이어의 dtype 처리는 호출부마다 캐스팅하지 않고 생성 함수 안에서 한 번 적용하여, 레이어가 어디서 만들어져도 컴파일 dtype이 유지되게 한다. — [#674](https://github.com/rebellions-sw/optimum-rbln/pull/674)
- `DEC-007` 생성 가능 여부는 클래스에 고정하지 않고 컴파일된 phase(예: decode 포함 여부)에 따라 정한다. — [#818](https://github.com/rebellions-sw/optimum-rbln/pull/818)
- `DEC-011` 모델의 어텐션 의미(인과/양방향)는 모델 클래스가 아니라 부모가 주입한 설정 플래그 조합으로 판정하며, 그 판정식은 한 곳에만 정의한다. — [#646](https://github.com/rebellions-sw/optimum-rbln/pull/646)
- `DEC-012` 패딩 길이가 0일 수 있는 자리에서 음수 슬라이스 끝을 쓰지 않고 None으로 가드한다. — [#657](https://github.com/rebellions-sw/optimum-rbln/pull/657)
- `DEC-016` decoder-only 모델 구현에서는 cache_position과 position_ids가 항상 같다고 가정하지 않는다. — [#133](https://github.com/rebellions-sw/optimum-rbln/pull/133)
- `DEC-017` 언어 모델의 forward 처리 중 최적화 관련 로직은 RBLN* 모델 클래스가 아니라 RBLN*Runtime 클래스에 둔다. — [#12](https://github.com/rebellions-sw/optimum-rbln/pull/12)

## attention · KV cache

- `ATT-018` 커스텀 op 시그니처는 vllm-rbln 쪽 인터페이스와 맞추고, 호출은 위치 인자 대신 키워드 인자로 쓰며 해당 없는 인자는 None으로 명시한다. — [#566](https://github.com/rebellions-sw/optimum-rbln/pull/566)
- `ATT-001` HF 레퍼런스와 같은 수치 동작을 따른다: rotary cos/sin은 활성 dtype으로 캐스트하고, softmax는 fp32로 계산한 뒤 query dtype으로 되돌리며, 스케일 상수는 HF… — [#711](https://github.com/rebellions-sw/optimum-rbln/pull/711)
- `ATT-007` 마스크는 결과에 영향이 없는 축을 뺀 형태로 보낸다(예: 패딩된 쿼리 행의 출력이 버려지면 키 패딩 마스크만 전송한다). — [#796](https://github.com/rebellions-sw/optimum-rbln/pull/796)
- `ATT-012` KV 캐시 블록 수 추정은 NPU별 DRAM 크기, 양자화 모델의 파라미터 크기, 슬라이딩 윈도우 어텐션의 KV 캐시 크기 변화를 모두 고려한다. — [#354](https://github.com/rebellions-sw/optimum-rbln/pull/354)
- `ATT-014` KV 캐시 추정에서 칩 종류별 전용 분기나 별도의 추정 알고리즘을 추가하지 않고, 하나의 추정 경로 안에서 처리한다. — [#584](https://github.com/rebellions-sw/optimum-rbln/pull/584)
- `ATT-015` vLLM 연동처럼 특정 모델에만 필요한 블록 할당 예외는 그 모델의 설정에만 적용하고, 다른 모델의 할당 로직은 바꾸지 않는다. — [#560](https://github.com/rebellions-sw/optimum-rbln/pull/560)
- `ATT-016` 커스텀 attention op는 최신 PyTorch custom op 규격에 맞춰 스키마를 정의하고, 출력 텐서는 입력과 저장 공간을 공유하지 않도록 새로 할당한다. — [#74](https://github.com/rebellions-sw/optimum-rbln/pull/74)
- `ATT-017` 커스텀 op의 이름은 Python 측 함수 이름과 일치시키고, op의 docstring은 실제 op 명세와 맞춘다. — [#62](https://github.com/rebellions-sw/optimum-rbln/pull/62)
- `ATT-020` 컴파일러 연산에 새 인자를 넘길 때는 연산 스키마를 먼저 확인해 지원하는 경우에만 전달한다. — [#807](https://github.com/rebellions-sw/optimum-rbln/pull/807)
- `ATT-025` static 주소로 표시하는 텐서는 항상 이름 인자를 함께 넘겨 식별 가능하게 한다. — [#677](https://github.com/rebellions-sw/optimum-rbln/pull/677)
- `ATT-026` 종류가 다른 캐시 메타는 문자열 태그가 아니라 타입(서브클래스)으로 구분하고, 분기에 쓰는 속성은 서브클래스가 정의한다. — [#666](https://github.com/rebellions-sw/optimum-rbln/pull/666)
- `ATT-027` 호스트 환경에 따라 달라지는 어텐션 구현체는 로드 시점에 고정해 컴파일 결과가 환경에 좌우되지 않게 한다. — [#625](https://github.com/rebellions-sw/optimum-rbln/pull/625)
- `SUB-007` 자동 KV 캐시 블록 수 산정은 같은 디바이스를 공유하는 다른 서브모듈이 쓰는 메모리를 예약하고, 디바이스 목록이 겹치지 않으면 아무것도 예약하지 않는다. — [#766](https://github.com/rebellions-sw/optimum-rbln/pull/766)

## diffusers

- `DIF-003` 배치 정합성 검사 로직은 파이프라인별 동작에 맞는 개별 함수로 구현하고, 새 파이프라인을 추가하거나 diffusers 버전이 바뀌면 해당 로직을 점검한다. — [#58](https://github.com/rebellions-sw/optimum-rbln/pull/58)
- `DIF-005` 배치 크기가 지정되지 않은 diffusers 서브모듈은 Hugging Face 라이브러리의 기본값을 참조해 컴파일한다. — [#104](https://github.com/rebellions-sw/optimum-rbln/pull/104)
- `DIF-006` 파이프라인이 쓰지 않는 서브모듈은 컴파일하지 않으며, 예를 들어 VQ 인코더는 파이프라인이 이미지 입력을 필요로 할 때만 컴파일한다. — [#102](https://github.com/rebellions-sw/optimum-rbln/pull/102)
- `DIF-008` 파이프라인 클래스는 자신의 설정 클래스를 클래스 멤버로 노출해 IDE에서 설정 항목을 탐색할 수 있게 한다. — [#110](https://github.com/rebellions-sw/optimum-rbln/pull/110)
- `DIF-009` 파이프라인 컴파일 설정의 하위 모듈 설정에도 NPU 종류와 텐서 병렬 크기를 전달한다. — [#164](https://github.com/rebellions-sw/optimum-rbln/pull/164)
- `DIF-011` 런타임에 값이 바뀔 수 있는 스케일 계수는 컴파일 상수로 굳히지 않고 입력 텐서로 전달해 재컴파일 없이 조정하게 한다. — [#725](https://github.com/rebellions-sw/optimum-rbln/pull/725)
- `DIF-012` 정밀도에 민감한 작은 연산(patch embedding, RoPE 테이블, timestep embedding)은 호스트에서 fp32로 수행한다. — [#694](https://github.com/rebellions-sw/optimum-rbln/pull/694)
- `DIF-013` 호스트와 디바이스 사이로 넘기는 텐서는 작은 형태로 보내고, 공간 방향 브로드캐스트는 그래프 안에서 수행한다. — [#694](https://github.com/rebellions-sw/optimum-rbln/pull/694)
- `DIF-014` 정적 그래프는 batch 1로 컴파일하고, 더 큰 파이프라인 배치는 샘플 단위로 나눠 실행한다. — [#694](https://github.com/rebellions-sw/optimum-rbln/pull/694)
- `DIF-015` 정확도 손실이 있는 최적화는 기본값에서 제외하고 옵트인으로 두며, 측정한 오차를 함께 기록한다. — [#775](https://github.com/rebellions-sw/optimum-rbln/pull/775)
- `DIF-016` 체크포인트 템플릿에 따라 달라지는 값(예: 템플릿이 감싸는 토큰 수)은 상수로 적지 않고 export 시 토크나이저에서 읽는다. — [#780](https://github.com/rebellions-sw/optimum-rbln/pull/780)
- `DIF-017` diffusers 쪽 forward에는 transformers 규약의 return_dict 폴백을 적용하지 않고, diffusers 고유의 return_dict 기본값 관례를 따른다. — [#715](https://github.com/rebellions-sw/optimum-rbln/pull/715)
- `DIF-020` 표준 PreTrainedModel을 상속하지 않는 컴포넌트(예: 가드레일 모델)는 파이프라인의 선택 컴포넌트 목록으로 등록해 파이프라인 kwargs로 전달한다. — [#106](https://github.com/rebellions-sw/optimum-rbln/pull/106)
- `DIF-024` diffusers 파이프라인은 공식 Hub 레이아웃(리비전, 별도 가중치, 가드레일 포함)을 그대로 따라 지원한다. — [#694](https://github.com/rebellions-sw/optimum-rbln/pull/694)

## 공개 API · 이름

- `API-010` 설정 클래스의 정수형 인자는 기본값을 None으로 두고 내부에서 기본값(예: 1)으로 해석하는 저장소 관례를 따른다. — [#682](https://github.com/rebellions-sw/optimum-rbln/pull/682)
- `API-012` 사용자에게 노출하지 않는 메서드는 '_' 접두사를 붙인다. — [#337](https://github.com/rebellions-sw/optimum-rbln/pull/337)
- `API-013` **kwargs 타입 주석에는 값의 타입만 쓰고 Dict로 감싸지 않는다. — [#234](https://github.com/rebellions-sw/optimum-rbln/pull/234)
- `API-018` CLI에서 Hugging Face 모델 로더 인자는 --hf-* 접두사로 전달하고, 나머지 인자는 컴파일 설정 객체로 보낸다. — [#426](https://github.com/rebellions-sw/optimum-rbln/pull/426)
- `ATT-023` 외부 통합(예: vLLM)이 사용하는 어텐션 구현 방식과 KV 블록 수는 getter 메서드로 노출한다. — [#112](https://github.com/rebellions-sw/optimum-rbln/pull/112)

## 테스트

- `TEST-003` 공통 테스트 베이스(tests/test_base.py)를 바꾸는 변경은 모든 테스트 스위트에 영향을 주는 것으로 취급해 전체를 실행한다. — [#731](https://github.com/rebellions-sw/optimum-rbln/pull/731)
- `TEST-021` 같은 컴파일 모델을 같은 장치에서 동시에 로드하는 테스트를 작성하지 않는다. — [#410](https://github.com/rebellions-sw/optimum-rbln/pull/410)
- `TEST-023` 아티팩트 저장과 로드를 검증하는 테스트(BC 테스트, requires_compile 클래스)는 가짜 컴파일 환경 변수와 무관하게 실제 컴파일을 사용한다. — [#772](https://github.com/rebellions-sw/optimum-rbln/pull/772)
- `TEST-027` NPU 없이 컴파일할 때는 대상 SoC를 환경 변수로 명시해, 호스트나 스케줄러가 넘긴 값을 상속하지 않게 한다. — [#753](https://github.com/rebellions-sw/optimum-rbln/pull/753)
- `TEST-030` 특정 NPU에서 지원되지 않는 모델은 해당 장치의 CI/CD 대상에서 임시로 제외한다. — [#152](https://github.com/rebellions-sw/optimum-rbln/pull/152)
- `TEST-031` 서빙 엔진(vLLM)의 기능은 optimum-rbln 테스트가 아니라 해당 엔진의 CI에서 검증한다. — [#327](https://github.com/rebellions-sw/optimum-rbln/pull/327)
- `TEST-033` 테스트는 컴파일 산출물을 저장하고 재사용한다. — [#404](https://github.com/rebellions-sw/optimum-rbln/pull/404)
- `TEST-035` 계약 조항(해석, 상한, 기본값, 일관성 등)마다 검증 테스트 함수를 하나씩 두어 실패 지점이 어느 조항인지 드러나게 한다. — [#729](https://github.com/rebellions-sw/optimum-rbln/pull/729)
- `TEST-040` 가중치를 제자리에서 바꾸는 전처리 헬퍼는 값 보존, 가중치 공유 유지, 멱등성을 단위 테스트로 검증한다. — [#600](https://github.com/rebellions-sw/optimum-rbln/pull/600)

## docstring · 문서

- `DIF-021` 새 diffusers 파이프라인을 추가할 때는 예제(examples)도 함께 추가하고 갱신한다. — [#54](https://github.com/rebellions-sw/optimum-rbln/pull/54)
- `DOC-006` 예제 코드는 NVIDIA나 diffusers의 예제와 크게 다르지 않은 형태로 작성하고, 컴파일은 from_pretrained 호출 안에서 처리한다. — [#168](https://github.com/rebellions-sw/optimum-rbln/pull/168)
- `DOC-007` docstring 예제는 실제로 실행해 통과를 확인하고, 예제의 컴파일 설정은 모델 예제 저장소의 컴파일 스크립트 설정과 같게 맞춘다. — [#788](https://github.com/rebellions-sw/optimum-rbln/pull/788)
- `DOC-008` 예제 스크립트는 컴파일 여부를 인자로 받지 않고 모델 id인지 컴파일 산출물 디렉터리인지로 자동 판별하며, 헤더에 동등한 CLI 명령을 함께 적는다. — [#635](https://github.com/rebellions-sw/optimum-rbln/pull/635)
- `DOC-010` 모델 디렉터리마다 파이프라인 사용법과 의존성 표를 담은 README 문서를 함께 둔다. — [#533](https://github.com/rebellions-sw/optimum-rbln/pull/533)
- `DOC-013` docstring 검사의 예외 목록에서 파일이 검사를 통과하게 되면 해당 예외를 즉시 제거한다. — [#731](https://github.com/rebellions-sw/optimum-rbln/pull/731)
- `DOC-015` 이름이 같은 서로 다른 개념(예: 외부 라이브러리의 설정값과 RBLN 고유 옵션)은 문서에서 어느 쪽을 가리키는지 명시해 구분한다. — [#625](https://github.com/rebellions-sw/optimum-rbln/pull/625)
- `DOC-019` 문서와 docstring에서 Hugging Face 표기는 HuggingFace로 통일한다. — [#113](https://github.com/rebellions-sw/optimum-rbln/pull/113)

## 의존성 · 버전

- `DEP-004` 버전에 따라 경로나 이름이 달라지는 외부 API는 호환 모듈 한 곳에 모으고 나머지 코드는 정식 이름 하나만 import한다. — [#564](https://github.com/rebellions-sw/optimum-rbln/pull/564)
- `DEP-013` 테스트 의존 패키지(pytest, torchcodec)는 버전을 고정하고 컴파일러 버전에 맞춰 함께 관리한다. — [#529](https://github.com/rebellions-sw/optimum-rbln/pull/529)
- `DEP-031` transformers 버전을 올릴 때는 새 버전과 호환되지 않는 모델 구현(VLM 계열, 디코더 전용, T5 아키텍처)을 함께 수정하고 vLLM 및 모델별 최신 지원 여부를 점검한다. — [#282](https://github.com/rebellions-sw/optimum-rbln/pull/282)
- `DEP-021` 최소 지원 컴파일러 버전이 어떤 기능을 항상 제공하게 되면 그 기능의 존재를 확인하는 호환 분기와 폴백 코드는 제거한다. — [#626](https://github.com/rebellions-sw/optimum-rbln/pull/626)
- `DEP-006` 지원 범위 안의 외부 라이브러리 메이저 버전마다 PR CI 매트릭스에서 각각 테스트한다. — [#564](https://github.com/rebellions-sw/optimum-rbln/pull/564)
- `DEP-014` 컴파일러와 optimum-rbln 사이의 버전 호환 검사는 post-release 접미사를 제외한 기본 버전만 비교하며, 버전 호환성 목록은 코드에 하드코딩하지 않는다. — [#109](https://github.com/rebellions-sw/optimum-rbln/pull/109)
- `DEP-022` ruff 버전은 pre-commit 훅, uv.lock, quality 의존성 그룹에서 같은 값으로 고정한다. — [#733](https://github.com/rebellions-sw/optimum-rbln/pull/733)
- `DEP-024` 의존성 잠금 파일이나 프로젝트 메타데이터가 바뀌면 CI 실행 환경 이미지를 재빌드하여 푸시한다. — [#415](https://github.com/rebellions-sw/optimum-rbln/pull/415)
- `DEP-028` 컴파일러가 이미 등록하는 커스텀 op 정의는 이 라이브러리에서 중복 등록하지 않고 컴파일러를 단일 원천으로 둔다. — [#621](https://github.com/rebellions-sw/optimum-rbln/pull/621)
- `DEP-029` 특정 (모델 × 메이저 버전) 조합에 알려진 호환 문제가 있으면 컴파일 깊은 곳에서 실패하기 전에 런타임 에러로 막는 허용목록을 둔다. — [#564](https://github.com/rebellions-sw/optimum-rbln/pull/564)
- `DEP-030` transformers 메이저 릴리스 대응은 'transformers/vX' 형태의 전용 브랜치를 base로 하는 별도 준비 PR로 진행한다. — [#479](https://github.com/rebellions-sw/optimum-rbln/pull/479)
- `DEP-036` 정식 릴리스 이후의 수정만 담은 패치 릴리스는 버전에 .postN 접미사를 붙여 표기한다. — [#384](https://github.com/rebellions-sw/optimum-rbln/pull/384)
- `DEP-037` CI와 pyproject.toml의 지원 Python 버전은 함께 갱신한다. — [#233](https://github.com/rebellions-sw/optimum-rbln/pull/233)
- `DEP-039` Python 클래스 레벨 property 패턴은 사용하지 않는다. — [#40](https://github.com/rebellions-sw/optimum-rbln/pull/40)

## PR · CI · 브랜치 운영

- `CFG-035` 컴파일 설정 객체의 필드 변경이 rbln_config.json 저장 포맷에 영향을 주면 PR 본문에 변경 후 직렬화 결과 예시를 보여준다. — [#560](https://github.com/rebellions-sw/optimum-rbln/pull/560)
- `DEC-006` 기본 집합 통신 레이아웃에서 벗어나는 최적화는 해당 구조에만 적용하고, 일반 패턴으로 확장하지 않는다는 근거를 PR에 적는다. — [#802](https://github.com/rebellions-sw/optimum-rbln/pull/802)
- `DIF-022` 생성형 diffusers 파이프라인 PR에서는 실제 생성 결과물을 공유해 동작을 검증한다. — [#38](https://github.com/rebellions-sw/optimum-rbln/pull/38)
- `LIFE-029` 모델 동작을 수정하는 버그 수정은 vLLM 연동 쪽에 문제가 없는지 확인한다. — [#16](https://github.com/rebellions-sw/optimum-rbln/pull/16)
- `PRC-011` 오래된 PR은 닫고, 구현을 다시 진행할 때는 최신 main에서 새 브랜치를 만든다. — [#83](https://github.com/rebellions-sw/optimum-rbln/pull/83)
- `PRC-013` PR 본문 첫머리에 변경 범위(변경한 파일 종류와 제외한 영역)를 명시한다. — [#711](https://github.com/rebellions-sw/optimum-rbln/pull/711)
- `PRC-016` 대량 수정은 대상 선정 방법(예: AST 스캔)과 위양성으로 제외한 항목을 PR 본문에 명시하고, 수정 후 같은 스캔을 다시 실행해 잔여 대상이 0인지 확인한다. — [#715](https://github.com/rebellions-sw/optimum-rbln/pull/715)
- `PRC-017` 원인 가설은 추론만으로 확정하지 않고 계측이나 실험으로 확인하며, 가설별 결과를 PR에 남긴다. — [#678](https://github.com/rebellions-sw/optimum-rbln/pull/678)
- `PRC-021` 버그 수정 PR 본문에는 해결하는 이슈의 링크를 Fixes 형식으로 적는다. — [#7](https://github.com/rebellions-sw/optimum-rbln/pull/7)
- `PRC-027` 잘못 들어간 커밋은 히스토리를 고쳐 쓰지 않고 revert 커밋으로 남긴다. — [#618](https://github.com/rebellions-sw/optimum-rbln/pull/618)
- `PRC-028` 릴리스에 포함하지 않을 대규모 리팩터링은 릴리스 태그 이후에 머지한다. — [#307](https://github.com/rebellions-sw/optimum-rbln/pull/307)
- `PRC-030` 기존 CI 실행 이후 병합된 PR들의 결합 결과는 검증되지 않았을 수 있으므로, 병합 후 기본 브랜치의 lint와 빌드를 다시 확인한다. — [#821](https://github.com/rebellions-sw/optimum-rbln/pull/821)
- `PRC-041` 개발자용 도구는 scripts/ 최상위에 두고, Buildkite 작업 안에서만 의미가 있는 헬퍼는 scripts/ci/에 둔다. — [#733](https://github.com/rebellions-sw/optimum-rbln/pull/733)
- `PRC-043` CI 스크립트에서 따옴표가 포함된 값은 셸 명령에 직접 넣지 않고 환경 변수로 전달한다. — [#361](https://github.com/rebellions-sw/optimum-rbln/pull/361)
- `STY-025` 수동 실행(workflow_dispatch)에서도 대상 브랜치가 바뀌지 않도록 워크플로우의 ref와 base를 고정 값으로 명시한다. — [#696](https://github.com/rebellions-sw/optimum-rbln/pull/696)

## 코드 스타일 · 기타

- `STY-016` 범용 유틸리티는 'utils/'에 두고, 특정 모델 전용 로직은 해당 모델 디렉터리에 둔다. — [#760](https://github.com/rebellions-sw/optimum-rbln/pull/760)
- `STY-017` 호스트 코드의 분기는 디바이스 출력 값을 읽어 결정하지 않고, 호스트가 이미 아는 값은 직접 계산한다. — [#678](https://github.com/rebellions-sw/optimum-rbln/pull/678)
- `STY-021` '# type: ignore' 주석에는 반드시 에러 코드를 명시한다. — [#736](https://github.com/rebellions-sw/optimum-rbln/pull/736)
- `STY-023` 린트 자동 수정 결과는 일괄 적용하지 않고 diff에서 검토한 뒤 반영하며, 동작이 바뀌는 수정은 되돌린다. — [#733](https://github.com/rebellions-sw/optimum-rbln/pull/733)
- `STY-026` 함수나 메서드의 기본 인자로 가변 객체(list, dict 등)를 쓰지 않으며 린터(bugbear 규칙)로 검출한다. — [#376](https://github.com/rebellions-sw/optimum-rbln/pull/376)
- `STY-028` 실험적 기능은 실험적이라고 명시하고 지원 범위를 함께 적는다. — [#248](https://github.com/rebellions-sw/optimum-rbln/pull/248)
- `STY-029` 컴파일 시간처럼 원인 추적이 어려운 경로에는 문제 추적용 로그를 추가한다. — [#21](https://github.com/rebellions-sw/optimum-rbln/pull/21)
