# 2단계 코드 대조 결과 (stage D)

작성일 2026-10-22. 대상: mechanical+supported 룰 41개. 각 룰을 재현 가능한 코드 질의로 번역해 현재 코드의 준수/위반을 셌다. 질의는 `rules.yaml`의 `check.command`에 있고, 위반 위치는 `violations_known`에 있다.

## 0. 측정 기준과 주요 발견

**측정 기준.** 저장소의 기본 브랜치이자 유일한 장기 브랜치는 `dev`다(`main`은 PR 727에서 삭제, `release-vX.Y.Z` 브랜치는 존재). 코드 룰은 dev tip 74e97fa4(2026-10-21)에서 쟀고(`grounding.measured_on`), PR 운영 룰은 2025-01 이후 머지된 PR 코퍼스에서 쟀다. 처음 main(2026-07-31)에서 잰 배치는 dev에서 다시 쟀으며 판정은 하나도 바뀌지 않았고 수치만 갱신됐다.

**행동이 필요한 발견 (룰과 별개로 이슈 후보).**

1. **설정 우선순위 불일치 (CFG-022).** docstring(`configuration_utils.py:408`)과 `tests/test_config.py::test_config_priority`는 "`rbln_*` 인자가 `rbln_config`보다 우선"이라 하고 `RBLNModelConfig.from_pretrained`(`configuration_utils.py:1044`)도 그렇게 동작하지만, 모델 로드 경로 `RBLNBaseModel.from_pretrained`(`modeling_base.py:246-247`)는 dict로 받은 `rbln_config`를 `rbln_*` kwargs로 바꿔 `kwargs.update`하므로 **dict가 사용자의 명시 인자를 덮어쓴다.** 테스트는 config 객체 경로만 검사해 dict 경로의 불일치를 못 잡는다. "동시 지정 시 ValueError" 룰(PR 445)은 머지되지 않았으므로 rejected이고, 현행 룰은 "rbln_* 우선"이다.
2. **PR 템플릿이 낡았다.** `.github/pull_request_template.md`는 여전히 "Release (dev → main merge)"와 "hotfix는 main으로"를 안내한다. 현행 릴리스는 태그 기반이다(PRC-005 refined 참조).
3. **만료된 deprecation 8건 (DEP-001).** 제거 예정 버전이 현재 버전 이하인데 남아 있는 항목. `violations.md` 참조.
4. **테스트 미등록 모델 클래스 30개 (TEST-008).** export된 사용자용 클래스 115개 중 30개가 `tests/test_*.py`에 `RBLN_CLASS`로 등록돼 있지 않다. 목록은 `violations.md`.
5. **저작권 헤더 불일치 47파일 (STY-005).** 연도 또는 블록 형식이 기존 파일과 다르다.
6. **죽은 문서 URL 4건 (DOC-012).** 런타임 메시지가 가리키는 docs.rbln.ai 경로가 SDK 문서 트리에 없다(`about_atom/release_note.html` 3건, `vllm_support/vllm-rbln.html` 1건).
7. **PR 제목 검사는 필수 체크가 아니다 (PRC-001).** `pr-title-check.yaml`은 돌지만 dev 보호 규칙은 Buildkite 4개만 요구하므로 제목 위반이 머지를 막지 못한다.
8. **monkey patch 복원 위험 (LIFE-007).** decoderonly의 `F.linear` 패치는 `original_linear`를 `try` 안에서 잡아, 그 전에 실패하면 `finally`가 복원할 값이 없다.

**룰 자체에 대한 발견.** rejected 5개 중 3개(PRC-005, PRC-023, PRC-002)는 룰의 대상이 사라졌거나 관행이 바뀐 것이고, 2개(CFG-019, CFG-022)는 머지되지 않은 PR의 제안이 룰로 채굴된 것이다. **머지되지 않은 PR의 본문은 룰 근거로 쓰면 안 된다**는 교훈이며, 1단계 파이프라인의 PR 본문 추출에 `mergedAt` 필터를 추가해야 한다(현재 코퍼스는 open/closed PR을 포함).

## 1. 판정 집계

| 판정 | 수 | 뜻 |
|---|---|---|
| verified | 25 | 코드가 따른다. 남은 위반은 기술부채 |
| aspirational | 8 | 팀 의도는 분명하나 코드가 아직 안 따른다 |
| interview | 2 | 코드에 두 패턴이 공존하거나 적용 경계를 코드로 정할 수 없다 |
| rejected | 5 | 코드가 일관되게 반대로 하고 근거가 얇거나 낡았다 |
| unverifiable | 1 | 검사로 번역 불가 → kind를 judgment로 바꿈 |

## 2. 룰별 결과

| id | 영역 | 판정 | 준수 | 위반 | 검사 종류 | 룰 |
|---|---|---|---|---|---|---|
| API-003 | api/naming | verified | 162 | 0 | script | API·설정 구조·공통 런타임 인터페이스·모델 출력 타입을 제거·교체·변경할 때는 라이브러리 내부 호출부, 하위 클래스와 멀티모달 래퍼, vLL |
| API-011 | api/naming | verified | 132 | 1 | script | 새 모델 클래스와 파이프라인용 diffusers 구성요소 클래스는 기존 RBLN 접두사 패턴(RBLN*For* 등)을 따라 이름 짓고 노출한다. |
| API-014 | api/naming | verified | 305 | 3 | script | 공개 객체는 정의한 모듈의 `__all__`에 등록하고 패키지 `__init__.py`는 고정 템플릿으로 두며, 새 모델은 패키지 최상위 exp |
| API-015 | api/naming | verified | 125 | 1 | ast | 내부 모듈 경로를 옮기거나 모델 클래스를 배치할 때 호환용 shim 모듈이나 별도 alias 파일을 남기지 않고, 공개 API의 재export만 |
| CFG-018 | rbln_config | verified | 10 | 0 | script | 런타임 전용 값(콜백 등)과 컴파일 시점에만 쓰는 추정 입력은 subclass_non_save_attributes에 등록해 rbln_config |
| DEP-008 | dependency/version | verified | 10 | 3 | ast | 모델 구현에 필요한 새 외부 의존성은 pyproject.toml에 명시하되, 필수 기능이 아닌 패키지는 기본 의존성에 넣지 않는다. |
| DEP-011 | dependency/version | verified | 25 | 2 | ast | torch.load는 weights_only를 명시적으로 지정한다. |
| DEP-012 | dependency/version | verified | 3 | 0 | script | PR CI가 쓰는 컴파일러 버전은 .github/version.yaml 한 곳에서 고정하고 모든 CI가 그 값을 읽게 하며, 컴파일러 업데이트가 |
| DEP-015 | dependency/version | verified | 37 | 0 | ast | 외부 라이브러리가 deprecated로 표시한 속성, 호출 방식, import 경로는 권장 대체 방식으로 옮긴다. |
| DEP-033 | dependency/version | verified | 42 | 0 | corpus | 의존성 봇이 만든 업데이트 PR은 자동 병합하지 않고 검토 후 사람이 직접 병합한다. |
| DOC-011 | docstring/docs | verified | 1138 | 7 | script | Google 스타일 docstring에서 파라미터 선언(이름과 괄호 속 타입)과 Returns 설명은 각각 한 줄에 쓰고, 파라미터 이름은 첫  |
| DOC-012 | docstring/docs | verified | 42 | 2 | ast | docstring의 문서 링크는 상대 경로가 아니라 절대 URL로 쓰고, SDK 문서 빌드 결과에 실제로 존재하는 경로만 가리킨다. |
| LIFE-007 | modeling_base/lifecycle | verified | 3 | 0 | ast | monkey patch로 전역 함수나 메서드를 바꾼 경우 실행 뒤 원래대로 복원하며, 컴파일이 실패해도 복원되도록 try/finally로 감싼다 |
| LIFE-020 | modeling_base/lifecycle | verified | 118 | 1 | ast | 오버라이드되는 메서드(get_compiled_model, get_input_info 등)는 공통 계약을 유지하여, 개별 값을 여러 인자로 받지  |
| PRC-001 | pr_process | verified | 610 | 21 | corpus | PR 제목은 `type(optional scope): description` 형식의 conventional commit을 따르고, type은 P |
| STY-004 | style/other | verified | 2154 | 2 | ast | 사용하지 않는 코드(미사용 변수·import·메서드·클래스, 부모와 동일한 override, 레거시 코드, 효과 없는 컴파일 옵션, 오래되었거나 |
| STY-010 | style/other | verified | 103 | 1 | ast | 타입 annotation에만 쓰이는 심볼이 순환 참조를 만들거나 외부 패키지 타입일 때만 import를 `TYPE_CHECKING` 블록으로 옮 |
| STY-011 | style/other | verified | 1705 | 4 | ast | 함수 시그니처에는 실제로 전달되는 객체와 일치하는 타입 힌트를 명시하고, Python 3.10 이상의 네이티브 문법(`X | None`, `li |
| STY-012 | style/other | verified | 264 | 0 | ast | `__init__` 목록, export 목록, 키 목록 같은 나열형 목록은 정렬된 순서를 유지한다. |
| STY-013 | style/other | verified | 56 | 6 | ast | 로그와 경고는 built-in logging/warnings 대신 optimum-rbln의 `utils.logging`을 사용한다. |
| STY-015 | pr_process | verified | 64 | 2 | grep | CI 워크플로우에서 시크릿은 `secrets` 컨텍스트로 전달하고, 내부 엔드포인트·클러스터 고유 경로나 주소는 저장소에 평문으로 적지 않고 C |
| STY-018 | style/other | verified | 627 | 4 | script | 머지 전에 PR 변경 코드 전체에 `ruff format`과 `ruff check`를 통과시키고 타입 검사와 독스트링 검사도 통과시킨다. |
| STY-027 | modeling_base/lifecycle | verified | 3 | 0 | grep | torch.export 경로를 지원할 때는 새 커스텀 op의 fake 함수와 인덱스 값의 동적 값 제약을 함께 제공한다. |
| TEST-007 | testing | verified | 7 | 0 | grep | pytest는 각 테스트 케이스의 실행 시간을 보고하도록 verbosity를 설정하고, 스케줄 실행은 첫 실패에서 중단하지 않는다. |
| TEST-019 | testing | verified | 23 | 3 | ast | 테스트 클래스는 기반 클래스나 클래스가 소유한 공유 설정 dict를 in-place로 수정하지 않고 새 dict를 만들거나 자기 속성으로 재바인 |
| CFG-010 | rbln_config | aspirational | 10 | 6 | script | 설정 키나 장치 지원을 폐지하거나 이름을 바꿀 때는 deprecate 데코레이터로 경고 후 옛 키를 새 키에 재매핑해 기존 동작을 유지하고(생성 |
| DEP-001 | dependency/version | aspirational | 7 | 8 | ast | deprecated 인자·API는 deprecate 데코레이터로 표시하고 제거 예정 버전을 명시하며, 해당 버전에 도달하면 제거한다. 이전 메이 |
| PRC-003 | pr_process | aspirational | 51 | 537 | corpus | PR 본문은 템플릿의 절(변경 유형, 변경 개요, Motivation, Verification, 관련 이슈)을 갖추고, Type of Chang |
| PRC-004 | pr_process | aspirational | 8 | 5 | corpus | 기존 동작의 호환성을 깨는 변경은 PR 본문의 Type of Change에서 Breaking change로 표시한다. |
| PRC-008 | pr_process | aspirational | 11 | 25 | corpus | 선행 PR이나 다른 저장소의 PR에 의존하는 PR은 base 브랜치를 선행 PR 브랜치로 두고, 본문에 선행·관련 PR 번호와 링크, 머지 순서 |
| STY-005 | style/other | aspirational | 290 | 47 | script | 새로 추가하는 소스 파일에는 기존 파일과 같은 형식의 Rebellions 저작권 헤더를 현재 연도로 붙인다. |
| SUB-008 | multimodal/submodule | aspirational | 7 | 4 | grep | HF가 사용하는 dtype 동작(예: fp32로 계산하는 비전 회전 위치 임베딩 테이블, 모듈 입력 dtype)을 그대로 따르며, 활성 dtyp |
| TEST-008 | testing | aspirational | 99 | 31 | script | 새 모델 클래스나 파이프라인을 추가하거나 변경할 때는 해당 테스트 목록(test_transformers, test_diffusers, pytes |
| ATT-006 | attention/kvcache | interview | 4 | 3 | grep | 정적 캐시의 마지막 축과 인코더 시퀀스 길이는 디바이스·컴파일러가 요구하는 단위(예: 64의 배수)에 맞춰 설계하거나 올림한다. |
| LIFE-002 | modeling_base/lifecycle | interview | 48 | 19 | script | 모델 forward의 입력 인자·순서·반환 타입은 원본 HF 모델 구현과 일치시키고(예: token_type_ids 누락 금지), 사용자가 지정 |
| CFG-019 | rbln_config | rejected | 0 | 36 | ast | 서브모듈을 가진 설정 클래스는 서브모듈 목록과 설정 클래스 매핑(submodule_config_classes)을 선언하고 타입 주석을 `RBLN |
| CFG-022 | rbln_config | rejected | 0 | 6 | script | 설정 키는 rbln_config와 from_pretrained의 rbln_ 접두사 인자 두 경로로 모두 지정할 수 있게 하되, 둘을 동시에 주면 |
| PRC-002 | pr_process | rejected | 4 | 34 | corpus | 작업 중인 PR은 제목에 (WIP)를 붙인다. |
| PRC-005 | pr_process | rejected | 0 | 2 | script | 릴리스 PR은 dev를 main으로 병합하는 PR로 만들고, base를 main으로 두며, 제목을 `release: vX.Y.Z` 형식으로 쓰고 |
| PRC-023 | pr_process | rejected | 0 | 0 | script | dev에서 main으로 가는 동기화·백머지와 그래프 토폴로지를 유지해야 하는 릴리스 계열 PR은 squash나 rebase가 아닌 merge c |
| DEP-016 | dependency/version | unverifiable | 10 | 0 | script | torch 버전을 올릴 때는 모델 구현의 혼합 dtype 행렬곱 피연산자를 점검하여 명시적으로 같은 dtype에 맞춘다. |

## 3. 판정 사유와 보정된 룰

### API-003 — verified

- 룰: API·설정 구조·공통 런타임 인터페이스·모델 출력 타입을 제거·교체·변경할 때는 라이브러리 내부 호출부, 하위 클래스와 멀티모달 래퍼, vLLM 등 소비 측 코드, README·문서 예제, model-zoo 예제를 같은 변경에서 함께 고치고 비호환이 없는지 확인한다.
- 검사: dev 트리를 import해 (1) src에서 import한 RBLN 클래스에 대한 Class.attr 접근 중 존재하지 않는 속성(제거된 메서드 호출 등), (2) README.md·examples/의 RBLN* 식별자와 RBLN*.attr 참조 중 optimum.rbln에 없는 것을 위반으로 센다; vLLM·model-zoo 등 외부 저장소 동시 수정은 이 저장소에서 확인 불가.
- 사유: src의 RBLN 클래스 속성 참조 23곳과 README/examples의 RBLN 참조 139곳이 모두 현재 API로 해석되며(PR #612/#614가 고친 RBLNAutoConfig.load 같은 잔존 호출 없음), 저장소 내부 불변식은 지켜진다. 다만 규칙의 vLLM·model-zoo·외부 문서 동시 수정 부분은 다른 저장소라 코드로 검증할 수 없다(judgment/PR 리뷰 항목).
- **보정된 룰:** 공개 API·설정 키·출력 타입을 제거·변경하는 PR은 같은 PR에서 저장소 내부 호출부·하위 클래스·README·examples를 고쳐 제거된 이름을 참조하는 곳이 0이 되게 하고, vLLM·model-zoo 영향 여부는 PR 본문에 명시한다.
- 테스트 제안: tests/test_conventions.py::test_no_stale_rbln_api_references — src/optimum/rbln/**/*.py에서 'from ... import RBLNX'로 들여온 이름의 Attribute 접근(RBLNX.attr)을 AST로 모아 hasattr(getattr(optimum.rbln, 'RBLNX'), attr)를 assert하고, README.md와 examples/**/*.py의 RBLN\w+ 토큰(파일 안에서 직접 class 정의한 이름 제외)이 optimum.rbln에서 resolve되는지 확인. 현재 위반 없음.

### API-011 — verified

- 룰: 새 모델 클래스와 파이프라인용 diffusers 구성요소 클래스는 기존 RBLN 접두사 패턴(RBLN*For* 등)을 따라 이름 짓고 노출한다.
- 검사: optimum.rbln에서 export되는 모든 RBLNBaseModel/RBLNDiffusionMixin 하위 클래스 이름이 'RBLN'+(transformers·diffusers 클래스명 또는 HF 모듈 내부/원격코드 클래스명)인지, 모든 RBLNModelConfig 하위 export가 RBLN*Config인지 확인한다; 공통 베이스(RBLNBaseModel·RBLNModel·RBLNDiffusionMixin·RBLNDecoderOnlyModel[ForCausalLM])는 제외하고 HF 대응명이 없는 이름이 위반.
- 사유: export된 모델·파이프라인 클래스 138개 중 공통 베이스 5개를 빼면 126개가 transformers/diffusers 최상위 클래스명, 6개(Exaone/Midm 원격코드, Qwen2/2.5 VisionTransformer, GroundingDino Encoder/Decoder)가 HF 모듈 내부 클래스명에 RBLN을 붙인 형태이고, 설정 클래스 135개는 모두 RBLN*Config다. HF 대응명이 없는 것은 RBLNGroundingDinoTextModel 1개뿐이다(근거 PR #533은 머지되지 않고 CLOSED).
- **보정된 룰:** 공개 모델·파이프라인·diffusers 구성요소 클래스는 'RBLN' + 대응하는 HF(transformers/diffusers, 원격코드 포함) 클래스명 그대로, 설정은 그 뒤에 'Config'를 붙여 짓는다; 자동 클래스가 convert_hf_to_rbln_model_name으로 이 이름을 찾으므로 이름 규칙이 곧 로딩 계약이다.
- 테스트 제안: tests/test_conventions.py::test_public_model_names_mirror_hf — optimum.rbln의 공개 이름 중 RBLNBaseModel/RBLNDiffusionMixin 하위 클래스(RBLNAuto*·공통 베이스 제외)에 대해 name[4:]이 transformers/diffusers 최상위 속성이거나 cls.get_hf_class().__name__과 같은지 확인하고, 각 클래스에 대해 name+'Config'가 export되는지(MultiControlNet 제외) assert. 허용 목록: RBLNGroundingDinoTextModel, RBLNExaoneForCausalLM, RBLNMidmLMHeadModel(원격코드).

### API-014 — verified

- 룰: 공개 객체는 정의한 모듈의 `__all__`에 등록하고 패키지 `__init__.py`는 고정 템플릿으로 두며, 새 모델은 패키지 최상위 export와 모델 레지스트리에 함께 등록한다.
- 검사: dev 트리에서 (a) 저장소의 scripts/check_inits.py(TYPE_CHECKING 블록·__all__ 정의 여부·lazy 템플릿) 실행, (b) 모든 __all__ 항목이 그 모듈에서 정의된 이름인지(재export 금지), (c) 패키지 __init__.py가 고정 lazy 템플릿과 같은지, (d) HF 클래스명에 대응하는 leaf RBLN 클래스가 optimum.rbln 최상위로 export되는지 확인한다.
- 사유: PR #734가 들어간 origin/dev에서 check_inits.py가 rc=0으로 통과(CI 게이트)하고, __all__ 306개 항목 중 305개가 정의 모듈에 있으며, 패키지 __init__ 77개 중 74개가 고정 템플릿, HF 대응 leaf 모델 149개가 모두 최상위 export(자동 클래스는 이름 기반으로 optimum.rbln에서 찾으므로 export가 곧 레지스트리 등록)된다. 단 워크트리 HEAD(main, 2026-07-31)는 #734 이전이라 손으로 관리하는 _import_structure를 쓰며 __all__ 보유 파일이 6개뿐이다.
- **보정된 룰:** 공개 객체는 정의 모듈의 __all__에만 등록하고(재export 금지) 패키지 __init__.py는 define_import_structure 기반 lazy 템플릿으로 두며, scripts/check_inits.py --fix로 TYPE_CHECKING 블록을 맞춘다; 자동 클래스는 이름(RBLN+HF 클래스명)으로 optimum.rbln에서 찾으므로 최상위 export가 곧 등록이고, optimum.rbln 밖의 사용자 정의 클래스만 RBLNAutoModel.register/RBLNAutoConfig.register를 쓴다.
- 테스트 제안: tests/test_conventions.py::test_import_structure — scripts/check_inits.py의 main()이 0을 반환하는지 확인하고, 추가로 src/optimum/rbln/**/*.py(비 __init__)의 __all__ 항목이 해당 모듈의 ClassDef/FunctionDef/Assign 이름인지(import 이름 불가) 검사하며, transformers/diffusers에 같은 이름(접두사 RBLN 제거)이 있는 모든 RBLN 클래스가 hasattr(optimum.rbln, name)인지 assert. 허용 목록: modeling.py의 RBLNBaseModel 재export, seq2seq/__init__.py, diffusers/__init__.py(LOADABLE_CLASSES 등록).

### API-015 — verified

- 룰: 내부 모듈 경로를 옮기거나 모델 클래스를 배치할 때 호환용 shim 모듈이나 별도 alias 파일을 남기지 않고, 공개 API의 재export만 유지한다.
- 검사: src 모듈 중 import/__all__만 있는 shim 모듈, 모듈 수준 클래스 별칭 대입(RBLNX = RBLNY), sys.modules 별칭 조작을 찾고, 추가로 HF 클래스에 대응하는 leaf RBLN 클래스가 HF와 같은 모델 디렉터리(transformers/models/<dir>, diffusers/<subpath>)에 놓였는지 비교해 어긋난 배치를 위반으로 센다.
- 사유: dev의 비 __init__ 모듈 245개(main 228개) 중 재import만 하는 shim 모듈과 모듈 수준 클래스 별칭은 0개이고 __init__ 밖의 sys.modules 조작도 없다; 별칭 클래스는 PR #157 방식대로 모델별 디렉터리의 빈 본문 하위 클래스(57개)로 존재한다. HF 대응 leaf 클래스 126개 중 125개가 HF와 같은 디렉터리에 있고, RBLNMultiControlNetModel만 PR #726 이동 때 pipelines/controlnet에 남았다.
- **보정된 룰:** 모듈을 옮길 때 옛 경로에 재export shim을 남기지 않고, 모델 클래스는 HF와 같은 디렉터리 구조(transformers/models/<model>/, diffusers/models|pipelines/<...>)에 두며, 이름 별칭이 필요하면 별칭 파일 대신 해당 모델 디렉터리에 빈 하위 클래스로 정의한다.
- 테스트 제안: tests/test_conventions.py::test_no_shim_modules_and_hf_mirrored_layout — src/optimum/rbln/**/*.py(비 __init__) 중 최상위 문장이 Import/ImportFrom/__all__/docstring뿐인 모듈이 없음을 assert하고, 모듈 수준 'CamelName = OtherCamelName' 대입을 금지하며, transformers/diffusers에 대응 클래스가 있는 RBLN 클래스의 파일 경로가 HF 클래스 __module__의 디렉터리와 같은지 비교. 허용 목록: diffusers/pipelines/controlnet/multicontrolnet.py.

### ATT-006 — interview

- 룰: 정적 캐시의 마지막 축과 인코더 시퀀스 길이는 디바이스·컴파일러가 요구하는 단위(예: 64의 배수)에 맞춰 설계하거나 올림한다.
- 검사: 정적 캐시(key_value_states / cross-KV / conv_state·recurrent_state) 입력의 마지막 축과 인코더·디코더 시퀀스 길이 기본값이 64의 배수로 설계되거나 올림되는지 확인한다. 64 배수가 아닌 기본값/마지막 축은 위반(또는 반례).
- 사유: dev에서도 main과 동일하다: 64 단위 정렬은 prefill_chunk_size 검증(modeling_attention_utils.py:95), TST dec_max_seq_len 올림(:237), Qwen3.5 conv_state(innermost=conv_dim, 'a multiple of 64 as RBLN requires')·recurrent_state 레이아웃 4곳에서 지켜지지만, Whisper 인코더 1500·seq2seq 인코더 길이는 올림하지 않고 TST cross-KV 캐시 마지막 축은 head_dim 13이다. 근거 PR #804·#816은 여전히 OPEN이다. 팀 질문: 64 배수 요구가 모든 정적 캐시에 적용되는가, 아니면 레이아웃을 직접 설계하는 커스텀 상태 캐시(conv/recurrent처럼 head_dim이 모델로 고정되지 않는 것)에만 해당하는가? 인코더 시퀀스 길이 올림은 특정 파이프라인 규칙인가 일반 규칙인가?
- **보정된 룰:** 레이아웃을 직접 설계하는 디바이스 상주 정적 상태 캐시(conv cache, recurrent state 등)는 64의 배수인 축을 가장 안쪽(마지막) 축에 둔다. 사용자 값이 아닌 기본값으로 정하는 시퀀스 길이·청크 크기는 64의 배수로 올림하거나 검증한다(prefill_chunk_size, TST dec_max_seq_len 선례).

### CFG-010 — aspirational

- 룰: 설정 키나 장치 지원을 폐지하거나 이름을 바꿀 때는 deprecate 데코레이터로 경고 후 옛 키를 새 키에 재매핑해 기존 동작을 유지하고(생성자 인자와 setattr 경로 포함), 구현 제거는 이후 별도의 breaking change로 한다.
- 검사: 폐지·개명된 설정 키/인자/장치·아티팩트 지원을 모두 찾아, (a) deprecate 헬퍼(deprecate_kwarg·deprecate_method·warn_deprecated_npu)로 경고하는지 (b) 생성자·setattr·rbln_ 접두사 경로 모두에서 옛 키가 새 키로 재매핑되는지 프로브로 확인한다. 수동 logger.warning이거나 setattr 경로에서 재매핑이 안 되면 위반.
- 사유: dev에서 헬퍼 사용 사이트가 6→10으로 늘었다(#666 kvcache_metas→cache_metas deprecate_kwarg, Cosmos 설정 2곳의 fps deprecate_kwarg, qwen3_5 _mask_padded_keys_and_queries deprecate_method; 기존 tensor_parallel_size·_torch_dtype·model_input_shapes·CA02·torch_dtype setter·서브모듈 tensor_parallel_size 경로 유지, 버전은 0.12.0→0.13.0으로 연장). 그러나 16개 사이트 중 6개(37.5%)가 여전히 위반이다: 수동 logger.warning 4곳(optimize_host_memory, precision, torch_dtype×2)과 gemma4 아티팩트 dtype 경고, 그리고 프로브 결과 `c.tensor_parallel_size=2`→num_devices None·직렬화됨, `C.initialize_from_kwargs(C(), rbln_tensor_parallel_size=2)`→num_devices None, `c.kvcache_metas=[]`도 직렬화됨으로 setattr 경로 재매핑이 main(6/5)과 동일하게 빠져 있다. #760(setattr 경로 지적)은 아직 OPEN.
- **보정된 룰:** 설정 키를 폐지·개명할 때는 즉시 제거하지 않고 경고 후 새 키로 재매핑해 기존 설정 파일과 사용자 코드를 유지한다. RBLNModelConfig 생성자 인자는 deprecate_kwarg(version=...)로, 속성 대입(setattr)·인스턴스+rbln_ 접두사 경로는 property setter 등으로 함께 덮어야 하며, 구 아티팩트·장치 지원 종료는 deprecate_method / warn_deprecated_npu 같은 헬퍼로 경고한다. 구현 제거는 해당 version 이후 별도 breaking change로 한다.
- 테스트 제안: tests/test_conventions.py::test_deprecations_use_helpers — src/optimum/rbln 전체(.py; diffusers/pipelines의 upstream 복사 pipeline_*.py와 utils/deprecation.py, HF 메시지를 그대로 옮긴 blip_2 제외)에서 'deprecated' 문자열을 담은 logger.warning/warning_once 호출을 AST로 찾아 실패시키고, 현재 위반(configuration_utils.py optimize_host_memory·torch_dtype property, rbln_quantization.py precision/torch_dtype, gpt_oss _get_dtype, gemma4 image_features dtype)을 allowlist로 둔다. 추가로 RBLNModelConfig 하위 클래스의 모든 deprecate_kwarg(new_name=...) 대상(tensor_parallel_size→num_devices, kvcache_metas→cache_metas)에 대해 `cfg=C(); setattr(cfg, old, v)` 후 `getattr(cfg, new)==v and old not in cfg._prepare_for_serialization()`을 검사하는 파라미터화 테스트를 두고 두 항목 모두 xfail로 시작한다.

### CFG-018 — verified

- 룰: 런타임 전용 값(콜백 등)과 컴파일 시점에만 쓰는 추정 입력은 subclass_non_save_attributes에 등록해 rbln_config.json에 직렬화되지 않게 하고, 하위 클래스가 이 목록을 재정의하면 상위 항목을 함께 포함한다.
- 검사: 모든 RBLNModelConfig 하위 클래스를 import해 subclass_non_save_attributes를 재정의한 클래스가 직계 부모의 유효 목록을 모두 포함하는지, Callable 타입 __init__ 인자/비공개 런타임 속성(self._x)·추정 입력이 비저장 목록에 등록돼 있는지 검사한다. 부모 항목 누락이나 미등록 런타임 전용 속성은 위반.
- 사유: dev에서 설정 클래스가 153→171개로 늘었지만 subclass_non_save_attributes를 재정의한 10개 모두 부모 항목(memory_budget 등)을 포함하고, 비저장 대상 비공개 속성(_load_visual_runtime, _batch_size_is_specified)과 추정 입력(memory_budget)은 할당하는 모든 파일에서 등록돼 있다(main과 동일 판정). 새로 생긴 _requires_batch_sort(#559)는 등록되지 않았지만 modeling_decoderonly.py 주석대로 'serialized so a loaded model knows to sort' — 컴파일 시점에 결정되는 값을 의도적으로 저장하는 것이라 위반이 아니다. Callable 타입 인자는 여전히 없다(#623·#750은 CLOSED 미병합).
- 테스트 제안: tests/test_conventions.py::test_subclass_non_save_attributes_not_shadowed — optimum.rbln의 모든 *Config를 import해 RBLNModelConfig 하위 클래스를 재귀 수집하고, `'subclass_non_save_attributes' in cls.__dict__`인 클래스마다 `set(cls.__mro__[1].subclass_non_save_attributes) <= set(cls.subclass_non_save_attributes)`를 단언한다. 함께 inspect.signature(cls.__init__)에서 Callable 주석 인자를 찾아 non_save_attributes ∪ subclass_non_save_attributes에 있는지 확인한다. allowlist 불필요.

### CFG-019 — rejected

- 룰: 서브모듈을 가진 설정 클래스는 서브모듈 목록과 설정 클래스 매핑(submodule_config_classes)을 선언하고 타입 주석을 `RBLNModelConfig | None`으로 쓰며, dict에서 설정 객체로의 변환은 각 클래스에서 수동으로 하지 않고 공통 검증 훅에서 자동 처리한다.
- 검사: configuration_*.py에서 submodules를 선언한 클래스마다 submodule_config_classes 매핑 존재 여부, 서브모듈 __init__ 인자의 타입 주석(영역별), __init__에서 initialize_submodule_config를 직접 호출하는지를 AST로 센다. 매핑 없이 클래스별 __init__에서 dict→설정 변환을 하면 위반.
- 사유: dev(74e97fa4)에서도 `submodule_config_classes`·`model_validator`는 src/tests 어디에도 없고(0건), 서브모듈을 가진 36개 클래스(main 32) 모두 매핑이 없으며 36개 전부가 __init__에서 `self.initialize_submodule_config(...)`를 직접 호출한다(main 31/32 → EXAONE-4.5도 헬퍼로 전환). 공통 경로의 실제 계약은 docstring "Returns the submodule config instance when its class is known (an instance was passed, or `cls_name` was given), otherwise the merged kwargs dict for the model class to resolve later."와 `__setattr__`의 `if key in self.submodules and isinstance(value, dict) and (cls_name := value.get("cls_name")):`로, dict는 cls_name이 있을 때만 변환되고 나머지는 모델 클래스가 나중에 해석한다(그래서 gemma4 설정은 `if isinstance(vt_cfg, dict):`로 수동 분기). 근거 PR #495는 2026-07-02 CLOSED(미병합). 타입 주석만 보면 transformers 32/34가 `RBLNModelConfig | None`, diffusers는 45/48이 구체 클래스라 영역별로 갈린다(main 31/33, 36/39).
- **보정된 룰:** 서브모듈을 가진 설정 클래스는 `submodules` 클래스 속성을 선언하고, __init__에서 각 서브모듈을 공통 헬퍼 `self.initialize_submodule_config(submodule_config, ...)`로 처리해 부모의 런타임 옵션(npu, num_devices, create_runtimes 등)을 전파한다. 클래스를 모르는 dict(cls_name 없음)는 dict로 남겨 모델 클래스가 나중에 해석하므로, 설정 클래스 안에서 서브모듈 값을 읽는 코드는 dict와 설정 객체 둘 다 처리해야 한다. transformers 쪽 서브모듈 인자 주석은 `RBLNModelConfig | None`을, diffusers 파이프라인은 구체 설정 클래스를 쓴다.
- 테스트 제안: tests/test_conventions.py::test_submodules_initialized_via_helper — configuration_*.py를 AST로 순회해 `submodules = [...]`가 비어있지 않은 클래스 중 __init__을 정의한 클래스는 initialize_submodule_config 호출이 최소 1회 있어야 한다고 단언한다(dev 기준 36/36 통과, allowlist 불필요). 보조로 transformers/ 아래 클래스의 서브모듈 인자 주석이 `RBLNModelConfig | None`인지 검사하고 grounding_dino의 encoder/decoder 2개를 allowlist로 둔다.

### CFG-022 — rejected

- 룰: 설정 키는 rbln_config와 from_pretrained의 rbln_ 접두사 인자 두 경로로 모두 지정할 수 있게 하되, 둘을 동시에 주면 ValueError를 발생시킨다.
- 검사: rbln_config(dict 또는 인스턴스)와 rbln_ 접두사 인자를 동시에 넘겼을 때 ValueError가 나는지 컴파일 경로(initialize_from_kwargs)와 로드 경로(RBLNModelConfig.from_pretrained, RBLNBaseModel.from_pretrained의 dict→kwargs 변환 에뮬레이션)에서 실제 호출로 확인한다. 오류 없이 병합되면 위반.
- 사유: dev에서도 동시 지정 시 ValueError를 내는 경로는 없다(#445는 2026-01-23 CLOSED 미병합). 대신 공식 계약은 우선순위다: 클래스 docstring "Method 4: Combining a config object with override parameters  # (rbln_ prefixed parameters take precedence over rbln_config values)"(configuration_utils.py:407-408)와 test_config_priority, 그리고 #702가 RBLNModelConfig.from_pretrained의 `if key not in kwargs:`를 `if f"rbln_{key}" not in kwargs:`로 고쳐 로드 경로도 접두사 우선으로 맞췄다(프로브 `from_pretrained(d, rbln_config={'device':0}, rbln_device=1).device`: main 0 → dev 1). 단 RBLNBaseModel.from_pretrained는 여전히 `kwargs.update(rbln_config_as_kwargs)`로 dict가 접두사를 덮어써 에뮬레이션 결과 device=0이며, 컴파일 시점 키를 저장값과 다르게 주면 `ValueError("Cannot set the following arguments ...")`가 나는 것이 유일한 오류 경로다.
- **보정된 룰:** 설정 키는 rbln_config(dict 또는 RBLN*Config 인스턴스)와 from_pretrained의 rbln_ 접두사 인자 두 경로로 지정할 수 있으며, 둘을 함께 주면 오류 없이 병합되고 rbln_ 접두사 인자가 우선한다(dict 병합은 nested_update처럼 키 단위로 덮어씀). 컴파일된 모델을 로드할 때 저장된 컴파일 시점 키를 다른 값으로 바꾸려 하면 ValueError("Cannot set the following arguments")가 난다. 현재 RBLNBaseModel.from_pretrained 로드 경로(modeling_base.py:246-247)는 dict가 이기므로 이 계약과 어긋난다.
- 테스트 제안: tests/test_conventions.py::test_rbln_prefix_overrides_rbln_config — 컴파일 없이 RBLNResNetForImageClassificationConfig.initialize_from_kwargs({'batch_size': 4}, rbln_batch_size=16)과 인스턴스 경로에서 batch_size==16을, 저장된 설정 디렉터리에 대해 RBLNResNetForImageClassificationConfig.from_pretrained(d, rbln_config={'device': 0}, rbln_device=1).device==1을 단언한다. RBLNBaseModel.from_pretrained 로드 경로의 dict→kwargs 변환(modeling_base.py:246-247)이 접두사를 덮어쓰지 않는지 검사하는 케이스는 xfail로 시작한다.

### DEP-001 — aspirational

- 룰: deprecated 인자·API는 deprecate 데코레이터로 표시하고 제거 예정 버전을 명시하며, 해당 버전에 도달하면 제거한다. 이전 메이저 버전 전용 기능도 같은 방식으로 deprecate 처리한다.
- 검사: src/ 전체를 AST로 훑어 deprecate_kwarg/deprecate_method 데코레이터 호출의 version 인자를 현재 패키지 버전과 비교하고(≤ 현재 버전이면 위반), 데코레이터 없이 logger.warning*로 'deprecat' 메시지를 내는 ad-hoc deprecation(업스트림 HF 복사 메시지 제외)을 위반으로 센다.
- 사유: dev tip(74e97fa, 버전 0.12.1.dev)에서 데코레이터 사용 7곳은 모두 제거 버전 0.13.0 > 현재로 정상이지만, 데코레이터 없이 버전도 없는 ad-hoc deprecation 경고가 8곳(53%) 남아 있다(worktree HEAD v0.11.1 기준 3 대 7). 또 '해당 버전 도달 시 제거'는 실제로는 #785('move the 0.12.0 deprecation cutoff to 0.13.0')처럼 제거 대신 컷오프를 미루는 방식으로 처리되었다.
- **보정된 룰:** 사용자 공개 kwarg·메서드의 deprecation은 utils/deprecation.py의 deprecate_kwarg/deprecate_method로 표시하고 version을 반드시 적는다. 데코레이터로 표현할 수 없는 경우(property, 클래스 생성자, kwargs pop)에도 경고 메시지에 제거 예정 버전을 명시한다. 현재 버전이 제거 버전에 도달하면 릴리스 전에 제거하거나, 연기한다면 별도 PR로 컷오프를 명시적으로 옮긴다.
- 테스트 제안: tests/test_conventions.py::test_deprecations_have_live_removal_version — src/optimum/rbln/**/*.py를 AST로 순회해 (1) deprecate_kwarg/deprecate_method 호출의 version이 importlib.metadata의 optimum-rbln base 버전보다 큰지 확인(같거나 작으면 실패 = 제거할 때가 됨), (2) logger.warning*/warnings.warn 문자열에 'deprecat'가 있으면서 r'\d+\.\d+\.\d+' 버전 표기가 없는 호출을 실패로 처리한다. 업스트림 복사 메시지(blip_2의 'v4.50')는 제외하고, 현재 위반 8곳(configuration_utils.py optimize_host_memory·torch_dtype property 2곳, decoderonly_architecture.py DecoderOnlyFlashAttention, gpt_oss/modeling_gpt_oss.py, rbln_quantization.py 2곳; gemma4는 버전이 있으므로 (2)에서 통과)을 파일:함수 단위 allowlist로 둔다.

### DEP-008 — verified

- 룰: 모델 구현에 필요한 새 외부 의존성은 pyproject.toml에 명시하되, 필수 기능이 아닌 패키지는 기본 의존성에 넣지 않는다.
- 검사: src/의 모든 import를 AST로 모아 stdlib·내부·전이 의존성을 뺀 서드파티 배포판이 pyproject.toml의 dependencies/optional-dependencies에 선언되었는지, 기본 의존성이 아닌 패키지는 모듈 최상위가 아닌 가드(is_*_available) 아래에서 import되는지 확인한다.
- 사유: dev tip에서 필수 서드파티 10개 패키지(torch, transformers, diffusers 등; rebel-compiler는 사내 인덱스라 의도적으로 제외)는 모두 기본 의존성에 선언되어 있고, 비필수 패키지(cosmos_guardrail, retinaface)는 기본 의존성에 없이 is_cosmos_guardrail_available() 가드 뒤에서만 import된다. 위반으로 잡힌 3곳은 '명시' 요건만 어긴 것으로, #168 리뷰('not a necessary dependency')의 결정과 일치한다. 근거 PR #533(einops/trimesh 추가)은 병합되지 않았다(CLOSED).
- **보정된 룰:** 모델 구현에 반드시 필요한 외부 패키지는 pyproject.toml 기본 dependencies에 버전과 함께 선언한다. 특정 파이프라인 전용 비필수 패키지(예: cosmos guardrail)는 기본 의존성에 넣지 않고 utils의 is_*_available() 가드 아래에서 import하며, 미설치 시 대체 클래스로 실패를 늦춘다(optional extra 선언은 현재 관행이 아니다).
- 테스트 제안: tests/test_conventions.py::test_third_party_imports_declared_or_guarded — src/optimum/rbln/**/*.py의 import를 AST로 모아 stdlib/optimum/rebel 및 전이 의존성 허용 목록(numpy, huggingface_hub, safetensors, PIL 등)을 빼고, 남은 배포판이 pyproject [project].dependencies에 없으면 해당 import 노드의 조상에 `if is_*_available()` 또는 try/except ImportError가 있는지 확인해 없으면 실패한다. 현재 allowlist 없음(cosmos_guardrail, retinaface는 가드 조건으로 통과).

### DEP-011 — verified

- 룰: torch.load는 weights_only를 명시적으로 지정한다.
- 검사: src/, tests/, examples/, scripts/의 torch.load 호출과 partial(torch.load, ...)을 AST로 찾아 weights_only 키워드가 없는 호출을 위반으로 센다.
- 사유: dev tip에서 torch.load 27곳 중 25곳(93%)이 weights_only를 명시하며(자체 저장 artifact는 False, 외부 체크포인트는 True), worktree HEAD도 23/2로 같다. 남은 2곳은 LoRA 어댑터 .bin 로더로 torch 2.6+ 기본값 True에 기대고 있다.
- **보정된 룰:** torch.load는 항상 weights_only를 명시한다: 라이브러리가 직접 저장한 torch_artifacts.pth 등은 weights_only=False, 사용자·외부에서 받은 체크포인트(LoRA 어댑터 등)는 weights_only=True.
- 테스트 제안: tests/test_conventions.py::test_torch_load_explicit_weights_only — src/, examples/를 AST로 순회해 torch.load(...) 호출과 functools.partial(torch.load, ...)에 weights_only 키워드가 없으면 실패한다. 현재 allowlist: src/optimum/rbln/transformers/models/decoderonly/lora_architecture.py 79·80행(수정 후 제거).

### DEP-012 — verified

- 룰: PR CI가 쓰는 컴파일러 버전은 .github/version.yaml 한 곳에서 고정하고 모든 CI가 그 값을 읽게 하며, 컴파일러 업데이트가 필요하면 그 파일에 반영하고 pytest에서 쓰는 배포 컴파일러 버전과 일치하는지 확인한다.
- 검사: CI 설정(.github/workflows, .buildkite, scripts/ci)에서 rebel-compiler를 설치하거나 버전을 결정하는 모든 지점이 .github/version.yaml의 rebel_compiler_version을 읽는지, 그리고 버전 문자열을 하드코딩한 곳(rebel-compiler==<숫자>)이 없는지 확인한다.
- 사유: dev tip에서 컴파일러 버전을 결정하는 3곳(scripts/ci/install-rebel-compiler.sh, .github/workflows/auto_compiler_update.yaml, scripts/ci/slack-report.py)이 모두 .github/version.yaml을 읽고 하드코딩된 핀은 0건이다. worktree HEAD(v0.11.1)에서도 GHA 5개 워크플로(rbln_trigger_on_pr, rbln_scheduled_test_impl, bc_compile, bc_inference, auto_compiler_update_impl)가 같은 grep으로 읽고 rbln_optimum_pytest.yaml 한 곳에서만 설치한다. 단 REBEL_COMPILER_VERSION 환경변수 override(rebel-compiler-override.sh)가 수동 실행용 탈출구로 존재하며, '배포 컴파일러 버전과 일치 확인'은 정적으로 검증할 수 없다.
- **보정된 룰:** CI가 설치하는 rebel-compiler 버전은 .github/version.yaml의 rebel_compiler_version 한 곳에서만 정하며, CI 스크립트·워크플로는 이 값을 읽거나 명시적 override 변수(REBEL_COMPILER_VERSION)로만 바꾼다. 버전 갱신은 auto_compiler_update 봇 PR 또는 이 파일 수정으로만 한다.
- 테스트 제안: tests/test_conventions.py::test_compiler_version_single_source — .github/workflows/*.y*ml, .buildkite/**/*.yml, scripts/ci/*.sh, pyproject.toml, tests/**/*.py를 텍스트로 읽어 r'rebel-compiler\s*(==|>=|~=)\s*\d' 패턴(하드코딩된 핀)이 있으면 실패시키고, 'rebel-compiler==' 설치 명령이 있는 파일은 같은 파일 안에 'version.yaml' 또는 '${{ inputs.rebel_compiler_version }}'/'${REBEL_COMPILER_VERSION}' 참조가 있어야 통과한다. 또 .github/version.yaml이 정확히 한 줄 'rebel_compiler_version: <PEP440>' 형식인지 검사한다. 현재 allowlist 없음.

### DEP-015 — verified

- 룰: 외부 라이브러리가 deprecated로 표시한 속성, 호출 방식, import 경로는 권장 대체 방식으로 옮긴다.
- 검사: src/를 AST로 훑어 현재 핀된 transformers/diffusers/torch에서 deprecated인 사용(config.use_return_dict 읽기, 단수형 diffusers.models.controlnet import, from_pretrained의 torch_dtype=/use_auth_token=, torch.cuda.amp)을 위반으로, config.return_dict 읽기와 diffusers.models.controlnets import를 준수로 센다.
- 사유: dev tip(74e97fa)에서는 #808 이후 config.use_return_dict 0건, 준수 37곳, 위반 0건이다. 다만 worktree HEAD(main v0.11.1, #808 이전)에는 use_return_dict 읽기 13곳이 남아 있어(준수 2) main은 다음 릴리스 병합 때 해소된다. 검사 대상 패턴 목록은 알려진 deprecation만 다루는 근사치다.
- 테스트 제안: tests/test_conventions.py::test_no_deprecated_upstream_apis — src/optimum/rbln/**/*.py를 AST로 읽어 (a) Attribute 'use_return_dict' 접근, (b) 'diffusers.models.controlnet'(복수형 controlnets 제외) import, (c) from_pretrained 호출의 torch_dtype=/use_auth_token= 키워드, (d) torch.cuda.amp 접근을 찾으면 실패한다. 패턴 목록은 상단 DEPRECATED_UPSTREAM 표에 (패턴, 대체, 근거 PR)로 두고 의존성 bump PR마다 추가한다. dev 기준 allowlist 없음.

### DEP-016 — unverifiable

- 룰: torch 버전을 올릴 때는 모델 구현의 혼합 dtype 행렬곱 피연산자를 점검하여 명시적으로 같은 dtype에 맞춘다.
- 검사: src/의 행렬곱 지점(torch.matmul/bmm/baddbmm/einsum/mm, @ 연산자)을 AST로 찾아 피연산자나 직전 3줄의 바인딩에 .to()/.type()/.float() 등 명시적 dtype 정렬이 보이는지 세는 근사 검사다. 피연산자의 실제 dtype은 정적으로 알 수 없어 위반 판정은 하지 않는다.
- 사유: dev tip(torch 2.13 핀)에서 행렬곱 40곳 중 10곳에 명시적 dtype 정렬이 보이고(HEAD torch 2.11에서는 38곳 중 3곳; #773·#774 수정 반영 확인), 나머지 30곳은 피연산자가 같은 dtype인지 정적으로 판정할 수 없다. 실제 보장은 torch bump 후 컴파일 테스트 통과로만 확인되므로 이 규칙은 코드 검사보다 절차(judgment) 성격이다.
- **보정된 룰:** torch 메이저/마이너 bump PR에서는 전체 컴파일 테스트([pytest-full])를 돌려 혼합 dtype 행렬곱 실패를 찾고, 실패 지점은 attention 확률·마스크 등을 query/value의 dtype으로 명시적으로 .to(dtype) 캐스팅해 고친다.

### DEP-033 — verified

- 룰: 의존성 봇이 만든 업데이트 PR은 자동 병합하지 않고 검토 후 사람이 직접 병합한다.
- 검사: 2025-01 이후 병합된 PR 중 본문에 Renovate 서명(renovatebot/renovate) 또는 create-pull-request 서명이 있는 의존성 봇 PR에 대해 gh로 mergedBy와 타임라인의 auto_merge_enabled 이벤트를 조회해, 자동 병합이 켜졌거나 봇 계정이 병합한 경우를 위반으로 센다.
- 사유: 2025-01 이후 병합된 봇 생성 의존성 PR 42건(Renovate 7, auto_compiler_update create-pull-request 35) 모두 auto-merge 이벤트가 없고 rebel-kblee/rebel-jongho/rebel-thkim/rebel-dkhong/rebel-seinpark가 직접 병합했다(100%). .github/renovate.json에도 automerge 설정이 없다(기본값 비활성).
- **보정된 룰:** Renovate(transformers/diffusers)와 auto_compiler_update 워크플로가 만든 dependency: PR은 auto-merge를 켜지 않고, [pytest-full] 테스트 결과를 확인한 팀원이 직접 병합한다.
- 테스트 제안: tests/test_conventions.py::test_renovate_automerge_disabled — .github/renovate.json을 json으로 읽어 최상위와 모든 packageRules에 automerge/platformAutomerge가 true로 설정되어 있지 않은지 확인하고, .github/workflows/auto_compiler_update*.yaml에 'gh pr merge --auto' 또는 peter-evans/enable-pull-request-automerge 사용이 없는지 텍스트 검사한다. 병합 방식 자체는 코드로 강제할 수 없으므로 설정 회귀만 막는다. allowlist 없음.

### DOC-011 — verified

- 룰: Google 스타일 docstring에서 파라미터 선언(이름과 괄호 속 타입)과 Returns 설명은 각각 한 줄에 쓰고, 파라미터 이름은 첫 `:` 앞에서 끝나게 하며, 기본값은 "Defaults to"로 명시한다.
- 검사: src/optimum/rbln의 공개 클래스·함수·메서드(__init__/__call__ 포함 1,817개, docstring 551개)를 griffe Google 파서(mkdocstrings와 동일)로 파싱한다. 위반은 세 가지다. ① 파싱된 파라미터 이름이 식별자가 아님(선언이 두 줄로 나뉘었거나 첫 `:` 앞에서 이름이 끝나지 않음) ② 반환 주석이 tuple이 아닌데 Returns가 여러 행으로 파싱됨 ③ 파라미터 기본값을 밝히면서 'Defaults to'가 아닌 표현(by default, default=, will default to)을 씀. ③에서 기본값이 아니라 동작을 설명하는 문장 2건(GroundingDino input_ids, Qwen3_5 vision batch_size)은 수작업으로 제외했다.
- 사유: dev 팁에서 공개 docstring 파라미터 944개 중 942개가 한 줄 선언으로 식별자 이름까지 파싱된다. Returns 섹션 79개 중 77개가 한 행으로 파싱되고, 기본값을 밝힌 설명 123개 중 119개가 'Defaults to'를 쓴다. 위반은 main의 13건에서 7건(0.6%)으로 줄었다. ControlNet 분할 선언 6건은 #783에서 고쳐졌고, dev에서 새로 추가된 코드에는 위반이 없다. 검증기는 dev에도 남아 있다(scripts/validate_docstrings.py + scripts/check-docstrings.sh, .buildkite/pr.yml은 변경 파일만, nightly.yml은 DOCSTRINGS_ALL_FILES=1로 전체 파일 검사). 하지만 infer_module_name이 파일명에 configuration/modeling이 없는 파일(pipeline_*.py, utils/runtime_utils.py)을 건너뛴다. 또 여러 행 Returns는 griffe 경고를 내지 않아 --strict 빌드를 통과한다. 그래서 남은 7건이 게이트에 걸리지 않는다.
- **보정된 룰:** 공개 API docstring은 Google 스타일만 쓴다(numpy 스타일 `name : type` 금지). 파라미터 선언 `name (type): 설명`의 이름과 타입은 한 줄에 쓰고, 이어지는 설명은 더 깊게 들여쓴다. Returns는 `type: 설명` 한 행으로 쓰고, 이어지는 줄은 더 들여쓴다(같은 들여쓰기는 별도 반환값으로 파싱됨). 기본값은 'Defaults to X.'로 쓴다. 검사 범위는 렌더되는 모든 파일(pipeline_*.py, utils/* 포함)과 __call__/__init__이다.
- 테스트 제안: tests/test_conventions.py::test_google_docstring_shape — griffe(mkdocstrings-python 의존성이므로 quality 그룹에 추가하거나 importorskip)로 `griffe.load('optimum.rbln', search_paths=['src'], allow_inspection=False)`를 하고 모든 공개 객체(이름이 _로 시작하지 않거나 dunder인 class/function, 공개 모듈 경로)의 docstring을 `griffe.parse_google`로 파싱한다. (a) parameters 섹션 항목 이름이 `^\*{0,2}[A-Za-z_]\w*$`가 아니면 실패한다. (b) returns 섹션이 2행 이상인데 객체의 returns 주석이 tuple로 시작하지 않으면 실패한다. (c) 설명에 `default\s*[=:]|will default to|(?:set to|is) \S+ by default`가 있고 `defaults to`가 없으면 실패한다(동작을 설명하는 'ignored by default'류는 걸리지 않게 좁힌다). 허용목록은 dev 기준 runtime_utils.get_available_dram_per_chiplet(numpy 스타일), RBLNModel.forward Returns, RBLNStableDiffusionXLControlNetImg2ImgPipeline.__call__ Returns, RBLNModelConfig.__init__.timeout, ControlNet SDXL 두 파이프라인의 target_size다. 아울러 scripts/validate_docstrings.py의 infer_module_name이 pipeline_*·utils 파일을 건너뛰는 점을 고치도록 제안한다.

### DOC-012 — verified

- 룰: docstring의 문서 링크는 상대 경로가 아니라 절대 URL로 쓰고, SDK 문서 빌드 결과에 실제로 존재하는 경로만 가리킨다.
- 검사: src/optimum/rbln의 모든 모듈·클래스·함수 docstring에서 마크다운 링크 `[text](target)`를 추출한다. target이 http(s)/#/mailto로 시작하지 않는 상대 경로이거나, docs.rbln.ai URL인데 rbln_sdk_doc의 docs/ 트리에 대응하는 .md가 없으면 위반이다. 참고용으로 docstring이 아닌 런타임 메시지 문자열의 docs.rbln.ai URL도 같은 방식으로 확인한다.
- 사유: dev 팁에서 docstring 링크 44개 중 42개가 절대 URL이고, docs.rbln.ai 대상은 모두 SDK 문서 트리(2026-10-13 체크아웃)에 존재한다. 결과는 main과 같다. 위반 2건은 T5의 `../../../index.md`뿐이다. scripts/check-docstrings.sh가 modeling_t5.py를 하드코딩으로 제외하며 'drop this once the link is absolute'라고 적어 두었으므로, 의도가 확인된 미해결 부채다. 참고로 docstring 밖 런타임 메시지의 docs.rbln.ai URL 중 4개는 SDK 문서 트리에 대응 페이지가 없다. 3개는 about_atom/release_note.html(import_utils.py:180·188, dev에서 새로 생긴 modeling_attention_utils.py:484)이고 실제 페이지는 supports/release_note다. 나머지 1개는 vllm_support/vllm-rbln.html(decoderonly_runtime_utils.py:34)이다.
- **보정된 룰:** docstring(그리고 사용자에게 보이는 에러·경고 메시지)의 SDK 문서 링크는 https://docs.rbln.ai/... 절대 URL로 쓴다. 대상 경로는 rbln_sdk_doc 빌드에 존재하는 페이지여야 하며, .html 경로는 docs/<path>.{en.,}md로 대응된다. 상대 경로는 throwaway mkdocs 검사(scripts/validate_docstrings.py)에서 깨지므로 금지한다.
- 테스트 제안: tests/test_conventions.py::test_docstring_links_are_absolute — src/optimum/rbln/**/*.py를 ast로 읽어 Module/ClassDef/FunctionDef의 첫 문자열 상수(docstring)에서 `\[[^\]]*\]\(([^)\s]+)\)`를 찾는다. target이 `https?://|#|mailto:`로 시작하지 않으면 실패한다. 허용목록은 transformers/models/t5/modeling_t5.py의 2건이고, 고쳐지면 scripts/check-docstrings.sh의 exclude 목록과 함께 지운다. docs.rbln.ai 경로의 존재 확인은 SDK 문서 체크아웃이 필요하므로, RBLN_SDK_DOC_ROOT 환경변수가 있을 때만 실행하는 선택 테스트로 둔다(<SCRATCH>/stage_d/scripts/test_doc_doc012.py의 exists() 로직). 이 선택 테스트는 런타임 메시지 문자열의 URL도 함께 검사한다.

### LIFE-002 — interview

- 룰: 모델 forward의 입력 인자·순서·반환 타입은 원본 HF 모델 구현과 일치시키고(예: token_type_ids 누락 금지), 사용자가 지정한 입력 이름이 forward 시그니처에 없으면 에러를 raise하며, 입력에 token_type_ids가 있는 경우와 없는 경우를 모두 검증한다.
- 검사: origin/dev 트리를 import해 forward를 직접 정의한 RBLN 모델 클래스마다 get_hf_class()의 HF forward와 파라미터를 비교한다: 공통 입력의 위치 순서가 다르거나(ORDER), HF의 데이터 입력이 RBLN에 없고 **kwargs도 없으면(MISSING) 위반이며, 별도로 generic 경로의 입력 이름 검증(raise)과 token_type_ids 유/무 테스트 존재를 grep으로 확인한다.
- 사유: 입력 집합 일치(MISSING은 Blip2QFormer 1건)와 사용자 지정 입력 이름 검증 raise(modeling_generic.py 143/151행, compile 입력 순서도 HF 시그니처 순서로 정렬), token_type_ids 유/무 테스트(BERT 계열 vs ModernBERT)는 지켜지지만, Python forward의 위치 인자 순서는 forward 정의 67개 중 18개(클래스 97개 중 43개)가 HF와 다르다(대부분 DecoderOnly·Qwen-VL 계열이 inputs_embeds를 2번째에 둠). 팀이 정할 질문: '순서 일치'가 컴파일 그래프 입력 순서(PR #11, 현재 준수)만 뜻하는가, 아니면 사용자에게 노출되는 forward의 위치 인자 순서까지 HF와 맞춰야 하는가?
- **보정된 룰:** RBLN forward는 HF forward가 받는 데이터 입력(token_type_ids 등)을 모두 받거나 **kwargs로 흡수해야 하고, 컴파일 입력(rbln_model_input_names)은 HF forward 시그니처 순서로 정렬하며 시그니처에 없는 이름은 ValueError로 거부한다. (Python forward 위치 인자 순서까지 맞출지는 미정)
- 테스트 제안: tests/test_conventions.py::test_forward_accepts_hf_inputs — optimum.rbln의 모든 RBLNBaseModel 하위 공개 클래스 중 forward를 직접 정의한 클래스에 대해 inspect.signature로 get_hf_class().forward와 비교하여, HF의 데이터 입력(input_ids, attention_mask, token_type_ids, position_ids, inputs_embeds, pixel_values 등)이 RBLN 시그니처에 없고 VAR_KEYWORD도 없으면 실패. 허용 목록: RBLNBlip2QFormerModel(attention_mask). 위치 순서 검사는 팀 결정 후 ORDER 위반 18개 forward 정의(decoderonly 671/878, qwen*_vl, exaone4_5, gemma3/4, llava_next, colqwen2, cosmos transformer, qwen3_asr)를 allowlist로 두고 추가.

### LIFE-007 — verified

- 룰: monkey patch로 전역 함수나 메서드를 바꾼 경우 실행 뒤 원래대로 복원하며, 컴파일이 실패해도 복원되도록 try/finally로 감싼다.
- 검사: import된 모듈/클래스(전역 상태)의 속성에 대입하거나 setattr하는 지점을 AST로 찾고, 패치 대입이 finally 블록이 있는 try 본문 안에 있고 같은 속성을 finally에서 원복하는지(또는 try/finally로 감싼 데코레이터 경유인지) 확인한다; 원복 없음·finally 없음이 위반.
- 사유: 전역 monkey patch는 3곳(decoderonly의 torch.nn.functional.linear, cosmos VAE의 CosmosCausalConv3d.forward, grounding_dino의 HF 클래스 forward 4개)뿐이며 모두 try/finally(grounding_dino는 monkey_patch_decorator 내부 try/finally)로 컴파일 실패 시에도 원복한다(dev·main 동일). logging.py의 sys.stderr=None 대체는 HF 관례의 영구 초기화라 제외했고, bart/t5/swin의 인스턴스 forward 교체(MethodType)는 전역이 아니라 범위 밖이다. 참고로 decoderonly는 original_linear를 try 안에서 캡처해 앞선 줄이 실패하면 finally에서 NameError가 날 수 있다.
- **보정된 룰:** 모듈 함수나 import한 클래스의 메서드(전역 상태)를 바꿀 때는 원본을 try 이전에 캡처하고 try/finally로 감싸 컴파일 실패 시에도 원복한다; 컴파일 대상 모델 인스턴스의 서브모듈 forward 교체(MethodType)는 이 규칙의 대상이 아니다.
- 테스트 제안: tests/test_conventions.py::test_global_monkey_patches_restored_in_finally — src/optimum/rbln/**/*.py를 AST로 순회하며 import된 이름을 루트로 하는 Attribute 대입/setattr을 찾고, 각 대입이 finalbody를 가진 ast.Try의 body 안에 있으면서 finalbody에 같은 dotted target 대입이 있는지 확인. 허용 목록: utils/logging.py의 sys.stderr, grounding_dino_architecture.py의 monkey_patch/restore_monkey_patch(데코레이터 monkey_patch_decorator가 try/finally임을 별도 assert).

### LIFE-020 — verified

- 룰: 오버라이드되는 메서드(get_compiled_model, get_input_info 등)는 공통 계약을 유지하여, 개별 값을 여러 인자로 받지 않고 모델 설정 객체와 컴파일 설정 객체를 받으며 쓰지 않는 키워드 인자도 받아서 무시한다.
- 검사: 오버라이드 계약 메서드(get_compiled_model, get_input_info, _update_rbln_config, _wrap_model_if_needed 등)의 모든 정의를 AST로 모아 파라미터 목록이 베이스 계약과 같은지 비교하고, 개별 값 인자나 다른 시그니처를 쓰는 오버라이드를 위반으로 센다.
- 사유: dev 기준 get_compiled_model 11/11이 (cls, model, rbln_config), get_input_info 7/7이 (cls, batch_size, query_length, rbln_config, model_config), _update_rbln_config 54/55가 (cls, preprocessors, model, model_config, rbln_config), _wrap_model_if_needed 46/46(+6개는 classmethod 첫 인자를 self로 명명)으로 계약이 균일하다(main도 동일). 다만 '안 쓰는 키워드도 받아 무시'의 근거인 PR #766은 아직 OPEN이라 그 부분은 코드에 없다.
- **보정된 룰:** 오버라이드는 베이스 시그니처(설정 객체 model_config/rbln_config를 받는 형태)를 그대로 유지하고, 계약에 인자를 추가할 때는 모든 오버라이드에 같은 인자를 함께 추가한다.
- 테스트 제안: tests/test_conventions.py::test_override_signatures_match_contract — src/optimum/rbln/**/*.py에서 클래스 안의 get_compiled_model/get_input_info/_update_rbln_config/_wrap_model_if_needed 정의를 AST로 수집해 첫 인자를 제외한 파라미터 이름 튜플이 각각 modeling.py RBLNModel / decoderonly RBLNDecoderOnlyModel의 베이스 정의와 같은지 검사. 허용 목록: multicontrolnet.py RBLNMultiControlNetModel._update_rbln_config.

### PRC-001 — verified

- 룰: PR 제목은 `type(optional scope): description` 형식의 conventional commit을 따르고, type은 PR 템플릿이 정의한 목록(model, performance, refactor, doc, dependency, release, other)에서 변경 성격에 맞게 고르며 모델 버그 수정도 model로 분류한다.
- 검사: dev@74e97fa4에서 제목 검사가 어디 있는지(.github/workflows vs .buildkite) 찾고, 그 검사(amannn/action-semantic-pull-request@v5, types=release/model/performance/refactor/doc/dependency/other)의 기본 headerPattern으로 2025-01 이후 머지 PR 제목을 검사한다; 매치하지 않거나 type이 목록 밖이면 위반. dev 브랜치 보호의 required check에 포함되는지도 확인한다.
- 사유: 판정 변화 없음(verified, 610/21 그대로). 제목 검사는 Buildkite로 옮겨지지 않았고 dev@74e97fa4에도 GitHub Actions(.github/workflows/pr-title-check.yaml)로 남아 있으며 2026-10-22까지 모든 PR에서 실행되고 있다. 머지된 631건 중 610건(96.7%)이 통과하고, main 삭제 이후(2026-09-11~) 머지된 66건과 2026-08 이후 113건은 위반이 0이다. 다만 dev 브랜치 보호의 required check는 Buildkite의 [OB] 4개(BC·pytest·code quality·docstrings)뿐이라 'Check PR title'이 실패해도 머지는 막히지 않는다(강제가 아니라 권고 수준의 CI 신호).
- **보정된 룰:** PR 제목은 GHA pr-title-check.yaml이 검사하는 `type(scope): description` 형식을 따른다(type ∈ release/model/performance/refactor/doc/dependency/other). release-vX.Y.Z 브랜치로 가는 cherry-pick PR도 원래 변경의 type(doc:, other: 등)을 쓴다.
- 테스트 제안: pytest 대신 'Check PR title'(GHA)을 dev와 release-* 브랜치의 required status check로 추가하는 것을 권한다. CI가 Buildkite로 옮겨가는 중이므로, 옮길 경우 pr.yml의 gate 단계에 BUILDKITE_PULL_REQUEST 제목을 같은 정규식으로 검사하는 스텝을 두고 [OB] 컨텍스트로 보고하게 해야 required check에 넣을 수 있다. Revert 기본 제목 예외 여부도 정해야 한다.

### PRC-002 — rejected

- 룰: 작업 중인 PR은 제목에 (WIP)를 붙인다.
- 검사: GitHub에서 draft 상태인 PR 전부(작업 중 PR의 대리 지표)와 제목에 WIP가 들어간 PR을 대조해, draft PR이 제목에 (WIP)를 붙였는지 본다; draft인데 (WIP)가 없으면 위반.
- 사유: draft PR 38건 중 (WIP)를 붙인 것은 4건(390·391·490·496)이고 모두 한 작성자(rebel-seinpark)다. 나머지 34건은 GitHub draft 상태만 쓰거나 [DRAFT]·(DO NOT MERGE)를 붙였다. 근거 3건도 같은 작성자이며, "(WIP) model: …" 제목은 pr-title-check.yaml의 headerPattern(^(\w*)…)에 걸려 실패하므로 PRC-001과 충돌한다.
- **보정된 룰:** 작업 중인 PR은 제목에 표식을 달지 않고 GitHub Draft로 연다(제목은 conventional commit 형식 유지).

### PRC-003 — aspirational

- 룰: PR 본문은 템플릿의 절(변경 유형, 변경 개요, Motivation, Verification, 관련 이슈)을 갖추고, Type of Change에서 해당 유형을 하나 체크하며(Other는 내용을 구체적으로 적음), Motivation에는 채택한 방식이 필요한 이유(문제의 원인이나 기대 효과)를 적는다.
- 검사: 2025-01 이후 머지된 사람 작성 PR(자동 dependency 봇·sync 미러 제외 588건)의 본문에서 템플릿 절(Type of Change/Changes Overview/Motivation/Related Issues) 존재, Type of Change 체크 정확히 1개, Other 체크 시 설명 기재, Motivation 절 내용(주석 제외) 존재를 모두 만족하는지 본다.
- 사유: 템플릿이 자동으로 채워지므로 절 헤더는 497/588건에 있고, 유형 체크도 템플릿을 쓴 509건 중 448건이 정확히 1개다. 그러나 Motivation을 실제로 쓴 PR은 67건(11%)뿐이고, Other를 체크한 145건 중 135건은 설명이 없어 전체 준수는 51건(8.7%)이다. 2026Q3에는 템플릿 없이 Summary/Why/Test plan 형식으로 쓴 PR이 40건으로 템플릿 사용(46건)과 맞먹는다. Verification 절은 템플릿에 없다.
- **보정된 룰:** PR 본문에는 무엇을 바꿨는지와 왜 그렇게 했는지(원인이나 기대 효과)를 적는다. 템플릿을 쓸 때는 Type of Change를 하나만 체크하고, Other를 체크하면 내용을 적는다. 템플릿 대신 Summary/Why/Test plan 형식을 허용할지는 팀이 결정해야 한다.

### PRC-004 — aspirational

- 룰: 기존 동작의 호환성을 깨는 변경은 PR 본문의 Type of Change에서 Breaking change로 표시한다.
- 검사: 2025-01 이후 머지 PR에서 Breaking change 체크 PR을 세고, 본문(템플릿 체크박스 줄 제외)·제목에 breaking/no longer/removed/rename/default 변경 신호가 있는 PR을 수작업으로 판정해 사용자 관점의 호환성 파괴인데 Breaking을 체크하지 않은 PR을 위반으로 센다.
- 사유: Breaking change를 체크한 PR은 8건(50·108·171·183·225·356·559·770)이다. 신호 기반 판정으로 찾은 사용자 관점 호환성 파괴 13건 중 5건(38%)은 체크하지 않았다(645·271·605·128·714). 특히 deprecation 정책에 따른 API 제거(645)도 Breaking으로 표시되지 않았다. 판정은 휴리스틱과 수작업 확인이라 실제 누락은 더 많을 수 있다.
- **보정된 룰:** 공개 API 제거(deprecation 창 만료 포함), 공개 인자 이름 변경, 기본 동작 변경처럼 기존 사용자 코드가 그대로 동작하지 않는 PR은 Type of Change에서 Breaking change를 체크한다.

### PRC-005 — rejected

- 룰: 릴리스 PR은 dev를 main으로 병합하는 PR로 만들고, base를 main으로 두며, 제목을 `release: vX.Y.Z` 형식으로 쓰고 Type of Change에서 Release를 체크한다.
- 검사: main 삭제 후 릴리스가 어떻게 나가는지 gh(기본 브랜치·main 존재·release-* 브랜치·ruleset·GitHub release의 target_commitish·base=release-v0.12.0 PR)와 dev@74e97fa4의 deploy-on-tag.yaml·bc-compile-steps.sh·PR 템플릿으로 확인하고, 2026-08-28 이후 정식 릴리스 중 dev→main `release: vX.Y.Z` PR로 나간 것이 있는지 센다.
- 사유: stale: main branch removed in PR 727. 이전 판정 interview(7/9) → rejected. 원격 main은 404이고 기본 브랜치는 dev이며, PR 727 본문이 "`dev` is now the default branch and `main` has been removed"라고 적는다. main 삭제 이후 정식 릴리스 2개(v0.11.2, v0.12.0)는 모두 dev→main PR 없이 release-vX.Y.Z 브랜치에서 태그로 나갔고, corpus에서 2026-07-31(#661) 이후 `release:` 제목 PR은 0건이다. PR 템플릿(.github/pull_request_template.md)은 아직 'Release (dev → main merge…)'와 'hotfix는 main으로'를 안내하고 있어 낡았다.
- **보정된 룰:** (저장소가 보여주는 현행 절차, 2026-08 이후) 릴리스는 PR이 아니라 태그로 낸다. 프리릴리스(vX.Y.ZaN, .devN)는 dev에 직접 태그한다(GitHub release target=dev). 정식 릴리스는 dev의 rc 지점(예: v0.12.0rc1=522f1bb8)에서 release-vX.Y.Z 브랜치를 잘라 내고, 필요한 수정은 base=release-vX.Y.Z인 cherry-pick PR(제목은 원래 type 유지, 예: `doc: … (#783)`, squash)로 넣은 뒤 그 브랜치에 vX.Y.ZrcN·vX.Y.Z 태그를 단다. release-* 브랜치와 release* 태그는 ruleset으로 삭제·강제푸시·업데이트가 막혀 있다(우회 사용자 1명). v* 태그 푸시가 GHA deploy-on-tag.yaml → deploy.yaml(PyPI 업로드)과 vllm-rbln 패키지 갱신 dispatch를 일으키고, Buildkite bc-compile.yml은 정식 버전 태그(bc-compile-steps.sh의 ^[0-9.]+(postN)?$)에서만 BC 아티팩트를 컴파일한다. PR 템플릿의 Release 항목과 main 안내는 이 절차에 맞게 고쳐야 한다.

### PRC-008 — aspirational

- 룰: 선행 PR이나 다른 저장소의 PR에 의존하는 PR은 base 브랜치를 선행 PR 브랜치로 두고, 본문에 선행·관련 PR 번호와 링크, 머지 순서, 순서가 어긋났을 때의 대응(닫기 등)을 명시한다.
- 검사: 2025-01 이후 머지된 PR 중 미머지 선행 PR에 의존하는 PR(다른 PR의 head 브랜치를 base로 둔 PR + 본문에 depends/stacked/based on/need #N 선언이 있는 PR)을 모아, base가 선행 PR 브랜치인지와 본문에 선행 PR 번호가 있는지를 본다; 둘 중 하나라도 빠지면 위반.
- 사유: 의존 PR 36건 중 base=선행 브랜치이면서 본문에 선행 PR을 밝힌 것은 11건(31%)이다. 2026-08 이후 머지분은 10건 중 9건이 준수(668만 base=dev)하지만, 2025~2026-07의 feature-branch 하위 PR 23건은 본문에 선행 PR을 적지 않았다. 머지 순서는 13건 중 3건(638·738의 retarget, 668의 merge after)만, 순서가 어긋났을 때의 대응은 692 한 건만 명시해 이 부분은 근거가 약하다.
- **보정된 룰:** 미머지 PR 위에 쌓는 PR은 base를 선행 PR의 head 브랜치로 두고 본문 첫 줄에 "Stacked on #N"을 적는다(머지 후 GitHub가 dev로 retarget). 머지 순서와 순서가 어긋났을 때의 대응(닫기 등)은 main/release 브랜치에 out-of-band로 들어가는 PR에만 요구한다.

### PRC-023 — rejected

- 룰: dev에서 main으로 가는 동기화·백머지와 그래프 토폴로지를 유지해야 하는 릴리스 계열 PR은 squash나 rebase가 아닌 merge commit으로 병합한다.
- 검사: main 삭제 후에도 브랜치 간 조상 관계를 유지해야 하는 PR(dev↔main sync/back-merge, release→dev back-merge)이 존재하는지 확인한다: 원격 main 존재 여부, 저장소 merge 설정, base=release-v0.12.0 PR의 merge commit 부모 수, release-* → dev PR 유무.
- 사유: stale: main branch removed in PR 727. 이전 판정 verified(21/3) → rejected. 규칙의 대상인 dev↔main 동기화·백머지가 더 이상 존재할 수 없다: 원격 main은 404이고, 저장소는 allow_merge_commit=false(squash만 허용)이다. 현행 release-vX.Y.Z 브랜치로 가는 PR(#786 #789 #805)은 dev 커밋의 cherry-pick이라 모두 squash(부모 1개)로 머지되었고, release 브랜치에서 dev로 되돌아오는 back-merge PR은 0건이다(수정은 dev에 먼저 들어가므로 필요 없음).
- **보정된 룰:** (현행) 장기 브랜치는 dev 하나이고 릴리스는 dev에서 잘라낸 release-vX.Y.Z 브랜치에 태그로 낸다. 수정은 dev에 먼저 머지한 뒤 base=release-vX.Y.Z cherry-pick PR로 squash 머지하며, 저장소 설정상 merge commit은 쓸 수 없다. 조상 관계를 보존해야 하는 병합(이전 main↔dev 동기화)은 이 절차에서 생기지 않으므로 규칙은 폐기한다. 만약 release 브랜치에만 있는 수정이 생기면 dev로도 별도 PR(cherry-pick)로 넣는다는 규칙으로 대체할지는 팀이 정해야 한다.

### STY-004 — verified

- 룰: 사용하지 않는 코드(미사용 변수·import·메서드·클래스, 부모와 동일한 override, 레거시 코드, 효과 없는 컴파일 옵션, 오래되었거나 중복된 CI 워크플로)는 남기지 않고 제거한다.
- 검사: ruff F401/F841/F811/RUF059/PLW0127로 미사용 import·변수를 세고, AST로 docstring 없이 super()에 인자를 그대로 넘기기만 하는 override와 저장소 어디서도(src/tests/examples) 참조되지 않는 private(_name) 함수·메서드·클래스를 위반으로 센다.
- 사유: dev 최신에서 ruff 미사용 import/변수 검사는 0건(__init__.py의 F401은 설정상 의도적 제외), 메서드 1685개 중 무의미한 pass-through override 1개(docstring을 붙이기 위한 12개는 제외), private 정의 471개 중 미참조 1개(_postprocess_outputs는 HF 훅이라 제외)뿐이다. 레거시 코드·효과 없는 컴파일 옵션·중복 CI 워크플로는 기계적으로 판정할 수 없어 측정하지 않았다.
- **보정된 룰:** 미사용 import·변수·unpack 변수는 ruff(F401, F841, RUF059)가, 부모와 동일한 override와 미참조 private 헬퍼는 리뷰가 잡는다. docstring만 덧붙이려는 pass-through override는 문서화 목적이므로 허용한다. '레거시 코드'·'효과 없는 컴파일 옵션'·'오래된 CI 워크플로'는 리뷰 판단 항목이다.
- 테스트 제안: tests/test_conventions.py에 src/optimum/rbln 전체를 ast로 읽어, docstring·데코레이터(classmethod/staticmethod 제외) 없이 본문이 `[return] super().<같은이름>(<같은 인자 그대로>)` 한 줄뿐인 메서드를 찾는 테스트를 둔다. 허용 목록은 현재 위반 1건(transformer_sd3.py RBLNSD3Transformer2DModel.__post_init__)이며 수정 후 비운다. 미참조 private 검사는 HF 훅 오버라이드 오탐이 있어 테스트로 두지 않는다.

### STY-005 — aspirational

- 룰: 새로 추가하는 소스 파일에는 기존 파일과 같은 형식의 Rebellions 저작권 헤더를 현재 연도로 붙인다.
- 검사: src/·tests/의 모든 .py 첫 줄(shebang 제외)과 이어지는 13줄을 표준 Rebellions Apache 헤더와 비교하고, 헤더 연도를 `git log --diff-filter=A`(--follow/--no-renames 둘 다)로 구한 추가 연도와 대조한다; 헤더 없음·형식 불일치·연도 불일치를 위반으로 센다(HF 유래 헤더, 빈 파일, 2023 초기 import 후 #19에서 2025로 일괄 갱신된 파일은 준수로 본다).
- 사유: dev 최신 src의 Rebellions 소유 파일 337개 중 290개(86%)만 표준 헤더와 맞는 연도를 갖고, 헤더 없음 11·형식 불일치 20·연도 불일치 16건이 있다. 2026년에 새로 추가된 src 파일만 봐도 78개 중 59개(76%)만 준수해 새 파일에서도 형식('#' 빈줄 변형 9개)과 연도(2025 복사 5개) 이탈이 계속된다. tests/는 12개 중 헤더가 있는 파일이 0개다.
- **보정된 룰:** src/optimum/rbln 아래 새 파일은 첫 줄 `# Copyright <추가 연도> Rebellions Inc. All rights reserved.`, 빈 줄, 그리고 기존 파일과 바이트 단위로 같은 Apache-2.0 블록(주석 사이 빈 줄은 '#'이 아닌 진짜 빈 줄)을 붙인다. HuggingFace 코드를 옮긴 파일은 원 저작권 헤더를 유지하고, tests/는 현재 헤더를 붙이지 않는 것이 관행이다(적용 범위 확인 필요).
- 테스트 제안: tests/test_conventions.py에 src/optimum/rbln/**/*.py 중 비어 있지 않은 파일의 첫 13줄(shebang 제외)이 정규식 `# Copyright 20\d\d Rebellions Inc. All rights reserved.` + 표준 Apache 블록과 정확히 일치하는지 검사하는 테스트를 둔다(서드파티 헤더 파일 9개는 'Copyright .* (HuggingFace|Optuna)' 로 면제). 연도-추가일 대조는 git 이력이 필요해 테스트에서 빼고, 현재 형식 위반·헤더 없음 31개 파일을 허용 목록으로 시작해 줄여 간다.

### STY-010 — verified

- 룰: 타입 annotation에만 쓰이는 심볼이 순환 참조를 만들거나 외부 패키지 타입일 때만 import를 `TYPE_CHECKING` 블록으로 옮기고 annotation은 문자열로 감싸며, 같은 이름을 런타임 import와 `TYPE_CHECKING` 블록에 중복 import하지 않고 TC001~TC003 규칙으로 일괄 이동하지 않는다.
- 검사: __init__.py를 제외한 모듈의 `if TYPE_CHECKING:` 본문 import 이름마다 모듈 수준 런타임 import와의 중복, annotation 밖 런타임 사용, `from __future__` 없이 따옴표 없는 annotation 사용, 미사용 여부를 검사하고, pyproject의 ruff ignore에 TC001~TC003이 있는지 확인한다; first-party TYPE_CHECKING import가 실제 순환을 피하는지도 정적 import 그래프로 시도했다.
- 사유: dev 최신에서 53개 모듈의 TYPE_CHECKING 이름 104개 중 런타임 중복 import는 0건(#733 직전 origin/dev에는 15개 모듈 18건)이고, annotation 밖 런타임 사용은 지역 재-import 1건뿐이며, 따옴표 없는 사용은 평가되지 않는 지역 변수 annotation 1건이다. pyproject는 TC001~TC003·TC006을 ignore하고 TC004·TC005만 켠다. '순환 참조이거나 외부 타입일 때만'이라는 조건은 정적 그래프가 #556의 실제 순환을 재현하지 못해 코드로 판정하지 못했다(TC import 104개 중 third-party 73, first-party 31).
- **보정된 룰:** annotation 전용 import를 `if TYPE_CHECKING:`으로 옮길 때는 annotation을 문자열로 감싸고, 같은 이름을 모듈 수준 런타임 import와 중복하지 않는다; ruff TC001~TC003/TC006은 끄고 TC004/TC005만 켠다. first-party 심볼을 TYPE_CHECKING으로 옮기는 근거(순환 회피)는 PR 설명에 적는 판단 항목이다.
- 테스트 제안: tests/test_conventions.py에 src/optimum/rbln의 __init__.py가 아닌 모듈을 ast로 읽어 `if TYPE_CHECKING:` 본문에서 import한 이름이 같은 모듈의 최상위 런타임 import에도 있으면 실패하는 테스트를 둔다(허용 목록 없음; TC004가 반대 방향을 이미 막는다). 추가로 pyproject.toml의 [tool.ruff.lint] ignore에 TC001·TC002·TC003이 있고 select에 TC가 있는지 확인한다.

### STY-011 — verified

- 룰: 함수 시그니처에는 실제로 전달되는 객체와 일치하는 타입 힌트를 명시하고, Python 3.10 이상의 네이티브 문법(`X | None`, `list`)을 쓰며 typing의 Optional·List 별칭과 암묵적 Optional은 쓰지 않고, 이를 위해 `from __future__ import annotations`를 일괄 추가하지 않는다.
- 검사: src/optimum/rbln의 모든 Subscript에서 typing Optional/Union/List/Dict/Tuple 사용과 인자에 문자열 forward-ref가 있는지, `X | None`·내장 제네릭 사용 수, 기본값 None인데 annotation이 None을 허용하지 않는 암묵적 Optional 파라미터, `from __future__ import annotations` 파일 수를 센다.
- 사유: dev 최신에서 기본값 None인 annotated 파라미터 1709개 중 암묵적 Optional은 4개(main HEAD는 #736 이전이라 242개)이고, `X | None` 1832회·내장 제네릭 666회에 비해 남은 Optional/Union 86회는 전부 문자열 forward-ref를 감싼 경우다(문자열 없는 사용 0건). `from __future__ import annotations`는 1개 파일뿐이다. forward-ref에서는 `Optional["X"]` 72회 대 `"X | None"` 17회로 typing 별칭이 우세하다.
- **보정된 룰:** annotation은 `X | None`·`list[...]` 등 3.10+ 네이티브 문법으로 쓰고, 기본값이 None이면 반드시 `| None`을 명시한다. 단, 인자가 문자열 forward-ref여서 `|`로 쓸 수 없는 경우(`Optional["PreTrainedModel"]`, `Union[dict[str, Any], "RBLNModelConfig"]`)는 예외로 남아 있다 — 이 경우 전체를 `"X | None"`처럼 문자열로 감싸는 쪽으로 통일할지는 팀 결정이 필요하다.
- 테스트 제안: tests/test_conventions.py에 src/optimum/rbln 전체를 ast로 순회해 (1) 모든 함수 인자 중 기본값이 None 상수인데 annotation 문자열에 'None'/'Optional'/'Any'가 없는 것, (2) Optional/Union/List/Dict/Tuple Subscript 중 slice에 문자열 상수가 없는 것을 실패로 보고하는 테스트를 둔다. 허용 목록은 cosmos2 파이프라인 4개 파일의 safety_checker 파라미터. (mypy no_implicit_optional이 이미 핵심 모듈을 검사하지만 models.* 는 느슨하게 설정돼 있어 테스트가 보완한다.)

### STY-012 — verified

- 룰: `__init__` 목록, export 목록, 키 목록 같은 나열형 목록은 정렬된 순서를 유지한다.
- 검사: 모든 모듈의 `__all__`, 패키지 __init__.py의 TYPE_CHECKING 블록 `from .x import *` 순서, (main의) `_import_structure` 키·값 목록, 그리고 문자열 3개 이상을 담은 모듈 수준 list/dict를 sorted()/RUF022(isort 스타일)/대소문자 무시 정렬 중 하나와 일치하는지 검사한다.
- 사유: dev 최신에서 `__all__` 185개와 __init__ TYPE_CHECKING star-import 블록 79개가 전부 정렬돼 있고 RUF022와 scripts/check_inits.py가 이를 강제한다. 정렬되지 않은 모듈 상수 7개(log_levels, _BYTE_UNITS, RBLN_VERSION_COMPATS, ROPE_INIT_FUNCTIONS, QUANTIZED_WEIGHTS 등)는 의미 순서를 따르는 표라 대상이 아니다. main HEAD에는 #733 이전 `_import_structure` 형식이라 미정렬 목록 37개가 남아 있다.
- **보정된 룰:** export·등록 목록(`__all__`, 패키지 __init__의 re-export, 모델/설정 등록 목록)은 정렬 순서를 유지한다(RUF022 isort 스타일). 로그 레벨·단위·버전 호환표처럼 순서 자체가 의미인 상수 표는 예외다.
- 테스트 제안: RUF022와 scripts/check_inits.py가 CI에서 이미 강제하므로 별도 테스트는 선택 사항이다. 둔다면 tests/test_conventions.py에서 src/optimum/rbln/**/__init__.py의 `if TYPE_CHECKING:` 본문 star-import 모듈 이름 목록이 sorted()와 같은지 확인한다(허용 목록 없음).

### STY-013 — verified

- 룰: 로그와 경고는 built-in logging/warnings 대신 optimum-rbln의 `utils.logging`을 사용한다.
- 검사: src/optimum/rbln(utils/logging.py 제외)에서 stdlib `import logging`/`import warnings`, `logging.getLogger()`, `warnings.warn()` 호출과 transformers/diffusers logging import를 찾고, `utils.logging.get_logger`를 쓰는 파일 수와 비교한다.
- 사유: 로그를 남기는 56개 파일이 `utils.logging.get_logger`를 쓰고, stdlib 로거를 직접 만드는 곳은 time_series_transformer 1곳뿐이다. built-in warnings.warn은 3개 파일 6곳(그중 4곳은 패키지 import 시 버전 호환성 ImportWarning)으로 소수다. modeling_attention_utils.py의 `import logging`은 `logging.DEBUG` 상수 참조용이라 위반에서 뺐다.
- **보정된 룰:** 로거는 항상 `from ...utils.logging import get_logger`로 만들고 stdlib `logging.getLogger`를 쓰지 않는다(`logging.DEBUG` 같은 레벨 상수 참조는 허용). 사용자 경고도 `logger.warning`/`warning_once`로 낸다. 단 import 시점의 버전 호환성 경고(`warnings.warn(..., ImportWarning)`)를 예외로 둘지는 팀 확인이 필요하다.
- 테스트 제안: tests/test_conventions.py에 src/optimum/rbln/**/*.py(utils/logging.py 제외)를 ast로 읽어 `logging.getLogger(...)` 호출과 `warnings.warn(...)` 호출을 실패로 보고하는 테스트를 둔다. 허용 목록: modeling_time_series_transformer.py:46(수정 대상), auto_factory.py의 warnings.warn 1곳, utils/import_utils.py의 ImportWarning 4곳(팀이 예외로 확정하면 영구 허용).

### STY-015 — verified

- 룰: CI 워크플로우에서 시크릿은 `secrets` 컨텍스트로 전달하고, 내부 엔드포인트·클러스터 고유 경로나 주소는 저장소에 평문으로 적지 않고 CI 변수로 주입하며, 필수 변수가 비어 있으면 빈 값으로 진행하지 않고 업로드 단계에서 실패시킨다.
- 검사: dev@74e97fa4의 .buildkite/**, .github/**(workflows·renovate·템플릿), scripts/ci/*에서 평문 IP·내부 호스트(*.rebellions.in, k8s.)·클러스터 마운트 경로·토큰을 찾고, 자격증명이 ${{ secrets.* }}(GHA) 또는 Buildkite secrets: 매핑으로, 이미지·클러스터 경로·사내 PyPI 엔드포인트가 에이전트 변수로 주입되는지, 필수 변수에 ${VAR:?not set} 가드가 있는지 센다. 평문 내부 호스트·경로가 위반.
- 사유: 판정 유지(verified), 수치 변경 54/1 → 64/2(파일은 이전 gh 스냅숏과 동일하고, 내부 호스트 grep과 Buildkite secrets 매핑 집계를 추가한 결과). GHA 자격증명 9건은 모두 ${{ secrets.* }}(Obedients URL도 secret), Buildkite 자격증명 18건은 모두 secrets: 매핑, 이미지·클러스터 경로 34건과 사내 PyPI 엔드포인트 3건은 에이전트 변수로 주입되며 평문 IP·마운트 경로는 없다. 필수 변수 가드는 bc-steps.sh·bc-compile-steps.sh(BC_BASE_PATH), install-rebel-compiler.sh(엔드포인트·계정 3개)에 있다. 위반은 Obedients API URL 하드코딩과 override 어노테이션의 사내 PyPI 호스트 문구 2건.
- **보정된 룰:** CI(GHA·Buildkite·scripts/ci) 자격증명은 GHA secrets 컨텍스트 또는 Buildkite secrets: 매핑으로만 받고, 내부 엔드포인트·클러스터 경로·이미지는 에이전트/CI 변수로 주입하며 로그·어노테이션 문구에도 평문으로 적지 않는다. 필수 변수는 ${VAR:?not set}로 가드한다.
- 테스트 제안: tests/test_conventions.py::test_ci_has_no_plaintext_endpoints — .github/**/*.y*ml, .buildkite/**/*.yml, scripts/ci/*를 읽어 (1) IPv4 리터럴, (2) *.rebellions.in / *.k8s.* / *.svc / *.cluster.local 호스트(주석 제외, 문자열 포함), (3) /mnt|/nfs|/data로 시작하는 절대경로, (4) token/password 키에 ${{ secrets. }}·${VAR}·Buildkite secret 이름이 아닌 리터럴 값이 있으면 실패. allowlist: scripts/ci/slack-report.py:22 API 상수, scripts/ci/rebel-compiler-override.sh:12 어노테이션 문구(변수화 전까지).

### STY-018 — verified

- 룰: 머지 전에 PR 변경 코드 전체에 `ruff format`과 `ruff check`를 통과시키고 타입 검사와 독스트링 검사도 통과시킨다.
- 검사: dev 최신 트리에서 CI 게이트와 같은 `ruff format --check`/`ruff check`를 돌려 실패 파일을 세고, 2025-01 이후 머지 PR 중 머지 후 lint/docstring 실패를 고치는 후속 PR(제목 매칭)을 위반으로 센다.
- 사유: dev 최신(74e97fa4) 398개 파일과 worktree HEAD(main) 345개 파일 모두 ruff format/check를 통과하고, CI 게이트(.buildkite/pr.yml: ruff format·ruff check·mypy·check_inits·pre-commit, 별도 docstrings 단계; main은 check_code_quality.yml+test-docstrings.yml)가 머지 전에 강제한다. 2025-01 이후 631개 머지 PR 중 lint/docstring 후속 수정 PR은 4건뿐이다. mypy는 로컬에 설치돼 있지 않아 직접 실행하지 못했다.
- **보정된 룰:** 머지 전에 CI 품질 게이트(ruff format --check, ruff check, mypy, scripts/check_inits.py, mkdocstrings docstring 검사)를 통과시킨다. 다른 PR이 먼저 머지되어 base가 바뀌었으면(특히 lint 규칙을 바꾸는 PR 직후) 최신 base에서 CI를 다시 돌린 뒤 머지한다.
- 테스트 제안: 규칙 자체는 CI 게이트가 강제하므로 별도 pytest는 불필요하다. 회귀 방지를 원하면 tests/test_conventions.py에 .buildkite/pr.yml을 yaml로 읽어 'code quality' 단계 command에 `ruff format --check .`, `ruff check .`, `mypy`, `scripts/check_inits.py`가, 'docstrings' 단계가 존재하는지 확인하는 테스트를 둔다(허용 목록 없음).

### STY-027 — verified

- 룰: torch.export 경로를 지원할 때는 새 커스텀 op의 fake 함수와 인덱스 값의 동적 값 제약을 함께 제공한다.
- 검사: 이 저장소 안의 커스텀 op 등록(torch.library/register_fake)과, 아키텍처 코드에서 .item()으로 꺼낸 데이터 의존 인덱스가 torch._check_is_size/torch._check 제약과 함께 쓰이는지 grep으로 확인한다; 제약 없는 인덱스 .item()이 위반.
- 사유: 데이터 의존 인덱스 3곳(t5 cache_pos, time_series embed_idx, grounding_dino spatial_shapes 합)은 모두 torch.compiler.is_exporting() 분기에서 torch._check_is_size/_check 제약을 건다(나머지 .item()은 scale 스칼라). fake 함수 부분은 커밋 ff8c3f6f(refactor(ops): drop duplicate custom op registration, defer to rebel-compiler)로 src/optimum/rbln/ops가 삭제되어 이제 rebel-compiler 소관이며 이 저장소에는 등록 지점이 0개다.
- **보정된 룰:** torch.export 경로에서 텐서 값을 .item()으로 인덱스로 쓸 때는 is_exporting() 분기에서 torch._check_is_size와 범위 torch._check를 함께 건다; 커스텀 op와 그 fake 함수는 rebel-compiler에서 등록한다(이 저장소에서 재등록 금지).
- 테스트 제안: tests/test_conventions.py::test_export_item_indices_constrained — src/optimum/rbln/**/*_architecture.py를 AST로 순회해 x = <expr>.item() 대입 뒤 같은 함수 안에서 x가 torch._check_is_size(x) 또는 torch._check(...)에 쓰이는지 확인(scale= 키워드 인자로 바로 쓰이는 .item()은 제외). 함께 src 아래에 torch.library.custom_op/register_fake가 없음을 assert. 현재 위반 없음.

### SUB-008 — aspirational

- 룰: HF가 사용하는 dtype 동작(예: fp32로 계산하는 비전 회전 위치 임베딩 테이블, 모듈 입력 dtype)을 그대로 따르며, 활성 dtype으로 반올림하거나 암묵적 승격에 맡기지 않고 명시적으로 선언·캐스트한다.
- 검사: 비전 rotary를 가진 계열마다 (a) 회전 적용 함수가 HF apply_rotary_pos_emb_vision처럼 fp32 업캐스트 후 한 번 반올림하는지 (b) 호스트가 graph 입력 cos/sin을 HF와 같은 dtype(Qwen 계열 fp32, Gemma4는 x.dtype)으로 선언·캐스트하는지 확인한다. HF가 fp32로 두는 표를 rbln_config.dtype으로 반올림하면 위반.
- 사유: main(1/5)에서 크게 개선됐다: #711(2026-10-13 병합)로 5개 계열(qwen2_vl, qwen2_5_vl, qwen3_vl, qwen3_5, exaone4_5) 모두 아키텍처 파일에 "Mirrors HF `apply_rotary_pos_emb_vision`: rotate in fp32 and round once." 함수를 갖게 됐고, Qwen3.5는 cos/sin 입력을 torch.float32로 선언하며, Gemma4 비전은 HF Gemma4VisionRotaryEmbedding이 `.to(dtype=x.dtype)`하므로 rbln_config.dtype 캐스트가 HF와 일치한다. 그러나 표를 fp32로 넘기는 부분은 #764가 OPEN이라 Qwen2-VL·Qwen2.5-VL·Qwen3-VL(·MoE 상속)·EXAONE-4.5 4곳이 여전히 `torch.cat((cos, cos), dim=-1).to(self.rbln_config.dtype)`로 활성 dtype 반올림을 한다(11개 사이트 중 4개, 36%).
- **보정된 룰:** HF가 명시적으로 두는 dtype 경계를 그대로 따른다: 비전 rotary 적용은 계열별 apply_rotary_pos_emb_vision에서 q/k/cos/sin을 fp32로 올려 계산 후 한 번 반올림하고, 호스트가 graph에 넘기는 cos/sin 표는 HF가 fp32로 두는 계열(Qwen-VL 계열·EXAONE-4.5)에서는 fp32로 선언·전달하며, HF가 x.dtype으로 캐스트하는 계열(Gemma4)만 rbln_config.dtype을 쓴다.
- 테스트 제안: tests/test_conventions.py::test_vision_rotary_matches_hf_dtype — (1) transformers/models/*/*_architecture.py 중 비전 어텐션이 있는 파일(qwen2_vl, qwen2_5_vl, qwen3_vl, qwen3_5, exaone4_5)에 apply_rotary_pos_emb_vision 정의가 있고 본문에 `.float()`가 있는지 AST로 단언(현재 전부 통과). (2) 같은 계열의 modeling_*.py에서 _update_rbln_config input_info의 'cos'/'sin' dtype 표현식이 torch.float32인지, `torch.cat((cos, cos), dim=-1).to(self.rbln_config.dtype)` 패턴이 없는지 단언하고 qwen2_vl·qwen2_5_vl·qwen3_vl·exaone4_5를 allowlist로 둔다(#764 병합 시 제거). gemma4는 HF가 x.dtype으로 캐스트하므로 검사 대상에서 제외.

### TEST-007 — verified

- 룰: pytest는 각 테스트 케이스의 실행 시간을 보고하도록 verbosity를 설정하고, 스케줄 실행은 첫 실패에서 중단하지 않는다.
- 검사: CI 정의(.buildkite/**, scripts/ci/*.sh, .github/workflows)의 모든 pytest 실행 줄에 각 케이스 실행 시간을 보고하는 `--durations 0`과 `-vv`가 붙어 있는지 확인한다. nightly.yml이 올리는 pytest·BC 파이프라인에 첫 실패에서 멈추는 설정(cancel_on_build_failing, pytest -x/--exitfirst/--maxfail)이 없는지 확인한다. 하나라도 어기면 위반이다.
- 사유: dev는 GHA pytest 워크플로를 지우고 Buildkite로 옮겼다. 모든 pytest 호출은 scripts/ci/run-suite.sh:29-37의 5줄(unit-cpu, config, transformers, diffusers, llm)에 모여 있고, 5줄 모두 `-vv --durations 0`을 붙인다. nightly.yml이 올리는 두 경로(.buildkite/pipelines/pytest/pipeline.yml, bc-steps.sh --all)에는 cancel_on_build_failing이 없다. pytest에도 -x/--maxfail이 없고, 보고 단계는 `wait: continue_on_failure: true` 뒤에 있어 첫 실패에서 멈추지 않는다. 준수 수가 main의 8(pytest 5줄 + 스케줄 매트릭스 3곳 fail_fast: false)에서 7(pytest 5줄 + nightly 업로드 2곳)로 바뀐 것은 측정 대상이 바뀌어서이고 위반은 여전히 0이다. 참고로 PR 경로도 이제 fail-fast가 없다.
- **보정된 룰:** CI의 모든 pytest 호출(dev: scripts/ci/run-suite.sh)은 `-vv --durations 0`을 붙인다. 정기(nightly) 실행은 잡 취소 없이 끝까지 돈다: Buildkite 파이프라인에 cancel_on_build_failing을 두지 않고, pytest에 -x/--maxfail을 쓰지 않으며, 보고 단계는 continue_on_failure: true인 wait 뒤에 둔다.
- 테스트 제안: tests/test_conventions.py::test_ci_pytest_reports_durations — scripts/ci/*.sh, .buildkite/**/*.yml, .github/workflows/*.y*ml에서 주석이 아닌 줄 중 `pytest `가 들어간 실행 줄을 모아, 각 줄에 `--durations`와 `-vv`가 있고 `-x`/`--exitfirst`/`--maxfail`이 없는지 단언한다. 최소 1줄은 찾아야 한다(빈 매칭으로 통과하지 않게). test_nightly_is_not_fail_fast는 .buildkite/nightly.yml과 .buildkite/pipelines/pytest/pipeline.yml 텍스트에 `cancel_on_build_failing: true`가 없는지, nightly.yml의 report 앞 wait 단계에 `continue_on_failure: true`가 있는지 yaml로 확인한다. 현재 허용목록은 필요 없다.

### TEST-008 — aspirational

- 룰: 새 모델 클래스나 파이프라인을 추가하거나 변경할 때는 해당 테스트 목록(test_transformers, test_diffusers, pytest 스케줄 등)에도 등록하며, 멀티모달 모델은 멀티모달 플래그를 설정한다.
- 검사: dev는 _import_structure 대신 모듈별 __all__(define_import_structure)로 export하므로 src/optimum/rbln/{transformers,diffusers}/**/*.py의 __all__에서 RBLN* 모델·파이프라인 클래스(Config·Auto·기반 클래스 제외)를 모은다. 서브모듈·믹스인(Vision/Encoder/UNet 등)은 분모에서 뺀다. tests/test_*.py에 `RBLN_CLASS = X`로 등록되지 않은 사용자용 클래스를 위반으로 센다. 추가로 test_llm.py에서 이미지·오디오 입력 모델의 테스트 클래스가 상속을 포함해 IS_MULTIMODAL=True로 해석되지 않으면 위반으로 센다.
- 사유: dev 팁에서 export된 사용자용 모델·파이프라인은 115개다. 이 중 85개가 등록되어 있고(직접 82, Kandinsky combined 경유 3) 30개(26%)가 미등록이다. main에서는 97개 중 67개 등록, 30개 미등록이었다. dev에서 늘어난 18개(Glm4·Granite·Olmo2·Phi3·SmolLM3·Starcoder2·ModernBert·Qwen3ASR·Cosmos2/2.5 등)는 모두 테스트에 등록되어 있어 신규 모델에서는 규칙이 지켜진다. 하지만 미등록 30개(Gemma/Gemma2/Gemma4·GptOss·Mixtral·Midm·Detr·DepthAnything·PaliGemma·Exaone4.5·*VLModel 5종, Cosmos 1세대 2종, inpaint·img2img 파이프라인 8종)는 그대로이고, test_diffusers.py:10의 `FIXME: add inpainting tests`도 남아 있다. 2025-01 이후 신규 모델 PR 43건 중 17건이 tests/를 건드리지 않았다. 멀티모달 플래그는 VLM 테스트 14개 중 13개와 오디오 모델 Qwen3ASR이 지키며, Qwen2VL만 get_tokenizer 오버라이드로 우회한다.
- **보정된 룰:** export된 사용자용 모델·파이프라인 클래스(서브모듈 Vision/Encoder/UNet·믹스인 제외)는 각각 tests/test_{transformers,diffusers,llm}.py 중 한 곳에 `RBLN_CLASS = <클래스>`로 등록된 테스트 클래스를 가진다. combined 파이프라인이 내부에서 만드는 하위 파이프라인은 combined 테스트로 갈음한다. 이미지·오디오 입력을 받는 모델의 테스트 클래스는 IS_MULTIMODAL=True를 선언하며, get_tokenizer 오버라이드로 대신하지 않는다.
- 테스트 제안: tests/test_conventions.py::test_every_exported_model_has_a_test — dev 레이아웃에 맞춰 src/optimum/rbln/{transformers,diffusers}/**/*.py의 모듈 수준 `__all__`을 ast로 모은다(main처럼 __init__.py에 `_import_structure`가 있으면 그것을 쓴다). RBLN*로 시작하고 Config/Auto/RBLNBaseModel/RBLNModel이 아닌 이름 중 COMPONENT 정규식(Vision|QFormer|TextModel|Encoder$|Decoder$|Backbone|TransformerPretrainedModel|^RBLNDecoderOnly|UNet|ControlNetModel$|VQModel|PriorTransformer|Transformer[23]DModel|SafetyChecker|Mixin|AutoencoderKL|AudioEncoder)에 걸리지 않는 클래스마다 tests/test_*.py에 `RBLN_CLASS\s*=\s*<name>`이 있는지 단언한다. 현재 미등록 30개는 KNOWN_UNTESTED 허용목록으로 두고, 허용목록의 클래스가 등록되면 목록에서 지우라는 실패를 내서 목록이 줄어들기만 하게 한다. 같은 파일의 test_multimodal_tests_declare_flag는 test_llm.py 클래스를 ast로 상속 해석해, RBLN_AUTO_CLASS가 RBLNAutoModelForImageTextToText 또는 RBLNAutoModelForMultimodalLM인 클래스의 IS_MULTIMODAL이 True인지 확인한다(허용목록 TestQwen2VLForConditionalGeneration).

### TEST-019 — verified

- 룰: 테스트 클래스는 기반 클래스나 클래스가 소유한 공유 설정 dict를 in-place로 수정하지 않고 새 dict를 만들거나 자기 속성으로 재바인딩하며, from_pretrained를 직접 호출할 때는 클래스의 HF_CONFIG_KWARGS를 함께 넘긴다.
- 검사: tests/test_*.py의 테스트 클래스 메서드에서 cls./self.로 접근하는 대문자 클래스 속성 dict를 in-place로 쓰는 곳(.update/.pop/[k]= 등)을 찾는다. 그 dict를 같은 클래스 본문이 선언했고 쓰기가 setUpClass 안이면 준수, 상속받은 dict이거나 테스트 메서드 안에서 쓰면 위반이다. 또 HF_MODEL_ID로 컴파일하는 RBLN_CLASS.from_pretrained 호출에 **HF_CONFIG_KWARGS가 없으면 위반이다.
- 사유: dev 팁에서 in-place 쓰기 17곳은 모두 자기 클래스가 선언한 dict를 setUpClass에서 고친다. HF_MODEL_ID 컴파일 호출 9곳 중 6곳이 HF_CONFIG_KWARGS를 넘긴다(23/3, 이전 origin/dev 측정과 같다). main은 #741·#745 이전이라 30곳 중 13곳이 위반이었다. 남은 3곳은 빈 base dict를 쓰는 클래스(ResNet, KandinskyV22)이거나 컴파일 전에 ValueError를 기대하는 호출(LlavaNext, revision 핀 누락)이다. tests/unit/cpu에는 공유 dict나 from_pretrained가 없다.
- **보정된 룰:** 테스트 클래스는 공유 설정 dict(HF_CONFIG_KWARGS, RBLN_CLASS_KWARGS 등)를 in-place로 고치려면 그 dict를 자기 클래스 본문에서 선언하고 setUpClass 안에서만 고친다. 테스트 메서드에서는 복사본(`{**self.X[...]}`)을 만들어 쓴다. 상속받은 dict는 재바인딩(`cls.X = {**cls.X, ...}`)한다. HF_MODEL_ID에서 컴파일하는 from_pretrained 직접 호출에는 **self.HF_CONFIG_KWARGS를 넘긴다.
- 테스트 제안: tests/test_conventions.py::test_no_shared_class_dict_mutation — <SCRATCH>/stage_d/scripts/test_doc_test019.py의 로직을 옮긴다. tests/test_*.py를 ast로 읽어 중첩 클래스(BaseTest.TestModel, LLMTest.TestLLM)까지 이름으로 인덱싱한다. 각 메서드에서 `cls.X`/`self.X`(X는 대문자) 루트에 대한 .update/.setdefault/.pop/.clear 호출과 Subscript 대입·삭제를 찾고, X가 그 클래스 본문의 Assign 대상이 아니거나 메서드명이 setUpClass가 아니면 실패한다. test_from_pretrained_passes_hf_config_kwargs는 receiver가 RBLN_CLASS로 끝나고 인자에 HF_MODEL_ID가 있는 from_pretrained 호출에 `**...HF_CONFIG_KWARGS` 키워드가 없으면 실패한다. 허용목록은 (TestResNetModel.test_failed_to_create_runtime, TestLlavaNextForConditionalGeneration의 batch_size 충돌 테스트, TestKandinskyV22Model 호출)이다.
