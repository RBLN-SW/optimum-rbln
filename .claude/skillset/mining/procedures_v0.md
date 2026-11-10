# 절차 v0 (optimum-rbln)

입력 절차 추출 76건을 8개 절차로 병합했다. 각 단계 끝의 `[#n]`은 해당 단계를 뒷받침하는 PR 번호다. 인용문은 입력 evidence에서 그대로 복사했다.

| # | 절차 | 병합된 입력 수 |
|---|---|---|
| 1 | 새 decoder-only 모델 추가 | 10 |
| 2 | 새 멀티모달(VLM) 모델 추가 | 4 |
| 3 | 새 diffusers 파이프라인/모델 추가 | 9 |
| 4 | transformers 버전 상향 (torch/diffusers/컴파일러/Python 등 의존성 상향 포함) | 20 |
| 5 | 릴리스(dev→main) | 15 |
| 6 | deprecation/인자 제거 | 5 |
| 7 | 성능 최적화 작업 | 4 |
| 8 | 기타 | 9 |

---

## 1. 새 decoder-only 모델 추가

1. 개발 중인 모델이면 dev를 대상으로 PR을 열고 `New Model Support` 유형을 선택하며 제목에 Draft를 표기한다. 최적화 등 개발이 끝나기 전에는 공개 안내와 model zoo 예제를 올리지 않는다. [#368, #369]
2. 기존 모델 클래스를 재사용할 수 있는지 먼저 확인하고, 차이가 나는 구성 요소만 새 모듈로 구현한다. [#734, #735]
3. `models/<model>/`에 `__init__.py`, `configuration_<model>.py`, `modeling_<model>.py`, `<model>_architecture.py`를 만들고 공통 decoder-only 기반 클래스를 상속한다. [#222, #253, #260, #177, #441, #326, #136, #83, #563, #799, #800]
4. 체크포인트 구조 차이(스케일, 정규화, 회전 방식, 융합 projection)는 architecture 훅으로 처리하고, 기본 래퍼가 처리하는 부분은 오버라이드하지 않는다. 래퍼/레이어에서 `_original_mod` 참조를 제거하고 getter가 쓰는 하위 모듈은 `__init__`에서 등록한다. [#799, #800, #444]
5. 공통 decoder-only 로직이나 op가 필요하면 `ops/`와 `decoderonly/` 공통 파일을 함께 수정한다. `torch.export`가 필요한 커스텀 op는 fake 함수를 함께 구현한다. [#563, #260, #222, #253]
6. 모델을 등록한다. `models/__init__.py`, `transformers/__init__.py`, `rbln/__init__.py`(필요 시 `modeling.py`)에 export하고 `auto/modeling_auto.py`에 아키텍처 매핑을 추가한다. 최근에는 새 모듈에 `__all__`을 쓰고 `python scripts/check_inits.py --fix`로 import 구조를 갱신하며 패키지 `__init__.py`를 손으로 고치지 않는다. [#222, #253, #260, #177, #441, #326, #136, #83, #563, #734, #735]
7. 컴파일러 측 관련 PR 링크, Hugging Face 모델 주소, 클래스명, 예제 실행 명령, 지원 기능(예: 새 RoPE 방식) 목록을 PR 본문에 적는다. [#441, #136, #83, #563]
8. 여러 텐서 병렬 크기에서 컴파일하고 집합 통신 레이아웃을 확인한 뒤, HF 대비 생성 토큰 일치와 logits pearson을 PR 설명에 기록한다. [#799, #800]
9. 테스트를 추가한다. 일반 모델은 `test_transformers` 목록과 모델 전용 pytest, decoder-only LLM은 `tests/test_llm.py` 테스트 클래스를 쓰며, 병렬 PR과 import 줄이 겹치면 rebase한다. 멀티모달이면 pytest 스케줄에 멀티모달 플래그를 설정한다. [#260, #222, #253, #799, #800, #326]
10. docstring만 따로 다루는 변경은 `doc` 타입 PR로 분리한다. [#326]

대표 경로: `src/optimum/rbln/transformers/models/<model>/*.py`, `src/optimum/rbln/transformers/models/__init__.py`, `src/optimum/rbln/transformers/__init__.py`, `src/optimum/rbln/__init__.py`, `src/optimum/rbln/transformers/models/auto/modeling_auto.py`, `src/optimum/rbln/transformers/models/decoderonly/*.py`, `src/optimum/rbln/ops/*.py`, `tests/test_llm.py`, `tests/test_transformers.py`

인용:
> "write `__all__` in the new modules and run `python scripts/check_inits.py --fix`." (#734)
> "The test uses 4 layers with `no_rope_layers=[1, 1, 1, 0]` so both paths are compiled." (#799)
> "Registered required components at init-time (e.g., layer norms, projections, embeddings) so downstream code does not depend on _original_mod indirection." (#444)

근거 PR: #83, #136, #177, #222, #253, #260, #326, #368, #369, #441, #444, #563, #734, #735, #799, #800

---

## 2. 새 멀티모달(VLM) 모델 추가

1. 기존 sibling 모델 패턴을 따라 `configuration`, `modeling`, `architecture` 모듈을 만든다. 3D 모델처럼 구조가 다르면 `get_pytorch_model`, `_update_rbln_config`, `_create_runtimes`를 기존 RBLN 모델 패턴대로 구현한다. [#625, #533]
2. NPU에서 실행하기 어려운 후처리는 CPU 대체 경로로 구현하고, 무거운 의존성은 모듈 최상단에서 import하며 `pyproject.toml`에 추가한다. [#533]
3. HF 구현이 마스크나 어텐션 구현을 내부에서 고르면 컴파일용 wrapper와 어텐션 구현 고정으로 결정성을 확보한다. [#625]
4. 복합 VLM 설정에서는 최상위 `batch_size`를 `force_kwargs` 없이 서브모듈 기본값으로 전달하고, 비전 타워처럼 값이 고정돼야 하는 서브모듈에만 `force_kwargs`로 리터럴을 고정한다. 최상위만 지정, 서브모듈만 지정, 둘 다 지정(같은 값/다른 값)을 config 레벨 테스트로 parametrize한다. [#681]
5. 배치 정렬(CR13 이상)이 필요하면 게이트를 모델 설정에 연결하고(composition 모델은 내부 언어 모델 설정을 읽음), 이미지 우선 텐서를 샘플 소유 관계로 행에 매핑하되 매핑 불가 시 에러로 중단한다. forward에 `inputs_sorted` 플러밍과 미정렬 다중 배치 가드를 추가하고, 게이트가 꺼진 경우와 배치 1에서 no-op임을 확인한다. [#706, #707]
6. 모델 패키지의 export 목록 세 곳(패키지, transformers, 최상위)에 클래스를 등록하고 Auto 클래스가 네이밍 규칙으로 연결되는지 확인한다. 파일 목록에 따라 `cli.py`를 갱신한다. [#625, #533]
7. sibling 테스트를 본뜬 테스트를 tiny `hf-internal-testing` 모델로 `tests/test_transformers.py`에 추가하고, 배치 정렬은 호스트 단위 테스트(`tests/test_batch_sort_vlm_*.py`)로 검증한다. [#625, #706, #707]
8. 실제 NPU에서 컴파일하고 CPU 기준 출력과 비교한 결과를 PR 본문에 기록하며, NPU 검증을 하지 않은 경로는 PR 설명에 명시한다. [#625, #706, #707]
9. 모델 README 문서를 작성한다. [#533]

대표 경로: `src/optimum/rbln/transformers/models/<model>/`, `src/optimum/rbln/transformers/utils/generation_multimodal.py`, `src/optimum/rbln/transformers/models/decoderonly/generation_decoderonly.py`, `src/optimum/rbln/__init__.py`, `src/optimum/rbln/cli.py`, `tests/test_transformers.py`, `tests/test_config.py`

인용:
> "Everything is a no-op when the gate is off (CA targets, old configs, batch=1)." (#706)
> "parametrized over the six composite configs: top-level-only propagates, submodule-only works" (#681)

근거 PR: #533, #625, #681, #706, #707

---

## 3. 새 diffusers 파이프라인/모델 추가

1. 구성 클래스를 추가한다. 서브모듈별 설정은 `configurations/models/configuration_<name>.py`(모델 파일 이름과 일치), 파이프라인 설정은 `configurations/pipelines/configuration_<name>.py`에 두고 각 `__init__`에서 export한다. 파이프라인 설정 클래스는 파이프라인 클래스의 멤버로 노출한다. [#114, #106, #199, #168, #38, #42, #726, #694, #816, #804]
2. 서브모듈 래퍼를 `diffusers/models/<범주>/`(autoencoders, transformers, unets, controlnets) 아래에 upstream diffusers 레이아웃에 맞춰 둔다. 기존 RBLN 클래스(텍스트 인코더, transformer, VAE)는 재사용하고, 기존 그래프를 바꾸는 변형은 별도 래퍼로 분리한다. [#726, #694, #816, #15, #114]
3. 파이프라인을 `diffusers/pipelines/<name>/`에 구현한다. connected 파이프라인이면 기존 단일 파이프라인 구현을 재사용하고, 컴파일된 서브모듈 로드에는 `optimize_host_memory`를 사용한다. 표준 `PreTrainedModel`을 상속하지 않는 컴포넌트는 optional 서브모듈 목록(현재 이름 `_optional_submodules`)에 등록한다. [#38, #42, #206, #198, #114]
4. 파이프라인 설정 인자 이름은 diffusers 호출 인자 이름과 맞추고, 파이프라인이 고정하는 서브모듈 설정은 `force_kwargs`로 강제한다. 파이프라인별 compiled batch size와 런타임 인자의 정합성 검사를 구현하고 불일치 시 에러를 낸다. [#816, #804, #38, #42]
5. `modeling_diffusers.py`, `diffusers/__init__.py`, `rbln/__init__.py`에 등록한다. [#168, #114, #15, #38, #42]
6. 정밀도에 민감한 연산은 호스트 fp32로 두고 나머지는 그래프에서 수행하며, 공식 Hub 리비전과 별도 가중치 레이아웃을 따른다. [#694]
7. 예제 스크립트와 테스트를 추가한다. `tests/test_diffusers.py`에 tiny-random 파이프라인으로 네이티브 diffusers 출력과 비교하는 테스트(export, generation, reload 포함)를, `tests/test_config.py`에 서브모듈 크기 파생 테스트를 둔다. 기존 모델의 컴파일 그래프가 바이트 단위로 같은지 확인한다. [#187, #15, #38, #816, #725, #694, #106]
8. 생성 결과물을 PR에 공유하고, 필요한 컴파일러 hash 또는 버전을 PR 본문에 명시한다. [#38, #42]

대표 경로: `src/optimum/rbln/diffusers/configurations/models/`, `src/optimum/rbln/diffusers/configurations/pipelines/`, `src/optimum/rbln/diffusers/models/**`, `src/optimum/rbln/diffusers/pipelines/<name>/`, `src/optimum/rbln/diffusers/modeling_diffusers.py`, `src/optimum/rbln/diffusers/__init__.py`, `tests/test_diffusers.py`, `tests/test_config.py`

인용:
> "with `force_kwargs`, so a conflicting VAE setting raises" (#804)
> "Predict graphs and their input order stay byte-identical" (#725)

근거 PR: #15, #38, #42, #106, #114, #168, #187, #198, #199, #206, #694, #725, #726, #804, #816

---

## 4. transformers 버전 상향 (torch/diffusers/컴파일러/Python 등 의존성 상향 포함)

1. 선행 조건을 확인한다. 배포 컴파일러 버전이 pytest가 쓰는 버전과 일치하는지, 같은 torch를 검증한 컴파일러 빌드가 있는지 확인한다. 메이저 대응은 `transformers/vX` 브랜치를 base로 한 준비 PR에서 시작하고, 컴파일러 측 변경이 필요하면 컴파일러 브랜치 또는 hash를 PR 본문에 명시한다. [#310, #50, #54, #807, #479, #480, #171, #79, #769]
2. 의존성 핀/범위를 갱신한다. `pyproject.toml`의 transformers/torch/diffusers 버전을 올리고(transformers는 단일 `==` 핀 대신 지원 범위 선언 가능), `uv.lock`을 재생성해 바뀐 패키지가 의도한 것뿐인지 확인한다. Python 상한을 올릴 때는 `uv sync --python <ver> --dry-run`과 classifier 추가를 함께 한다. [#282, #301, #564, #582, #769, #773, #442, #434, #549, #13, #671]
3. Renovate가 연 상향 PR은 자동 머지되지 않으므로 Release Notes를 확인하고 사람이 직접 병합한다. 핵심 의존성(예: transformers)은 제목에 `[pytest-full]` 접두어를 붙여 전체 테스트를 돌린다. 충돌 시 rebase 체크박스로 봇이 다시 rebase하게 한다. [#476, #372, #340, #341, #306, #297, #582]
4. 깨지는 코드를 고친다. `modeling_*.py`, `*_architecture.py`, rope 공통 유틸을 새 버전에 맞추고, 버전 의존 API는 호환 모듈 한 곳으로 모으며 분기는 공용 버전 비교 헬퍼로 한다. diffusers 상향 시 deprecated import 경로와 type annotation 오류를 수정하고, 이전 버전 전용 기능은 deprecate하며 대응용 임시 패치는 업데이트 후 되돌린다. [#282, #301, #310, #330, #171, #549, #564, #186, #155, #773, #13]
5. 지원 범위를 관리한다. 지원하는 메이저 버전마다 PR CI 매트릭스에 항목을 추가하고, 알려진 호환 문제가 있는 (모델 x 메이저 버전) 조합은 런타임 허용목록에 등록해 막는다. [#564, #582]
6. 컴파일러 버전을 반영한다. `version.yml`(과 `utils/import_utils.py`)의 컴파일러 버전, PR CI가 쓰는 컴파일러 버전(기준 버전 또는 nightly)을 함께 갱신한다. [#50, #54, #807, #59, #23]
7. 재구현한 그래프 코드를 대상 transformers 소스 기준으로 dtype 재감사한다. 명시적 dtype 캐스트와 상수 표현을 항목별로 대조해 HF 형태로 복원하고(복원하지 않는 항목은 사유 기록), 헬퍼는 CPU fp16/bf16에서 HF 함수와 비트 단위 일치를 확인한다. [#711]
8. 테스트한다. CPU 단위 테스트와 샘플 모델 테스트, 영향받은 모델 디렉터리 테스트, 전체 pytest를 실행하고, 혼합 dtype 행렬곱처럼 새 torch에서 실패하는 구현을 고친다. 검증하지 못한 모델과 미완 커버리지는 PR 본문에 명시한다. [#769, #773, #282, #301, #310, #171, #711]
9. 릴리스 전에 테스트 전용으로 열어 둔 설정(bc test, 컴파일러 testing 버전)을 prod로 원복한다. [#549]

대표 경로: `pyproject.toml`, `uv.lock`, `version.yml`, `.github/version.yaml`, `src/optimum/rbln/utils/import_utils.py`, `src/optimum/rbln/utils/transformers_compat.py`, `src/optimum/rbln/transformers/modeling_rope_utils.py`, `src/optimum/rbln/transformers/models/*/modeling_*.py`, `src/optimum/rbln/transformers/models/*/*_architecture.py`, `src/optimum/rbln/diffusers/models/**`, `src/optimum/rbln/ops/*.py`

인용:
> "Please merge this manually once you are satisfied." (#582)
> "This PR re-audits the graph code we re-implement against transformers 5.8.1 and restores the HF form." (#711)
> "`uv.lock` regenerated; only the three packages above changed." (#769)

근거 PR: #13, #23, #50, #54, #59, #79, #155, #171, #186, #282, #297, #301, #306, #310, #330, #340, #341, #372, #434, #442, #476, #479, #480, #549, #564, #582, #671, #711, #769, #773, #807

---

## 5. 릴리스(dev→main)

주의: #727 이후 dev가 기본 브랜치가 되고 main이 제거되었다. 아래 절차는 그 이전 관행을 기록한 것이며 현행 여부는 인터뷰에서 확인한다.

1. dev를 base=main으로 하는 PR을 연다. 제목은 `release: v<버전>` 형식이며 패치 릴리스는 `.postN` 접미사를 붙인다. Type of Change에서 `Release (dev → main merge for production release)`를 체크하고, 수정이 포함되면 Bug fix도 체크한다. 변경 개요(Changes Overview)에 릴리스 포함 변경을 정리한다. [#482, #484, #512, #541, #554, #578, #384, #388, #342]
2. 파일 목록에 버전 정의 파일, `cli.py`, diffusers 구성 파일 등 릴리스 대상 변경이 포함되는지 검토한다. [#541, #342, #384, #388, #482]
3. dev와 main이 충돌하면 base=dev인 `release: resolve conflicts between dev & main` PR로 해결해 릴리스 PR에 반영한다. 릴리스 이후 빠진 커밋은 `release: missed commit in v<버전>` PR로 main에 반영한다. [#511, #521, #512]
4. 릴리스 PR은 merge commit으로 병합하고, squash sync를 쓴 경우 sync 직후 main을 dev로 백머지해 merge base를 관리한다. 백머지는 조상 관계를 잇는 것이 목적이므로 squash하지 않는다. (저장소 설정이 squash만 허용해 충돌이 있었음: #695) [#615, #617, #695, #696]
5. 릴리스(및 핫픽스) 후 main의 변경을 dev로 백머지한다. 핫픽스는 base=main PR로 먼저 병합하고, 같은 변경을 base=dev PR로 제목에 `(back merge)`를 붙여 반영한다. [#482, #484, #451, #454, #492, #493]
6. 배포 후 수정은 릴리스 태그 기준 브랜치에서 만들어 릴리스 라인에 먼저 반영하고, 그 수정을 dev에 반영하며 중복 커밋은 revert로 정리한다. 이미 배포된 PyPI wheel은 git 정리와 무관하게 변경하지 않는다. [#615, #616, #617]
7. main에 잘못 머지된 변경은 main에서 revert PR로 되돌리고, 원본 squash 커밋을 `cherry-pick -x`로 dev에 옮겨 `git patch-id`가 같은지 확인한 뒤 main을 dev로 백머지한다. dev→main 동기화 PR은 base를 main으로 명시한다. [#692, #693, #695, #696]
8. 릴리스 브랜치에 독스트링 전용 수정이 필요하면 dev 커밋을 `cherry-pick -x`로 적용해 원본 커밋을 기록하고, diff 동일 여부와 ruff를 확인한다. [#805]
9. 릴리스 태그 빌드(또는 `BC_TAG`)로 BC(하위 호환성) 기준선 아티팩트를 만든다. 릴리스 태그에 대해서만 컴파일을 트리거하고, 해당 릴리스의 version pin 컴파일러를 설치해 대상 SoC를 명시하고 런타임을 생성하지 않은 채 save 모드로 컴파일한 뒤 BC 기준선 경로에 업로드한다. 산출물은 추론 테스트에서 재사용한다. [#404, #753]

대표 경로: `src/optimum/rbln/__init__.py`, `src/optimum/rbln/configuration_utils.py`, `src/optimum/rbln/cli.py`, `src/optimum/rbln/diffusers/**`, `.buildkite/bc-compile.yml`, `.buildkite/scripts/bc-compile-steps.sh`, `.github/workflows/bc_*.yaml`

인용:
> "- [x] Release (dev → main merge for production release)" (#578)
> "backmerge immediately after every squash sync" (#695)
> "keep the PyPI wheel 0.11.0.post1 as-is (treated as immutable, unrelated to git cleanup)." (#616)

근거 PR: #342, #384, #388, #404, #451, #454, #482, #484, #492, #493, #511, #512, #521, #541, #554, #578, #615, #616, #617, #692, #693, #695, #696, #753, #805

---

## 6. deprecation/인자 제거

1. 폐기 대상을 deprecate 유틸(`deprecate_kwarg` 등)로 명시하고 제거 예정 버전을 적는다. 생성자 인자에는 `deprecate_kwarg`를 `raise_if_greater_or_equal_version=False`와 함께 붙인다. 지원 중단 예정 디바이스는 경고를 추가하는 PR을 먼저 낸다. [#760, #645, #597, #228]
2. 이름을 바꿀 때는 새 이름을 도입하고 이전 이름은 폐기 별칭으로 유지하며 중첩 서브모듈 설정 경로에서도 별칭이 동작하는지 확인한다. 직렬화 키를 바꿀 때는 옛 키를 새 키로 재매핑하고, 같은 키를 속성으로 설정하는 경로는 `deprecate_method` 기반 setter로 새 위치에 전달한다. [#597, #601, #645, #760]
3. 폐기 버전 이후까지 경고와 함께 동작을 유지하고, 폐기 버전에서 한 릴리스 동안 에러를 낸 뒤 다음 릴리스에서 제거한다. [#760, #645]
4. 제거는 경고 릴리스 이후 별도 breaking change PR로 내고 릴리스 노트에 제거 사실을 기록한다. [#225, #228]
5. 이전 minor 아티팩트가 남긴 설정 키를 계속 읽어야 하면 컷오프를 다음 minor로 옮긴다. `deprecate_kwarg` 사이트와 제거 안내 문구의 버전을 함께 바꾼다. [#785, #784]
6. 테스트를 추가한다. 별칭 매핑과 서브모듈 경로를 직접 호출하는 회귀 테스트(`tests/test_config.py`), 컷오프 전후 버전을 순회하는 테스트(`tests/unit/cpu/`)를 둔다. [#601, #597, #784]
7. CLI 도움말, docstring, 예제, 테스트를 새 이름으로 바꾼다. [#597]

대표 경로: `src/optimum/rbln/utils/deprecation.py`, `src/optimum/rbln/utils/depreacate_utils.py`, `src/optimum/rbln/configuration_utils.py`, `src/optimum/rbln/transformers/configuration_generic.py`, `src/optimum/rbln/transformers/models/*/configuration_*.py`, `src/optimum/rbln/cli.py`, `tests/test_config.py`

인용:
> "raise an error for one release at the deprecated version, then remove in the following release" (#645)
> "The old name is kept as a deprecated alias via `deprecate_kwarg`, so both" (#597)
> "The decorators have no test that walks a version across a cutoff" (#784)

근거 PR: #225, #228, #597, #601, #645, #760, #784, #785

---

## 7. 성능 최적화 작업

1. 최적화 대상의 기준(HF 참조 결과)과 측정 항목을 정한다. 메모리 최적화는 커밋 단계별로 래핑 peak, 상시 증가분, 컴파일 peak를 측정한다. [#714]
2. 구현한다. MoE는 fused `gate_up_proj`를 gate/up 두 절반으로 분할하되 추가 clone을 제거하고, `down_proj`와 router 가중치는 원본 저장소를 공유하며, 분할 후 쓰이지 않는 `nn.Linear` 껍데기 없이 텐서를 버퍼로 직접 등록하고 HF의 fused 원본을 해제한다. 호스트 rotary는 cos/sin 테이블을 로드 시점에 최대 seq 길이 기준으로 한 번 만들고 호출마다 위치 id로 gather하되, 정적 `inv_freq`가 아닌 rope 타입이나 범위 밖 위치는 경고와 함께 동적 경로로 폴백한다. [#714, #663]
3. 원본 모델 속성이 `None`이 되는 등 동작 변경은 PR 본문에 별도 절로 기록한다. [#714]
4. 정합성을 검증한다. 지정 dtype으로 디바이스 컴파일 후 같은 dtype의 HF 참조를 CPU에서 실행해 로짓 최대 절대 오차, 코사인 유사도, argmax 일치를 비교한다. 테이블 경로와 폴백 경로는 비트 단위 일치, 모듈 출력과는 1-ulp 이내를 확인하고, 결과 표를 PR 본문에 첨부하며 실행하지 못한 경로를 명시한다. [#711, #663]
5. NPU 없이 버그나 최적화를 재현해야 하면 실제 `prefill_forward`를 stub 런타임과 위치 코드 값을 쓰는 가짜 런타임으로 구동한다. 길이(청크 배수 포함), 패딩 방향, `output_hidden_states` 조합을 순회하고, 수정 전 출력을 같은 스크립트로 재현해 결함을 확인한 뒤 해당 테스트 클래스에 회귀 테스트로 추가한다. [#657]

대표 경로: `src/optimum/rbln/transformers/utils/moe.py`, `src/optimum/rbln/transformers/models/*moe*/*_architecture.py`, `src/optimum/rbln/transformers/models/gemma4/gemma4_architecture.py`, `src/optimum/rbln/transformers/models/gpt_oss/gpt_oss_architecture.py`, `src/optimum/rbln/transformers/modeling_rope_utils.py`, `src/optimum/rbln/transformers/models/decoderonly/decoderonly_runtime_utils.py`, `tests/test_llm.py`

인용:
> "Only the two gate/up halves are copied now;" (#714)
> "cos/sin tables built once at load (via `np_cos`/`np_sin`, so still bit-reproducible) and gathered per call" (#663)

근거 PR: #657, #663, #711, #714

---

## 8. 기타

### 8.1 CI/인프라 변경

1. GHA 워크플로를 Buildkite(Obedients)로 이전할 때는 PR 게이트와 pytest suite를 기존 GHA 잡과 나란히 추가하고, 같은 커밋에서 결과가 일치하면 대체된 GHA 잡을 제거한다. 야간 빌드와 BC 단계도 같은 방식으로 옮기고 야간 빌드가 여러 번 green이 된 뒤 대체된 워크플로 파일을 삭제한다. 배포, 릴리스, 의존성 봇 워크플로는 GHA에 남긴다. [#728, #737]
2. CI 실행 환경 이미지는 잠금 파일 기반 Dockerfile(`docker/ci/Dockerfile`)로 유지하고, 잠금 파일이 바뀌면 이미지를 빌드/푸시하는 워크플로가 돌게 하며 `container.image`는 저장소 변수로 지정하고 설치 명령은 `uv pip`으로 통일한다. [#415]
3. 린트/포맷터(ruff) 규칙을 확장하거나 버전을 올릴 때는 `pyproject.toml` 규칙 집합을 확장하고 무시하는 규칙마다 근거를 기록하며, pre-commit 훅, `uv.lock`, quality 의존성 그룹의 ruff 버전을 같은 값으로 올린다. 자동 수정은 diff에서 검토해 동작이 바뀌는 것은 되돌리고 `pre-commit run --all-files`, `ruff format --check .`, `ruff check .`를 실행한다. [#733]

### 8.2 코드/문서 변경

4. 예제 스크립트 갱신: 인자 파싱을 argparse로 통일하고 컴파일 여부 인자 없이 헤더에 동등한 CLI 명령을 적으며, 모든 예제를 실제 NPU에서 컴파일과 추론까지 실행해 확인하고 README 설명을 맞춘다. [#635]
5. 커스텀 op 시그니처 변경: `ops/` op 시그니처를 vllm-rbln 쪽 인터페이스와 맞추고 새 선택 인자는 기본값 `None`으로 두며, 모델 `*_architecture.py` 호출부를 키워드 인자로 바꾸고 해당 없는 인자는 `None`을 명시한다. op 이름은 바꾸지 않고, 최소 하나의 attention 변형과 MoE 모델로 컴파일 스모크 테스트를 한다. attention 커널 변경 시 커널 스키마와 Python 함수 이름, 입력 shape을 컴파일러 규격에 맞추고 각 아키텍처 호출부를 동기화한다. [#566, #579, #24]
6. alias 파일의 모델 클래스를 모델 폴더로 이동: `modeling_alias.py`의 대상 클래스를 모델 폴더 `modeling` 파일로 옮기고 모델 폴더 `__init__`과 상위 `__init__`의 import를 새 위치로 정리한다. [#17]
7. `rbln_config` 구조 변경(저장/로드 형식, `_create_runtimes`): 변경 범위를 PR에 명시하고 내부 CI/CD 도구의 호환성을 함께 갱신하며 모델 코드와 model-zoo 스크립트의 비호환을 확인하고 담당 팀에 리뷰를 요청한다. [#67]
8. PR 본문 작성: 제목은 conventional commit 형식, Type of Change 체크리스트 표시, 버그 수정이면 `Fixes` 이슈 링크, 여러 저장소에 걸치면 related PR 링크를 모두 적는다. [#7]

대표 경로: `.buildkite/**`, `.github/workflows/*.yaml`, `scripts/ci/**`, `docker/ci/Dockerfile`, `pyproject.toml`, `.pre-commit-config.yaml`, `examples/**`, `README.md`, `src/optimum/rbln/ops/*.py`, `src/optimum/rbln/configuration_utils.py`

인용:
> "it stays on until Obedients results match on the same commits." (#728)
> "All 33 example scripts were run end-to-end (compile + inference) on RBLN NPUs and pass." (#635)

근거 PR: #7, #17, #24, #67, #415, #566, #579, #635, #728, #733, #737
