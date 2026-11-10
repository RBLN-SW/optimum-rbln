# 결정 이력 v0 (optimum-rbln)

입력 결정 87건 → 중복 병합 후 86건. 날짜 오름차순. `(superseded?)`는 이후 결정이 뒤집은 것으로 보이는 항목(추정).

## 날짜순

- 2025-02-03 · PR #1 · 패키지 업로드 대상을 내부 PyPI에서 공식 PyPI로 바꾸기로 했다. — "기존에 사용하던 internal pypi 대신 official pypi로 upload하도록 workflow를 변경합니다."
- 2025-02-04 · PR #9 · 누락된 ruff format을 전체 코드에 적용하기로 했다. — "몇몇 코드에 ruff format이 누락되어있던 것을 적용합니다."
- 2025-02-11 · PR #17 · 기존 alias 파일의 모델 클래스를 해당 모델 폴더로 이동하기로 했다. — "transformers/models/bert.py이 존재하므로 해당 위치로 이동시키겠습니다."
- 2025-02-13 · PR #19 · 프로젝트 라이선스를 Apache License 2.0으로 확정했다. — "We are pleased to announce that this project is now officially licensed under the Apache License, Version 2.0."
- 2025-02-17 · PR #14 · 모델 저장 방식을 임시 디렉터리에 복사한 뒤 rename하는 방식으로 바꾸기로 했다. (superseded?) — "대신 temp dir에 copy한 다음, rename하는 식으로 변경합니다."
- 2025-03-24 · PR #39 · 연결된 파이프라인 지원을 위해 새로 추가한 구현은 제거하고 기존 단일 파이프라인 구현을 재사용하기로 했다. — "remove newly added implementations that support a connected pipeline structure"
- 2025-03-26 · PR #63 · PR CI 시간을 줄이기 위해 대상 영역별 skip 태그를 추가하기로 했다. (superseded?) — "pr ci가 오래걸려서... [skip-diffusers], [skip-transformers], [skip-llms] 추가합니다."
- 2025-04-22 · PR #88 · 이 PR은 이번 달 릴리스 이후에 병합한다. — "This PR will be merged after the release of this month."
- 2025-04-23 · PR #108 · 런타임 생성이 실패하면 즉시 에러를 발생시키도록 변경하기로 했다. — "so we update to raise an error immediately when runtime creation fails."
- 2025-04-28 · PR #104 · 배치 크기가 지정되지 않은 diffusers 서브모듈은 Hugging Face 라이브러리 기본값을 참조해 컴파일하기로 했다. — "This PR changes it to compile by referencing the default value of the Huggingface library unless a batch size is otherwise specified."
- 2025-04-29 · PR #109 · 0.7.2부터 optimum-rbln과 컴파일러 패키지의 버전 번호를 같게 유지하기로 했다. — "from 0.7.2, we decided to keep same version number for optimum-rbln and rebel-compiler"
- 2025-05-26 · PR #136 · 릴리스 일정을 맞추기 위해 미완성 Gemma3 코드를 우선 머지하고 후속 PR에서 일반화하기로 했다. — "To meet the release schedule, it seems we need to merge it for now"
- 2025-05-27 · PR #140 · 배포 이후에는 모델 경로에 rbln_config에 없는 컴파일 산출물 파일이 있으면 오류를 내도록 변경한다. — "After release, I will modify it to raise an error if a file other than the compilation model included in rbln_config is in the model path."
- 2025-05-29 · PR #152 · Gemma3는 RBLN-CA02에서 미지원이므로 해당 장치의 CI/CD에서 임시로 제외한다. — "Currently, Gemma3 is not supported in RBLN-CA02."
- 2025-06-13 · PR #157 · 별칭(alias) 클래스를 제거하고 모델별 클래스를 모델 디렉터리 구조로 옮기기로 했다. — "it was awkward to have an alias class to do the above, so we removed alias"
- 2025-06-13 · PR #157 · 공식 문서를 장기적으로 코드베이스 위에서 빌드하는 것을 목표로 한다. — "in the future we aim for official documentation to be built on top of the codebase"
- 2025-06-25 · PR #168 · RBLNCosmosSafetyChecker는 기존 diffusion 파이프라인 안에서 CosmosSafetyChecker를 대체해야 한다. — "So RBLNCosmosSafetyChecker should go into the existing Diffusion pipeline and replace CosmosSafetyChecker."
- 2025-07-04 · PR #203 · SDXL 추론 테스트는 스케줄 CI 전용으로만 실행하기로 했다. — "SDXL inference test to scheduled test only"
- 2025-07-15 · PR #215 · save_pretrained 시 원본 파일 복사 대신 HF 설정을 저장하기로 했다. — "Save HF config when save_pretrained is called, instead of copying original file."
- 2025-07-17 · PR #219 · Model과 CausalLM 래퍼를 분리하지 않고 현재 플래그 구조를 유지한 뒤 리팩터링하기로 했다. — "so we decided to maintain the current state and refactor."
- 2025-07-22 · PR #225 · CA02 관련 구현을 제거하기로 했다(breaking change). — "remove CA02-related implementations"
- 2025-08-11 · PR #246 · save_pretrained는 기존 디렉터리를 삭제하지 않고 저장 대상 파일만 덮어쓰도록 바꾸기로 했다. — "This is unsafe, so we changed it to overwrite only the files to be saved in the directory."
- 2025-08-29 · PR #259 · torch_dtype='auto' 지원을 향후 모든 RBLN 모델로 확장하기로 했다. — "We need to extend this to all RBLN models in the future."
- 2025-09-05 · PR #243 · export 인자는 사용자가 지정하지 않아도 자동으로 추론하기로 했다. — "other(base): infer `export` automatically."
- 2025-09-23 · PR #297 · 의존성 봇은 당분간 transformers와 diffusers만 추적하도록 임시로 제한한다. — "Temporarily turned off all but transformers and diffusers to avoid creating unrelated PRs."
- 2025-09-26 · PR #302 · 보안 취약점 패키지의 자동 업데이트 전략은 최소 버전(lowest)이 아니라 최신 버전(latest)으로 설정하기로 했다. — "When we find a package with a security vulnerability, it is set to lowest, which doesn't match our intent. Fix it to latest."
- 2025-10-23 · PR #327 · prefix caching은 optimum-rbln API 범위가 아니므로 vLLM CI에서 검증한다. — "Prefix caching is not part of the optimum-rbln API, and it should be viewed in the VLLM CI."
- 2025-10-24 · PR #271 · fusion 컴파일을 기본 적용으로 두고 non-fusion 컴파일은 지원하지 않는 것으로 정리한다. — "`fusion` is now applied by default, non-fusion compilation is not supported anymore."
- 2025-11-03 · PR #337 · 원본 모델 수정 방식으로 컴파일되는 모델에서 `from_model` 지원을 추가하기로 했다. — "We will modify this to also support `from_model`."
- 2025-11-17 · PR #362 · PR과 관련된 CI 패치는 dev에 머지하는 것으로 충분하다고 판단한다. — "Since this is a patch related to PR, merging it into dev is sufficient."
- 2025-11-24 · PR #367 · diffusers 파이프라인 docstring은 현재 diffusers 모델 docstring 갱신이 리뷰된 뒤에 갱신하기로 한다. — "docstrings for diffusers pipelines will be updated after current update is reviewed."
- 2025-12-17 · PR #369 · 개발 중인 MoE 모델은 최적화 등 개발이 끝난 뒤 rbln-model-zoo를 통해 공지하기로 한다. — "we plan to announce it via rbln-model-zoo"
- 2026-01-02 · PR #415 · CI의 rebel-compiler 설치는 uv pip으로 통일하기로 했다. — "Switched rebel-compiler installation to use uv pip in CI workflows for consistency."
- 2026-01-14 · PR #444 · 앞으로 아키텍처 서브클래스는 forward() 동작이 바뀌거나 구조가 실제로 다를 때만 메서드를 오버라이드하기로 했다. — "The rule of thumb going forward should be:"
- 2026-01-21 · PR #462 · unittest 실행 순서 의존 테스트는 이름을 바꿔 push 테스트가 pull 테스트보다 먼저 실행되도록 보장한다. — "Rename `test_pull_compiled_model_from_hub` to `test_z_pull_compiled_model_from_hub` to guarantee it runs after `test_push_to_hub`."
- 2026-01-23 · PR #440 · 스케줄 워크플로의 실행 로직은 dev에서 관리하고 main에는 트리거만 두기로 했다. (superseded?) — "Execution logic maintained and iterated on dev"
- 2026-01-23 · PR #468 · 폐지된 rbln-stat 대신 현재 도구 이름으로 오류 메시지를 갱신한다. — "As `rbln-stat` is deprecated, Error messages are updated."
- 2026-01-23 · PR #473 · load() 메서드는 deprecate 처리하고 from_pretrained를 새 설정 로딩 API로 도입한다. (superseded?) — "Existing `load()` methods are marked as deprecated and will be removed in version 0.12.0"
- 2026-02-25 · PR #508 · 내부 버그로 gpt-oss BF16 컴파일 수정은 다음 달로 연기하기로 했고, 이 PR은 그 변경을 반영하는 패치다. — "Due to an internal bug, the bugfix for gpt-oss's compile with BF16 has been postponed until next month."
- 2026-04-22 · PR #537 · 컴파일러 샤딩과 호환되지 않는 ViT 인코더 어텐션 구조 때문에 RSD를 지원하도록 모델을 재설계하기로 했다. — "the model has been redesigned to support RSD."
- 2026-05-15 · PR #560 · vLLM 연동용 prefill 패딩 블록 추가는 whisper 모델에만 적용한다. — "whisper 모델에 한하여 블록 1개 추가 할당하는 PR입니다."
- 2026-06-09 · PR #549 · transformers 5.8.1 상향과 함께 transformers 4.x 전용 기능은 deprecate 처리한다. — "deprecate 4.xx.x transformers features"
- 2026-06-22 · PR #597 · 컴파일러 인자를 개명할 때 이전 이름은 호환성을 위해 폐기 표기로 남기기로 했다. — "However, for backward compatibility, we are marking it as deprecated."
- 2026-06-27 · PR #615 · dev → main 및 main → dev 두 단계 정리로 기존 PR을 대체한다. — "replaced by a two-step cleanup (dev → main + main → dev)."
- 2026-06-28 · PR #617 · 배포 태그는 릴리스 라인의 merge 커밋 위치로 옮기기로 했다. — "will be **moved** from `235075e4` (dev tip) to the new main merge SHA"
- 2026-07-01 · PR #621 · 커스텀 op 등록은 컴파일러를 단일 원천으로 두고 이 라이브러리의 중복 등록을 제거하기로 했다. — "This PR makes rebel-compiler the single source of truth and removes optimum's duplicate registrations."
- 2026-07-16 · PR #632 · 컴파일러 버전은 dev 브랜치의 최신 통과 nightly 빌드에서 가져오기로 했다. — "Resolve the version from the **latest passed `nightly` build on `dev`** in Obedients"
- 2026-07-23 · PR #624 · 타입 주석 현대화에서는 future import를 추가하지 않기로 했다. — "It does **not** add `from __future__ import annotations`."
- 2026-07-24 · PR #642 · 검증 우회 수단을 없애고 검증을 항상 적용하기로 했다. — "validation is always enforced again (including `validate_sliding_window`)."
- 2026-07-28 · PR #645 · v0.11.1 릴리스에 맞춰 폐기 기간이 끝난 항목을 모두 제거하기로 했다. — "this removes everything whose window closed at v0.11.0 or earlier, ahead of the v0.11.1 release."
- 2026-08-13 · PR #666 · 캐시 메타 생성은 서브클래스별 from_config() 팩토리로 한다. — "Construction goes through per-subclass `from_config()` factories."
- 2026-08-13 · PR #669 · hatchling 상한은 임시 조치이며, 배포 스택이 twine 7 이상으로 올라가면 해제하기로 했다. — "Conservative stopgap: lift the cap once the deploy stack moves to `twine>=7`"
- 2026-08-25 · PR #682 · 비전 배치 처리는 qwen3.5와 같은 형태로 통일하기로 했다. — "This PR generalizes the vision batch handling to the **same form as qwen3.5**"
- 2026-08-26 · PR #692, #693 · main은 dev 동기화 병합으로만 전진시키고, 수정은 dev에 먼저 머지한 뒤 dev 동기화로 main에 전달한다. (superseded?) — "main should only advance via dev sync merges, e.g. #661"
- 2026-09-01 · PR #707 · 배치 정렬 기계 장치는 generation_multimodal 모듈 하나가 소유한다. — "One module now owns the machinery: `src/optimum/rbln/transformers/utils/generation_multimodal.py`"
- 2026-09-03 · PR #710 · 단일 dtype 유지를 위해 제거했던 fp32 캐스트는 디바이스 실험으로 안전이 확인되면 HF 충실도를 택해 복원하기로 했다. — "so we take the HF-faithful form:"
- 2026-09-08 · PR #719 · gdn_chunk_size 기본값은 prefill_chunk_size에서 상속하지 않고 고정 상수 128로 둔다. — "Default `gdn_chunk_size` to a fixed `MAX_GDN_CHUNK_SIZE` (128) in the config constructors"
- 2026-09-08 · PR #724 · uv 호출은 pyproject의 >=0.11.25 버전 게이트를 통과하도록 강제한다. — "Gating every uv invocation on `>=0.11.25` keeps `uv.lock` diffs clean and consistent"
- 2026-09-11 · PR #727 · dev가 기본 브랜치가 되었으므로 워크플로의 디스패처/impl 분리를 제거하고 본문을 디스패처에 합친다. — "`dev` is now the default branch and `main` has been removed, so the indirection no longer serves a purpose."
- 2026-09-11 · PR #737 · Obedients로 옮긴 파이프라인이 대체하는 GHA 워크플로는 같은 PR에서 삭제한다. — "the GHA workflows they replace are deleted here."
- 2026-09-14 · PR #742 · 컴파일러 CI 트리거 메시지에 [pytest-full] 마커를 넣어 전체 테스트를 실행하며, 파이프라인 파일은 변경하지 않는다. — "So the marker goes in the message the compiler triggers with, and no gate is added here."
- 2026-09-15 · PR #734 · 모델 추가 시 패키지 __init__.py 파일들을 수정하지 않고, 새 모듈의 __all__과 check_inits 스크립트로 처리한다. — "Adding a model no longer touches four `__init__.py` files: write `__all__` in the new modules and run `python scripts/check_inits.py --fix`."
- 2026-09-15 · PR #748 · 경로 기반 CI 조건부 실행은 고치지 않고 제거하며, 모든 테스트 스위트를 모든 PR에서 실행한다. — "Path gating is removed rather than repaired."
- 2026-09-15 · PR #753 · 릴리스 판정 정책은 외부 필터가 아니라 저장소 안에 둔다. — "The release policy moves into the repo."
- 2026-09-16 · PR #702 · 로드 시 rbln_config 객체 계약은 엄격하게 유지하기로 했다. — "The load-time contract for `rbln_config` objects is kept strict"
- 2026-09-16 · PR #755 · CI 자동 재시도는 도입하지 않는다. — "Worth adding narrowly against `agent_stop` if one ever shows up, not before."
- 2026-09-30 · PR #754 · 컨테이너가 필요 없는 가벼운 CI 워크플로우는 일반 러너 풀에서 실행한다. — "All six workflows now run there."
- 2026-09-30 · PR #770 · 컴파일 형상은 호출자가 생략한 경우에만 채운다. — "Compile-time shapes are now filled in **only when the caller left them out**."
- 2026-10-06 · PR #785 · 0.11.x 아티팩트와의 호환을 유지하기 위해 deprecation 컷오프를 0.13.0으로 옮긴다. — "0.12.0 is required to stay backward compatible with 0.11.x artifacts, so this PR moves the cutoff to `0.13.0`:"
- 2026-10-13 · PR #711 · 재구현한 그래프 코드는 transformers 버전 기준으로 재감사해 HF 형태로 복원하기로 했다. — "This PR re-audits the graph code we re-implement against transformers 5.8.1 and restores the HF form."
- 2026-10-13 · PR #711 · 호스트 측 변경과 이전 아티팩트 호환성이 얽힌 항목은 별도 PR로 분리하기로 했다. — "Moved to #764 (independent, dev-based) together with the artifact-compatibility handling, since it touches the host side and older artifacts."
- 2026-10-14 · PR #805 · 릴리스 브랜치의 독스트링도 dev와 같은 수정을 받는다. — "so the release branch needs the same fixes as `dev`."
- 2026-10-16 · PR #807 · 새 설정을 추가하지 않고 기존 requires_batch_sort 값을 정렬 결정으로 사용한다. — "No new config: `requires_batch_sort` already is optimum's sort decision"
- 2026-10-19 · PR #809 · fork PR 대신 upstream 브랜치에서 CI가 돌도록 PR을 새로 올려 대체한다. — "This PR replaces the fork PR #790 so CI can run on an upstream branch."
- 2026-10-19 · PR #813 · trust_remote_code를 강제하지 않고 호출자가 준 값을 따른다. — "It honors the caller's value, like transformers' Auto classes."

## 날짜 미상 (PR 번호순; 미병합 표기 포함)

- 날짜 미상 · PR #103 · optimum-rbln은 배치 크기를 암묵적으로 추론하지 않도록 변경되었다. — "since optimum-rbln has been updated not to infer batch size implicitly"
- 날짜 미상 · PR #303 · 모델 트레이싱을 torch.jit.trace에서 torch.export로 전환하여 custom linear op 사용을 중단한다. — "But now the models are now traced with torch.export, so the custom linear op is not necessary anymore."
- 날짜 미상 · PR #344 · 자동 코드 리뷰 워크플로는 Copilot으로 대체되어 제거한다. — "auto-code-review (replaced by copilot)"
- 날짜 미상 · PR #445 · 두 설정 입력 경로를 상호 배타로 만들어 사용자가 한 가지 방식만 선택하도록 강제하기로 했다. — "To provide a clearer API, we now enforce that users must choose one approach or the other."
- 날짜 미상 · PR #564 · optimum-rbln은 transformers 4.57.x와 5.x를 모두 지원하며, v4를 고정 pin하지 않기로 했다. — "so we need to operate under both rather than force-pin v4 forever." (순서 불명: #549와 상충 가능)
- 날짜 미상 · PR #566 · custom op 시그니처 정리 작업에서 op 이름 변경은 범위에서 제외한다. — "Op renames (e.g., `paged_attn_decode` -> `attention_naive_decode`) — tied to RBLN compiler pattern matching."
- 날짜 미상 · PR #567 · RBLNModel.get_pytorch_model의 dtype 기본값은 체크포인트 dtype을 따르는 `auto`로 한다. — "Default `dtype` to `"auto"` in `RBLNModel.get_pytorch_model` so HF"
- 날짜 미상 · PR #630 · tag 릴리스 시 모델 생성 빌드는 워크플로 디스패치 대신 Obedients API로 직접 생성하기로 했다. — "to **create a Buildkite (Obedients) build directly** instead of a GitHub Actions `workflow_dispatch`." (미병합 PR)
- 날짜 미상 · PR #696 · 릴리스 PR(dev → main)은 이제부터 base를 main으로 명시한다. — "Release PRs (`dev` → `main`) will need `--base main` explicitly from now on." (superseded?)
- 날짜 미상 · PR #696 · 기여자 안내의 기준 브랜치는 main이 아니라 dev다. — "`CONTRIBUTING.md` now points contributors at `dev` instead of `main`."
- 날짜 미상 · PR #760 · Qwen3-VL과 Qwen3.5의 _load_visual_runtime 표기는 visual.create_runtimes 표기로 대체하고 폐지한다. — "on the Qwen3-VL and Qwen3.5 configs in favour of that spelling."

## superseded? 쌍 (이전 결정 → 이후 결정)

1. PR #440 (2026-01-23, 스케줄 워크플로 실행 로직은 dev, main에는 트리거만) → PR #727 (2026-09-11, dev가 기본 브랜치가 되고 main이 제거되어 디스패처/impl 분리를 합침)
2. PR #692, #693 (2026-08-26, main은 dev 동기화 병합으로만 전진) → PR #727 (2026-09-11, main 제거)
3. PR #696 (날짜 미상, 릴리스 PR은 `--base main` 명시) → PR #727 (2026-09-11, main 제거로 base=main 릴리스 흐름 소멸). 같은 PR #696의 "기여자 안내 기준 브랜치는 dev"는 유지됨
4. PR #473 (2026-01-23, `load()` 제거 예정 버전 0.12.0) → PR #785 (2026-10-06, deprecation 컷오프를 0.13.0으로 이동)
5. PR #14 (2025-02-17, 임시 디렉터리에 복사 후 rename) → PR #246 (2025-08-11, 기존 디렉터리를 지우지 않고 저장 대상 파일만 덮어쓰기). 부분 번복 가능성, 확인 필요
6. PR #63 (2025-03-26, 영역별 skip 태그로 PR CI 단축) → PR #748 (2026-09-15, 경로 기반 조건부 실행 제거, 모든 PR에서 전체 스위트 실행). 부분 번복 가능성, 확인 필요

참고(번복 쌍으로 확정하지 않음): PR #564(날짜 미상, transformers 4.57.x와 5.x 동시 지원)와 PR #549(2026-06-09, 5.8.1 상향과 함께 4.x 전용 기능 deprecate)는 같은 주제이나 선후 관계가 불명이다. PR #630(Obedients API로 직접 빌드 생성)은 미병합이다.
