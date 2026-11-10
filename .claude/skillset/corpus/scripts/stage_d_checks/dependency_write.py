import json
SD = "<SCRATCH>/stage_d"
SC = SD + "/scripts"
TIP = "mkdir -p /tmp/devtip && gh api repos/rebellions-sw/optimum-rbln/tarball/dev | tar xz --strip-components=1 -C /tmp/devtip"
out = [
{
 "id": "DEP-001",
 "check": {"kind": "ast",
  "description": "src/ 전체를 AST로 훑어 deprecate_kwarg/deprecate_method 데코레이터 호출의 version 인자를 현재 패키지 버전과 비교하고(≤ 현재 버전이면 위반), 데코레이터 없이 logger.warning*로 'deprecat' 메시지를 내는 ad-hoc deprecation(업스트림 HF 복사 메시지 제외)을 위반으로 센다.",
  "command": f"python3 {SC}/dependency_dep001.py  # worktree HEAD (v0.11.1); dev tip: {TIP} && cd /tmp/devtip && python3 {SC}/dependency_dep001.py 0.12.1"},
 "compliant": 7,
 "violations": [
  {"path": "src/optimum/rbln/configuration_utils.py", "line": 767, "snippet": "logger.warning(\"`optimize_host_memory` is deprecated and will be removed in future versions.\")", "note": "ad-hoc warning, no decorator, no removal version"},
  {"path": "src/optimum/rbln/configuration_utils.py", "line": 814, "snippet": "logger.warning_once(\"`torch_dtype` is deprecated. Use `dtype` instead.\")", "note": "property getter deprecation without removal version"},
  {"path": "src/optimum/rbln/configuration_utils.py", "line": 819, "snippet": "logger.warning_once(\"`torch_dtype` is deprecated. Use `dtype` instead.\")", "note": "property setter deprecation without removal version"},
  {"path": "src/optimum/rbln/transformers/models/decoderonly/decoderonly_architecture.py", "line": 921, "snippet": "\"DecoderOnlyFlashAttention is deprecated and may not work as expected. Use DecoderOnlyAttention instead.\"", "note": "class deprecation in __init__, no decorator or removal version"},
  {"path": "src/optimum/rbln/transformers/models/gemma4/modeling_gemma4.py", "line": 716, "snippet": "\"`image_features` dtype differs from `rbln_config.dtype` is deprecated and will be \"", "note": "has version but hand-written warning, not decorator"},
  {"path": "src/optimum/rbln/transformers/models/gpt_oss/modeling_gpt_oss.py", "line": 105, "snippet": "logger.warning_once(\"`torch_dtype` is deprecated! Use `dtype` instead!\")", "note": "kwarg deprecation bypasses deprecate_kwarg, no removal version"},
  {"path": "src/optimum/rbln/transformers/utils/rbln_quantization.py", "line": 85, "snippet": "logger.warning(\"The `precision` argument is deprecated. Use `weights` and `activations` instead.\")", "note": "kwarg deprecation bypasses deprecate_kwarg, no removal version"},
  {"path": "src/optimum/rbln/transformers/utils/rbln_quantization.py", "line": 182, "snippet": "logger.warning_once(\"torch_dtype is deprecated. Use dtype instead.\")", "note": "kwarg deprecation bypasses deprecate_kwarg, no removal version"}
 ],
 "violations_total": 8,
 "verdict": "aspirational",
 "verdict_reason": "dev tip(74e97fa, 버전 0.12.1.dev)에서 데코레이터 사용 7곳은 모두 제거 버전 0.13.0 > 현재로 정상이지만, 데코레이터 없이 버전도 없는 ad-hoc deprecation 경고가 8곳(53%) 남아 있다(worktree HEAD v0.11.1 기준 3 대 7). 또 '해당 버전 도달 시 제거'는 실제로는 #785('move the 0.12.0 deprecation cutoff to 0.13.0')처럼 제거 대신 컷오프를 미루는 방식으로 처리되었다.",
 "rule_refinement": "사용자 공개 kwarg·메서드의 deprecation은 utils/deprecation.py의 deprecate_kwarg/deprecate_method로 표시하고 version을 반드시 적는다. 데코레이터로 표현할 수 없는 경우(property, 클래스 생성자, kwargs pop)에도 경고 메시지에 제거 예정 버전을 명시한다. 현재 버전이 제거 버전에 도달하면 릴리스 전에 제거하거나, 연기한다면 별도 PR로 컷오프를 명시적으로 옮긴다.",
 "suggested_test": "tests/test_conventions.py::test_deprecations_have_live_removal_version — src/optimum/rbln/**/*.py를 AST로 순회해 (1) deprecate_kwarg/deprecate_method 호출의 version이 importlib.metadata의 optimum-rbln base 버전보다 큰지 확인(같거나 작으면 실패 = 제거할 때가 됨), (2) logger.warning*/warnings.warn 문자열에 'deprecat'가 있으면서 r'\\d+\\.\\d+\\.\\d+' 버전 표기가 없는 호출을 실패로 처리한다. 업스트림 복사 메시지(blip_2의 'v4.50')는 제외하고, 현재 위반 8곳(configuration_utils.py optimize_host_memory·torch_dtype property 2곳, decoderonly_architecture.py DecoderOnlyFlashAttention, gpt_oss/modeling_gpt_oss.py, rbln_quantization.py 2곳; gemma4는 버전이 있으므로 (2)에서 통과)을 파일:함수 단위 allowlist로 둔다."
},
{
 "id": "DEP-012",
 "check": {"kind": "script",
  "description": "CI 설정(.github/workflows, .buildkite, scripts/ci)에서 rebel-compiler를 설치하거나 버전을 결정하는 모든 지점이 .github/version.yaml의 rebel_compiler_version을 읽는지, 그리고 버전 문자열을 하드코딩한 곳(rebel-compiler==<숫자>)이 없는지 확인한다.",
  "command": f"grep -rnE 'rebel_compiler_version|version\\.yaml|rebel-compiler==' .github scripts .buildkite 2>/dev/null; grep -rnE 'rebel-compiler\\s*(==|>=|~=)\\s*[0-9]' .github scripts .buildkite pyproject.toml tests 2>/dev/null  # run in worktree HEAD and in dev tip ({TIP} && cd /tmp/devtip)"},
 "compliant": 3,
 "violations": [],
 "violations_total": 0,
 "verdict": "verified",
 "verdict_reason": "dev tip에서 컴파일러 버전을 결정하는 3곳(scripts/ci/install-rebel-compiler.sh, .github/workflows/auto_compiler_update.yaml, scripts/ci/slack-report.py)이 모두 .github/version.yaml을 읽고 하드코딩된 핀은 0건이다. worktree HEAD(v0.11.1)에서도 GHA 5개 워크플로(rbln_trigger_on_pr, rbln_scheduled_test_impl, bc_compile, bc_inference, auto_compiler_update_impl)가 같은 grep으로 읽고 rbln_optimum_pytest.yaml 한 곳에서만 설치한다. 단 REBEL_COMPILER_VERSION 환경변수 override(rebel-compiler-override.sh)가 수동 실행용 탈출구로 존재하며, '배포 컴파일러 버전과 일치 확인'은 정적으로 검증할 수 없다.",
 "rule_refinement": "CI가 설치하는 rebel-compiler 버전은 .github/version.yaml의 rebel_compiler_version 한 곳에서만 정하며, CI 스크립트·워크플로는 이 값을 읽거나 명시적 override 변수(REBEL_COMPILER_VERSION)로만 바꾼다. 버전 갱신은 auto_compiler_update 봇 PR 또는 이 파일 수정으로만 한다.",
 "suggested_test": "tests/test_conventions.py::test_compiler_version_single_source — .github/workflows/*.y*ml, .buildkite/**/*.yml, scripts/ci/*.sh, pyproject.toml, tests/**/*.py를 텍스트로 읽어 r'rebel-compiler\\s*(==|>=|~=)\\s*\\d' 패턴(하드코딩된 핀)이 있으면 실패시키고, 'rebel-compiler==' 설치 명령이 있는 파일은 같은 파일 안에 'version.yaml' 또는 '${{ inputs.rebel_compiler_version }}'/'${REBEL_COMPILER_VERSION}' 참조가 있어야 통과한다. 또 .github/version.yaml이 정확히 한 줄 'rebel_compiler_version: <PEP440>' 형식인지 검사한다. 현재 allowlist 없음."
},
{
 "id": "DEP-033",
 "check": {"kind": "corpus",
  "description": "2025-01 이후 병합된 PR 중 본문에 Renovate 서명(renovatebot/renovate) 또는 create-pull-request 서명이 있는 의존성 봇 PR에 대해 gh로 mergedBy와 타임라인의 auto_merge_enabled 이벤트를 조회해, 자동 병합이 켜졌거나 봇 계정이 병합한 경우를 위반으로 센다.",
  "command": f"python3 {SC}/dependency_dep033.py {SD.rsplit('/stage_d',1)[0]}/corpus/prs.jsonl"},
 "compliant": 42,
 "violations": [],
 "violations_total": 0,
 "verdict": "verified",
 "verdict_reason": "2025-01 이후 병합된 봇 생성 의존성 PR 42건(Renovate 7, auto_compiler_update create-pull-request 35) 모두 auto-merge 이벤트가 없고 rebel-kblee/rebel-jongho/rebel-thkim/rebel-dkhong/rebel-seinpark가 직접 병합했다(100%). .github/renovate.json에도 automerge 설정이 없다(기본값 비활성).",
 "rule_refinement": "Renovate(transformers/diffusers)와 auto_compiler_update 워크플로가 만든 dependency: PR은 auto-merge를 켜지 않고, [pytest-full] 테스트 결과를 확인한 팀원이 직접 병합한다.",
 "suggested_test": "tests/test_conventions.py::test_renovate_automerge_disabled — .github/renovate.json을 json으로 읽어 최상위와 모든 packageRules에 automerge/platformAutomerge가 true로 설정되어 있지 않은지 확인하고, .github/workflows/auto_compiler_update*.yaml에 'gh pr merge --auto' 또는 peter-evans/enable-pull-request-automerge 사용이 없는지 텍스트 검사한다. 병합 방식 자체는 코드로 강제할 수 없으므로 설정 회귀만 막는다. allowlist 없음."
},
{
 "id": "DEP-015",
 "check": {"kind": "ast",
  "description": "src/를 AST로 훑어 현재 핀된 transformers/diffusers/torch에서 deprecated인 사용(config.use_return_dict 읽기, 단수형 diffusers.models.controlnet import, from_pretrained의 torch_dtype=/use_auth_token=, torch.cuda.amp)을 위반으로, config.return_dict 읽기와 diffusers.models.controlnets import를 준수로 센다.",
  "command": f"python3 {SC}/dependency_dep015.py .  # worktree HEAD; dev tip: {TIP} && python3 {SC}/dependency_dep015.py /tmp/devtip"},
 "compliant": 37,
 "violations": [],
 "violations_total": 0,
 "verdict": "verified",
 "verdict_reason": "dev tip(74e97fa)에서는 #808 이후 config.use_return_dict 0건, 준수 37곳, 위반 0건이다. 다만 worktree HEAD(main v0.11.1, #808 이전)에는 use_return_dict 읽기 13곳이 남아 있어(준수 2) main은 다음 릴리스 병합 때 해소된다. 검사 대상 패턴 목록은 알려진 deprecation만 다루는 근사치다.",
 "rule_refinement": None,
 "suggested_test": "tests/test_conventions.py::test_no_deprecated_upstream_apis — src/optimum/rbln/**/*.py를 AST로 읽어 (a) Attribute 'use_return_dict' 접근, (b) 'diffusers.models.controlnet'(복수형 controlnets 제외) import, (c) from_pretrained 호출의 torch_dtype=/use_auth_token= 키워드, (d) torch.cuda.amp 접근을 찾으면 실패한다. 패턴 목록은 상단 DEPRECATED_UPSTREAM 표에 (패턴, 대체, 근거 PR)로 두고 의존성 bump PR마다 추가한다. dev 기준 allowlist 없음."
},
{
 "id": "DEP-008",
 "check": {"kind": "ast",
  "description": "src/의 모든 import를 AST로 모아 stdlib·내부·전이 의존성을 뺀 서드파티 배포판이 pyproject.toml의 dependencies/optional-dependencies에 선언되었는지, 기본 의존성이 아닌 패키지는 모듈 최상위가 아닌 가드(is_*_available) 아래에서 import되는지 확인한다.",
  "command": f"python3 {SC}/dependency_dep008.py .  # worktree HEAD; dev tip: {TIP} && python3 {SC}/dependency_dep008.py /tmp/devtip"},
 "compliant": 10,
 "violations": [
  {"path": "src/optimum/rbln/diffusers/pipelines/cosmos/cosmos_guardrail.py", "line": 34, "snippet": "from cosmos_guardrail import CosmosSafetyChecker", "note": "not declared in pyproject (guarded by is_cosmos_guardrail_available)"},
  {"path": "src/optimum/rbln/diffusers/pipelines/cosmos/cosmos_guardrail.py", "line": 35, "snippet": "from cosmos_guardrail.cosmos_guardrail import (", "note": "not declared in pyproject (guarded import)"},
  {"path": "src/optimum/rbln/diffusers/pipelines/cosmos/cosmos_guardrail.py", "line": 47, "snippet": "from retinaface.data import cfg_re50", "note": "transitive of cosmos_guardrail, undeclared (guarded import)"}
 ],
 "violations_total": 3,
 "verdict": "verified",
 "verdict_reason": "dev tip에서 필수 서드파티 10개 패키지(torch, transformers, diffusers 등; rebel-compiler는 사내 인덱스라 의도적으로 제외)는 모두 기본 의존성에 선언되어 있고, 비필수 패키지(cosmos_guardrail, retinaface)는 기본 의존성에 없이 is_cosmos_guardrail_available() 가드 뒤에서만 import된다. 위반으로 잡힌 3곳은 '명시' 요건만 어긴 것으로, #168 리뷰('not a necessary dependency')의 결정과 일치한다. 근거 PR #533(einops/trimesh 추가)은 병합되지 않았다(CLOSED).",
 "rule_refinement": "모델 구현에 반드시 필요한 외부 패키지는 pyproject.toml 기본 dependencies에 버전과 함께 선언한다. 특정 파이프라인 전용 비필수 패키지(예: cosmos guardrail)는 기본 의존성에 넣지 않고 utils의 is_*_available() 가드 아래에서 import하며, 미설치 시 대체 클래스로 실패를 늦춘다(optional extra 선언은 현재 관행이 아니다).",
 "suggested_test": "tests/test_conventions.py::test_third_party_imports_declared_or_guarded — src/optimum/rbln/**/*.py의 import를 AST로 모아 stdlib/optimum/rebel 및 전이 의존성 허용 목록(numpy, huggingface_hub, safetensors, PIL 등)을 빼고, 남은 배포판이 pyproject [project].dependencies에 없으면 해당 import 노드의 조상에 `if is_*_available()` 또는 try/except ImportError가 있는지 확인해 없으면 실패한다. 현재 allowlist 없음(cosmos_guardrail, retinaface는 가드 조건으로 통과)."
},
{
 "id": "DEP-011",
 "check": {"kind": "ast",
  "description": "src/, tests/, examples/, scripts/의 torch.load 호출과 partial(torch.load, ...)을 AST로 찾아 weights_only 키워드가 없는 호출을 위반으로 센다.",
  "command": f"python3 {SC}/dependency_dep011.py .  # worktree HEAD; dev tip: {TIP} && python3 {SC}/dependency_dep011.py /tmp/devtip"},
 "compliant": 25,
 "violations": [
  {"path": "src/optimum/rbln/transformers/models/decoderonly/lora_architecture.py", "line": 79, "snippet": "(\"adapter_model.bin\", lambda p: torch.load(p, map_location=\"cpu\")),", "note": "torch.load without explicit weights_only"},
  {"path": "src/optimum/rbln/transformers/models/decoderonly/lora_architecture.py", "line": 80, "snippet": "(\"pytorch_model.bin\", lambda p: torch.load(p, map_location=\"cpu\")),", "note": "torch.load without explicit weights_only"}
 ],
 "violations_total": 2,
 "verdict": "verified",
 "verdict_reason": "dev tip에서 torch.load 27곳 중 25곳(93%)이 weights_only를 명시하며(자체 저장 artifact는 False, 외부 체크포인트는 True), worktree HEAD도 23/2로 같다. 남은 2곳은 LoRA 어댑터 .bin 로더로 torch 2.6+ 기본값 True에 기대고 있다.",
 "rule_refinement": "torch.load는 항상 weights_only를 명시한다: 라이브러리가 직접 저장한 torch_artifacts.pth 등은 weights_only=False, 사용자·외부에서 받은 체크포인트(LoRA 어댑터 등)는 weights_only=True.",
 "suggested_test": "tests/test_conventions.py::test_torch_load_explicit_weights_only — src/, examples/를 AST로 순회해 torch.load(...) 호출과 functools.partial(torch.load, ...)에 weights_only 키워드가 없으면 실패한다. 현재 allowlist: src/optimum/rbln/transformers/models/decoderonly/lora_architecture.py 79·80행(수정 후 제거)."
},
{
 "id": "DEP-016",
 "check": {"kind": "script",
  "description": "src/의 행렬곱 지점(torch.matmul/bmm/baddbmm/einsum/mm, @ 연산자)을 AST로 찾아 피연산자나 직전 3줄의 바인딩에 .to()/.type()/.float() 등 명시적 dtype 정렬이 보이는지 세는 근사 검사다. 피연산자의 실제 dtype은 정적으로 알 수 없어 위반 판정은 하지 않는다.",
  "command": f"python3 {SC}/dependency_dep016.py .  # worktree HEAD; dev tip: {TIP} && python3 {SC}/dependency_dep016.py /tmp/devtip"},
 "compliant": 10,
 "violations": [],
 "violations_total": 0,
 "verdict": "unverifiable",
 "verdict_reason": "dev tip(torch 2.13 핀)에서 행렬곱 40곳 중 10곳에 명시적 dtype 정렬이 보이고(HEAD torch 2.11에서는 38곳 중 3곳; #773·#774 수정 반영 확인), 나머지 30곳은 피연산자가 같은 dtype인지 정적으로 판정할 수 없다. 실제 보장은 torch bump 후 컴파일 테스트 통과로만 확인되므로 이 규칙은 코드 검사보다 절차(judgment) 성격이다.",
 "rule_refinement": "torch 메이저/마이너 bump PR에서는 전체 컴파일 테스트([pytest-full])를 돌려 혼합 dtype 행렬곱 실패를 찾고, 실패 지점은 attention 확률·마스크 등을 query/value의 dtype으로 명시적으로 .to(dtype) 캐스팅해 고친다.",
 "suggested_test": None
}
]
p = SD + "/out/batch_dependency.json"
json.dump(out, open(p, "w"), ensure_ascii=False, indent=1)
json.load(open(p))
for r in out: print(r["id"], r["verdict"], f'{r["compliant"]}/{r["violations_total"]}')
