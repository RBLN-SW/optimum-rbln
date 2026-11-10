import json, sys
SP = "<SCRATCH>/stage_d/scripts"
GRIFFE_PY = "python"
missing = []
for l in open(f"{SP}/test_doc_dev_test008_missing.txt"):
    p, ln, s = l.rstrip("\n").split(":", 2)
    missing.append({"path": p, "line": int(ln), "snippet": s[:120], "note": "exported user-facing class never set as RBLN_CLASS in tests/"})
t008_v = missing[:24] + [{"path": "tests/test_llm.py", "line": 896, "snippet": "class TestQwen2VLForConditionalGeneration(LLMTest.TestLLM):",
                          "note": "VLM test leaves IS_MULTIMODAL=False; overrides get_tokenizer with AutoProcessor instead"}]

out = [
{
 "id": "TEST-008",
 "check": {"kind": "script",
  "description": "dev는 _import_structure 대신 모듈별 __all__(define_import_structure)로 export하므로 src/optimum/rbln/{transformers,diffusers}/**/*.py의 __all__에서 RBLN* 모델·파이프라인 클래스(Config·Auto·기반 클래스 제외)를 모은다. 서브모듈·믹스인(Vision/Encoder/UNet 등)은 분모에서 뺀다. tests/test_*.py에 `RBLN_CLASS = X`로 등록되지 않은 사용자용 클래스를 위반으로 센다. 추가로 test_llm.py에서 이미지·오디오 입력 모델의 테스트 클래스가 상속을 포함해 IS_MULTIMODAL=True로 해석되지 않으면 위반으로 센다.",
  "command": f"python3 {SP}/test_doc_test008.py . && python3 {SP}/test_doc_test008_mm.py . | grep -v '^OK' && python3 {SP}/test_doc_test008_corpus.py <corpus>/prs.jsonl"},
 "compliant": 99,
 "violations": t008_v,
 "violations_total": 31,
 "verdict": "aspirational",
 "verdict_reason": "dev 팁에서 export된 사용자용 모델·파이프라인은 115개다. 이 중 85개가 등록되어 있고(직접 82, Kandinsky combined 경유 3) 30개(26%)가 미등록이다. main에서는 97개 중 67개 등록, 30개 미등록이었다. dev에서 늘어난 18개(Glm4·Granite·Olmo2·Phi3·SmolLM3·Starcoder2·ModernBert·Qwen3ASR·Cosmos2/2.5 등)는 모두 테스트에 등록되어 있어 신규 모델에서는 규칙이 지켜진다. 하지만 미등록 30개(Gemma/Gemma2/Gemma4·GptOss·Mixtral·Midm·Detr·DepthAnything·PaliGemma·Exaone4.5·*VLModel 5종, Cosmos 1세대 2종, inpaint·img2img 파이프라인 8종)는 그대로이고, test_diffusers.py:10의 `FIXME: add inpainting tests`도 남아 있다. 2025-01 이후 신규 모델 PR 43건 중 17건이 tests/를 건드리지 않았다. 멀티모달 플래그는 VLM 테스트 14개 중 13개와 오디오 모델 Qwen3ASR이 지키며, Qwen2VL만 get_tokenizer 오버라이드로 우회한다.",
 "rule_refinement": "export된 사용자용 모델·파이프라인 클래스(서브모듈 Vision/Encoder/UNet·믹스인 제외)는 각각 tests/test_{transformers,diffusers,llm}.py 중 한 곳에 `RBLN_CLASS = <클래스>`로 등록된 테스트 클래스를 가진다. combined 파이프라인이 내부에서 만드는 하위 파이프라인은 combined 테스트로 갈음한다. 이미지·오디오 입력을 받는 모델의 테스트 클래스는 IS_MULTIMODAL=True를 선언하며, get_tokenizer 오버라이드로 대신하지 않는다.",
 "suggested_test": "tests/test_conventions.py::test_every_exported_model_has_a_test — dev 레이아웃에 맞춰 src/optimum/rbln/{transformers,diffusers}/**/*.py의 모듈 수준 `__all__`을 ast로 모은다(main처럼 __init__.py에 `_import_structure`가 있으면 그것을 쓴다). RBLN*로 시작하고 Config/Auto/RBLNBaseModel/RBLNModel이 아닌 이름 중 COMPONENT 정규식(Vision|QFormer|TextModel|Encoder$|Decoder$|Backbone|TransformerPretrainedModel|^RBLNDecoderOnly|UNet|ControlNetModel$|VQModel|PriorTransformer|Transformer[23]DModel|SafetyChecker|Mixin|AutoencoderKL|AudioEncoder)에 걸리지 않는 클래스마다 tests/test_*.py에 `RBLN_CLASS\\s*=\\s*<name>`이 있는지 단언한다. 현재 미등록 30개는 KNOWN_UNTESTED 허용목록으로 두고, 허용목록의 클래스가 등록되면 목록에서 지우라는 실패를 내서 목록이 줄어들기만 하게 한다. 같은 파일의 test_multimodal_tests_declare_flag는 test_llm.py 클래스를 ast로 상속 해석해, RBLN_AUTO_CLASS가 RBLNAutoModelForImageTextToText 또는 RBLNAutoModelForMultimodalLM인 클래스의 IS_MULTIMODAL이 True인지 확인한다(허용목록 TestQwen2VLForConditionalGeneration)."
},
{
 "id": "TEST-019",
 "check": {"kind": "ast",
  "description": "tests/test_*.py의 테스트 클래스 메서드에서 cls./self.로 접근하는 대문자 클래스 속성 dict를 in-place로 쓰는 곳(.update/.pop/[k]= 등)을 찾는다. 그 dict를 같은 클래스 본문이 선언했고 쓰기가 setUpClass 안이면 준수, 상속받은 dict이거나 테스트 메서드 안에서 쓰면 위반이다. 또 HF_MODEL_ID로 컴파일하는 RBLN_CLASS.from_pretrained 호출에 **HF_CONFIG_KWARGS가 없으면 위반이다.",
  "command": f"python3 {SP}/test_doc_test019.py ."},
 "compliant": 23,
 "violations": [
  {"path": "tests/test_transformers.py", "line": 114, "snippet": "_ = self.RBLN_CLASS.from_pretrained(", "note": "compiles HF_MODEL_ID without HF_CONFIG_KWARGS (dict empty, harmless today)"},
  {"path": "tests/test_llm.py", "line": 742, "snippet": "_ = self.RBLN_CLASS.from_pretrained(model_id=self.HF_MODEL_ID, **rbln_class_kwargs)", "note": "LlavaNext revision pin dropped; call expects ValueError before compile"},
  {"path": "tests/test_diffusers.py", "line": 291, "snippet": "_ = self.RBLN_CLASS.from_pretrained(model_id=self.HF_MODEL_ID, **rbln_class_kwargs_copy)", "note": "compiles HF_MODEL_ID without HF_CONFIG_KWARGS (dict empty, harmless today)"}
 ],
 "violations_total": 3,
 "verdict": "verified",
 "verdict_reason": "dev 팁에서 in-place 쓰기 17곳은 모두 자기 클래스가 선언한 dict를 setUpClass에서 고친다. HF_MODEL_ID 컴파일 호출 9곳 중 6곳이 HF_CONFIG_KWARGS를 넘긴다(23/3, 이전 origin/dev 측정과 같다). main은 #741·#745 이전이라 30곳 중 13곳이 위반이었다. 남은 3곳은 빈 base dict를 쓰는 클래스(ResNet, KandinskyV22)이거나 컴파일 전에 ValueError를 기대하는 호출(LlavaNext, revision 핀 누락)이다. tests/unit/cpu에는 공유 dict나 from_pretrained가 없다.",
 "rule_refinement": "테스트 클래스는 공유 설정 dict(HF_CONFIG_KWARGS, RBLN_CLASS_KWARGS 등)를 in-place로 고치려면 그 dict를 자기 클래스 본문에서 선언하고 setUpClass 안에서만 고친다. 테스트 메서드에서는 복사본(`{**self.X[...]}`)을 만들어 쓴다. 상속받은 dict는 재바인딩(`cls.X = {**cls.X, ...}`)한다. HF_MODEL_ID에서 컴파일하는 from_pretrained 직접 호출에는 **self.HF_CONFIG_KWARGS를 넘긴다.",
 "suggested_test": f"tests/test_conventions.py::test_no_shared_class_dict_mutation — {SP}/test_doc_test019.py의 로직을 옮긴다. tests/test_*.py를 ast로 읽어 중첩 클래스(BaseTest.TestModel, LLMTest.TestLLM)까지 이름으로 인덱싱한다. 각 메서드에서 `cls.X`/`self.X`(X는 대문자) 루트에 대한 .update/.setdefault/.pop/.clear 호출과 Subscript 대입·삭제를 찾고, X가 그 클래스 본문의 Assign 대상이 아니거나 메서드명이 setUpClass가 아니면 실패한다. test_from_pretrained_passes_hf_config_kwargs는 receiver가 RBLN_CLASS로 끝나고 인자에 HF_MODEL_ID가 있는 from_pretrained 호출에 `**...HF_CONFIG_KWARGS` 키워드가 없으면 실패한다. 허용목록은 (TestResNetModel.test_failed_to_create_runtime, TestLlavaNextForConditionalGeneration의 batch_size 충돌 테스트, TestKandinskyV22Model 호출)이다."
},
{
 "id": "TEST-007",
 "check": {"kind": "grep",
  "description": "CI 정의(.buildkite/**, scripts/ci/*.sh, .github/workflows)의 모든 pytest 실행 줄에 각 케이스 실행 시간을 보고하는 `--durations 0`과 `-vv`가 붙어 있는지 확인한다. nightly.yml이 올리는 pytest·BC 파이프라인에 첫 실패에서 멈추는 설정(cancel_on_build_failing, pytest -x/--exitfirst/--maxfail)이 없는지 확인한다. 하나라도 어기면 위반이다.",
  "command": "grep -rnE '(^|[^-_a-z])pytest ' .buildkite scripts .github/workflows | grep -vE ':\\s*#'; grep -rnE -i 'cancel_on_build_failing|fail.fast|--maxfail|(^|\\s)-x(\\s|$)|--exitfirst' .buildkite scripts .github/workflows pyproject.toml tests/conftest.py"},
 "compliant": 7,
 "violations": [],
 "violations_total": 0,
 "verdict": "verified",
 "verdict_reason": "dev는 GHA pytest 워크플로를 지우고 Buildkite로 옮겼다. 모든 pytest 호출은 scripts/ci/run-suite.sh:29-37의 5줄(unit-cpu, config, transformers, diffusers, llm)에 모여 있고, 5줄 모두 `-vv --durations 0`을 붙인다. nightly.yml이 올리는 두 경로(.buildkite/pipelines/pytest/pipeline.yml, bc-steps.sh --all)에는 cancel_on_build_failing이 없다. pytest에도 -x/--maxfail이 없고, 보고 단계는 `wait: continue_on_failure: true` 뒤에 있어 첫 실패에서 멈추지 않는다. 준수 수가 main의 8(pytest 5줄 + 스케줄 매트릭스 3곳 fail_fast: false)에서 7(pytest 5줄 + nightly 업로드 2곳)로 바뀐 것은 측정 대상이 바뀌어서이고 위반은 여전히 0이다. 참고로 PR 경로도 이제 fail-fast가 없다.",
 "rule_refinement": "CI의 모든 pytest 호출(dev: scripts/ci/run-suite.sh)은 `-vv --durations 0`을 붙인다. 정기(nightly) 실행은 잡 취소 없이 끝까지 돈다: Buildkite 파이프라인에 cancel_on_build_failing을 두지 않고, pytest에 -x/--maxfail을 쓰지 않으며, 보고 단계는 continue_on_failure: true인 wait 뒤에 둔다.",
 "suggested_test": "tests/test_conventions.py::test_ci_pytest_reports_durations — scripts/ci/*.sh, .buildkite/**/*.yml, .github/workflows/*.y*ml에서 주석이 아닌 줄 중 `pytest `가 들어간 실행 줄을 모아, 각 줄에 `--durations`와 `-vv`가 있고 `-x`/`--exitfirst`/`--maxfail`이 없는지 단언한다. 최소 1줄은 찾아야 한다(빈 매칭으로 통과하지 않게). test_nightly_is_not_fail_fast는 .buildkite/nightly.yml과 .buildkite/pipelines/pytest/pipeline.yml 텍스트에 `cancel_on_build_failing: true`가 없는지, nightly.yml의 report 앞 wait 단계에 `continue_on_failure: true`가 있는지 yaml로 확인한다. 현재 허용목록은 필요 없다."
},
{
 "id": "DOC-011",
 "check": {"kind": "script",
  "description": "src/optimum/rbln의 공개 클래스·함수·메서드(__init__/__call__ 포함 1,817개, docstring 551개)를 griffe Google 파서(mkdocstrings와 동일)로 파싱한다. 위반은 세 가지다. ① 파싱된 파라미터 이름이 식별자가 아님(선언이 두 줄로 나뉘었거나 첫 `:` 앞에서 이름이 끝나지 않음) ② 반환 주석이 tuple이 아닌데 Returns가 여러 행으로 파싱됨 ③ 파라미터 기본값을 밝히면서 'Defaults to'가 아닌 표현(by default, default=, will default to)을 씀. ③에서 기본값이 아니라 동작을 설명하는 문장 2건(GroundingDino input_ids, Qwen3_5 vision batch_size)은 수작업으로 제외했다.",
  "command": f"{GRIFFE_PY} {SP}/test_doc_doc011.py .   # any python with griffe (mkdocstrings-python)"},
 "compliant": 1138,
 "violations": [
  {"path": "src/optimum/rbln/utils/runtime_utils.py", "line": 63, "snippet": "num_chiplets : int", "note": "numpy-style 'name : type' parsed as name 'num_chiplets '"},
  {"path": "src/optimum/rbln/utils/runtime_utils.py", "line": 65, "snippet": "npu : str | None, default=None", "note": "numpy-style declaration and 'default=' instead of 'Defaults to'"},
  {"path": "src/optimum/rbln/modeling.py", "line": 320, "snippet": "If `return_dict=True`, Returns a dictionary-like object (e.g., BaseModelOutput,", "note": "Returns continues on same-indent lines; parsed as 3 return rows"},
  {"path": "src/optimum/rbln/diffusers/pipelines/controlnet/pipeline_controlnet_sd_xl_img2img.py", "line": 461, "snippet": "[`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] if `return_dict` is True, otherwise a `tuple`", "note": "Returns description on same indent; parsed as 3 return rows"},
  {"path": "src/optimum/rbln/configuration_utils.py", "line": 740, "snippet": "timeout (int | None): The timeout for the runtime in seconds. If it isn't provided, it will be set to 60 by default.", "note": "default stated as 'by default' not 'Defaults to 60.'"},
  {"path": "src/optimum/rbln/diffusers/pipelines/controlnet/pipeline_controlnet_sd_xl.py", "line": 396, "snippet": "not specified it will default to `(height, width)`. Part of SDXL's micro-conditioning as explained in", "note": "'will default to' instead of 'Defaults to' (upstream diffusers text)"},
  {"path": "src/optimum/rbln/diffusers/pipelines/controlnet/pipeline_controlnet_sd_xl_img2img.py", "line": 421, "snippet": "not specified it will default to `(height, width)`. Part of SDXL's micro-conditioning as explained in", "note": "'will default to' instead of 'Defaults to' (upstream diffusers text)"}
 ],
 "violations_total": 7,
 "verdict": "verified",
 "verdict_reason": "dev 팁에서 공개 docstring 파라미터 944개 중 942개가 한 줄 선언으로 식별자 이름까지 파싱된다. Returns 섹션 79개 중 77개가 한 행으로 파싱되고, 기본값을 밝힌 설명 123개 중 119개가 'Defaults to'를 쓴다. 위반은 main의 13건에서 7건(0.6%)으로 줄었다. ControlNet 분할 선언 6건은 #783에서 고쳐졌고, dev에서 새로 추가된 코드에는 위반이 없다. 검증기는 dev에도 남아 있다(scripts/validate_docstrings.py + scripts/check-docstrings.sh, .buildkite/pr.yml은 변경 파일만, nightly.yml은 DOCSTRINGS_ALL_FILES=1로 전체 파일 검사). 하지만 infer_module_name이 파일명에 configuration/modeling이 없는 파일(pipeline_*.py, utils/runtime_utils.py)을 건너뛴다. 또 여러 행 Returns는 griffe 경고를 내지 않아 --strict 빌드를 통과한다. 그래서 남은 7건이 게이트에 걸리지 않는다.",
 "rule_refinement": "공개 API docstring은 Google 스타일만 쓴다(numpy 스타일 `name : type` 금지). 파라미터 선언 `name (type): 설명`의 이름과 타입은 한 줄에 쓰고, 이어지는 설명은 더 깊게 들여쓴다. Returns는 `type: 설명` 한 행으로 쓰고, 이어지는 줄은 더 들여쓴다(같은 들여쓰기는 별도 반환값으로 파싱됨). 기본값은 'Defaults to X.'로 쓴다. 검사 범위는 렌더되는 모든 파일(pipeline_*.py, utils/* 포함)과 __call__/__init__이다.",
 "suggested_test": "tests/test_conventions.py::test_google_docstring_shape — griffe(mkdocstrings-python 의존성이므로 quality 그룹에 추가하거나 importorskip)로 `griffe.load('optimum.rbln', search_paths=['src'], allow_inspection=False)`를 하고 모든 공개 객체(이름이 _로 시작하지 않거나 dunder인 class/function, 공개 모듈 경로)의 docstring을 `griffe.parse_google`로 파싱한다. (a) parameters 섹션 항목 이름이 `^\\*{0,2}[A-Za-z_]\\w*$`가 아니면 실패한다. (b) returns 섹션이 2행 이상인데 객체의 returns 주석이 tuple로 시작하지 않으면 실패한다. (c) 설명에 `default\\s*[=:]|will default to|(?:set to|is) \\S+ by default`가 있고 `defaults to`가 없으면 실패한다(동작을 설명하는 'ignored by default'류는 걸리지 않게 좁힌다). 허용목록은 dev 기준 runtime_utils.get_available_dram_per_chiplet(numpy 스타일), RBLNModel.forward Returns, RBLNStableDiffusionXLControlNetImg2ImgPipeline.__call__ Returns, RBLNModelConfig.__init__.timeout, ControlNet SDXL 두 파이프라인의 target_size다. 아울러 scripts/validate_docstrings.py의 infer_module_name이 pipeline_*·utils 파일을 건너뛰는 점을 고치도록 제안한다."
},
{
 "id": "DOC-012",
 "check": {"kind": "ast",
  "description": "src/optimum/rbln의 모든 모듈·클래스·함수 docstring에서 마크다운 링크 `[text](target)`를 추출한다. target이 http(s)/#/mailto로 시작하지 않는 상대 경로이거나, docs.rbln.ai URL인데 rbln_sdk_doc의 docs/ 트리에 대응하는 .md가 없으면 위반이다. 참고용으로 docstring이 아닌 런타임 메시지 문자열의 docs.rbln.ai URL도 같은 방식으로 확인한다.",
  "command": f"python3 {SP}/test_doc_doc012.py . <WORKSPACE>"},
 "compliant": 42,
 "violations": [
  {"path": "src/optimum/rbln/transformers/models/t5/modeling_t5.py", "line": 54, "snippet": "based on the model size. Please refer to the [Optimum RBLN Overview](../../../index.md) for guidance on choosing the", "note": "relative docs link; breaks validate_docstrings, file excluded from gate"},
  {"path": "src/optimum/rbln/transformers/models/t5/modeling_t5.py", "line": 99, "snippet": "based on the model size. Please refer to the [Optimum RBLN Overview](../../../index.md) for guidance on choosing the", "note": "relative docs link; breaks validate_docstrings, file excluded from gate"}
 ],
 "violations_total": 2,
 "verdict": "verified",
 "verdict_reason": "dev 팁에서 docstring 링크 44개 중 42개가 절대 URL이고, docs.rbln.ai 대상은 모두 SDK 문서 트리(2026-10-13 체크아웃)에 존재한다. 결과는 main과 같다. 위반 2건은 T5의 `../../../index.md`뿐이다. scripts/check-docstrings.sh가 modeling_t5.py를 하드코딩으로 제외하며 'drop this once the link is absolute'라고 적어 두었으므로, 의도가 확인된 미해결 부채다. 참고로 docstring 밖 런타임 메시지의 docs.rbln.ai URL 중 4개는 SDK 문서 트리에 대응 페이지가 없다. 3개는 about_atom/release_note.html(import_utils.py:180·188, dev에서 새로 생긴 modeling_attention_utils.py:484)이고 실제 페이지는 supports/release_note다. 나머지 1개는 vllm_support/vllm-rbln.html(decoderonly_runtime_utils.py:34)이다.",
 "rule_refinement": "docstring(그리고 사용자에게 보이는 에러·경고 메시지)의 SDK 문서 링크는 https://docs.rbln.ai/... 절대 URL로 쓴다. 대상 경로는 rbln_sdk_doc 빌드에 존재하는 페이지여야 하며, .html 경로는 docs/<path>.{en.,}md로 대응된다. 상대 경로는 throwaway mkdocs 검사(scripts/validate_docstrings.py)에서 깨지므로 금지한다.",
 "suggested_test": f"tests/test_conventions.py::test_docstring_links_are_absolute — src/optimum/rbln/**/*.py를 ast로 읽어 Module/ClassDef/FunctionDef의 첫 문자열 상수(docstring)에서 `\\[[^\\]]*\\]\\(([^)\\s]+)\\)`를 찾는다. target이 `https?://|#|mailto:`로 시작하지 않으면 실패한다. 허용목록은 transformers/models/t5/modeling_t5.py의 2건이고, 고쳐지면 scripts/check-docstrings.sh의 exclude 목록과 함께 지운다. docs.rbln.ai 경로의 존재 확인은 SDK 문서 체크아웃이 필요하므로, RBLN_SDK_DOC_ROOT 환경변수가 있을 때만 실행하는 선택 테스트로 둔다({SP}/test_doc_doc012.py의 exists() 로직). 이 선택 테스트는 런타임 메시지 문자열의 URL도 함께 검사한다."
}
]
path = sys.argv[1]
json.dump(out, open(path, "w"), ensure_ascii=False, indent=1)
d = json.load(open(path))
for x in d: print(x["id"], x["verdict"], f'{x["compliant"]}/{x["violations_total"]}', len(x["violations"]))
