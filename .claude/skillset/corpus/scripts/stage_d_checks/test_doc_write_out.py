import json, sys
SP = "<SCRATCH>/stage_d/scripts"
missing = [
 ("src/optimum/rbln/diffusers/pipelines/cosmos/pipeline_cosmos_text2world.py",33,"class RBLNCosmosTextToWorldPipeline(RBLNDiffusionMixin, CosmosTextToWorldPipeline):"),
 ("src/optimum/rbln/diffusers/pipelines/cosmos/pipeline_cosmos_video2world.py",33,"class RBLNCosmosVideoToWorldPipeline(RBLNDiffusionMixin, CosmosVideoToWorldPipeline):"),
 ("src/optimum/rbln/transformers/models/depth_anything/modeling_depth_anything.py",22,"class RBLNDepthAnythingForDepthEstimation(RBLNModelForDepthEstimation):"),
 ("src/optimum/rbln/transformers/models/detr/modeling_detr.py",31,"class RBLNDetrForObjectDetection(RBLNModel):"),
 ("src/optimum/rbln/transformers/models/exaone4_5/modeling_exaone4_5.py",461,"class RBLNExaone4_5_ForConditionalGeneration(RBLNExaone4_5_Model, RBLNDecoderOnlyModelForCausalLM):"),
 ("src/optimum/rbln/transformers/models/exaone4_5/modeling_exaone4_5.py",308,"class RBLNExaone4_5_Model(RBLNDecoderOnlyModel):"),
 ("src/optimum/rbln/transformers/models/gemma2/modeling_gemma2.py",27,"class RBLNGemma2ForCausalLM(RBLNDecoderOnlyModelForCausalLM):"),
 ("src/optimum/rbln/transformers/models/gemma2/modeling_gemma2.py",83,"class RBLNGemma2Model(RBLNDecoderOnlyModel):"),
 ("src/optimum/rbln/transformers/models/gemma4/modeling_gemma4.py",233,"class RBLNGemma4ForCausalLM(RBLNDecoderOnlyModelForCausalLM):"),
 ("src/optimum/rbln/transformers/models/gemma4/modeling_gemma4.py",592,"class RBLNGemma4ForConditionalGeneration(RBLNModel, RBLNDecoderOnlyGenerationMixin):"),
 ("src/optimum/rbln/transformers/models/gemma/modeling_gemma.py",23,"class RBLNGemmaForCausalLM(RBLNDecoderOnlyModelForCausalLM):"),
 ("src/optimum/rbln/transformers/models/gemma/modeling_gemma.py",86,"class RBLNGemmaModel(RBLNDecoderOnlyModel):"),
 ("src/optimum/rbln/transformers/models/gpt_oss/modeling_gpt_oss.py",40,"class RBLNGptOssForCausalLM(RBLNDecoderOnlyModelForCausalLM):"),
 ("src/optimum/rbln/diffusers/pipelines/kandinsky2_2/pipeline_kandinsky2_2_combined.py",152,"class RBLNKandinskyV22InpaintCombinedPipeline(RBLNDiffusionMixin, KandinskyV22InpaintCombinedPipeline):"),
 ("src/optimum/rbln/diffusers/pipelines/kandinsky2_2/pipeline_kandinsky2_2_inpaint.py",21,"class RBLNKandinskyV22InpaintPipeline(RBLNDiffusionMixin, KandinskyV22InpaintPipeline):"),
 ("src/optimum/rbln/transformers/models/midm/modeling_midm.py",32,"class RBLNMidmLMHeadModel(RBLNDecoderOnlyModelForCausalLM):"),
 ("src/optimum/rbln/transformers/models/mixtral/modeling_mixtral.py",19,"class RBLNMixtralForCausalLM(RBLNDecoderOnlyModelForCausalLM):"),
 ("src/optimum/rbln/transformers/models/paligemma/modeling_paligemma.py",65,"class RBLNPaliGemmaForConditionalGeneration(RBLNModel, RBLNDecoderOnlyGenerationMixin):"),
 ("src/optimum/rbln/transformers/models/paligemma/modeling_paligemma.py",367,"class RBLNPaliGemmaModel(RBLNModel):"),
 ("src/optimum/rbln/diffusers/pipelines/stable_diffusion_3/pipeline_stable_diffusion_3_inpaint.py",21,"class RBLNStableDiffusion3InpaintPipeline(RBLNDiffusionMixin, StableDiffusion3InpaintPipeline):"),
 ("src/optimum/rbln/diffusers/pipelines/controlnet/pipeline_controlnet_img2img.py",50,"class RBLNStableDiffusionControlNetImg2ImgPipeline(RBLNDiffusionMixin, StableDiffusionControlNetImg2ImgPipeline):"),
 ("src/optimum/rbln/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion_inpaint.py",21,"class RBLNStableDiffusionInpaintPipeline(RBLNDiffusionMixin, StableDiffusionInpaintPipeline):"),
 ("src/optimum/rbln/diffusers/pipelines/stable_diffusion_xl/pipeline_stable_diffusion_xl_img2img.py",21,"class RBLNStableDiffusionXLImg2ImgPipeline(RBLNDiffusionMixin, StableDiffusionXLImg2ImgPipeline):"),
 ("src/optimum/rbln/diffusers/pipelines/stable_diffusion_xl/pipeline_stable_diffusion_xl_inpaint.py",21,"class RBLNStableDiffusionXLInpaintPipeline(RBLNDiffusionMixin, StableDiffusionXLInpaintPipeline):"),
]
t008_v = [{"path":p,"line":l,"snippet":s[:120],"note":"exported user-facing class never set as RBLN_CLASS in tests/"} for p,l,s in missing]
t008_v.append({"path":"tests/test_llm.py","line":705,"snippet":"class TestQwen2VLForConditionalGeneration(LLMTest.TestLLM):","note":"VLM test class leaves IS_MULTIMODAL=False; overrides get_tokenizer instead"})

out = [
{
 "id":"TEST-008",
 "check":{"kind":"script",
  "description":"src/optimum/rbln/__init__.py의 _import_structure(dev는 모듈별 __all__)에서 RBLN* 모델/파이프라인 클래스(Config·Auto·기반 클래스 제외)를 모으고, 서브모듈·믹스인(Vision/Encoder/UNet 등)을 분모에서 뺀 뒤 tests/test_*.py에 `RBLN_CLASS = X`로 등록되지 않은 사용자용 클래스를 위반으로 센다. 추가로 test_llm.py의 VLM 테스트 클래스가 상속을 포함해 IS_MULTIMODAL=True로 해석되지 않으면 위반으로 센다.",
  "command":f"python3 {SP}/test_doc_test008.py . && python3 {SP}/test_doc_test008_mm.py . | grep MISMATCH && python3 {SP}/test_doc_test008_corpus.py <corpus>/prs.jsonl"},
 "compliant":80,
 "violations":t008_v[:25],
 "violations_total":31,
 "verdict":"aspirational",
 "verdict_reason":"export된 사용자용 모델·파이프라인 97개 중 67개만 테스트에 등록되어(직접 64, Kandinsky combined 경유 3) 30개(31%)가 미등록이다: Gemma/Gemma2/Gemma4·GptOss·Mixtral·Midm·Detr·DepthAnything·PaliGemma·Exaone4.5·*VLModel 계열, Cosmos 2종, SD/SDXL/SD3/Kandinsky의 inpaint·img2img 파이프라인 8종. origin/dev에서도 30개로 같다. 코퍼스로 보면 2025-01 이후 신규 모델 PR 43건 중 17건이 tests/를 건드리지 않았다(#563 gemma4, #593 exaone4.5, #441 detr, #249 depth anything 등). PR #15에서 요청한 Kandinsky inpaint 테스트는 아직 `# FIXME: add inpainting tests`로 남아 있다. 멀티모달 플래그는 VLM 테스트 14개 중 13개가 지키며, Qwen2VL만 get_tokenizer를 오버라이드하는 방식으로 우회한다.",
 "rule_refinement":"export된 사용자용 모델·파이프라인 클래스(서브모듈 Vision/Encoder/UNet·믹스인 제외)는 각각 tests/test_{transformers,diffusers,llm}.py 중 한 곳에 `RBLN_CLASS = <클래스>`로 등록된 테스트 클래스를 가진다. combined 파이프라인이 내부에서 만드는 하위 파이프라인은 combined 테스트로 갈음한다. 이미지·오디오 입력을 받는 모델의 테스트 클래스는 IS_MULTIMODAL=True를 선언하며, get_tokenizer 오버라이드로 대신하지 않는다.",
 "suggested_test":"tests/test_conventions.py::test_every_exported_model_has_a_test — src/optimum/rbln/__init__.py의 `_import_structure`를 ast로 읽는다(dev처럼 __all__ 기반이면 src/optimum/rbln/{transformers,diffusers}/**/*.py의 __all__을 모은다). RBLN*로 시작하고 Config/Auto/RBLNBaseModel/RBLNModel이 아닌 이름 중 COMPONENT 정규식(Vision|QFormer|TextModel|Encoder$|Decoder$|Backbone|TransformerPretrainedModel|^RBLNDecoderOnly|UNet|ControlNetModel$|VQModel|PriorTransformer|Transformer[23]DModel|SafetyChecker|Mixin)에 걸리지 않는 클래스마다, tests/test_*.py에 `RBLN_CLASS\\s*=\\s*<name>`이 있는지 단언한다. 현재 미등록 30개(위 목록 + Qwen2VLModel/Qwen2_5_VLModel/Qwen3VLModel/Qwen3VLMoeModel/Qwen3_5Model/SDXLControlNetImg2Img)는 KNOWN_UNTESTED 허용목록으로 두고, 허용목록에 있는 클래스가 등록되면 목록에서 지우라는 실패를 내서 목록이 줄어들기만 하게 한다. 같은 파일의 test_vlm_tests_declare_multimodal은 test_llm.py의 클래스를 ast로 상속 해석해, RBLN_AUTO_CLASS가 RBLNAutoModelForImageTextToText인 클래스의 IS_MULTIMODAL이 True인지 확인한다(허용목록 TestQwen2VLForConditionalGeneration)."
},
{
 "id":"TEST-019",
 "check":{"kind":"ast",
  "description":"tests/test_*.py의 테스트 클래스 메서드에서 cls./self.로 접근하는 대문자 클래스 속성 dict를 in-place로 쓰는 곳(.update/.pop/[k]= 등)을 찾는다. 그 dict를 같은 클래스 본문이 선언했고 쓰기가 setUpClass 안이면 준수, 상속받은 dict이거나 테스트 메서드 안에서 쓰면 위반이다. 또 HF_MODEL_ID로 컴파일하는 RBLN_CLASS.from_pretrained 호출에 **HF_CONFIG_KWARGS가 없으면 위반이다.",
  "command":f"python3 {SP}/test_doc_test019.py . origin/dev   # worktree HEAD: python3 {SP}/test_doc_test019.py ."},
 "compliant":23,
 "violations":[
  {"path":"origin/dev:tests/test_transformers.py","line":115,"snippet":"_ = self.RBLN_CLASS.from_pretrained(","note":"compiles HF_MODEL_ID without HF_CONFIG_KWARGS (dict empty, harmless today)"},
  {"path":"origin/dev:tests/test_llm.py","line":638,"snippet":"_ = self.RBLN_CLASS.from_pretrained(model_id=self.HF_MODEL_ID, **rbln_class_kwargs)","note":"LlavaNext revision pin dropped; call expects ValueError before compile"},
  {"path":"origin/dev:tests/test_diffusers.py","line":291,"snippet":"_ = self.RBLN_CLASS.from_pretrained(model_id=self.HF_MODEL_ID, **rbln_class_kwargs_copy)","note":"compiles HF_MODEL_ID without HF_CONFIG_KWARGS (dict empty, harmless today)"}
 ],
 "violations_total":3,
 "verdict":"verified",
 "verdict_reason":"규칙을 만든 #741·#745는 dev에만 머지되어 있다. origin/dev 기준으로 in-place 쓰기 17곳은 모두 자기 클래스가 선언한 dict를 setUpClass에서 고치고, HF_MODEL_ID 컴파일 호출 9곳 중 6곳이 HF_CONFIG_KWARGS를 넘긴다. 남은 3곳은 빈 base dict를 쓰는 클래스이거나 컴파일 전에 예외를 기대하는 호출이다. worktree HEAD(main, 2026-07-31)는 두 PR 이전이라 30곳 중 13곳이 위반이다: 상속 dict 변경 3곳(GroundingDino 2, Qwen3Moe 1), test_propagate_config의 RBLN_CLASS_KWARGS 변경 3곳, HF_CONFIG_KWARGS 누락 7곳.",
 "rule_refinement":"테스트 클래스는 공유 설정 dict(HF_CONFIG_KWARGS, RBLN_CLASS_KWARGS 등)를 in-place로 고치려면 그 dict를 자기 클래스 본문에서 선언하고 setUpClass 안에서만 고친다. 테스트 메서드에서는 복사본(`{**self.X[...]}`)을 만들어 쓴다. 상속받은 dict는 재바인딩(`cls.X = {**cls.X, ...}`)한다. HF_MODEL_ID에서 컴파일하는 from_pretrained 직접 호출에는 **self.HF_CONFIG_KWARGS를 넘긴다.",
 "suggested_test":f"tests/test_conventions.py::test_no_shared_class_dict_mutation — {SP}/test_doc_test019.py의 로직을 옮긴다. tests/test_*.py를 ast로 읽어 중첩 클래스(BaseTest.TestModel, LLMTest.TestLLM)까지 이름으로 인덱싱한다. 각 메서드에서 `cls.X`/`self.X`(X는 대문자) 루트에 대한 .update/.setdefault/.pop/.clear 호출과 Subscript 대입·삭제를 찾고, X가 그 클래스 본문의 Assign 대상이 아니거나 메서드명이 setUpClass가 아니면 실패한다. test_from_pretrained_passes_hf_config_kwargs는 receiver가 RBLN_CLASS로 끝나고 인자에 HF_MODEL_ID가 있는 from_pretrained 호출에 `**...HF_CONFIG_KWARGS` 키워드가 없으면 실패한다. 허용목록은 (TestResNetModel.test_failed_to_create_runtime, TestLlavaNextForConditionalGeneration.test_complicate_config, TestKandinskyV22Model 호출)이다."
},
{
 "id":"TEST-007",
 "check":{"kind":"grep",
  "description":"CI 정의의 모든 pytest 실행 줄에 각 케이스 실행 시간을 보고하는 `--durations 0`과 `-vv`가 붙어 있는지 확인한다. 스케줄(nightly) 경로가 호출하는 pytest 매트릭스가 fail-fast를 끄는지(fail_fast: false / fail-fast: false, Buildkite는 cancel 없음) 확인한다. 하나라도 빠지면 위반이다.",
  "command":"grep -rn 'pytest ' .github/workflows | grep -v description; grep -rn 'fail_fast:\\|fail-fast:' .github/workflows; git grep -n 'pytest ' origin/dev -- .buildkite/scripts/run-suite.sh; git grep -n -i 'cancel_on\\|fail.fast' origin/dev -- .buildkite"},
 "compliant":8,
 "violations":[],
 "violations_total":0,
 "verdict":"verified",
 "verdict_reason":"HEAD의 pytest 실행 5줄(rbln_optimum_pytest.yaml:173-187)이 모두 `-vv --durations 0`을 붙인다. 스케줄 경로(rbln_scheduled_test_impl → rbln_optimum_full_test, bc_full_inference → bc_inference)의 매트릭스 3곳은 모두 fail_fast/fail-fast: false다. PR 경로의 기본값 true는 규칙 범위 밖이다. dev는 CI를 Buildkite로 옮겼는데 run-suite.sh의 pytest 5줄이 같은 플래그를 유지하고, nightly.yml에는 cancel_on_build_failing이 없어 첫 실패에서 멈추지 않는다.",
 "rule_refinement":"CI의 모든 pytest 호출은 `-vv --durations 0`을 붙인다. 정기(nightly/scheduled) 실행 경로는 매트릭스·잡 취소 없이 끝까지 돈다(GHA fail_fast: false, Buildkite cancel_on_build_failing 미설정). PR 경로의 fail-fast는 허용한다.",
 "suggested_test":"tests/test_conventions.py::test_ci_pytest_reports_durations — .github/workflows/*.y*ml과 .buildkite/**/*.{yml,sh}의 텍스트에서 `pytest `가 들어간 실행 줄(주석·description 제외)을 모아, 각 줄에 `--durations`와 `-vv`가 있는지 단언한다. test_scheduled_ci_is_not_fail_fast는 yaml을 읽어 rbln_optimum_full_test.yaml·bc_inference.yaml·bc_compile.yaml의 `fail_fast` 입력이 false인지, bc_full_inference.yaml strategy.fail-fast가 false인지 확인한다. Buildkite가 있으면 nightly.yml에 `cancel_on_build_failing: true`가 없는지 확인한다. 현재 허용목록은 필요 없다."
},
{
 "id":"DOC-011",
 "check":{"kind":"script",
  "description":"src/optimum/rbln의 공개 클래스·함수·메서드(__init__/__call__ 포함 1,605개, docstring 474개)를 griffe Google 파서(mkdocstrings와 동일)로 파싱한다. 위반은 세 가지다. ① 파싱된 파라미터 이름이 식별자가 아님(선언이 두 줄로 나뉘었거나 첫 `:` 앞에서 이름이 끝나지 않음) ② 반환 주석이 tuple이 아닌데 Returns가 여러 행으로 파싱됨 ③ 기본값을 밝히면서 'Defaults to'가 아닌 표현(by default, default=, will default to)을 씀.",
  "command":f"python {SP}/test_doc_doc011.py .   # any python with griffe (mkdocstrings-python)"},
 "compliant":1054,
 "violations":[
  {"path":"src/optimum/rbln/diffusers/pipelines/controlnet/pipeline_controlnet.py","line":271,"snippet":"image (`torch.FloatTensor`, `PIL.Image.Image`, `np.ndarray`, `list[torch.FloatTensor]`, `list[PIL.Image.Image]`, `l","note":"type split over two lines with stray ':'; name parsed wrongly (fixed on dev #783)"},
  {"path":"src/optimum/rbln/diffusers/pipelines/controlnet/pipeline_controlnet_img2img.py","line":264,"snippet":"image (`torch.FloatTensor`, `PIL.Image.Image`, `np.ndarray`, `list[torch.FloatTensor]`, `list[PIL.Image.Image]`, `l","note":"split declaration; parsed name not an identifier (fixed on dev #783)"},
  {"path":"src/optimum/rbln/diffusers/pipelines/controlnet/pipeline_controlnet_img2img.py","line":268,"snippet":"control_image (`torch.FloatTensor`, `PIL.Image.Image`, `np.ndarray`, `list[torch.FloatTensor]`, `list[PIL.Image.Imag","note":"split declaration; parsed name not an identifier (fixed on dev #783)"},
  {"path":"src/optimum/rbln/diffusers/pipelines/controlnet/pipeline_controlnet_sd_xl.py","line":308,"snippet":"image (`torch.FloatTensor`, `PIL.Image.Image`, `np.ndarray`, `list[torch.FloatTensor]`, `list[PIL.Image.Image]`, `l","note":"split declaration; parsed name not an identifier (fixed on dev #783)"},
  {"path":"src/optimum/rbln/diffusers/pipelines/controlnet/pipeline_controlnet_sd_xl_img2img.py","line":324,"snippet":"image (`torch.FloatTensor`, `PIL.Image.Image`, `np.ndarray`, `list[torch.FloatTensor]`, `list[PIL.Image.Image]`, `l","note":"split declaration; parsed name not an identifier (fixed on dev #783)"},
  {"path":"src/optimum/rbln/diffusers/pipelines/controlnet/pipeline_controlnet_sd_xl_img2img.py","line":328,"snippet":"control_image (`torch.FloatTensor`, `PIL.Image.Image`, `np.ndarray`, `list[torch.FloatTensor]`, `list[PIL.Image.Imag","note":"split declaration; parsed name not an identifier (fixed on dev #783)"},
  {"path":"src/optimum/rbln/utils/runtime_utils.py","line":64,"snippet":"num_chiplets : int","note":"numpy-style 'name : type' parsed as name 'num_chiplets '"},
  {"path":"src/optimum/rbln/utils/runtime_utils.py","line":66,"snippet":"npu : str | None, default=None","note":"numpy-style declaration and 'default=' instead of 'Defaults to'"},
  {"path":"src/optimum/rbln/modeling.py","line":302,"snippet":"If `return_dict=True`, Returns a dictionary-like object (e.g., BaseModelOutput,","note":"Returns continues on same-indent lines; parsed as 3 return rows"},
  {"path":"src/optimum/rbln/diffusers/pipelines/controlnet/pipeline_controlnet_sd_xl_img2img.py","line":471,"snippet":"[`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] if `return_dict` is True, otherwise a `tuple`","note":"Returns description on same indent; parsed as 3 return rows"},
  {"path":"src/optimum/rbln/configuration_utils.py","line":694,"snippet":"timeout (int | None): The timeout for the runtime in seconds. If it isn't provided, it will be set to 60 by default.","note":"default stated as 'by default' not 'Defaults to 60.'"},
  {"path":"src/optimum/rbln/diffusers/pipelines/controlnet/pipeline_controlnet_sd_xl.py","line":405,"snippet":"not specified it will default to `(height, width)`. Part of SDXL's micro-conditioning as explained in","note":"'will default to' instead of 'Defaults to' (upstream diffusers text)"},
  {"path":"src/optimum/rbln/diffusers/pipelines/controlnet/pipeline_controlnet_sd_xl_img2img.py","line":431,"snippet":"not specified it will default to `(height, width)`. Part of SDXL's micro-conditioning as explained in","note":"'will default to' instead of 'Defaults to' (upstream diffusers text)"}
 ],
 "violations_total":13,
 "verdict":"verified",
 "verdict_reason":"공개 docstring 파라미터 884개 중 876개가 한 줄 선언으로 식별자 이름까지 파싱되고, Returns 섹션 75개 중 73개가 한 행으로 파싱된다. 기본값을 밝힌 설명 108개 중 105개가 'Defaults to'를 쓴다. 위반은 13건(1.2%)이다. ControlNet 분할 선언 6건은 dev의 #783에서 고쳐져 dev 기준 위반은 7건이다. Returns 검사는 #783 직전 커밋(029069ea^)으로 canary해 get_load_overrides의 두 행 파싱을 실제로 잡는 것을 확인했다. validate_docstrings.py는 파일명에 configuration/modeling이 들어간 파일만 `filters: [\"!^_\"]`로 렌더하므로 pipeline_*.py와 __call__ docstring은 검사하지 못한다. ControlNet 결함이 CI를 통과한 원인이 이것이다.",
 "rule_refinement":"공개 API docstring은 Google 스타일만 쓴다(numpy 스타일 `name : type` 금지). 파라미터 선언 `name (type): 설명`의 이름과 타입은 한 줄에 쓰고, 이어지는 설명은 더 깊게 들여쓴다. Returns는 `type: 설명` 한 행으로 쓰고, 이어지는 줄은 더 들여쓴다(같은 들여쓰기는 별도 반환값으로 파싱됨). 기본값은 'Defaults to X.'로 쓴다. 검사 범위는 렌더되는 모든 파일(pipeline_*.py 포함)과 __call__/__init__이다.",
 "suggested_test":"tests/test_conventions.py::test_google_docstring_shape — griffe(mkdocstrings-python 의존성이므로 quality/tests 그룹에 추가하거나 importorskip)로 `griffe.load('optimum.rbln', search_paths=['src'], allow_inspection=False)`를 하고 모든 공개 객체(이름이 _로 시작하지 않거나 dunder인 class/function, 공개 모듈 경로)의 docstring을 `griffe.parse_google`로 파싱한다. (a) parameters 섹션 항목 이름이 `^\\*{0,2}[A-Za-z_]\\w*$`가 아니면 실패한다. (b) returns 섹션이 2행 이상인데 객체의 returns 주석이 tuple로 시작하지 않으면 실패한다. (c) 설명에 `by default|default\\s*[=:]|will default to`가 있고 `defaults to`가 없으면 실패한다. 허용목록은 dev 기준 runtime_utils.get_available_dram_per_chiplet(numpy 스타일), RBLNModel.forward Returns, RBLNStableDiffusionXLControlNetImg2ImgPipeline.__call__ Returns, RBLNModelConfig.__init__.timeout, ControlNet SDXL 두 파이프라인의 original_size/target_size다. 아울러 test-docstrings 게이트의 infer_module_name이 pipeline_* 파일을 건너뛰는 점을 고치도록 제안한다."
},
{
 "id":"DOC-012",
 "check":{"kind":"ast",
  "description":"src/optimum/rbln의 모든 모듈·클래스·함수 docstring에서 마크다운 링크 `[text](target)`를 추출한다. target이 http(s)/#/mailto로 시작하지 않는 상대 경로이거나, docs.rbln.ai URL인데 rbln_sdk_doc의 docs/ 트리에 대응하는 .md가 없으면 위반이다. 참고용으로 docstring이 아닌 런타임 메시지 문자열의 docs.rbln.ai URL도 같은 방식으로 확인한다.",
  "command":f"python3 {SP}/test_doc_doc012.py . <WORKSPACE>"},
 "compliant":42,
 "violations":[
  {"path":"src/optimum/rbln/transformers/models/t5/modeling_t5.py","line":59,"snippet":"based on the model size. Please refer to the [Optimum RBLN Overview](../../../index.md) for guidance on choosing the","note":"relative docs link; fails validate_docstrings, file excluded from gate"},
  {"path":"src/optimum/rbln/transformers/models/t5/modeling_t5.py","line":104,"snippet":"based on the model size. Please refer to the [Optimum RBLN Overview](../../../index.md) for guidance on choosing the","note":"relative docs link; fails validate_docstrings, file excluded from gate"}
 ],
 "violations_total":2,
 "verdict":"verified",
 "verdict_reason":"docstring 링크 44개 중 42개가 절대 URL이다. 위반 2건은 T5의 `../../../index.md`뿐이다. 이 상대 경로는 SDK 문서 빌드(docs/software/optimum/index.*.md)에서는 실제로 해석되지만 validate_docstrings의 throwaway mkdocs에서는 실패한다. 그래서 test-docstrings.yml(HEAD)과 dev의 .buildkite/scripts/check-docstrings.sh가 modeling_t5.py를 하드코딩으로 제외하고, dev 스크립트에는 'drop this once the link is absolute'라는 주석이 있다. 절대 URL화 PR #732는 머지되지 않고 닫혔다(2026-09-09). #550은 존재하는 경로로 고쳤지만 상대 형식을 유지했다. 따라서 이 2건은 의도가 확인된 미해결 기술 부채다. 참고로 docstring 밖 런타임 메시지의 docs.rbln.ai URL 3개(utils/import_utils.py:178·187 about_atom/release_note.html, decoderonly_runtime_utils.py:34 vllm_support/vllm-rbln.html)는 현재 SDK 문서 트리에 대응 페이지가 없다.",
 "rule_refinement":"docstring(그리고 사용자에게 보이는 에러·경고 메시지)의 SDK 문서 링크는 https://docs.rbln.ai/... 절대 URL로 쓴다. 대상 경로는 rbln_sdk_doc 빌드에 존재하는 페이지여야 하며 .html 경로는 docs/<path>.{en.,}md로 대응된다. 상대 경로는 throwaway mkdocs 검사(validate_docstrings.py)에서 깨지므로 금지한다.",
 "suggested_test":f"tests/test_conventions.py::test_docstring_links_are_absolute — src/optimum/rbln/**/*.py를 ast로 읽어 Module/ClassDef/FunctionDef의 첫 문자열 상수(docstring)에서 `\\[[^\\]]*\\]\\(([^)\\s]+)\\)`를 찾는다. target이 `https?://|#|mailto:`로 시작하지 않으면 실패한다. 허용목록은 transformers/models/t5/modeling_t5.py의 2건이고, 고쳐지면 check-docstrings.sh의 제외 목록과 함께 지운다. docs.rbln.ai 경로의 존재 확인은 SDK 문서 체크아웃이 필요하므로 rbln_sdk_doc 쪽 linkcheck에 맡기거나, RBLN_SDK_DOC_ROOT 환경변수가 있을 때만 실행하는 선택 테스트로 둔다({SP}/test_doc_doc012.py의 exists() 로직)."
}
]
path = sys.argv[1]
json.dump(out, open(path, "w"), ensure_ascii=False, indent=1)
d = json.load(open(path)); print(len(d), [ (x["id"], x["verdict"], x["compliant"], x["violations_total"], len(x["violations"])) for x in d])
