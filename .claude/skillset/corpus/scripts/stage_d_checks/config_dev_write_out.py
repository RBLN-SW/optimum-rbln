import json, os
S = "<SCRATCH>/stage_d/scripts"
SRC = f"PYTHONPATH={S}/config_dev_src"  # copy of devtree src/optimum/rbln (74e97fa4) + generated __version__.py
out = []

out.append({
 "id": "CFG-010",
 "check": {"kind": "script",
  "description": "폐지·개명된 설정 키/인자/장치·아티팩트 지원을 모두 찾아, (a) deprecate 헬퍼(deprecate_kwarg·deprecate_method·warn_deprecated_npu)로 경고하는지 (b) 생성자·setattr·rbln_ 접두사 경로 모두에서 옛 키가 새 키로 재매핑되는지 프로브로 확인한다. 수동 logger.warning이거나 setattr 경로에서 재매핑이 안 되면 위반.",
  "command": "grep -rnE 'deprecate_kwarg\\(|deprecate_method\\(|warn_deprecated_npu\\(|is deprecated|deprecated!' src/optimum/rbln --include=*.py | grep -v 'utils/deprecation.py' | grep -v 'diffusers/pipelines/.*/pipeline_' ; " + SRC + " python -W ignore " + S + "/config_dev_cfg010_probe.py"},
 "compliant": 10,
 "violations": [
  {"path": "src/optimum/rbln/configuration_utils.py", "line": 665, "snippet": "def __setattr__(self, key, value):", "note": "cfg.tensor_parallel_size=2 / cfg.kvcache_metas 대입 시 재매핑 없이 미지 키로 직렬화"},
  {"path": "src/optimum/rbln/configuration_utils.py", "line": 767, "snippet": "logger.warning(\"`optimize_host_memory` is deprecated and will be removed in future versions.\")", "note": "데코레이터 대신 수동 경고, 제거 버전 미명시"},
  {"path": "src/optimum/rbln/transformers/utils/rbln_quantization.py", "line": 85, "snippet": "logger.warning(\"The `precision` argument is deprecated. Use `weights` and `activations` instead.\")", "note": "재매핑은 하나 deprecate_kwarg 미사용 수동 경고"},
  {"path": "src/optimum/rbln/transformers/utils/rbln_quantization.py", "line": 182, "snippet": "logger.warning_once(\"torch_dtype is deprecated. Use dtype instead.\")", "note": "수동 kwarg 재매핑·경고, 데코레이터 미사용"},
  {"path": "src/optimum/rbln/transformers/models/gpt_oss/modeling_gpt_oss.py", "line": 105, "snippet": "logger.warning_once(\"`torch_dtype` is deprecated! Use `dtype` instead!\")", "note": "수동 kwarg 재매핑·경고, 데코레이터 미사용"},
  {"path": "src/optimum/rbln/transformers/models/gemma4/modeling_gemma4.py", "line": 710, "snippet": "logger.warning_once(", "note": "구 아티팩트 지원 폐지를 수동 경고로; qwen3_5는 같은 경우 deprecate_method 사용"}
 ],
 "violations_total": 6,
 "verdict": "aspirational",
 "verdict_reason": "dev에서 헬퍼 사용 사이트가 6→10으로 늘었다(#666 kvcache_metas→cache_metas deprecate_kwarg, Cosmos 설정 2곳의 fps deprecate_kwarg, qwen3_5 _mask_padded_keys_and_queries deprecate_method; 기존 tensor_parallel_size·_torch_dtype·model_input_shapes·CA02·torch_dtype setter·서브모듈 tensor_parallel_size 경로 유지, 버전은 0.12.0→0.13.0으로 연장). 그러나 16개 사이트 중 6개(37.5%)가 여전히 위반이다: 수동 logger.warning 4곳(optimize_host_memory, precision, torch_dtype×2)과 gemma4 아티팩트 dtype 경고, 그리고 프로브 결과 `c.tensor_parallel_size=2`→num_devices None·직렬화됨, `C.initialize_from_kwargs(C(), rbln_tensor_parallel_size=2)`→num_devices None, `c.kvcache_metas=[]`도 직렬화됨으로 setattr 경로 재매핑이 main(6/5)과 동일하게 빠져 있다. #760(setattr 경로 지적)은 아직 OPEN.",
 "rule_refinement": "설정 키를 폐지·개명할 때는 즉시 제거하지 않고 경고 후 새 키로 재매핑해 기존 설정 파일과 사용자 코드를 유지한다. RBLNModelConfig 생성자 인자는 deprecate_kwarg(version=...)로, 속성 대입(setattr)·인스턴스+rbln_ 접두사 경로는 property setter 등으로 함께 덮어야 하며, 구 아티팩트·장치 지원 종료는 deprecate_method / warn_deprecated_npu 같은 헬퍼로 경고한다. 구현 제거는 해당 version 이후 별도 breaking change로 한다.",
 "suggested_test": "tests/test_conventions.py::test_deprecations_use_helpers — src/optimum/rbln 전체(.py; diffusers/pipelines의 upstream 복사 pipeline_*.py와 utils/deprecation.py, HF 메시지를 그대로 옮긴 blip_2 제외)에서 'deprecated' 문자열을 담은 logger.warning/warning_once 호출을 AST로 찾아 실패시키고, 현재 위반(configuration_utils.py optimize_host_memory·torch_dtype property, rbln_quantization.py precision/torch_dtype, gpt_oss _get_dtype, gemma4 image_features dtype)을 allowlist로 둔다. 추가로 RBLNModelConfig 하위 클래스의 모든 deprecate_kwarg(new_name=...) 대상(tensor_parallel_size→num_devices, kvcache_metas→cache_metas)에 대해 `cfg=C(); setattr(cfg, old, v)` 후 `getattr(cfg, new)==v and old not in cfg._prepare_for_serialization()`을 검사하는 파라미터화 테스트를 두고 두 항목 모두 xfail로 시작한다."
})

out.append({
 "id": "CFG-018",
 "check": {"kind": "script",
  "description": "모든 RBLNModelConfig 하위 클래스를 import해 subclass_non_save_attributes를 재정의한 클래스가 직계 부모의 유효 목록을 모두 포함하는지, Callable 타입 __init__ 인자/비공개 런타임 속성(self._x)·추정 입력이 비저장 목록에 등록돼 있는지 검사한다. 부모 항목 누락이나 미등록 런타임 전용 속성은 위반.",
  "command": SRC + " python -W ignore " + S + "/config_cfg018_nonsave.py ; grep -rnE 'self\\._[a-z][a-z_0-9]* *=' src/optimum/rbln --include=configuration_*.py | grep -vE 'self\\._(attributes_map|frozen|runtime_options|compile_cfgs|dtype) *='"},
 "compliant": 10,
 "violations": [],
 "violations_total": 0,
 "verdict": "verified",
 "verdict_reason": "dev에서 설정 클래스가 153→171개로 늘었지만 subclass_non_save_attributes를 재정의한 10개 모두 부모 항목(memory_budget 등)을 포함하고, 비저장 대상 비공개 속성(_load_visual_runtime, _batch_size_is_specified)과 추정 입력(memory_budget)은 할당하는 모든 파일에서 등록돼 있다(main과 동일 판정). 새로 생긴 _requires_batch_sort(#559)는 등록되지 않았지만 modeling_decoderonly.py 주석대로 'serialized so a loaded model knows to sort' — 컴파일 시점에 결정되는 값을 의도적으로 저장하는 것이라 위반이 아니다. Callable 타입 인자는 여전히 없다(#623·#750은 CLOSED 미병합).",
 "rule_refinement": None,
 "suggested_test": "tests/test_conventions.py::test_subclass_non_save_attributes_not_shadowed — optimum.rbln의 모든 *Config를 import해 RBLNModelConfig 하위 클래스를 재귀 수집하고, `'subclass_non_save_attributes' in cls.__dict__`인 클래스마다 `set(cls.__mro__[1].subclass_non_save_attributes) <= set(cls.subclass_non_save_attributes)`를 단언한다. 함께 inspect.signature(cls.__init__)에서 Callable 주석 인자를 찾아 non_save_attributes ∪ subclass_non_save_attributes에 있는지 확인한다. allowlist 불필요."
})

out.append({
 "id": "CFG-019",
 "check": {"kind": "ast",
  "description": "configuration_*.py에서 submodules를 선언한 클래스마다 submodule_config_classes 매핑 존재 여부, 서브모듈 __init__ 인자의 타입 주석(영역별), __init__에서 initialize_submodule_config를 직접 호출하는지를 AST로 센다. 매핑 없이 클래스별 __init__에서 dict→설정 변환을 하면 위반.",
  "command": "python3 " + S + "/config_cfg019_submodules.py . | python3 " + S + "/config_dev_cfg019_summary.py ; grep -rn 'submodule_config_classes\\|model_validator' src tests | wc -l"},
 "compliant": 0,
 "violations": [
  {"path": "src/optimum/rbln/configuration_utils.py", "line": 574, "snippet": "def initialize_submodule_config(", "note": "공통 검증 훅이 아닌 각 __init__이 호출하는 헬퍼; cls_name 없으면 dict 반환"},
  {"path": "src/optimum/rbln/configuration_utils.py", "line": 680, "snippet": "if key in self.submodules and isinstance(value, dict) and (cls_name := value.get(\"cls_name\")):", "note": "자동 변환은 cls_name 있는 dict에만, 매핑 없음"},
  {"path": "src/optimum/rbln/transformers/models/gemma4/configuration_gemma4.py", "line": 231, "snippet": "if isinstance(vt_cfg, dict):", "note": "서브모듈이 아직 dict일 수 있어 클래스별로 수동 분기"},
  {"path": "src/optimum/rbln/transformers/models/llava/configuration_llava.py", "line": 24, "snippet": "class RBLNLlavaForConditionalGenerationConfig", "note": "submodule_config_classes 없음, __init__에서 수동 변환"},
  {"path": "src/optimum/rbln/transformers/models/exaone4_5/configuration_exaone4_5.py", "line": 53, "snippet": "self.visual = self.initialize_submodule_config(submodule_config=visual, batch_size=1, force_kwargs=True)", "note": "매핑 없음, 클래스별 수동 변환(main의 직접 대입은 헬퍼 호출로 바뀜)"},
  {"path": "src/optimum/rbln/diffusers/configurations/pipelines/configuration_stable_diffusion.py", "line": 22, "snippet": "class RBLNStableDiffusionPipelineBaseConfig", "note": "매핑 없음, 구체 클래스 타입 주석, 수동 변환"},
  {"path": "src/optimum/rbln/transformers/models/grounding_dino/configuration_grounding_dino.py", "line": None, "snippet": "encoder: Optional['RBLNGroundingDinoEncoderConfig']", "note": "transformers 쪽 유일한 구체 클래스 주석"}
 ],
 "violations_total": 36,
 "verdict": "rejected",
 "verdict_reason": "dev(74e97fa4)에서도 `submodule_config_classes`·`model_validator`는 src/tests 어디에도 없고(0건), 서브모듈을 가진 36개 클래스(main 32) 모두 매핑이 없으며 36개 전부가 __init__에서 `self.initialize_submodule_config(...)`를 직접 호출한다(main 31/32 → EXAONE-4.5도 헬퍼로 전환). 공통 경로의 실제 계약은 docstring \"Returns the submodule config instance when its class is known (an instance was passed, or `cls_name` was given), otherwise the merged kwargs dict for the model class to resolve later.\"와 `__setattr__`의 `if key in self.submodules and isinstance(value, dict) and (cls_name := value.get(\"cls_name\")):`로, dict는 cls_name이 있을 때만 변환되고 나머지는 모델 클래스가 나중에 해석한다(그래서 gemma4 설정은 `if isinstance(vt_cfg, dict):`로 수동 분기). 근거 PR #495는 2026-07-02 CLOSED(미병합). 타입 주석만 보면 transformers 32/34가 `RBLNModelConfig | None`, diffusers는 45/48이 구체 클래스라 영역별로 갈린다(main 31/33, 36/39).",
 "rule_refinement": "서브모듈을 가진 설정 클래스는 `submodules` 클래스 속성을 선언하고, __init__에서 각 서브모듈을 공통 헬퍼 `self.initialize_submodule_config(submodule_config, ...)`로 처리해 부모의 런타임 옵션(npu, num_devices, create_runtimes 등)을 전파한다. 클래스를 모르는 dict(cls_name 없음)는 dict로 남겨 모델 클래스가 나중에 해석하므로, 설정 클래스 안에서 서브모듈 값을 읽는 코드는 dict와 설정 객체 둘 다 처리해야 한다. transformers 쪽 서브모듈 인자 주석은 `RBLNModelConfig | None`을, diffusers 파이프라인은 구체 설정 클래스를 쓴다.",
 "suggested_test": "tests/test_conventions.py::test_submodules_initialized_via_helper — configuration_*.py를 AST로 순회해 `submodules = [...]`가 비어있지 않은 클래스 중 __init__을 정의한 클래스는 initialize_submodule_config 호출이 최소 1회 있어야 한다고 단언한다(dev 기준 36/36 통과, allowlist 불필요). 보조로 transformers/ 아래 클래스의 서브모듈 인자 주석이 `RBLNModelConfig | None`인지 검사하고 grounding_dino의 encoder/decoder 2개를 allowlist로 둔다."
})

out.append({
 "id": "CFG-022",
 "check": {"kind": "script",
  "description": "rbln_config(dict 또는 인스턴스)와 rbln_ 접두사 인자를 동시에 넘겼을 때 ValueError가 나는지 컴파일 경로(initialize_from_kwargs)와 로드 경로(RBLNModelConfig.from_pretrained, RBLNBaseModel.from_pretrained의 dict→kwargs 변환 에뮬레이션)에서 실제 호출로 확인한다. 오류 없이 병합되면 위반.",
  "command": SRC + " python -W ignore " + S + "/config_dev_cfg022_probe.py ; grep -nE 'rbln_config.update\\(rbln_kwargs\\)|setattr\\(rbln_config, key, value\\)|if f\"rbln_\\{key\\}\" not in kwargs|take precedence' src/optimum/rbln/configuration_utils.py ; grep -n 'kwargs.update(rbln_config_as_kwargs)' src/optimum/rbln/modeling_base.py ; grep -rn 'convert_rbln_config_dict' src tests"},
 "compliant": 0,
 "violations": [
  {"path": "src/optimum/rbln/configuration_utils.py", "line": 1109, "snippet": "rbln_config.update(rbln_kwargs)", "note": "dict+접두사 동시 지정 시 오류 없이 병합, 접두사 값이 우선"},
  {"path": "src/optimum/rbln/configuration_utils.py", "line": 1115, "snippet": "setattr(rbln_config, key, value)", "note": "인스턴스+접두사 동시 지정 시 setattr로 덮어씀"},
  {"path": "src/optimum/rbln/configuration_utils.py", "line": 1044, "snippet": "if f\"rbln_{key}\" not in kwargs:", "note": "로드 시 두 경로 병합; #702 이후 접두사 우선(main은 dict 우선)"},
  {"path": "src/optimum/rbln/modeling_base.py", "line": 247, "snippet": "kwargs.update(rbln_config_as_kwargs)", "note": "모델 로드 경로는 dict가 접두사를 덮어씀, 오류 없음"},
  {"path": "src/optimum/rbln/configuration_utils.py", "line": 1207, "snippet": "rbln_config.update(rbln_kwargs)", "note": "convert_rbln_config_dict도 병합(dev에선 호출처 없음)"},
  {"path": "tests/test_config.py", "line": 144, "snippet": "def test_config_priority(model_id):", "note": "접두사 인자가 rbln_config를 덮어쓰는 우선순위를 테스트로 고정"}
 ],
 "violations_total": 6,
 "verdict": "rejected",
 "verdict_reason": "dev에서도 동시 지정 시 ValueError를 내는 경로는 없다(#445는 2026-01-23 CLOSED 미병합). 대신 공식 계약은 우선순위다: 클래스 docstring \"Method 4: Combining a config object with override parameters  # (rbln_ prefixed parameters take precedence over rbln_config values)\"(configuration_utils.py:407-408)와 test_config_priority, 그리고 #702가 RBLNModelConfig.from_pretrained의 `if key not in kwargs:`를 `if f\"rbln_{key}\" not in kwargs:`로 고쳐 로드 경로도 접두사 우선으로 맞췄다(프로브 `from_pretrained(d, rbln_config={'device':0}, rbln_device=1).device`: main 0 → dev 1). 단 RBLNBaseModel.from_pretrained는 여전히 `kwargs.update(rbln_config_as_kwargs)`로 dict가 접두사를 덮어써 에뮬레이션 결과 device=0이며, 컴파일 시점 키를 저장값과 다르게 주면 `ValueError(\"Cannot set the following arguments ...\")`가 나는 것이 유일한 오류 경로다.",
 "rule_refinement": "설정 키는 rbln_config(dict 또는 RBLN*Config 인스턴스)와 from_pretrained의 rbln_ 접두사 인자 두 경로로 지정할 수 있으며, 둘을 함께 주면 오류 없이 병합되고 rbln_ 접두사 인자가 우선한다(dict 병합은 nested_update처럼 키 단위로 덮어씀). 컴파일된 모델을 로드할 때 저장된 컴파일 시점 키를 다른 값으로 바꾸려 하면 ValueError(\"Cannot set the following arguments\")가 난다. 현재 RBLNBaseModel.from_pretrained 로드 경로(modeling_base.py:246-247)는 dict가 이기므로 이 계약과 어긋난다.",
 "suggested_test": "tests/test_conventions.py::test_rbln_prefix_overrides_rbln_config — 컴파일 없이 RBLNResNetForImageClassificationConfig.initialize_from_kwargs({'batch_size': 4}, rbln_batch_size=16)과 인스턴스 경로에서 batch_size==16을, 저장된 설정 디렉터리에 대해 RBLNResNetForImageClassificationConfig.from_pretrained(d, rbln_config={'device': 0}, rbln_device=1).device==1을 단언한다. RBLNBaseModel.from_pretrained 로드 경로의 dict→kwargs 변환(modeling_base.py:246-247)이 접두사를 덮어쓰지 않는지 검사하는 케이스는 xfail로 시작한다."
})

out.append({
 "id": "SUB-008",
 "check": {"kind": "grep",
  "description": "비전 rotary를 가진 계열마다 (a) 회전 적용 함수가 HF apply_rotary_pos_emb_vision처럼 fp32 업캐스트 후 한 번 반올림하는지 (b) 호스트가 graph 입력 cos/sin을 HF와 같은 dtype(Qwen 계열 fp32, Gemma4는 x.dtype)으로 선언·캐스트하는지 확인한다. HF가 fp32로 두는 표를 rbln_config.dtype으로 반올림하면 위반.",
  "command": "grep -rnE 'def apply_rotary_pos_emb_vision' -A8 src/optimum/rbln/transformers/models ; grep -rnE '\\(\"(cos|sin)\"|^\\s+\"(cos|sin)\",' -A3 src/optimum/rbln/transformers/models/{qwen2_vl,qwen2_5_vl,qwen3_vl,qwen3_5,exaone4_5,gemma4}/modeling_*.py ; grep -rnE 'torch\\.cat\\(\\((cos, cos|sin, sin)\\), dim=-1\\)\\.to\\(self\\.rbln_config\\.dtype\\)|rotary_(cos|sin)_table\\[.*\\.to\\(self\\.rbln_config\\.dtype\\)' src/optimum/rbln"},
 "compliant": 7,
 "violations": [
  {"path": "src/optimum/rbln/transformers/models/qwen2_vl/modeling_qwen2_vl.py", "line": 192, "snippet": "torch.cat((cos, cos), dim=-1).to(self.rbln_config.dtype),", "note": "HF는 fp32 유지, 활성 dtype으로 반올림(입력도 rbln_config.dtype 선언)"},
  {"path": "src/optimum/rbln/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py", "line": 287, "snippet": "torch.cat((cos, cos), dim=-1).to(self.rbln_config.dtype),", "note": "HF는 fp32 유지, 활성 dtype으로 반올림"},
  {"path": "src/optimum/rbln/transformers/models/qwen3_vl/modeling_qwen3_vl.py", "line": 195, "snippet": "torch.cat((cos, cos), dim=-1).to(self.rbln_config.dtype),", "note": "HF는 fp32 유지, 활성 dtype으로 반올림(qwen3_vl_moe도 상속)"},
  {"path": "src/optimum/rbln/transformers/models/exaone4_5/modeling_exaone4_5.py", "line": 269, "snippet": "torch.cat((cos, cos), dim=-1).to(self.rbln_config.dtype),", "note": "HF는 fp32 유지, 활성 dtype으로 반올림"}
 ],
 "violations_total": 4,
 "verdict": "aspirational",
 "verdict_reason": "main(1/5)에서 크게 개선됐다: #711(2026-10-13 병합)로 5개 계열(qwen2_vl, qwen2_5_vl, qwen3_vl, qwen3_5, exaone4_5) 모두 아키텍처 파일에 \"Mirrors HF `apply_rotary_pos_emb_vision`: rotate in fp32 and round once.\" 함수를 갖게 됐고, Qwen3.5는 cos/sin 입력을 torch.float32로 선언하며, Gemma4 비전은 HF Gemma4VisionRotaryEmbedding이 `.to(dtype=x.dtype)`하므로 rbln_config.dtype 캐스트가 HF와 일치한다. 그러나 표를 fp32로 넘기는 부분은 #764가 OPEN이라 Qwen2-VL·Qwen2.5-VL·Qwen3-VL(·MoE 상속)·EXAONE-4.5 4곳이 여전히 `torch.cat((cos, cos), dim=-1).to(self.rbln_config.dtype)`로 활성 dtype 반올림을 한다(11개 사이트 중 4개, 36%).",
 "rule_refinement": "HF가 명시적으로 두는 dtype 경계를 그대로 따른다: 비전 rotary 적용은 계열별 apply_rotary_pos_emb_vision에서 q/k/cos/sin을 fp32로 올려 계산 후 한 번 반올림하고, 호스트가 graph에 넘기는 cos/sin 표는 HF가 fp32로 두는 계열(Qwen-VL 계열·EXAONE-4.5)에서는 fp32로 선언·전달하며, HF가 x.dtype으로 캐스트하는 계열(Gemma4)만 rbln_config.dtype을 쓴다.",
 "suggested_test": "tests/test_conventions.py::test_vision_rotary_matches_hf_dtype — (1) transformers/models/*/*_architecture.py 중 비전 어텐션이 있는 파일(qwen2_vl, qwen2_5_vl, qwen3_vl, qwen3_5, exaone4_5)에 apply_rotary_pos_emb_vision 정의가 있고 본문에 `.float()`가 있는지 AST로 단언(현재 전부 통과). (2) 같은 계열의 modeling_*.py에서 _update_rbln_config input_info의 'cos'/'sin' dtype 표현식이 torch.float32인지, `torch.cat((cos, cos), dim=-1).to(self.rbln_config.dtype)` 패턴이 없는지 단언하고 qwen2_vl·qwen2_5_vl·qwen3_vl·exaone4_5를 allowlist로 둔다(#764 병합 시 제거). gemma4는 HF가 x.dtype으로 캐스트하므로 검사 대상에서 제외."
})

out.append({
 "id": "ATT-006",
 "check": {"kind": "grep",
  "description": "정적 캐시(key_value_states / cross-KV / conv_state·recurrent_state) 입력의 마지막 축과 인코더·디코더 시퀀스 길이 기본값이 64의 배수로 설계되거나 올림되는지 확인한다. 64 배수가 아닌 기본값/마지막 축은 위반(또는 반례).",
  "command": "grep -rnE '% *64|multiple of 64|multiples of 64' src/optimum/rbln --include=*.py ; grep -rnE 'enc_max_seq_len = |cross_key_value_states' -A6 src/optimum/rbln/transformers/models/{seq2seq,whisper,time_series_transformer}/modeling_*.py"},
 "compliant": 4,
 "violations": [
  {"path": "src/optimum/rbln/transformers/models/whisper/modeling_whisper.py", "line": 284, "snippet": "rbln_config.enc_max_seq_len = model_config.max_source_positions", "note": "인코더 길이 1500(64 배수 아님)을 올림 없이 사용"},
  {"path": "src/optimum/rbln/transformers/models/seq2seq/modeling_seq2seq.py", "line": 228, "snippet": "enc_max_seq_len = enc_max_seq_len or tokenizer.model_max_length", "note": "인코더 길이를 64로 올림하지 않음"},
  {"path": "src/optimum/rbln/transformers/models/time_series_transformer/modeling_time_series_transformer.py", "line": 283, "snippet": "model_config.d_model // model_config.decoder_attention_heads,  # 13", "note": "정적 cross-KV 캐시 마지막 축이 head_dim(13), 64 배수 아님"}
 ],
 "violations_total": 3,
 "verdict": "interview",
 "verdict_reason": "dev에서도 main과 동일하다: 64 단위 정렬은 prefill_chunk_size 검증(modeling_attention_utils.py:95), TST dec_max_seq_len 올림(:237), Qwen3.5 conv_state(innermost=conv_dim, 'a multiple of 64 as RBLN requires')·recurrent_state 레이아웃 4곳에서 지켜지지만, Whisper 인코더 1500·seq2seq 인코더 길이는 올림하지 않고 TST cross-KV 캐시 마지막 축은 head_dim 13이다. 근거 PR #804·#816은 여전히 OPEN이다. 팀 질문: 64 배수 요구가 모든 정적 캐시에 적용되는가, 아니면 레이아웃을 직접 설계하는 커스텀 상태 캐시(conv/recurrent처럼 head_dim이 모델로 고정되지 않는 것)에만 해당하는가? 인코더 시퀀스 길이 올림은 특정 파이프라인 규칙인가 일반 규칙인가?",
 "rule_refinement": "레이아웃을 직접 설계하는 디바이스 상주 정적 상태 캐시(conv cache, recurrent state 등)는 64의 배수인 축을 가장 안쪽(마지막) 축에 둔다. 사용자 값이 아닌 기본값으로 정하는 시퀀스 길이·청크 크기는 64의 배수로 올림하거나 검증한다(prefill_chunk_size, TST dec_max_seq_len 선례).",
 "suggested_test": None
})

path = "<SCRATCH>/stage_d/out_dev/batch_config.json"
os.makedirs(os.path.dirname(path), exist_ok=True)
json.dump(out, open(path, "w"), ensure_ascii=False, indent=1)
d = json.load(open(path))
for r in d:
    assert all(len(v["snippet"]) <= 120 for v in r["violations"]), r["id"]
    assert len(r["violations"]) <= 25
    print(r["id"], r["verdict"], f'{r["compliant"]}/{r["violations_total"]}')
