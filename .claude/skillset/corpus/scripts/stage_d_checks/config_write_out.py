import json
S = "<SCRATCH>/stage_d/scripts"
SRC = f"PYTHONPATH={S}/config_src"  # copy of src/optimum/rbln + generated __version__.py (repo has no __version__.py)
out = []

out.append({
 "id": "CFG-010",
 "check": {"kind": "script",
  "description": "폐지·개명된 설정 키/인자/장치를 모두 찾아, (a) deprecate 헬퍼(deprecate_kwarg·deprecate_method·warn_deprecated_npu)로 경고하는지 (b) 생성자·setattr 경로 모두에서 옛 키가 새 키로 재매핑되는지 확인한다. 수동 logger.warning 이거나 setattr 경로에서 재매핑이 안 되면 위반.",
  "command": "grep -rnE 'deprecate_kwarg|deprecate_method|warn_deprecated_npu|is deprecated' src/optimum/rbln --include=*.py | grep -v 'utils/deprecation.py' | grep -v diffusers/pipelines ; " + SRC + " python -W ignore -c \"from optimum.rbln import RBLNResNetForImageClassificationConfig as C; import torch; c=C(tensor_parallel_size=2); print(c.num_devices); c=C(); c.tensor_parallel_size=2; print(c.num_devices, 'tensor_parallel_size' in c._prepare_for_serialization()); c=C(); c.torch_dtype=torch.bfloat16; print(c.dtype); c,_=C.initialize_from_kwargs(C(), rbln_tensor_parallel_size=2); print(c.num_devices)\""},
 "compliant": 6,
 "violations": [
  {"path": "src/optimum/rbln/configuration_utils.py", "line": 618, "snippet": "def __setattr__(self, key, value):", "note": "setattr로 tensor_parallel_size 지정 시 num_devices 재매핑 없이 미지 키로 저장됨"},
  {"path": "src/optimum/rbln/configuration_utils.py", "line": 720, "snippet": "logger.warning(\"`optimize_host_memory` is deprecated and will be removed in future versions.\")", "note": "데코레이터 대신 수동 경고, 제거 버전 미명시"},
  {"path": "src/optimum/rbln/transformers/utils/rbln_quantization.py", "line": 85, "snippet": "logger.warning(\"The `precision` argument is deprecated. Use `weights` and `activations` instead.\")", "note": "재매핑은 하나 deprecate_kwarg 미사용 수동 경고"},
  {"path": "src/optimum/rbln/transformers/utils/rbln_quantization.py", "line": 182, "snippet": "logger.warning_once(\"torch_dtype is deprecated. Use dtype instead.\")", "note": "수동 kwarg 재매핑·경고, 데코레이터 미사용"},
  {"path": "src/optimum/rbln/transformers/models/gpt_oss/modeling_gpt_oss.py", "line": 106, "snippet": "logger.warning_once(\"`torch_dtype` is deprecated! Use `dtype` instead!\")", "note": "수동 kwarg 재매핑·경고, 데코레이터 미사용"}
 ],
 "violations_total": 5,
 "verdict": "aspirational",
 "verdict_reason": "폐지 키를 즉시 제거하지 않고 경고+재매핑으로 동작을 유지하는 원칙은 모든 사이트에서 지켜진다(tensor_parallel_size·_torch_dtype·model_input_shapes는 deprecate_kwarg, torch_dtype는 property setter, CA02는 warn_deprecated_npu, 서브모듈 경로도 처리). 그러나 11개 사이트 중 5개가 데코레이터 대신 수동 logger.warning을 쓰고, 프로브 결과 setattr 경로(`cfg.tensor_parallel_size=2`, 인스턴스+rbln_ 접두사)는 재매핑되지 않는다. 근거 PR #666(kvcache_metas→cache_metas)·#760은 이 스냅샷(2026-07-31) 이후/미병합이다.",
 "rule_refinement": "설정 키를 폐지·개명할 때는 즉시 제거하지 않고 경고 후 새 키로 재매핑해 기존 설정 파일과 사용자 코드를 유지한다. RBLNModelConfig 생성자 인자는 deprecate_kwarg(version=...)로, 속성 접근/setattr 경로는 property setter(또는 deprecate_method)로 함께 덮어야 하며, 장치 지원 종료는 warn_deprecated_npu 같은 전용 헬퍼로 경고한다. 구현 제거는 해당 version 이후 별도 breaking change로 한다.",
 "suggested_test": "tests/test_conventions.py::test_deprecations_use_helpers — src/optimum/rbln 전체(.py, diffusers/pipelines 중 upstream 복사 코드와 utils/deprecation.py 제외)에서 문자열 'is deprecated'를 담은 logger.warning/warning_once 호출을 AST로 찾아 실패시키고, 현재 위반(configuration_utils.py optimize_host_memory, rbln_quantization.py precision/torch_dtype, gpt_oss torch_dtype, configuration_utils.py torch_dtype property)을 allowlist로 둔다. 추가로 RBLNModelConfig의 모든 deprecate_kwarg(new_name=...) 대상에 대해 `cfg=C(); setattr(cfg, old, v)` 후 `getattr(cfg, new)==v`를 검사하는 파라미터화 테스트를 두고 tensor_parallel_size는 xfail로 시작한다."
})

out.append({
 "id": "CFG-018",
 "check": {"kind": "script",
  "description": "모든 RBLNModelConfig 하위 클래스를 import해 subclass_non_save_attributes를 재정의한 클래스가 직계 부모의 유효 목록을 모두 포함하는지, 그리고 Callable 타입 __init__ 인자/비공개 런타임 속성(self._x)이 비저장 목록에 등록돼 있는지 검사한다. 부모 항목 누락이나 미등록 런타임 속성은 위반.",
  "command": SRC + " python -W ignore " + S + "/config_cfg018_nonsave.py ; grep -rnE 'self\\._[a-z][a-z_0-9]* *=' src/optimum/rbln --include=configuration_*.py | grep -vE 'self\\._(attributes_map|frozen|runtime_options|compile_cfgs|dtype) *='"},
 "compliant": 10,
 "violations": [],
 "violations_total": 0,
 "verdict": "verified",
 "verdict_reason": "153개 설정 클래스 중 subclass_non_save_attributes를 재정의한 10개 모두 부모 항목(memory_budget 등)을 포함하며, 비저장 대상 비공개 속성(_load_visual_runtime, _batch_size_is_specified)과 추정 입력(memory_budget)은 할당하는 7개 파일 모두에서 등록돼 있다. Callable 타입 설정 인자는 현재 없다(#623의 adjuster는 미병합).",
 "rule_refinement": None,
 "suggested_test": "tests/test_conventions.py::test_subclass_non_save_attributes_not_shadowed — optimum.rbln의 모든 *Config를 import해 RBLNModelConfig의 하위 클래스를 재귀 수집하고, `'subclass_non_save_attributes' in cls.__dict__`인 클래스마다 `set(cls.__mro__[1].subclass_non_save_attributes) <= set(cls.subclass_non_save_attributes)`를 단언한다. 함께 inspect.signature(cls.__init__)에서 Callable 주석 인자를 찾아 non_save_attributes ∪ subclass_non_save_attributes에 있는지 확인한다. allowlist 불필요."
})

out.append({
 "id": "CFG-019",
 "check": {"kind": "ast",
  "description": "configuration_*.py에서 submodules를 선언한 클래스마다 submodule_config_classes 매핑 존재 여부, 서브모듈 __init__ 인자의 타입 주석, __init__에서 initialize_submodule_config를 수동 호출하는지를 AST로 센다. 매핑 없이 클래스별로 dict→설정 변환을 하면 위반.",
  "command": "python3 " + S + "/config_cfg019_submodules.py . | python3 -c \"import json,sys; r=[x for x in json.load(sys.stdin) if x['submodules']]; print(len(r), sum(x['has_map'] for x in r), sum(x['n_init_calls']>0 for x in r))\""},
 "compliant": 0,
 "violations": [
  {"path": "src/optimum/rbln/transformers/models/llava/configuration_llava.py", "line": 24, "snippet": "class RBLNLlavaForConditionalGenerationConfig", "note": "submodule_config_classes 없음, __init__에서 initialize_submodule_config 수동 호출"},
  {"path": "src/optimum/rbln/transformers/models/qwen2_5_vl/configuration_qwen2_5_vl.py", "line": 21, "snippet": "class RBLNQwen2_5_VLForConditionalGenerationConfig", "note": "매핑 없음, 클래스별 수동 변환"},
  {"path": "src/optimum/rbln/transformers/models/blip_2/configuration_blip_2.py", "line": 78, "snippet": "class RBLNBlip2ForConditionalGenerationConfig", "note": "매핑 없음, 클래스별 수동 변환"},
  {"path": "src/optimum/rbln/diffusers/configurations/pipelines/configuration_stable_diffusion.py", "line": 22, "snippet": "class RBLNStableDiffusionPipelineBaseConfig", "note": "매핑 없음, 구체 클래스 타입 주석, 수동 변환"},
  {"path": "src/optimum/rbln/transformers/models/exaone4_5/configuration_exaone4_5.py", "line": 21, "snippet": "class RBLNExaone4_5_ForConditionalGenerationConfig", "note": "매핑 없음, self.visual = visual 직접 대입"}
 ],
 "violations_total": 32,
 "verdict": "rejected",
 "verdict_reason": "규칙의 핵심(submodule_config_classes 매핑 + Pydantic model_validator 공통 훅)은 PR #495에서 제안됐지만 이 PR은 미병합 종료(CLOSED)됐고, 현재 코드는 서브모듈을 가진 32개 클래스 모두 매핑이 없으며 31개가 __init__에서 self.initialize_submodule_config(...)를 직접 호출한다. 타입 주석도 RBLNModelConfig | None 34개 대 구체 클래스 38개로 갈린다.",
 "rule_refinement": "서브모듈을 가진 설정 클래스는 `submodules` 클래스 속성을 선언하고, __init__에서 각 서브모듈을 공통 헬퍼 `self.initialize_submodule_config(submodule_config, ...)`로 변환해 런타임 옵션(npu, num_devices, create_runtimes 등)이 서브모듈로 전파되게 한다.",
 "suggested_test": "tests/test_conventions.py::test_submodules_initialized_via_helper — configuration_*.py를 AST로 순회해 `submodules = [...]`가 비어있지 않은 클래스 중 __init__을 정의한 클래스는 initialize_submodule_config 호출이 최소 1회 있어야 한다고 단언. 현재 allowlist: RBLNExaone4_5_ForConditionalGenerationConfig(self.visual 직접 대입)."
})

out.append({
 "id": "CFG-022",
 "check": {"kind": "script",
  "description": "rbln_config(dict 또는 인스턴스)와 rbln_ 접두사 인자를 동시에 넘겼을 때 ValueError가 나는지 실제 호출로 확인하고, 병합 경로(initialize_from_kwargs, from_pretrained, convert_rbln_config_dict)를 센다. 오류 없이 병합되면 위반.",
  "command": SRC + " python -W ignore -c \"from optimum.rbln import RBLNResNetForImageClassificationConfig as C; c,_=C.initialize_from_kwargs({'batch_size':4}, rbln_batch_size=16); print(c.batch_size); c,_=C.initialize_from_kwargs(C(image_size=224), rbln_batch_size=16); print(c.batch_size)\" ; grep -nE 'rbln_config.update\\(rbln_kwargs\\)|setattr\\(rbln_config, key, value\\)|kwargs\\[f\\\"rbln_\\{key\\}\\\"\\]' src/optimum/rbln/configuration_utils.py"},
 "compliant": 0,
 "violations": [
  {"path": "src/optimum/rbln/configuration_utils.py", "line": 981, "snippet": "rbln_config.update(rbln_kwargs)", "note": "dict+접두사 동시 지정 시 오류 없이 병합, 접두사 값이 우선"},
  {"path": "src/optimum/rbln/configuration_utils.py", "line": 988, "snippet": "for key, value in rbln_kwargs.items():", "note": "인스턴스+접두사 동시 지정 시 setattr로 덮어씀"},
  {"path": "src/optimum/rbln/configuration_utils.py", "line": 914, "snippet": "if isinstance(rbln_config, dict):", "note": "from_pretrained(path)에서 두 경로를 병합"},
  {"path": "src/optimum/rbln/configuration_utils.py", "line": 1080, "snippet": "rbln_config.update(rbln_kwargs)", "note": "convert_rbln_config_dict도 병합"},
  {"path": "tests/test_config.py", "line": 143, "snippet": "def test_config_priority(model_id):", "note": "테스트가 접두사 인자가 rbln_config를 덮어쓰는 동작을 명시적으로 고정"}
 ],
 "violations_total": 5,
 "verdict": "rejected",
 "verdict_reason": "두 경로를 모두 허용하는 부분은 사실이지만, 상호 배타(ValueError)는 PR #445가 미병합 종료(CLOSED)돼 반영되지 않았다. 현재 코드의 4개 병합 경로는 모두 오류 없이 합치며 rbln_ 접두사 값이 이기고, tests/test_config.py::test_config_priority가 그 우선순위를 단언한다.",
 "rule_refinement": "설정 키는 rbln_config(dict 또는 RBLN*Config 인스턴스)와 from_pretrained의 rbln_ 접두사 인자 두 경로로 지정할 수 있으며, 둘을 함께 주면 병합되고 rbln_ 접두사 인자가 우선한다. 단, 컴파일된 모델을 로드할 때 저장된 컴파일 시점 키를 다른 값으로 바꾸려 하면 ValueError(\"Cannot set the following arguments\")가 난다.",
 "suggested_test": "tests/test_conventions.py::test_rbln_prefix_overrides_rbln_config — 컴파일 없이 RBLNResNetForImageClassificationConfig.initialize_from_kwargs({'batch_size': 4}, rbln_batch_size=16)과 인스턴스 경로 모두에서 batch_size==16을 단언해 현재 우선순위 계약을 고정한다(test_config.py의 컴파일 기반 테스트를 빠른 단위 테스트로 보완)."
})

out.append({
 "id": "SUB-008",
 "check": {"kind": "grep",
  "description": "비전 타워 호스트가 rotary cos/sin을 graph 입력으로 선언·캐스트할 때 HF처럼 fp32를 쓰는지, 아니면 rbln_config.dtype(활성 dtype)으로 반올림하는지, 그리고 회전 적용이 HF의 apply_rotary_pos_emb_vision처럼 명시적 fp32 업캐스트를 하는지 확인한다. 활성 dtype 캐스트나 암묵적 승격 의존은 위반.",
  "command": "grep -rnE '\\(\"(cos|sin)\"|^\\s+\"(cos|sin)\",' -A3 src/optimum/rbln/transformers/models/{qwen2_vl,qwen2_5_vl,qwen3_vl,qwen3_5,exaone4_5}/modeling_*.py ; grep -rnE 'emb\\.cos\\(\\)\\.to\\(self\\.rbln_config\\.dtype\\)' src/optimum/rbln --include=modeling_*.py ; grep -rn 'apply_rotary_pos_emb_vision' src/optimum/rbln"},
 "compliant": 1,
 "violations": [
  {"path": "src/optimum/rbln/transformers/models/qwen2_vl/modeling_qwen2_vl.py", "line": 191, "snippet": "position_embeddings = (emb.cos().to(self.rbln_config.dtype), emb.sin().to(self.rbln_config.dtype))", "note": "HF는 fp32 유지, 활성 dtype으로 반올림"},
  {"path": "src/optimum/rbln/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py", "line": 281, "snippet": "position_embeddings = (emb.cos().to(self.rbln_config.dtype), emb.sin().to(self.rbln_config.dtype))", "note": "HF는 fp32 유지, 활성 dtype으로 반올림"},
  {"path": "src/optimum/rbln/transformers/models/qwen3_vl/modeling_qwen3_vl.py", "line": 280, "snippet": "position_embeddings = (emb.cos().to(self.rbln_config.dtype), emb.sin().to(self.rbln_config.dtype))", "note": "HF는 fp32 유지, 활성 dtype으로 반올림"},
  {"path": "src/optimum/rbln/transformers/models/exaone4_5/modeling_exaone4_5.py", "line": 251, "snippet": "position_embeddings = (emb.cos().to(self.rbln_config.dtype), emb.sin().to(self.rbln_config.dtype))", "note": "HF는 fp32 유지, 활성 dtype으로 반올림"},
  {"path": "src/optimum/rbln/transformers/models/decoderonly/decoderonly_architecture.py", "line": 1356, "snippet": "def apply_rotary_pos_emb(q, k, cos, sin):", "note": "fp32 cos와 q 곱의 암묵적 승격에 의존; 비전 타워도 이 텍스트용 함수 사용"}
 ],
 "violations_total": 5,
 "verdict": "aspirational",
 "verdict_reason": "비전 rotary를 가진 5개 계열 중 Qwen3.5만 cos/sin을 torch.float32로 선언하고, Qwen2-VL·Qwen2.5-VL·Qwen3-VL·EXAONE-4.5는 rbln_config.dtype으로 캐스트하며, apply_rotary_pos_emb_vision 계열 함수는 하나도 없어 텍스트용 회전(암묵적 승격)을 공유한다. 근거 PR #711(2026-10 병합)·#764(미병합)는 이 스냅샷(2026-07-31) 이후라 의도는 분명하나 현재 코드는 아직 따르지 않는다.",
 "rule_refinement": None,
 "suggested_test": "tests/test_conventions.py::test_vision_rotary_inputs_fp32 — 비전 타워 호스트 modeling_*.py(qwen2_vl, qwen2_5_vl, qwen3_vl, qwen3_5, exaone4_5)를 AST로 읽어 _update_rbln_config의 input_info 튜플 중 이름이 'cos'/'sin'인 항목의 dtype 표현식이 torch.float32인지 단언하고, `emb.cos().to(self.rbln_config.dtype)` 패턴을 금지한다. 초기 allowlist: qwen2_vl, qwen2_5_vl, qwen3_vl, exaone4_5(#764 병합 시 제거)."
})

out.append({
 "id": "ATT-006",
 "check": {"kind": "grep",
  "description": "정적 캐시(mark_static_address / key_value_states / conv_state·recurrent_state) 입력의 마지막 축과 인코더·디코더 시퀀스 길이 기본값이 64의 배수로 설계되거나 올림되는지 확인한다. 64 배수가 아닌 기본값/마지막 축은 위반(또는 반례).",
  "command": "grep -rnE '% *64|multiple of 64' src/optimum/rbln --include=*.py ; grep -rnE 'enc_max_seq_len = |cross_key_value_states' -A6 src/optimum/rbln/transformers/models/{seq2seq,whisper,time_series_transformer}/modeling_*.py"},
 "compliant": 4,
 "violations": [
  {"path": "src/optimum/rbln/transformers/models/whisper/modeling_whisper.py", "line": 286, "snippet": "rbln_config.enc_max_seq_len = model_config.max_source_positions", "note": "인코더 길이 1500(64 배수 아님)을 올림 없이 사용"},
  {"path": "src/optimum/rbln/transformers/models/seq2seq/modeling_seq2seq.py", "line": 229, "snippet": "enc_max_seq_len = enc_max_seq_len or tokenizer.model_max_length", "note": "인코더 길이를 64로 올림하지 않음"},
  {"path": "src/optimum/rbln/transformers/models/time_series_transformer/modeling_time_series_transformer.py", "line": 282, "snippet": "model_config.d_model // model_config.decoder_attention_heads,  # 13", "note": "정적 cross-KV 캐시 마지막 축이 head_dim(13), 64 배수 아님"}
 ],
 "violations_total": 3,
 "verdict": "interview",
 "verdict_reason": "64 단위 정렬은 prefill_chunk_size 검증(modeling_attention_utils.py:43), TST dec_max_seq_len 올림, Qwen3.5 conv_state/recurrent_state 레이아웃에서 지켜지지만, Whisper 인코더 1500·seq2seq 인코더 길이는 올림하지 않고 TST cross-KV 캐시 마지막 축은 head_dim 13이다. 근거 PR #804·#816은 모두 미병합(OPEN)이다. 팀 질문: 64 배수 요구가 모든 정적 캐시에 적용되는가, 아니면 rbln_cache_update로 그래프 안에서 갱신하는 커스텀 상태 캐시(KV처럼 head_dim이 모델로 고정되지 않는 것)에만 해당하는가? 인코더 시퀀스 길이 올림은 특정 파이프라인 규칙인가 일반 규칙인가?",
 "rule_refinement": "레이아웃을 직접 설계하는 디바이스 상주 정적 상태 캐시(conv cache, recurrent state 등)는 64의 배수인 축을 가장 안쪽(마지막) 축에 둔다. 사용자 값이 아닌 기본값으로 정하는 시퀀스 길이·청크 크기는 64의 배수로 올림하거나 검증한다(prefill_chunk_size, TST dec_max_seq_len 선례).",
 "suggested_test": None
})

path = "<SCRATCH>/stage_d/out/batch_config.json"
json.dump(out, open(path, "w"), ensure_ascii=False, indent=1)
d = json.load(open(path))
for r in d:
    assert all(len(v["snippet"]) <= 120 for v in r["violations"]), r["id"]
    print(r["id"], r["verdict"], f'{r["compliant"]}/{r["violations_total"]}')
