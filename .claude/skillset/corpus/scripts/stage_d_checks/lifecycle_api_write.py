import json
S = "<SCRATCH>/stage_d"
SETUP = ('S=' + S + '; DEV=$S/lifecycle_api_dev; mkdir -p $DEV && git archive origin/dev src scripts README.md examples | tar -x -C $DEV '
         '&& printf \'__version__ = "0.12.0"\\n__version_tuple__ = (0, 12, 0)\\nversion = __version__\\nversion_tuple = __version_tuple__\\n\' > $DEV/src/optimum/rbln/__version__.py; ')
fwd = json.load(open(S + "/scripts/lifecycle_api_forward_viol.json"))
for v in fwd:
    n = v["note"]
    if n.startswith("'"):
        # "'inputs_embeds' at pos 1; HF X pos 2 (15 classes)"
        import re
        m = re.match(r"'(\w+)' at pos (\d+); HF (\w+) pos (\d+) \((\d+) classes\)", n)
        v["note"] = f"{m.group(1)} pos {m.group(2)} vs HF pos {m.group(4)}; affects {m.group(5)} classes"
    else:
        v["note"] = "HF Blip2QFormerModel.forward takes attention_mask; RBLN omits it"
    v["snippet"] = v["snippet"][:120]

out = []
out.append({
 "id": "LIFE-002",
 "check": {"kind": "script",
  "description": "origin/dev 트리를 import해 forward를 직접 정의한 RBLN 모델 클래스마다 get_hf_class()의 HF forward와 파라미터를 비교한다: 공통 입력의 위치 순서가 다르거나(ORDER), HF의 데이터 입력이 RBLN에 없고 **kwargs도 없으면(MISSING) 위반이며, 별도로 generic 경로의 입력 이름 검증(raise)과 token_type_ids 유/무 테스트 존재를 grep으로 확인한다.",
  "command": SETUP + "cd $S && PYTHONPATH=$DEV/src python3 $S/scripts/lifecycle_api_forward_parity.py 2>/dev/null > $S/scripts/lifecycle_api_forward_parity.out; grep -n 'Invalid model input names' $DEV/src/optimum/rbln/transformers/modeling_generic.py; grep -n 'token_type_ids' $DEV/tests/test_transformers.py"},
 "compliant": 48,
 "violations": fwd,
 "violations_total": 19,
 "verdict": "interview",
 "verdict_reason": "입력 집합 일치(MISSING은 Blip2QFormer 1건)와 사용자 지정 입력 이름 검증 raise(modeling_generic.py 143/151행, compile 입력 순서도 HF 시그니처 순서로 정렬), token_type_ids 유/무 테스트(BERT 계열 vs ModernBERT)는 지켜지지만, Python forward의 위치 인자 순서는 forward 정의 67개 중 18개(클래스 97개 중 43개)가 HF와 다르다(대부분 DecoderOnly·Qwen-VL 계열이 inputs_embeds를 2번째에 둠). 팀이 정할 질문: '순서 일치'가 컴파일 그래프 입력 순서(PR #11, 현재 준수)만 뜻하는가, 아니면 사용자에게 노출되는 forward의 위치 인자 순서까지 HF와 맞춰야 하는가?",
 "rule_refinement": "RBLN forward는 HF forward가 받는 데이터 입력(token_type_ids 등)을 모두 받거나 **kwargs로 흡수해야 하고, 컴파일 입력(rbln_model_input_names)은 HF forward 시그니처 순서로 정렬하며 시그니처에 없는 이름은 ValueError로 거부한다. (Python forward 위치 인자 순서까지 맞출지는 미정)",
 "suggested_test": "tests/test_conventions.py::test_forward_accepts_hf_inputs — optimum.rbln의 모든 RBLNBaseModel 하위 공개 클래스 중 forward를 직접 정의한 클래스에 대해 inspect.signature로 get_hf_class().forward와 비교하여, HF의 데이터 입력(input_ids, attention_mask, token_type_ids, position_ids, inputs_embeds, pixel_values 등)이 RBLN 시그니처에 없고 VAR_KEYWORD도 없으면 실패. 허용 목록: RBLNBlip2QFormerModel(attention_mask). 위치 순서 검사는 팀 결정 후 ORDER 위반 18개 forward 정의(decoderonly 671/878, qwen*_vl, exaone4_5, gemma3/4, llava_next, colqwen2, cosmos transformer, qwen3_asr)를 allowlist로 두고 추가."
})
out.append({
 "id": "LIFE-007",
 "check": {"kind": "ast",
  "description": "import된 모듈/클래스(전역 상태)의 속성에 대입하거나 setattr하는 지점을 AST로 찾고, 패치 대입이 finally 블록이 있는 try 본문 안에 있고 같은 속성을 finally에서 원복하는지(또는 try/finally로 감싼 데코레이터 경유인지) 확인한다; 원복 없음·finally 없음이 위반.",
  "command": SETUP + "python3 $S/scripts/lifecycle_api_monkeypatch.py $DEV/src/optimum/rbln; python3 $S/scripts/lifecycle_api_monkeypatch.py src/optimum/rbln; grep -n 'monkey_patch_decorator' -A12 $DEV/src/optimum/rbln/transformers/models/grounding_dino/grounding_dino_architecture.py"},
 "compliant": 3,
 "violations": [],
 "violations_total": 0,
 "verdict": "verified",
 "verdict_reason": "전역 monkey patch는 3곳(decoderonly의 torch.nn.functional.linear, cosmos VAE의 CosmosCausalConv3d.forward, grounding_dino의 HF 클래스 forward 4개)뿐이며 모두 try/finally(grounding_dino는 monkey_patch_decorator 내부 try/finally)로 컴파일 실패 시에도 원복한다(dev·main 동일). logging.py의 sys.stderr=None 대체는 HF 관례의 영구 초기화라 제외했고, bart/t5/swin의 인스턴스 forward 교체(MethodType)는 전역이 아니라 범위 밖이다. 참고로 decoderonly는 original_linear를 try 안에서 캡처해 앞선 줄이 실패하면 finally에서 NameError가 날 수 있다.",
 "rule_refinement": "모듈 함수나 import한 클래스의 메서드(전역 상태)를 바꿀 때는 원본을 try 이전에 캡처하고 try/finally로 감싸 컴파일 실패 시에도 원복한다; 컴파일 대상 모델 인스턴스의 서브모듈 forward 교체(MethodType)는 이 규칙의 대상이 아니다.",
 "suggested_test": "tests/test_conventions.py::test_global_monkey_patches_restored_in_finally — src/optimum/rbln/**/*.py를 AST로 순회하며 import된 이름을 루트로 하는 Attribute 대입/setattr을 찾고, 각 대입이 finalbody를 가진 ast.Try의 body 안에 있으면서 finalbody에 같은 dotted target 대입이 있는지 확인. 허용 목록: utils/logging.py의 sys.stderr, grounding_dino_architecture.py의 monkey_patch/restore_monkey_patch(데코레이터 monkey_patch_decorator가 try/finally임을 별도 assert)."
})
out.append({
 "id": "LIFE-020",
 "check": {"kind": "ast",
  "description": "오버라이드 계약 메서드(get_compiled_model, get_input_info, _update_rbln_config, _wrap_model_if_needed 등)의 모든 정의를 AST로 모아 파라미터 목록이 베이스 계약과 같은지 비교하고, 개별 값 인자나 다른 시그니처를 쓰는 오버라이드를 위반으로 센다.",
  "command": SETUP + "python3 $S/scripts/lifecycle_api_sigs.py $DEV/src/optimum/rbln; python3 $S/scripts/lifecycle_api_sigs.py src/optimum/rbln get_compiled_model get_input_info _update_rbln_config"},
 "compliant": 118,
 "violations": [{"path": "src/optimum/rbln/diffusers/pipelines/controlnet/multicontrolnet.py", "line": 81, "snippet": "def _update_rbln_config(cls, **rbln_config_kwargs):", "note": "no-op override drops (preprocessors, model, model_config, rbln_config) contract"}],
 "violations_total": 1,
 "verdict": "verified",
 "verdict_reason": "dev 기준 get_compiled_model 11/11이 (cls, model, rbln_config), get_input_info 7/7이 (cls, batch_size, query_length, rbln_config, model_config), _update_rbln_config 54/55가 (cls, preprocessors, model, model_config, rbln_config), _wrap_model_if_needed 46/46(+6개는 classmethod 첫 인자를 self로 명명)으로 계약이 균일하다(main도 동일). 다만 '안 쓰는 키워드도 받아 무시'의 근거인 PR #766은 아직 OPEN이라 그 부분은 코드에 없다.",
 "rule_refinement": "오버라이드는 베이스 시그니처(설정 객체 model_config/rbln_config를 받는 형태)를 그대로 유지하고, 계약에 인자를 추가할 때는 모든 오버라이드에 같은 인자를 함께 추가한다.",
 "suggested_test": "tests/test_conventions.py::test_override_signatures_match_contract — src/optimum/rbln/**/*.py에서 클래스 안의 get_compiled_model/get_input_info/_update_rbln_config/_wrap_model_if_needed 정의를 AST로 수집해 첫 인자를 제외한 파라미터 이름 튜플이 각각 modeling.py RBLNModel / decoderonly RBLNDecoderOnlyModel의 베이스 정의와 같은지 검사. 허용 목록: multicontrolnet.py RBLNMultiControlNetModel._update_rbln_config."
})
out.append({
 "id": "STY-027",
 "check": {"kind": "grep",
  "description": "이 저장소 안의 커스텀 op 등록(torch.library/register_fake)과, 아키텍처 코드에서 .item()으로 꺼낸 데이터 의존 인덱스가 torch._check_is_size/torch._check 제약과 함께 쓰이는지 grep으로 확인한다; 제약 없는 인덱스 .item()이 위반.",
  "command": SETUP + "grep -rnE 'torch\\.library|register_fake|impl_abstract|custom_op\\(' $DEV/src src --include=*.py; grep -rnE '\\.item\\(\\)' $DEV/src/optimum/rbln --include=*_architecture.py; grep -rnE 'torch\\._check(_is_size)?\\(' $DEV/src/optimum/rbln --include=*.py; git show --stat ff8c3f6f | head -20"},
 "compliant": 3,
 "violations": [],
 "violations_total": 0,
 "verdict": "verified",
 "verdict_reason": "데이터 의존 인덱스 3곳(t5 cache_pos, time_series embed_idx, grounding_dino spatial_shapes 합)은 모두 torch.compiler.is_exporting() 분기에서 torch._check_is_size/_check 제약을 건다(나머지 .item()은 scale 스칼라). fake 함수 부분은 커밋 ff8c3f6f(refactor(ops): drop duplicate custom op registration, defer to rebel-compiler)로 src/optimum/rbln/ops가 삭제되어 이제 rebel-compiler 소관이며 이 저장소에는 등록 지점이 0개다.",
 "rule_refinement": "torch.export 경로에서 텐서 값을 .item()으로 인덱스로 쓸 때는 is_exporting() 분기에서 torch._check_is_size와 범위 torch._check를 함께 건다; 커스텀 op와 그 fake 함수는 rebel-compiler에서 등록한다(이 저장소에서 재등록 금지).",
 "suggested_test": "tests/test_conventions.py::test_export_item_indices_constrained — src/optimum/rbln/**/*_architecture.py를 AST로 순회해 x = <expr>.item() 대입 뒤 같은 함수 안에서 x가 torch._check_is_size(x) 또는 torch._check(...)에 쓰이는지 확인(scale= 키워드 인자로 바로 쓰이는 .item()은 제외). 함께 src 아래에 torch.library.custom_op/register_fake가 없음을 assert. 현재 위반 없음."
})
out.append({
 "id": "API-003",
 "check": {"kind": "script",
  "description": "dev 트리를 import해 (1) src에서 import한 RBLN 클래스에 대한 Class.attr 접근 중 존재하지 않는 속성(제거된 메서드 호출 등), (2) README.md·examples/의 RBLN* 식별자와 RBLN*.attr 참조 중 optimum.rbln에 없는 것을 위반으로 센다; vLLM·model-zoo 등 외부 저장소 동시 수정은 이 저장소에서 확인 불가.",
  "command": SETUP + "cd $S && PYTHONPATH=$DEV/src python3 $S/scripts/lifecycle_api_stale_refs.py $DEV 2>/dev/null"},
 "compliant": 162,
 "violations": [],
 "violations_total": 0,
 "verdict": "verified",
 "verdict_reason": "src의 RBLN 클래스 속성 참조 23곳과 README/examples의 RBLN 참조 139곳이 모두 현재 API로 해석되며(PR #612/#614가 고친 RBLNAutoConfig.load 같은 잔존 호출 없음), 저장소 내부 불변식은 지켜진다. 다만 규칙의 vLLM·model-zoo·외부 문서 동시 수정 부분은 다른 저장소라 코드로 검증할 수 없다(judgment/PR 리뷰 항목).",
 "rule_refinement": "공개 API·설정 키·출력 타입을 제거·변경하는 PR은 같은 PR에서 저장소 내부 호출부·하위 클래스·README·examples를 고쳐 제거된 이름을 참조하는 곳이 0이 되게 하고, vLLM·model-zoo 영향 여부는 PR 본문에 명시한다.",
 "suggested_test": "tests/test_conventions.py::test_no_stale_rbln_api_references — src/optimum/rbln/**/*.py에서 'from ... import RBLNX'로 들여온 이름의 Attribute 접근(RBLNX.attr)을 AST로 모아 hasattr(getattr(optimum.rbln, 'RBLNX'), attr)를 assert하고, README.md와 examples/**/*.py의 RBLN\\w+ 토큰(파일 안에서 직접 class 정의한 이름 제외)이 optimum.rbln에서 resolve되는지 확인. 현재 위반 없음."
})
out.append({
 "id": "API-014",
 "check": {"kind": "script",
  "description": "dev 트리에서 (a) 저장소의 scripts/check_inits.py(TYPE_CHECKING 블록·__all__ 정의 여부·lazy 템플릿) 실행, (b) 모든 __all__ 항목이 그 모듈에서 정의된 이름인지(재export 금지), (c) 패키지 __init__.py가 고정 lazy 템플릿과 같은지, (d) HF 클래스명에 대응하는 leaf RBLN 클래스가 optimum.rbln 최상위로 export되는지 확인한다.",
  "command": SETUP + "cd $DEV && python3 scripts/check_inits.py; cd $S && PYTHONPATH=$DEV/src python3 $S/scripts/lifecycle_api_exports.py $DEV/src/optimum/rbln --import 2>/dev/null | python3 -c \"import json,sys; d=json.load(sys.stdin); print(d['B_ok'], d['B_bad'])\"; PYTHONPATH=$DEV/src python3 $S/scripts/lifecycle_api_leaf_exports.py $DEV/src/optimum/rbln 2>/dev/null"},
 "compliant": 305,
 "violations": [
  {"path": "src/optimum/rbln/modeling.py", "line": 398, "snippet": "__all__ = [", "note": "lists RBLNBaseModel, which is defined in modeling_base.py (re-export)"},
  {"path": "src/optimum/rbln/transformers/models/seq2seq/__init__.py", "line": 15, "snippet": "from .configuration_seq2seq import RBLNModelForSeq2SeqLMConfig", "note": "eager import, not the fixed lazy template"},
  {"path": "src/optimum/rbln/diffusers/__init__.py", "line": 23, "snippet": "LOADABLE_CLASSES[\"optimum.rbln\"] = {", "note": "template plus diffusers loader side-effect registration"}
 ],
 "violations_total": 3,
 "verdict": "verified",
 "verdict_reason": "PR #734가 들어간 origin/dev에서 check_inits.py가 rc=0으로 통과(CI 게이트)하고, __all__ 306개 항목 중 305개가 정의 모듈에 있으며, 패키지 __init__ 77개 중 74개가 고정 템플릿, HF 대응 leaf 모델 149개가 모두 최상위 export(자동 클래스는 이름 기반으로 optimum.rbln에서 찾으므로 export가 곧 레지스트리 등록)된다. 단 워크트리 HEAD(main, 2026-07-31)는 #734 이전이라 손으로 관리하는 _import_structure를 쓰며 __all__ 보유 파일이 6개뿐이다.",
 "rule_refinement": "공개 객체는 정의 모듈의 __all__에만 등록하고(재export 금지) 패키지 __init__.py는 define_import_structure 기반 lazy 템플릿으로 두며, scripts/check_inits.py --fix로 TYPE_CHECKING 블록을 맞춘다; 자동 클래스는 이름(RBLN+HF 클래스명)으로 optimum.rbln에서 찾으므로 최상위 export가 곧 등록이고, optimum.rbln 밖의 사용자 정의 클래스만 RBLNAutoModel.register/RBLNAutoConfig.register를 쓴다.",
 "suggested_test": "tests/test_conventions.py::test_import_structure — scripts/check_inits.py의 main()이 0을 반환하는지 확인하고, 추가로 src/optimum/rbln/**/*.py(비 __init__)의 __all__ 항목이 해당 모듈의 ClassDef/FunctionDef/Assign 이름인지(import 이름 불가) 검사하며, transformers/diffusers에 같은 이름(접두사 RBLN 제거)이 있는 모든 RBLN 클래스가 hasattr(optimum.rbln, name)인지 assert. 허용 목록: modeling.py의 RBLNBaseModel 재export, seq2seq/__init__.py, diffusers/__init__.py(LOADABLE_CLASSES 등록)."
})
out.append({
 "id": "API-015",
 "check": {"kind": "ast",
  "description": "src 모듈 중 import/__all__만 있는 shim 모듈, 모듈 수준 클래스 별칭 대입(RBLNX = RBLNY), sys.modules 별칭 조작을 찾고, 추가로 HF 클래스에 대응하는 leaf RBLN 클래스가 HF와 같은 모델 디렉터리(transformers/models/<dir>, diffusers/<subpath>)에 놓였는지 비교해 어긋난 배치를 위반으로 센다.",
  "command": SETUP + "python3 $S/scripts/lifecycle_api_shims.py $DEV/src/optimum/rbln; python3 $S/scripts/lifecycle_api_shims.py src/optimum/rbln; grep -rn 'sys.modules\\[' $DEV/src/optimum/rbln --include=*.py | grep -v __init__.py; cd $S && PYTHONPATH=$DEV/src python3 $S/scripts/lifecycle_api_placement.py $DEV/src/optimum/rbln 2>/dev/null"},
 "compliant": 125,
 "violations": [{"path": "src/optimum/rbln/diffusers/pipelines/controlnet/multicontrolnet.py", "line": 30, "snippet": "class RBLNMultiControlNetModel(RBLNModel):", "note": "HF moved it to diffusers.models.controlnets.multicontrolnet; RBLN copy stayed"}],
 "violations_total": 1,
 "verdict": "verified",
 "verdict_reason": "dev의 비 __init__ 모듈 245개(main 228개) 중 재import만 하는 shim 모듈과 모듈 수준 클래스 별칭은 0개이고 __init__ 밖의 sys.modules 조작도 없다; 별칭 클래스는 PR #157 방식대로 모델별 디렉터리의 빈 본문 하위 클래스(57개)로 존재한다. HF 대응 leaf 클래스 126개 중 125개가 HF와 같은 디렉터리에 있고, RBLNMultiControlNetModel만 PR #726 이동 때 pipelines/controlnet에 남았다.",
 "rule_refinement": "모듈을 옮길 때 옛 경로에 재export shim을 남기지 않고, 모델 클래스는 HF와 같은 디렉터리 구조(transformers/models/<model>/, diffusers/models|pipelines/<...>)에 두며, 이름 별칭이 필요하면 별칭 파일 대신 해당 모델 디렉터리에 빈 하위 클래스로 정의한다.",
 "suggested_test": "tests/test_conventions.py::test_no_shim_modules_and_hf_mirrored_layout — src/optimum/rbln/**/*.py(비 __init__) 중 최상위 문장이 Import/ImportFrom/__all__/docstring뿐인 모듈이 없음을 assert하고, 모듈 수준 'CamelName = OtherCamelName' 대입을 금지하며, transformers/diffusers에 대응 클래스가 있는 RBLN 클래스의 파일 경로가 HF 클래스 __module__의 디렉터리와 같은지 비교. 허용 목록: diffusers/pipelines/controlnet/multicontrolnet.py."
})
out.append({
 "id": "API-011",
 "check": {"kind": "script",
  "description": "optimum.rbln에서 export되는 모든 RBLNBaseModel/RBLNDiffusionMixin 하위 클래스 이름이 'RBLN'+(transformers·diffusers 클래스명 또는 HF 모듈 내부/원격코드 클래스명)인지, 모든 RBLNModelConfig 하위 export가 RBLN*Config인지 확인한다; 공통 베이스(RBLNBaseModel·RBLNModel·RBLNDiffusionMixin·RBLNDecoderOnlyModel[ForCausalLM])는 제외하고 HF 대응명이 없는 이름이 위반.",
  "command": SETUP + "cd $S && PYTHONPATH=$DEV/src python3 $S/scripts/lifecycle_api_naming.py 2>/dev/null; python3 -c \"import transformers.models.grounding_dino.modeling_grounding_dino as g, transformers.models.qwen2_vl.modeling_qwen2_vl as q; print(hasattr(g,'GroundingDinoEncoder'), hasattr(g,'GroundingDinoDecoder'), hasattr(g,'GroundingDinoTextModel'), hasattr(q,'Qwen2VisionTransformerPretrainedModel'))\""},
 "compliant": 132,
 "violations": [{"path": "src/optimum/rbln/transformers/models/grounding_dino/modeling_grounding_dino.py", "line": 61, "snippet": "class RBLNGroundingDinoTextModel(RBLNBertModel):", "note": "no HF GroundingDinoTextModel; wraps BertModel (_hf_class = BertModel)"}],
 "violations_total": 1,
 "verdict": "verified",
 "verdict_reason": "export된 모델·파이프라인 클래스 138개 중 공통 베이스 5개를 빼면 126개가 transformers/diffusers 최상위 클래스명, 6개(Exaone/Midm 원격코드, Qwen2/2.5 VisionTransformer, GroundingDino Encoder/Decoder)가 HF 모듈 내부 클래스명에 RBLN을 붙인 형태이고, 설정 클래스 135개는 모두 RBLN*Config다. HF 대응명이 없는 것은 RBLNGroundingDinoTextModel 1개뿐이다(근거 PR #533은 머지되지 않고 CLOSED).",
 "rule_refinement": "공개 모델·파이프라인·diffusers 구성요소 클래스는 'RBLN' + 대응하는 HF(transformers/diffusers, 원격코드 포함) 클래스명 그대로, 설정은 그 뒤에 'Config'를 붙여 짓는다; 자동 클래스가 convert_hf_to_rbln_model_name으로 이 이름을 찾으므로 이름 규칙이 곧 로딩 계약이다.",
 "suggested_test": "tests/test_conventions.py::test_public_model_names_mirror_hf — optimum.rbln의 공개 이름 중 RBLNBaseModel/RBLNDiffusionMixin 하위 클래스(RBLNAuto*·공통 베이스 제외)에 대해 name[4:]이 transformers/diffusers 최상위 속성이거나 cls.get_hf_class().__name__과 같은지 확인하고, 각 클래스에 대해 name+'Config'가 export되는지(MultiControlNet 제외) assert. 허용 목록: RBLNGroundingDinoTextModel, RBLNExaoneForCausalLM, RBLNMidmLMHeadModel(원격코드)."
})
p = S + "/out/batch_lifecycle_api.json"
json.dump(out, open(p, "w"), ensure_ascii=False, indent=1)
d = json.load(open(p))
for r in d:
    assert len(r["violations"]) <= 25
    for v in r["violations"]: assert len(v["snippet"]) <= 120, v
    print(r["id"], r["verdict"], f"{r['compliant']}/{r['violations_total']}")
