# 알려진 위반 목록 (stage D, 2026-10-22)

룰별로 현재 코드에서 룰을 어기는 위치. 상태가 verified인 룰의 위반은 기술부채 이슈 후보이고, aspirational인 룰의 위반은 룰 채택 시 함께 고쳐야 할 범위다.

## PRC-003 (aspirational) — 위반 537 / 준수 51

PR 본문은 템플릿의 절(변경 유형, 변경 개요, Motivation, Verification, 관련 이슈)을 갖추고, Type of Change에서 해당 유형을 하나 체크하며(Other는 내용을 구체적으로 적음), Motivation에는 채택한 방식이 필요한 이유(문제의 원인이나 기대 효과)를 적는다.

- `PR #819` — Motivation 비어 있음
    ```
    model: update Qwen3-VL methods to support embedding & reranker models
    ```
- `PR #814` — 유형 0개 체크, Motivation 비어 있음
    ```
    doc(README): Fix documentation links in README.md
    ```
- `PR #812` — 템플릿 절 누락, Motivation 비어 있음
    ```
    other(test): TimeSeries BC reload sanitizes NaN in the dummy outputs
    ```
- `PR #810` — 템플릿 절 누락, Motivation 비어 있음
    ```
    other(test): fix the BC v0.12.0 reload of Cosmos and the llm suite
    ```
- `PR #802` — 유형 2개 체크
    ```
    model: Add OLMo-2 decoder-only model
    ```
- `PR #796` — 템플릿 절 누락, Motivation 비어 있음
    ```
    model(qwen3.5): Send the vision attention mask as key padding
    ```
- `PR #785` — 템플릿 절 누락, Motivation 비어 있음
    ```
    other: move the 0.12.0 deprecation cutoff to 0.13.0
    ```
- `PR #783` — 템플릿 절 누락, Motivation 비어 있음
    ```
    doc: one line per parameter type, one line per return
    ```
- `PR #778` — 템플릿 절 누락
    ```
    other: marker commit for the dev0 tag
    ```
- `PR #777` — 템플릿 절 누락, Motivation 비어 있음
    ```
    model: Cosmos2.5 pipeline keeps the visual tower config it sets
    ```
- `PR #774` — 템플릿 절 누락, Motivation 비어 있음
    ```
    model: fix pixtral cast attention probabilities back to the query dtype for torch 2.13
    ```
- `PR #773` — 템플릿 절 누락, Motivation 비어 있음
    ```
    model: fix cast the strict-lower mask to the decay dtype for torch 2.13
    ```
- `PR #770` — 유형 2개 체크
    ```
    model: fill compile-time shapes only when the caller left them out
    ```
- `PR #769` — 템플릿 절 누락
    ```
    dependency: bump torch to 2.13.0 (torchvision 0.28.0, torchcodec 0.16.0)
    ```
- `PR #763` — 템플릿 절 누락, Motivation 비어 있음
    ```
    model(vlm): feed vision rotary cos/sin to the graph in fp32
    ```
- `PR #761` — Motivation 비어 있음
    ```
    refactor: use config.get_text_config() to reuse redundant methods
    ```
- `PR #758` — 템플릿 절 누락, Motivation 비어 있음
    ```
    other: ci - take the cluster paths from Obedients variables
    ```
- `PR #757` — Motivation 비어 있음
    ```
    other: align qwen3_vl left padding
    ```
- `PR #756` — 템플릿 절 누락, Motivation 비어 있음
    ```
    other: ci - report which test failed, not just which step
    ```
- `PR #755` — 템플릿 절 누락, Motivation 비어 있음
    ```
    other: ci - skip suites before they take an NPU, and name the tests that failed
    ```
- `PR #754` — 템플릿 절 누락, Motivation 비어 있음
    ```
    other: ci - drop the CI image, nothing left here needs one
    ```
- `PR #753` — 템플릿 절 누락, Motivation 비어 있음
    ```
    other: ci - compile the BC artifacts on Obedients, not GHA
    ```
- `PR #752` — 템플릿 절 누락, Motivation 비어 있음
    ```
    other: ci - restore the [slack-report] step on PR builds
    ```
- `PR #749` — 템플릿 절 누락, Motivation 비어 있음
    ```
    other: ci - point the team gate at the renamed icp team
    ```
- `PR #748` — 템플릿 절 누락, Motivation 비어 있음
    ```
    other: ci - drop path gating and run every suite on every PR
    ```
- … 외 512건 (검사 명령으로 전체 목록 재현 가능)

## STY-005 (aspirational) — 위반 47 / 준수 290

새로 추가하는 소스 파일에는 기존 파일과 같은 형식의 Rebellions 저작권 헤더를 현재 연도로 붙인다.

- `src/optimum/rbln/diffusers/models/controlnets/controlnet.py:1` — 헤더 연도가 파일 추가 연도와 다름 (hdr 2025, added 2026)
    ```
    # Copyright 2025 Rebellions Inc. All rights reserved.
    ```
- `src/optimum/rbln/diffusers/models/controlnets/multicontrolnet.py:1` — 헤더 연도가 파일 추가 연도와 다름 (hdr 2025, added 2026)
    ```
    # Copyright 2025 Rebellions Inc. All rights reserved.
    ```
- `src/optimum/rbln/transformers/models/detr/__init__.py:1` — 헤더 연도가 파일 추가 연도와 다름 (hdr 2025, added 2026)
    ```
    # Copyright 2025 Rebellions Inc. All rights reserved.
    ```
- `src/optimum/rbln/transformers/models/mixtral/__init__.py:1` — 헤더 연도가 파일 추가 연도와 다름 (hdr 2025, added 2026)
    ```
    # Copyright 2025 Rebellions Inc. All rights reserved.
    ```
- `src/optimum/rbln/transformers/models/modernbert/__init__.py:1` — 헤더 연도가 파일 추가 연도와 다름 (hdr 2025, added 2026)
    ```
    # Copyright 2025 Rebellions Inc. All rights reserved.
    ```
- `src/optimum/rbln/transformers/models/modernbert/configuration_modernbert.py:1` — 헤더 연도가 파일 추가 연도와 다름 (hdr 2025, added 2026)
    ```
    # Copyright 2025 Rebellions Inc. All rights reserved.
    ```
- `src/optimum/rbln/transformers/models/modernbert/modeling_modernbert.py:1` — 헤더 연도가 파일 추가 연도와 다름 (hdr 2025, added 2026)
    ```
    # Copyright 2025 Rebellions Inc. All rights reserved.
    ```
- `src/optimum/rbln/transformers/models/modernbert/modernbert_architecture.py:1` — 헤더 연도가 파일 추가 연도와 다름 (hdr 2025, added 2026)
    ```
    # Copyright 2025 Rebellions Inc. All rights reserved.
    ```
- `src/optimum/rbln/transformers/utils/moe.py:1` — 헤더 연도가 파일 추가 연도와 다름 (hdr 2025, added 2026)
    ```
    # Copyright 2025 Rebellions Inc. All rights reserved.
    ```
- `src/optimum/rbln/transformers/models/exaone4_5/__init__.py:1` — 헤더 형식이 표준과 다름(2024 단축형 또는 '#' 빈줄) (hdr 2026, added 2026)
    ```
    # Copyright 2026 Rebellions Inc. All rights reserved.
    ```
- `src/optimum/rbln/transformers/models/exaone4_5/configuration_exaone4_5.py:1` — 헤더 형식이 표준과 다름(2024 단축형 또는 '#' 빈줄) (hdr 2026, added 2026)
    ```
    # Copyright 2026 Rebellions Inc. All rights reserved.
    ```
- `src/optimum/rbln/transformers/models/exaone4_5/exaone4_5_architecture.py:1` — 헤더 형식이 표준과 다름(2024 단축형 또는 '#' 빈줄) (hdr 2026, added 2026)
    ```
    # Copyright 2026 Rebellions Inc. All rights reserved.
    ```
- `src/optimum/rbln/transformers/models/exaone4_5/modeling_exaone4_5.py:1` — 헤더 형식이 표준과 다름(2024 단축형 또는 '#' 빈줄) (hdr 2026, added 2026)
    ```
    # Copyright 2026 Rebellions Inc. All rights reserved.
    ```
- `src/optimum/rbln/transformers/models/qwen3_vl_moe/__init__.py:1` — 헤더 형식이 표준과 다름(2024 단축형 또는 '#' 빈줄) (hdr 2026, added 2026)
    ```
    # Copyright 2026 Rebellions Inc. All rights reserved.
    ```
- `src/optimum/rbln/transformers/models/qwen3_vl_moe/configuration_qwen3_vl_moe.py:1` — 헤더 형식이 표준과 다름(2024 단축형 또는 '#' 빈줄) (hdr 2026, added 2026)
    ```
    # Copyright 2026 Rebellions Inc. All rights reserved.
    ```
- `src/optimum/rbln/transformers/models/qwen3_vl_moe/modeling_qwen3_vl_moe.py:1` — 헤더 형식이 표준과 다름(2024 단축형 또는 '#' 빈줄) (hdr 2026, added 2026)
    ```
    # Copyright 2026 Rebellions Inc. All rights reserved.
    ```
- `src/optimum/rbln/transformers/models/qwen3_vl_moe/qwen3_vl_moe_architecture.py:1` — 헤더 형식이 표준과 다름(2024 단축형 또는 '#' 빈줄) (hdr 2026, added 2026)
    ```
    # Copyright 2026 Rebellions Inc. All rights reserved.
    ```
- `src/optimum/rbln/transformers/models/qwen3_vl_moe/qwen3_vl_moe_runtime_utils.py:1` — 헤더 형식이 표준과 다름(2024 단축형 또는 '#' 빈줄) (hdr 2026, added 2026)
    ```
    # Copyright 2026 Rebellions Inc. All rights reserved.
    ```
- `src/optimum/rbln/transformers/utils/qlinear.py:1` — 저작권 헤더 없음 (hdr None, added 2026)
    ```
    import torch
    ```
- `src/optimum/rbln/diffusers/configurations/__init__.py:1` — 헤더 연도가 파일 추가 연도와 다름 (hdr 2026, added 2025)
    ```
    # Copyright 2026 Rebellions Inc. All rights reserved.
    ```
- `src/optimum/rbln/diffusers/configurations/models/__init__.py:1` — 헤더 연도가 파일 추가 연도와 다름 (hdr 2026, added 2025)
    ```
    # Copyright 2026 Rebellions Inc. All rights reserved.
    ```
- `src/optimum/rbln/diffusers/configurations/pipelines/__init__.py:1` — 헤더 연도가 파일 추가 연도와 다름 (hdr 2026, added 2025)
    ```
    # Copyright 2026 Rebellions Inc. All rights reserved.
    ```
- `src/optimum/rbln/transformers/models/colpali/__init__.py:1` — 헤더 연도가 파일 추가 연도와 다름 (hdr 2026, added 2025)
    ```
    # Copyright 2026 Rebellions Inc. All rights reserved.
    ```
- `src/optimum/rbln/transformers/models/colqwen2/__init__.py:1` — 헤더 연도가 파일 추가 연도와 다름 (hdr 2026, added 2025)
    ```
    # Copyright 2026 Rebellions Inc. All rights reserved.
    ```
- `src/optimum/rbln/transformers/models/grounding_dino/__init__.py:1` — 헤더 연도가 파일 추가 연도와 다름 (hdr 2026, added 2025)
    ```
    # Copyright 2026 Rebellions Inc. All rights reserved.
    ```
- … 외 22건 (검사 명령으로 전체 목록 재현 가능)

## CFG-019 (rejected) — 위반 36 / 준수 0

서브모듈을 가진 설정 클래스는 서브모듈 목록과 설정 클래스 매핑(submodule_config_classes)을 선언하고 타입 주석을 `RBLNModelConfig | None`으로 쓰며, dict에서 설정 객체로의 변환은 각 클래스에서 수동으로 하지 않고 공통 검증 훅에서 자동 처리한다.

- `src/optimum/rbln/configuration_utils.py:574` — 공통 검증 훅이 아닌 각 __init__이 호출하는 헬퍼; cls_name 없으면 dict 반환
    ```
    def initialize_submodule_config(
    ```
- `src/optimum/rbln/configuration_utils.py:680` — 자동 변환은 cls_name 있는 dict에만, 매핑 없음
    ```
    if key in self.submodules and isinstance(value, dict) and (cls_name := value.get("cls_name")):
    ```
- `src/optimum/rbln/transformers/models/gemma4/configuration_gemma4.py:231` — 서브모듈이 아직 dict일 수 있어 클래스별로 수동 분기
    ```
    if isinstance(vt_cfg, dict):
    ```
- `src/optimum/rbln/transformers/models/llava/configuration_llava.py:24` — submodule_config_classes 없음, __init__에서 수동 변환
    ```
    class RBLNLlavaForConditionalGenerationConfig
    ```
- `src/optimum/rbln/transformers/models/exaone4_5/configuration_exaone4_5.py:53` — 매핑 없음, 클래스별 수동 변환(main의 직접 대입은 헬퍼 호출로 바뀜)
    ```
    self.visual = self.initialize_submodule_config(submodule_config=visual, batch_size=1, force_kwargs=True)
    ```
- `src/optimum/rbln/diffusers/configurations/pipelines/configuration_stable_diffusion.py:22` — 매핑 없음, 구체 클래스 타입 주석, 수동 변환
    ```
    class RBLNStableDiffusionPipelineBaseConfig
    ```
- `src/optimum/rbln/transformers/models/grounding_dino/configuration_grounding_dino.py` — transformers 쪽 유일한 구체 클래스 주석
    ```
    encoder: Optional['RBLNGroundingDinoEncoderConfig']
    ```
- … 외 29건 (검사 명령으로 전체 목록 재현 가능)

## PRC-002 (rejected) — 위반 34 / 준수 4

작업 중인 PR은 제목에 (WIP)를 붙인다.

- `PR #822` — draft PR인데 제목에 (WIP) 없음
    ```
    model(gemma3): run the standalone Gemma3 text prefill causal, as transformers does
    ```
- `PR #820` — draft PR인데 제목에 (WIP) 없음
    ```
    model: add multi_token_decoder for speculative decoding
    ```
- `PR #795` — draft PR인데 제목에 (WIP) 없음
    ```
    model: [mode-2] enable ALBERT DeBERTa-v2 DINOv2 ConvNeXt and Swin classification
    ```
- `PR #794` — draft PR인데 제목에 (WIP) 없음
    ```
    model: [mode-1] improve existing model export and runtime contracts
    ```
- `PR #768` — draft PR인데 제목에 (WIP) 없음
    ```
    performance(qwen3.5): Run the GDN decode short conv as the causal_conv1d_update custom op
    ```
- `PR #759` — draft PR인데 제목에 (WIP) 없음
    ```
    other: test - cover _prepare_prefill_inputs on CPU for every runtime class
    ```
- `PR #708` — draft PR인데 제목에 (WIP) 없음
    ```
    performance(memory): size the KV cache from the joint allocation, not the sum
    ```
- `PR #686` — draft PR인데 제목에 (WIP) 없음
    ```
    other(ci): warm the HF cache once, then run pytest fully offline
    ```
- `PR #680` — draft PR인데 제목에 (WIP) 없음
    ```
    fix: keep torch cos/sin for the compile-time decoderonly rotary constants
    ```
- `PR #656` — draft PR인데 제목에 (WIP) 없음
    ```
    fix: quote rebel.* union annotations to avoid runtime TypeError
    ```
- `PR #643` — draft PR인데 제목에 (WIP) 없음
    ```
    performance(weight-free): link deferred weights from Hugging Face checkpoints
    ```
- `PR #589` — draft PR인데 제목에 (WIP) 없음
    ```
    model(whisper): restore [batch, seq] long-form generate output rank
    ```
- `PR #487` — draft PR인데 제목에 (WIP) 없음
    ```
    other: fix deprecation version logic
    ```
- `PR #480` — draft PR인데 제목에 (WIP) 없음
    ```
    other: support non-fp32 dtype for transformers/v0.5.0
    ```
- `PR #479` — draft PR인데 제목에 (WIP) 없음
    ```
    other: prepare Transformers/v5.0.0rc3
    ```
- `PR #446` — draft PR인데 제목에 (WIP) 없음
    ```
    test
    ```
- `PR #424` — draft PR인데 제목에 (WIP) 없음
    ```
    (DO NOT MERGE) batched attention mask
    ```
- `PR #417` — draft PR인데 제목에 (WIP) 없음
    ```
    model: change condition for dequant
    ```
- `PR #398` — draft PR인데 제목에 (WIP) 없음
    ```
    test bc ci
    ```
- `PR #358` — draft PR인데 제목에 (WIP) 없음
    ```
    other: backward compatibility ci
    ```
- `PR #323` — draft PR인데 제목에 (WIP) 없음
    ```
    Gpt oss sink
    ```
- `PR #274` — draft PR인데 제목에 (WIP) 없음
    ```
    [DRAFT] Detach Cache Update from Custom Attention OP
    ```
- `PR #233` — draft PR인데 제목에 (WIP) 없음
    ```
    other: Update Python version
    ```
- `PR #210` — draft PR인데 제목에 (WIP) 없음
    ```
    other(ci): pyright checking
    ```
- `PR #192` — draft PR인데 제목에 (WIP) 없음
    ```
    model: Faster compilation of cosmos transformers
    ```
- … 외 9건 (검사 명령으로 전체 목록 재현 가능)

## TEST-008 (aspirational) — 위반 31 / 준수 99

새 모델 클래스나 파이프라인을 추가하거나 변경할 때는 해당 테스트 목록(test_transformers, test_diffusers, pytest 스케줄 등)에도 등록하며, 멀티모달 모델은 멀티모달 플래그를 설정한다.

- `src/optimum/rbln/diffusers/pipelines/cosmos/pipeline_cosmos_text2world.py:29` — exported user-facing class never set as RBLN_CLASS in tests/
    ```
    class RBLNCosmosTextToWorldPipeline(RBLNDiffusionMixin, CosmosTextToWorldPipeline):
    ```
- `src/optimum/rbln/diffusers/pipelines/cosmos/pipeline_cosmos_video2world.py:29` — exported user-facing class never set as RBLN_CLASS in tests/
    ```
    class RBLNCosmosVideoToWorldPipeline(RBLNDiffusionMixin, CosmosVideoToWorldPipeline):
    ```
- `src/optimum/rbln/transformers/models/depth_anything/modeling_depth_anything.py:22` — exported user-facing class never set as RBLN_CLASS in tests/
    ```
    class RBLNDepthAnythingForDepthEstimation(RBLNModelForDepthEstimation):
    ```
- `src/optimum/rbln/transformers/models/detr/modeling_detr.py:33` — exported user-facing class never set as RBLN_CLASS in tests/
    ```
    class RBLNDetrForObjectDetection(RBLNModel):
    ```
- `src/optimum/rbln/transformers/models/exaone4_5/modeling_exaone4_5.py:471` — exported user-facing class never set as RBLN_CLASS in tests/
    ```
    class RBLNExaone4_5_ForConditionalGeneration(
    ```
- `src/optimum/rbln/transformers/models/exaone4_5/modeling_exaone4_5.py:328` — exported user-facing class never set as RBLN_CLASS in tests/
    ```
    class RBLNExaone4_5_Model(RBLNDecoderOnlyModel):
    ```
- `src/optimum/rbln/transformers/models/gemma2/modeling_gemma2.py:27` — exported user-facing class never set as RBLN_CLASS in tests/
    ```
    class RBLNGemma2ForCausalLM(RBLNDecoderOnlyModelForCausalLM):
    ```
- `src/optimum/rbln/transformers/models/gemma2/modeling_gemma2.py:83` — exported user-facing class never set as RBLN_CLASS in tests/
    ```
    class RBLNGemma2Model(RBLNDecoderOnlyModel):
    ```
- `src/optimum/rbln/transformers/models/gemma4/modeling_gemma4.py:232` — exported user-facing class never set as RBLN_CLASS in tests/
    ```
    class RBLNGemma4ForCausalLM(RBLNMoeLoadMixin, RBLNDecoderOnlyModelForCausalLM):
    ```
- `src/optimum/rbln/transformers/models/gemma4/modeling_gemma4.py:589` — exported user-facing class never set as RBLN_CLASS in tests/
    ```
    class RBLNGemma4ForConditionalGeneration(RBLNMoeLoadMixin, RBLNModel, RBLNImageIndexedBatchSortMixin):
    ```
- `src/optimum/rbln/transformers/models/gemma/modeling_gemma.py:23` — exported user-facing class never set as RBLN_CLASS in tests/
    ```
    class RBLNGemmaForCausalLM(RBLNDecoderOnlyModelForCausalLM):
    ```
- `src/optimum/rbln/transformers/models/gemma/modeling_gemma.py:86` — exported user-facing class never set as RBLN_CLASS in tests/
    ```
    class RBLNGemmaModel(RBLNDecoderOnlyModel):
    ```
- `src/optimum/rbln/transformers/models/gpt_oss/modeling_gpt_oss.py:39` — exported user-facing class never set as RBLN_CLASS in tests/
    ```
    class RBLNGptOssForCausalLM(RBLNDecoderOnlyModelForCausalLM):
    ```
- `src/optimum/rbln/diffusers/pipelines/kandinsky2_2/pipeline_kandinsky2_2_combined.py:152` — exported user-facing class never set as RBLN_CLASS in tests/
    ```
    class RBLNKandinskyV22InpaintCombinedPipeline(RBLNDiffusionMixin, KandinskyV22InpaintCombinedPipeline):
    ```
- `src/optimum/rbln/diffusers/pipelines/kandinsky2_2/pipeline_kandinsky2_2_inpaint.py:21` — exported user-facing class never set as RBLN_CLASS in tests/
    ```
    class RBLNKandinskyV22InpaintPipeline(RBLNDiffusionMixin, KandinskyV22InpaintPipeline):
    ```
- `src/optimum/rbln/transformers/models/midm/modeling_midm.py:32` — exported user-facing class never set as RBLN_CLASS in tests/
    ```
    class RBLNMidmLMHeadModel(RBLNDecoderOnlyModelForCausalLM):
    ```
- `src/optimum/rbln/transformers/models/mixtral/modeling_mixtral.py:20` — exported user-facing class never set as RBLN_CLASS in tests/
    ```
    class RBLNMixtralForCausalLM(RBLNMoeLoadMixin, RBLNDecoderOnlyModelForCausalLM):
    ```
- `src/optimum/rbln/transformers/models/paligemma/modeling_paligemma.py:63` — exported user-facing class never set as RBLN_CLASS in tests/
    ```
    class RBLNPaliGemmaForConditionalGeneration(RBLNModel, RBLNImageIndexedBatchSortMixin):
    ```
- `src/optimum/rbln/transformers/models/paligemma/modeling_paligemma.py:371` — exported user-facing class never set as RBLN_CLASS in tests/
    ```
    class RBLNPaliGemmaModel(RBLNModel):
    ```
- `src/optimum/rbln/transformers/models/qwen2_vl/modeling_qwen2_vl.py:244` — exported user-facing class never set as RBLN_CLASS in tests/
    ```
    class RBLNQwen2VLModel(RBLNDecoderOnlyModel):
    ```
- `src/optimum/rbln/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py:357` — exported user-facing class never set as RBLN_CLASS in tests/
    ```
    class RBLNQwen2_5_VLModel(RBLNDecoderOnlyModel):
    ```
- `src/optimum/rbln/transformers/models/qwen3_vl/modeling_qwen3_vl.py:252` — exported user-facing class never set as RBLN_CLASS in tests/
    ```
    class RBLNQwen3VLModel(RBLNDecoderOnlyModel):
    ```
- `src/optimum/rbln/transformers/models/qwen3_vl_moe/modeling_qwen3_vl_moe.py:82` — exported user-facing class never set as RBLN_CLASS in tests/
    ```
    class RBLNQwen3VLMoeModel(RBLNMoeLoadMixin, RBLNQwen3VLModel):
    ```
- `src/optimum/rbln/transformers/models/qwen3_5/modeling_qwen3_5.py:546` — exported user-facing class never set as RBLN_CLASS in tests/
    ```
    class RBLNQwen3_5Model(RBLNDecoderOnlyModel):
    ```
- `tests/test_llm.py:896` — VLM test leaves IS_MULTIMODAL=False; overrides get_tokenizer with AutoProcessor instead
    ```
    class TestQwen2VLForConditionalGeneration(LLMTest.TestLLM):
    ```
- … 외 6건 (검사 명령으로 전체 목록 재현 가능)

## PRC-008 (aspirational) — 위반 25 / 준수 11

선행 PR이나 다른 저장소의 PR에 의존하는 PR은 base 브랜치를 선행 PR 브랜치로 두고, 본문에 선행·관련 PR 번호와 링크, 머지 순서, 순서가 어긋났을 때의 대응(닫기 등)을 명시한다.

- `PR #668` — Stacked on #662 라 적었지만 base는 처음부터 dev (662 미머지 상태)
    ```
    model: qwen3.5 aggregate full-sequence logits for the bare text model
    ```
- `PR #360` — need #359 & #356 선언, base는 main→dev; 선행 브랜치 아님
    ```
    model: refactor colpali with paligemma, paligemma2, gemma2
    ```
- `PR #640` — 선행 PR 브랜치에 stack했으나 본문에 선행 PR 번호 없음
    ```
    feat: enable tri update op (base=seinpark/qwen3_5)
    ```
- `PR #457` — 선행 PR 브랜치에 stack했으나 본문에 선행 PR 번호 없음
    ```
    model: decoder_only batch_attention_decode - sorting inputs & unsort output (base=yhboo_dev_base_batch_dyn)
    ```
- `PR #401` — 선행 PR 브랜치에 stack했으나 본문에 선행 PR 번호 없음
    ```
    model: gpt_oss routing patch (base=gpt_oss_mxfp4)
    ```
- `PR #277` — 선행 PR 브랜치에 stack했으나 본문에 선행 PR 번호 없음
    ```
    doc: update docstrings of multimodal models (base=jongho/update_docstring_of_auto)
    ```
- `PR #255` — 선행 PR 브랜치에 stack했으나 본문에 선행 PR 번호 없음
    ```
    model: handling DecoderOnlyModels's batch_size when can generate is False. (base=model/refac_llm)
    ```
- `PR #251` — 선행 PR 브랜치에 stack했으나 본문에 선행 PR 번호 없음
    ```
    model: decoderonly runtime refactoring (base=model/refac_llm)
    ```
- `PR #228` — 선행 PR 브랜치에 stack했으나 본문에 선행 PR 번호 없음
    ```
    other: deprecated warning for RBLN-CA02 (base=deprecate_ca02)
    ```
- `PR #191` — 선행 PR 브랜치에 stack했으나 본문에 선행 PR 번호 없음
    ```
    model(cosmos guardrail): add save and load logic (base=sync_cosmos)
    ```
- `PR #189` — 선행 PR 브랜치에 stack했으나 본문에 선행 PR 번호 없음
    ```
    model: colqwen2_5 refactor  (base=colqwen2_5)
    ```
- `PR #187` — 선행 PR 브랜치에 stack했으나 본문에 선행 PR 번호 없음
    ```
    model(cosmos): Refactor Guardrail (base=sync_cosmos)
    ```
- `PR #184` — 선행 PR 브랜치에 stack했으나 본문에 선행 PR 번호 없음
    ```
    model: refactor colqwen2_5 (base=colqwen2_5)
    ```
- `PR #167` — 선행 PR 브랜치에 stack했으나 본문에 선행 PR 번호 없음
    ```
    model: refactoring cosmos guardrail (base=sync_cosmos)
    ```
- `PR #129` — 선행 PR 브랜치에 stack했으나 본문에 선행 PR 번호 없음
    ```
    model: applying RBLN config to Stable Video Diffusion (base=stable-video-diffusion)
    ```
- `PR #121` — 선행 PR 브랜치에 stack했으나 본문에 선행 PR 번호 없음
    ```
    model: Bugfix cosmos decoder (base=integrations/cosmos)
    ```
- `PR #119` — 선행 PR 브랜치에 stack했으나 본문에 선행 PR 번호 없음
    ```
    model: Cosmos decoder config (base=integrations/cosmos)
    ```
- `PR #116` — 선행 PR 브랜치에 stack했으나 본문에 선행 PR 번호 없음
    ```
    model: add RBLNConfig for cosmos guardrail (base=dk/cosmos)
    ```
- `PR #114` — 선행 PR 브랜치에 stack했으나 본문에 선행 PR 번호 없음
    ```
    model: integrate Cosmos World Foundation Model (base=integrations/cosmos)
    ```
- `PR #111` — 선행 PR 브랜치에 stack했으나 본문에 선행 PR 번호 없음
    ```
    model: Cosmos VAE Decoder (base=dk/cosmos)
    ```
- `PR #107` — 선행 PR 브랜치에 stack했으나 본문에 선행 PR 번호 없음
    ```
    model: Merge common logic (base=add/qwen2_5vl)
    ```
- `PR #106` — 선행 PR 브랜치에 stack했으나 본문에 선행 PR 번호 없음
    ```
    model: Add New models for cosmos guardrails (base=dk/cosmos)
    ```
- `PR #73` — 선행 PR 브랜치에 stack했으나 본문에 선행 PR 번호 없음
    ```
    refactor: [WIP] rbln_config refactor (diffusers) (base=refactor/config_as_class)
    ```
- `PR #52` — 선행 PR 브랜치에 stack했으나 본문에 선행 PR 번호 없음
    ```
    model: add pytest to SVD (base=stable-video-diffusion)
    ```
- `PR #49` — 선행 PR 브랜치에 stack했으나 본문에 선행 PR 번호 없음
    ```
    other: causal paged attn (base=paged_attn_kvshape)
    ```

## PRC-001 (verified) — 위반 21 / 준수 610

PR 제목은 `type(optional scope): description` 형식의 conventional commit을 따르고, type은 PR 템플릿이 정의한 목록(model, performance, refactor, doc, dependency, release, other)에서 변경 성격에 맞게 고르며 모델 버그 수정도 model로 분류한다.

- `PR #32` — type 'fix' 은 허용 목록 밖
    ```
    fix: support connected pipe when optimize_host_memory=False
    ```
- `PR #95` — type 'Other' 은 허용 목록 밖
    ```
    Other(Seq2Seq): fix ruff formating for #68
    ```
- `PR #128` — scope 괄호/형식이 CI headerPattern과 불일치
    ```
    other(runtime)): change default behavior when no npu is available
    ```
- `PR #148` — type 'Model' 은 허용 목록 밖
    ```
    Model: Gemma3 eager attention, not supported yet
    ```
- `PR #149` — GitHub 기본 Revert 제목, type 없음
    ```
    Revert "Model: Gemma3 eager attention, not supported yet (#148)"
    ```
- `PR #150` — type 'Fix' 은 허용 목록 밖
    ```
    Fix: remove debugging code
    ```
- `PR #154` — type 'others' 은 허용 목록 밖
    ```
    others: add pytest for release model
    ```
- `PR #240` — scope 괄호/형식이 CI headerPattern과 불일치
    ```
    model(qwen&llava): add docstring
    ```
- `PR #299` — GitHub 기본 Revert 제목, type 없음
    ```
    Revert "model: Refactor multimodal"
    ```
- `PR #357` — GitHub 기본 Revert 제목, type 없음
    ```
    Revert "other: fix dependency pr title"
    ```
- `PR #427` — 테스트용 PR, type 없음
    ```
    Dev test
    ```
- `PR #428` — 테스트용 PR, type 없음
    ```
    Dev test
    ```
- `PR #429` — 테스트용 PR, type 없음
    ```
    Dev test
    ```
- `PR #430` — 테스트용 PR, type 없음
    ```
    Dev test2
    ```
- `PR #431` — 테스트용 PR, type 없음
    ```
    Dev test3
    ```
- `PR #489` — scope 괄호/형식이 CI headerPattern과 불일치
    ```
    Hotfix/deprecated logics
    ```
- `PR #516` — type 'hotfix' 은 허용 목록 밖
    ```
    hotfix: sync with dev
    ```
- `PR #520` — GitHub 기본 Revert 제목, type 없음
    ```
    Revert "Revert "model: gpt-oss not support bf16 yet (#508)""
    ```
- `PR #522` — scope 괄호/형식이 CI headerPattern과 불일치
    ```
    Dev
    ```
- `PR #619` — type 'chore' 은 허용 목록 밖
    ```
    chore: advance main to dev (SHA parity, no code change)
    ```
- `PR #640` — type 'feat' 은 허용 목록 밖
    ```
    feat: enable tri update op
    ```

## LIFE-002 (interview) — 위반 19 / 준수 48

모델 forward의 입력 인자·순서·반환 타입은 원본 HF 모델 구현과 일치시키고(예: token_type_ids 누락 금지), 사용자가 지정한 입력 이름이 forward 시그니처에 없으면 에러를 raise하며, 입력에 token_type_ids가 있는 경우와 없는 경우를 모두 검증한다.

- `src/optimum/rbln/transformers/models/blip_2/modeling_blip_2.py:242` — HF Blip2QFormerModel.forward takes attention_mask; RBLN omits it
    ```
    def forward(
    ```
- `src/optimum/rbln/transformers/models/colqwen2/modeling_colqwen2.py:122` — inputs_embeds pos 1 vs HF pos 2; affects 1 classes
    ```
    inputs_embeds: torch.FloatTensor | None = None,
    ```
- `src/optimum/rbln/diffusers/models/transformers/transformer_cosmos.py:622` — attention_mask pos 3 vs HF pos 4; affects 1 classes
    ```
    attention_mask: torch.Tensor | None = None,
    ```
- `src/optimum/rbln/transformers/models/exaone4_5/modeling_exaone4_5.py:556` — inputs_embeds pos 1 vs HF pos 2; affects 1 classes
    ```
    inputs_embeds: torch.FloatTensor | None = None,
    ```
- `src/optimum/rbln/transformers/models/exaone4_5/modeling_exaone4_5.py:406` — inputs_embeds pos 1 vs HF pos 2; affects 1 classes
    ```
    inputs_embeds: torch.FloatTensor | None = None,
    ```
- `src/optimum/rbln/transformers/models/decoderonly/modeling_decoderonly.py:881` — inputs_embeds pos 1 vs HF pos 4; affects 15 classes
    ```
    inputs_embeds: torch.Tensor | None = None,
    ```
- `src/optimum/rbln/transformers/models/decoderonly/modeling_decoderonly.py:674` — inputs_embeds pos 1 vs HF pos 3; affects 10 classes
    ```
    inputs_embeds: torch.Tensor | None = None,
    ```
- `src/optimum/rbln/transformers/models/gemma3/modeling_gemma3.py:272` — attention_mask pos 1 vs HF pos 2; affects 1 classes
    ```
    attention_mask: torch.Tensor | None = None,
    ```
- `src/optimum/rbln/transformers/models/gemma4/modeling_gemma4.py:944` — attention_mask pos 1 vs HF pos 4; affects 1 classes
    ```
    attention_mask: torch.Tensor | None = None,
    ```
- `src/optimum/rbln/transformers/models/llava_next/modeling_llava_next.py:472` — attention_mask pos 1 vs HF pos 3; affects 1 classes
    ```
    attention_mask: torch.LongTensor | None = None,
    ```
- `src/optimum/rbln/transformers/models/qwen2_vl/modeling_qwen2_vl.py:619` — inputs_embeds pos 1 vs HF pos 2; affects 1 classes
    ```
    inputs_embeds: torch.FloatTensor | None = None,
    ```
- `src/optimum/rbln/transformers/models/qwen2_vl/modeling_qwen2_vl.py:415` — inputs_embeds pos 1 vs HF pos 2; affects 1 classes
    ```
    inputs_embeds: torch.FloatTensor | None = None,
    ```
- `src/optimum/rbln/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py:740` — inputs_embeds pos 1 vs HF pos 2; affects 1 classes
    ```
    inputs_embeds: torch.FloatTensor | None = None,
    ```
- `src/optimum/rbln/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py:527` — inputs_embeds pos 1 vs HF pos 2; affects 1 classes
    ```
    inputs_embeds: torch.FloatTensor | None = None,
    ```
- `src/optimum/rbln/transformers/models/qwen3_asr/modeling_qwen3_asr.py:276` — inputs_embeds pos 1 vs HF pos 3; affects 1 classes
    ```
    inputs_embeds: torch.Tensor | None = None,
    ```
- `src/optimum/rbln/transformers/models/qwen3_vl/modeling_qwen3_vl.py:886` — inputs_embeds pos 1 vs HF pos 2; affects 2 classes
    ```
    inputs_embeds: torch.FloatTensor | None = None,
    ```
- `src/optimum/rbln/transformers/models/qwen3_vl/modeling_qwen3_vl.py:599` — inputs_embeds pos 1 vs HF pos 2; affects 2 classes
    ```
    inputs_embeds: torch.FloatTensor | None = None,
    ```
- `src/optimum/rbln/transformers/models/qwen3_5/modeling_qwen3_5.py:851` — inputs_embeds pos 1 vs HF pos 2; affects 1 classes
    ```
    inputs_embeds: torch.FloatTensor | None = None,
    ```
- `src/optimum/rbln/transformers/models/qwen3_5/modeling_qwen3_5.py:665` — inputs_embeds pos 1 vs HF pos 2; affects 1 classes
    ```
    inputs_embeds: torch.FloatTensor | None = None,
    ```

## DEP-001 (aspirational) — 위반 8 / 준수 7

deprecated 인자·API는 deprecate 데코레이터로 표시하고 제거 예정 버전을 명시하며, 해당 버전에 도달하면 제거한다. 이전 메이저 버전 전용 기능도 같은 방식으로 deprecate 처리한다.

- `src/optimum/rbln/configuration_utils.py:767` — ad-hoc warning, no decorator, no removal version
    ```
    logger.warning("`optimize_host_memory` is deprecated and will be removed in future versions.")
    ```
- `src/optimum/rbln/configuration_utils.py:814` — property getter deprecation without removal version
    ```
    logger.warning_once("`torch_dtype` is deprecated. Use `dtype` instead.")
    ```
- `src/optimum/rbln/configuration_utils.py:819` — property setter deprecation without removal version
    ```
    logger.warning_once("`torch_dtype` is deprecated. Use `dtype` instead.")
    ```
- `src/optimum/rbln/transformers/models/decoderonly/decoderonly_architecture.py:921` — class deprecation in __init__, no decorator or removal version
    ```
    "DecoderOnlyFlashAttention is deprecated and may not work as expected. Use DecoderOnlyAttention instead."
    ```
- `src/optimum/rbln/transformers/models/gemma4/modeling_gemma4.py:716` — has version but hand-written warning, not decorator
    ```
    "`image_features` dtype differs from `rbln_config.dtype` is deprecated and will be "
    ```
- `src/optimum/rbln/transformers/models/gpt_oss/modeling_gpt_oss.py:105` — kwarg deprecation bypasses deprecate_kwarg, no removal version
    ```
    logger.warning_once("`torch_dtype` is deprecated! Use `dtype` instead!")
    ```
- `src/optimum/rbln/transformers/utils/rbln_quantization.py:85` — kwarg deprecation bypasses deprecate_kwarg, no removal version
    ```
    logger.warning("The `precision` argument is deprecated. Use `weights` and `activations` instead.")
    ```
- `src/optimum/rbln/transformers/utils/rbln_quantization.py:182` — kwarg deprecation bypasses deprecate_kwarg, no removal version
    ```
    logger.warning_once("torch_dtype is deprecated. Use dtype instead.")
    ```

## DOC-011 (verified) — 위반 7 / 준수 1138

Google 스타일 docstring에서 파라미터 선언(이름과 괄호 속 타입)과 Returns 설명은 각각 한 줄에 쓰고, 파라미터 이름은 첫 `:` 앞에서 끝나게 하며, 기본값은 "Defaults to"로 명시한다.

- `src/optimum/rbln/utils/runtime_utils.py:63` — numpy-style 'name : type' parsed as name 'num_chiplets '
    ```
    num_chiplets : int
    ```
- `src/optimum/rbln/utils/runtime_utils.py:65` — numpy-style declaration and 'default=' instead of 'Defaults to'
    ```
    npu : str | None, default=None
    ```
- `src/optimum/rbln/modeling.py:320` — Returns continues on same-indent lines; parsed as 3 return rows
    ```
    If `return_dict=True`, Returns a dictionary-like object (e.g., BaseModelOutput,
    ```
- `src/optimum/rbln/diffusers/pipelines/controlnet/pipeline_controlnet_sd_xl_img2img.py:461` — Returns description on same indent; parsed as 3 return rows
    ```
    [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] if `return_dict` is True, otherwise a `tuple`
    ```
- `src/optimum/rbln/configuration_utils.py:740` — default stated as 'by default' not 'Defaults to 60.'
    ```
    timeout (int | None): The timeout for the runtime in seconds. If it isn't provided, it will be set to 60 by default.
    ```
- `src/optimum/rbln/diffusers/pipelines/controlnet/pipeline_controlnet_sd_xl.py:396` — 'will default to' instead of 'Defaults to' (upstream diffusers text)
    ```
    not specified it will default to `(height, width)`. Part of SDXL's micro-conditioning as explained in
    ```
- `src/optimum/rbln/diffusers/pipelines/controlnet/pipeline_controlnet_sd_xl_img2img.py:421` — 'will default to' instead of 'Defaults to' (upstream diffusers text)
    ```
    not specified it will default to `(height, width)`. Part of SDXL's micro-conditioning as explained in
    ```

## CFG-010 (aspirational) — 위반 6 / 준수 10

설정 키나 장치 지원을 폐지하거나 이름을 바꿀 때는 deprecate 데코레이터로 경고 후 옛 키를 새 키에 재매핑해 기존 동작을 유지하고(생성자 인자와 setattr 경로 포함), 구현 제거는 이후 별도의 breaking change로 한다.

- `src/optimum/rbln/configuration_utils.py:665` — cfg.tensor_parallel_size=2 / cfg.kvcache_metas 대입 시 재매핑 없이 미지 키로 직렬화
    ```
    def __setattr__(self, key, value):
    ```
- `src/optimum/rbln/configuration_utils.py:767` — 데코레이터 대신 수동 경고, 제거 버전 미명시
    ```
    logger.warning("`optimize_host_memory` is deprecated and will be removed in future versions.")
    ```
- `src/optimum/rbln/transformers/utils/rbln_quantization.py:85` — 재매핑은 하나 deprecate_kwarg 미사용 수동 경고
    ```
    logger.warning("The `precision` argument is deprecated. Use `weights` and `activations` instead.")
    ```
- `src/optimum/rbln/transformers/utils/rbln_quantization.py:182` — 수동 kwarg 재매핑·경고, 데코레이터 미사용
    ```
    logger.warning_once("torch_dtype is deprecated. Use dtype instead.")
    ```
- `src/optimum/rbln/transformers/models/gpt_oss/modeling_gpt_oss.py:105` — 수동 kwarg 재매핑·경고, 데코레이터 미사용
    ```
    logger.warning_once("`torch_dtype` is deprecated! Use `dtype` instead!")
    ```
- `src/optimum/rbln/transformers/models/gemma4/modeling_gemma4.py:710` — 구 아티팩트 지원 폐지를 수동 경고로; qwen3_5는 같은 경우 deprecate_method 사용
    ```
    logger.warning_once(
    ```

## CFG-022 (rejected) — 위반 6 / 준수 0

설정 키는 rbln_config와 from_pretrained의 rbln_ 접두사 인자 두 경로로 모두 지정할 수 있게 하되, 둘을 동시에 주면 ValueError를 발생시킨다.

- `src/optimum/rbln/configuration_utils.py:1109` — dict+접두사 동시 지정 시 오류 없이 병합, 접두사 값이 우선
    ```
    rbln_config.update(rbln_kwargs)
    ```
- `src/optimum/rbln/configuration_utils.py:1115` — 인스턴스+접두사 동시 지정 시 setattr로 덮어씀
    ```
    setattr(rbln_config, key, value)
    ```
- `src/optimum/rbln/configuration_utils.py:1044` — 로드 시 두 경로 병합; #702 이후 접두사 우선(main은 dict 우선)
    ```
    if f"rbln_{key}" not in kwargs:
    ```
- `src/optimum/rbln/modeling_base.py:247` — 모델 로드 경로는 dict가 접두사를 덮어씀, 오류 없음
    ```
    kwargs.update(rbln_config_as_kwargs)
    ```
- `src/optimum/rbln/configuration_utils.py:1207` — convert_rbln_config_dict도 병합(dev에선 호출처 없음)
    ```
    rbln_config.update(rbln_kwargs)
    ```
- `tests/test_config.py:144` — 접두사 인자가 rbln_config를 덮어쓰는 우선순위를 테스트로 고정
    ```
    def test_config_priority(model_id):
    ```

## STY-013 (verified) — 위반 6 / 준수 56

로그와 경고는 built-in logging/warnings 대신 optimum-rbln의 `utils.logging`을 사용한다.

- `src/optimum/rbln/transformers/models/time_series_transformer/modeling_time_series_transformer.py:46` — stdlib 로거 생성; get_logger 미사용
    ```
    logger = logging.getLogger(__name__)
    ```
- `src/optimum/rbln/transformers/models/auto/auto_factory.py:181` — UserWarning을 built-in warnings로 발행
    ```
    warnings.warn(
    ```
- `src/optimum/rbln/utils/import_utils.py:149` — ImportWarning을 built-in warnings로 발행
    ```
    warnings.warn(
    ```
- `src/optimum/rbln/utils/import_utils.py:162` — ImportWarning을 built-in warnings로 발행
    ```
    warnings.warn(
    ```
- `src/optimum/rbln/utils/import_utils.py:179` — ImportWarning을 built-in warnings로 발행
    ```
    warnings.warn(
    ```
- `src/optimum/rbln/utils/import_utils.py:187` — ImportWarning을 built-in warnings로 발행
    ```
    warnings.warn(
    ```

## PRC-004 (aspirational) — 위반 5 / 준수 8

기존 동작의 호환성을 깨는 변경은 PR 본문의 Type of Change에서 Breaking change로 표시한다.

- `PR #645` — refactor: 제거 창이 지난 deprecated API 일괄 제거, 유형 체크 없음
    ```
    refactor: remove deprecations past their removal window for v0.11.1
    ```
- `PR #271` — "Remove legacy apis"인데 Other로 체크
    ```
    other: Remove legacy apis
    ```
- `PR #605` — image_prefill_chunk_size 기본 버킷팅 제거(기본 동작 변경), Code refactoring 체크
    ```
    refactor: remove default bucketing for image_prefill_chunk_size & max_soft_tokens
    ```
- `PR #128` — NPU 없을 때 create_runtimes 기본 동작 변경, Bug fix 체크
    ```
    other(runtime)): change default behavior when no npu is available
    ```
- `PR #714` — 본문에 "Behavior change": from_model 원본 HF 모델 forward 불가, Breaking 미체크
    ```
    model: moe drop redundant expert weight copies and release HF fused gate_up_proj
    ```

## SUB-008 (aspirational) — 위반 4 / 준수 7

HF가 사용하는 dtype 동작(예: fp32로 계산하는 비전 회전 위치 임베딩 테이블, 모듈 입력 dtype)을 그대로 따르며, 활성 dtype으로 반올림하거나 암묵적 승격에 맡기지 않고 명시적으로 선언·캐스트한다.

- `src/optimum/rbln/transformers/models/qwen2_vl/modeling_qwen2_vl.py:192` — HF는 fp32 유지, 활성 dtype으로 반올림(입력도 rbln_config.dtype 선언)
    ```
    torch.cat((cos, cos), dim=-1).to(self.rbln_config.dtype),
    ```
- `src/optimum/rbln/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py:287` — HF는 fp32 유지, 활성 dtype으로 반올림
    ```
    torch.cat((cos, cos), dim=-1).to(self.rbln_config.dtype),
    ```
- `src/optimum/rbln/transformers/models/qwen3_vl/modeling_qwen3_vl.py:195` — HF는 fp32 유지, 활성 dtype으로 반올림(qwen3_vl_moe도 상속)
    ```
    torch.cat((cos, cos), dim=-1).to(self.rbln_config.dtype),
    ```
- `src/optimum/rbln/transformers/models/exaone4_5/modeling_exaone4_5.py:269` — HF는 fp32 유지, 활성 dtype으로 반올림
    ```
    torch.cat((cos, cos), dim=-1).to(self.rbln_config.dtype),
    ```

## STY-018 (verified) — 위반 4 / 준수 627

머지 전에 PR 변경 코드 전체에 `ruff format`과 `ruff check`를 통과시키고 타입 검사와 독스트링 검사도 통과시킨다.

- `PR #821` — #809·#813이 새 ruff 규칙 검사 없이 dev에 머지됨
    ```
    other: fix the syntax error and lint failures left on dev by #733
    ```
- `PR #665` — 머지된 코드의 lint 실패를 후속 PR로 수정
    ```
    other(model): fix lint in modeling modernbert
    ```
- `PR #95` — #68 머지 후 ruff format 누락을 후속 수정
    ```
    Other(Seq2Seq): fix ruff formating for #68
    ```
- `PR #467` — docstring 검사 실패를 후속 PR로 수정
    ```
    other: fix docstring fail cases
    ```

## STY-011 (verified) — 위반 4 / 준수 1705

함수 시그니처에는 실제로 전달되는 객체와 일치하는 타입 힌트를 명시하고, Python 3.10 이상의 네이티브 문법(`X | None`, `list`)을 쓰며 typing의 Optional·List 별칭과 암묵적 Optional은 쓰지 않고, 이를 위해 `from __future__ import annotations`를 일괄 추가하지 않는다.

- `src/optimum/rbln/diffusers/pipelines/cosmos/pipeline_cosmos2_5_predict.py:52` — 암묵적 Optional: 기본값 None인데 `| None` 없음
    ```
    safety_checker: RBLNCosmosSafetyChecker = None,
    ```
- `src/optimum/rbln/diffusers/pipelines/cosmos/pipeline_cosmos2_5_transfer.py:65` — 암묵적 Optional: 기본값 None인데 `| None` 없음
    ```
    safety_checker: RBLNCosmosSafetyChecker = None,
    ```
- `src/optimum/rbln/diffusers/pipelines/cosmos/pipeline_cosmos2_text2image.py:49` — 암묵적 Optional: 기본값 None인데 `| None` 없음
    ```
    safety_checker: RBLNCosmosSafetyChecker = None,
    ```
- `src/optimum/rbln/diffusers/pipelines/cosmos/pipeline_cosmos2_video2world.py:49` — 암묵적 Optional: 기본값 None인데 `| None` 없음
    ```
    safety_checker: RBLNCosmosSafetyChecker = None,
    ```

## ATT-006 (interview) — 위반 3 / 준수 4

정적 캐시의 마지막 축과 인코더 시퀀스 길이는 디바이스·컴파일러가 요구하는 단위(예: 64의 배수)에 맞춰 설계하거나 올림한다.

- `src/optimum/rbln/transformers/models/whisper/modeling_whisper.py:284` — 인코더 길이 1500(64 배수 아님)을 올림 없이 사용
    ```
    rbln_config.enc_max_seq_len = model_config.max_source_positions
    ```
- `src/optimum/rbln/transformers/models/seq2seq/modeling_seq2seq.py:228` — 인코더 길이를 64로 올림하지 않음
    ```
    enc_max_seq_len = enc_max_seq_len or tokenizer.model_max_length
    ```
- `src/optimum/rbln/transformers/models/time_series_transformer/modeling_time_series_transformer.py:283` — 정적 cross-KV 캐시 마지막 축이 head_dim(13), 64 배수 아님
    ```
    model_config.d_model // model_config.decoder_attention_heads,  # 13
    ```

## DEP-008 (verified) — 위반 3 / 준수 10

모델 구현에 필요한 새 외부 의존성은 pyproject.toml에 명시하되, 필수 기능이 아닌 패키지는 기본 의존성에 넣지 않는다.

- `src/optimum/rbln/diffusers/pipelines/cosmos/cosmos_guardrail.py:34` — not declared in pyproject (guarded by is_cosmos_guardrail_available)
    ```
    from cosmos_guardrail import CosmosSafetyChecker
    ```
- `src/optimum/rbln/diffusers/pipelines/cosmos/cosmos_guardrail.py:35` — not declared in pyproject (guarded import)
    ```
    from cosmos_guardrail.cosmos_guardrail import (
    ```
- `src/optimum/rbln/diffusers/pipelines/cosmos/cosmos_guardrail.py:47` — transitive of cosmos_guardrail, undeclared (guarded import)
    ```
    from retinaface.data import cfg_re50
    ```

## API-014 (verified) — 위반 3 / 준수 305

공개 객체는 정의한 모듈의 `__all__`에 등록하고 패키지 `__init__.py`는 고정 템플릿으로 두며, 새 모델은 패키지 최상위 export와 모델 레지스트리에 함께 등록한다.

- `src/optimum/rbln/modeling.py:398` — lists RBLNBaseModel, which is defined in modeling_base.py (re-export)
    ```
    __all__ = [
    ```
- `src/optimum/rbln/transformers/models/seq2seq/__init__.py:15` — eager import, not the fixed lazy template
    ```
    from .configuration_seq2seq import RBLNModelForSeq2SeqLMConfig
    ```
- `src/optimum/rbln/diffusers/__init__.py:23` — template plus diffusers loader side-effect registration
    ```
    LOADABLE_CLASSES["optimum.rbln"] = {
    ```

## TEST-019 (verified) — 위반 3 / 준수 23

테스트 클래스는 기반 클래스나 클래스가 소유한 공유 설정 dict를 in-place로 수정하지 않고 새 dict를 만들거나 자기 속성으로 재바인딩하며, from_pretrained를 직접 호출할 때는 클래스의 HF_CONFIG_KWARGS를 함께 넘긴다.

- `tests/test_transformers.py:114` — compiles HF_MODEL_ID without HF_CONFIG_KWARGS (dict empty, harmless today)
    ```
    _ = self.RBLN_CLASS.from_pretrained(
    ```
- `tests/test_llm.py:742` — LlavaNext revision pin dropped; call expects ValueError before compile
    ```
    _ = self.RBLN_CLASS.from_pretrained(model_id=self.HF_MODEL_ID, **rbln_class_kwargs)
    ```
- `tests/test_diffusers.py:291` — compiles HF_MODEL_ID without HF_CONFIG_KWARGS (dict empty, harmless today)
    ```
    _ = self.RBLN_CLASS.from_pretrained(model_id=self.HF_MODEL_ID, **rbln_class_kwargs_copy)
    ```

## DEP-011 (verified) — 위반 2 / 준수 25

torch.load는 weights_only를 명시적으로 지정한다.

- `src/optimum/rbln/transformers/models/decoderonly/lora_architecture.py:79` — torch.load without explicit weights_only
    ```
    ("adapter_model.bin", lambda p: torch.load(p, map_location="cpu")),
    ```
- `src/optimum/rbln/transformers/models/decoderonly/lora_architecture.py:80` — torch.load without explicit weights_only
    ```
    ("pytorch_model.bin", lambda p: torch.load(p, map_location="cpu")),
    ```

## PRC-005 (rejected) — 위반 2 / 준수 0

릴리스 PR은 dev를 main으로 병합하는 PR로 만들고, base를 main으로 두며, 제목을 `release: vX.Y.Z` 형식으로 쓰고 Type of Change에서 Release를 체크한다.

- `release v0.11.2 (2026-08-28)` — dev에서 release 브랜치를 잘라 태그, dev→main PR 없음
    ```
    v0.11.2 target=release-v0.11.2; release-v0.11.2 == 8861ed75 (dev 조상, cherry-pick 0)
    ```
- `release v0.12.0 (2026-10-16)` — release 브랜치 + cherry-pick PR, 제목 release: 아님
    ```
    v0.12.0 target=release-v0.12.0; PRs #786 #789 #805 base=release-v0.12.0 (squash cherry-pick)
    ```

## STY-015 (verified) — 위반 2 / 준수 64

CI 워크플로우에서 시크릿은 `secrets` 컨텍스트로 전달하고, 내부 엔드포인트·클러스터 고유 경로나 주소는 저장소에 평문으로 적지 않고 CI 변수로 주입하며, 필수 변수가 비어 있으면 빈 값으로 진행하지 않고 업로드 단계에서 실패시킨다.

- `scripts/ci/slack-report.py:22` — 내부 API 엔드포인트 평문; GHA는 같은 값을 secrets.OBEDIENTS_API_URL로 받음
    ```
    API = "https://obedients-api.k8s.rebellions.in/v2/organizations"
    ```
- `scripts/ci/rebel-compiler-override.sh:12` — 엔드포인트는 변수로 받지만 어노테이션 문구에 내부 호스트 평문
    ```
      "rebel-compiler overridden to \`${REBEL_COMPILER_VERSION}\` (pypi.rebellions.in/simple)" || true
    ```

## STY-004 (verified) — 위반 2 / 준수 2154

사용하지 않는 코드(미사용 변수·import·메서드·클래스, 부모와 동일한 override, 레거시 코드, 효과 없는 컴파일 옵션, 오래되었거나 중복된 CI 워크플로)는 남기지 않고 제거한다.

- `src/optimum/rbln/diffusers/models/transformers/transformer_sd3.py:78` — super().__post_init__(**kwargs)만 호출하는 부모와 동일한 override
    ```
    def __post_init__(self, **kwargs):
    ```
- `src/optimum/rbln/transformers/modeling_attention_utils.py:425` — 정의 외 참조가 src/tests/examples 어디에도 없는 private classmethod
    ```
    def _required_memory_at(
    ```

## DOC-012 (verified) — 위반 2 / 준수 42

docstring의 문서 링크는 상대 경로가 아니라 절대 URL로 쓰고, SDK 문서 빌드 결과에 실제로 존재하는 경로만 가리킨다.

- `src/optimum/rbln/transformers/models/t5/modeling_t5.py:54` — relative docs link; breaks validate_docstrings, file excluded from gate
    ```
    based on the model size. Please refer to the [Optimum RBLN Overview](../../../index.md) for guidance on choosing the
    ```
- `src/optimum/rbln/transformers/models/t5/modeling_t5.py:99` — relative docs link; breaks validate_docstrings, file excluded from gate
    ```
    based on the model size. Please refer to the [Optimum RBLN Overview](../../../index.md) for guidance on choosing the
    ```

## LIFE-020 (verified) — 위반 1 / 준수 118

오버라이드되는 메서드(get_compiled_model, get_input_info 등)는 공통 계약을 유지하여, 개별 값을 여러 인자로 받지 않고 모델 설정 객체와 컴파일 설정 객체를 받으며 쓰지 않는 키워드 인자도 받아서 무시한다.

- `src/optimum/rbln/diffusers/pipelines/controlnet/multicontrolnet.py:81` — no-op override drops (preprocessors, model, model_config, rbln_config) contract
    ```
    def _update_rbln_config(cls, **rbln_config_kwargs):
    ```

## API-015 (verified) — 위반 1 / 준수 125

내부 모듈 경로를 옮기거나 모델 클래스를 배치할 때 호환용 shim 모듈이나 별도 alias 파일을 남기지 않고, 공개 API의 재export만 유지한다.

- `src/optimum/rbln/diffusers/pipelines/controlnet/multicontrolnet.py:30` — HF moved it to diffusers.models.controlnets.multicontrolnet; RBLN copy stayed
    ```
    class RBLNMultiControlNetModel(RBLNModel):
    ```

## API-011 (verified) — 위반 1 / 준수 132

새 모델 클래스와 파이프라인용 diffusers 구성요소 클래스는 기존 RBLN 접두사 패턴(RBLN*For* 등)을 따라 이름 짓고 노출한다.

- `src/optimum/rbln/transformers/models/grounding_dino/modeling_grounding_dino.py:61` — no HF GroundingDinoTextModel; wraps BertModel (_hf_class = BertModel)
    ```
    class RBLNGroundingDinoTextModel(RBLNBertModel):
    ```

## STY-010 (verified) — 위반 1 / 준수 103

타입 annotation에만 쓰이는 심볼이 순환 참조를 만들거나 외부 패키지 타입일 때만 import를 `TYPE_CHECKING` 블록으로 옮기고 annotation은 문자열로 감싸며, 같은 이름을 런타임 import와 `TYPE_CHECKING` 블록에 중복 import하지 않고 TC001~TC003 규칙으로 일괄 이동하지 않는다.

- `src/optimum/rbln/utils/submodule.py:119` — TYPE_CHECKING 이름을 따옴표 없이 사용(지역 변수라 런타임 무해)
    ```
    submodule_cls: type[RBLNModel] = get_rbln_model_cls(submodule_rbln_config.rbln_model_cls_name)
    ```
