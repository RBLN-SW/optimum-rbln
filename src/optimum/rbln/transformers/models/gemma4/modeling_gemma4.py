# Copyright 2026 Rebellions Inc. All rights reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import importlib
import inspect
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional, Tuple, Union

import torch
from transformers import (
    AutoModel,
    AutoModelForImageTextToText,
    Gemma4ForConditionalGeneration,
    PretrainedConfig,
    PreTrainedModel,
)
from transformers.initialization import no_init_weights
from transformers.modeling_outputs import BaseModelOutputWithPooling

from ....configuration_utils import RBLNCompileConfig, RBLNModelConfig
from ....modeling import RBLNModel
from ....utils.logging import get_logger
from ...modeling_outputs import RBLNDecoderOnlyOutput
from ...utils.rbln_runtime_wrapper import LoopProcessor
from ..decoderonly.decoderonly_runtime_utils import RBLNPageTableManager
from ..decoderonly.generation_decoderonly import RBLNDecoderOnlyGenerationMixin
from ..decoderonly.modeling_decoderonly import RBLNDecoderOnlyModelForCausalLM
from .configuration_gemma4 import (
    RBLNGemma4ForCausalLMConfig,
    RBLNGemma4VisionModelConfig,
)
from .gemma4_architecture import (
    Gemma4ForCausalLMWrapper,
    Gemma4VisionModelWrapper,
)
from .gemma4_runtime_utils import RBLNGemma4RuntimeModel


logger = get_logger(__name__)


if TYPE_CHECKING:
    from transformers import AutoFeatureExtractor, AutoProcessor, AutoTokenizer


class LoopVisionTower(LoopProcessor):
    def __init__(self, vision_tower: "RBLNModel"):
        super().__init__(model=vision_tower)

    def _get_batch_size(self, pixel_values, **kwargs):
        return pixel_values.shape[0]

    def _prepare_inputs_for_iteration(self, index, common_inputs, pixel_values, pixel_position_ids, **kwargs):
        pixel_values_item = pixel_values[index : index + 1]
        pixel_position_ids_item = pixel_position_ids[index : index + 1]
        out_buffer = [tensor[index : index + 1] for tensor in kwargs["out"]]
        return ([pixel_values_item, pixel_position_ids_item], {"out": out_buffer})

    def _process_outputs(self, outputs: list, **kwargs) -> "BaseModelOutputWithPooling":
        output = kwargs["out"]
        return BaseModelOutputWithPooling(last_hidden_state=output[0])


class LoopProjector(LoopProcessor):
    def __init__(self, multi_modal_projector: "RBLNModel"):
        super().__init__(model=multi_modal_projector)

    def _get_batch_size(self, image_feature, **kwargs):
        return image_feature.shape[0]

    def _prepare_inputs_for_iteration(self, index, common_inputs, image_feature, **kwargs):
        image_feature_item = image_feature[index : index + 1]
        out_buffer = [tensor[index : index + 1] for tensor in kwargs["out"]]
        return ([image_feature_item], {"out": out_buffer})

    def _process_outputs(self, outputs: list, **kwargs):
        output = kwargs["out"]
        return output[0]


class RBLNGemma4VisionModel(RBLNModel):
    """
    Gemma4 vision encoder model optimized for RBLN NPU.

    This model inherits from [`RBLNModel`]. It implements the methods to convert and run
    pre-trained transformers based Gemma4 vision encoder model on RBLN devices by:

    - transferring the checkpoint weights of the original into an optimized RBLN graph,
    - compiling the resulting graph using the RBLN compiler.

    `patch_embedder` (per-patch linear projection + 2D position embedding lookup) and
    `rotary_emb` (multidimensional cos/sin tables) both run on the host (CPU). `patch_embedder`
    weights are persisted as a saved torch artifact; `rotary_emb` is recreated from config since
    its `inv_freq` buffer is non-persistent. The compiled `Gemma4VisionModelWrapper`
    (encoder-layers -> pooler) takes the host-computed `inputs_embeds`, `pixel_position_ids`,
    and `(cos, sin)` rotary tables as inputs. Padding within `max_patches` is handled by the
    encoder via `pixel_position_ids == -1` markers.
    """

    auto_model_class = AutoModel
    _supports_non_fp32 = True

    @classmethod
    def _wrap_model_if_needed(cls, model: PreTrainedModel, rbln_config: RBLNGemma4VisionModelConfig):
        return Gemma4VisionModelWrapper(model, num_soft_tokens=rbln_config.max_soft_tokens).eval()

    @classmethod
    def _update_rbln_config(
        cls,
        preprocessors: Optional[Union["AutoFeatureExtractor", "AutoProcessor", "AutoTokenizer"]],
        model: Optional[PreTrainedModel] = None,
        model_config: Optional[PretrainedConfig] = None,
        rbln_config: Optional[RBLNGemma4VisionModelConfig] = None,
    ) -> RBLNGemma4VisionModelConfig:
        if rbln_config.pooling_kernel_size is None:
            rbln_config.pooling_kernel_size = model_config.pooling_kernel_size
        if rbln_config.patch_size is None:
            rbln_config.patch_size = model_config.patch_size
        if rbln_config.max_soft_tokens is None:
            rbln_config.max_soft_tokens = 280

        max_patches = rbln_config.max_patches
        hidden_size = model_config.hidden_size
        head_dim = getattr(model_config, "head_dim", None) or (
            hidden_size // model_config.num_attention_heads
        )

        input_info = [
            ("inputs_embeds", [rbln_config.batch_size, max_patches, hidden_size], "float32"),
            ("pixel_position_ids", [rbln_config.batch_size, max_patches, 2], "int64"),
            ("attn_mask", [rbln_config.batch_size, 1, max_patches, max_patches], "float32"),
            ("padding_positions", [rbln_config.batch_size, max_patches], "bool"),
            ("cos", [rbln_config.batch_size, max_patches, head_dim], rbln_config.dtype),
            ("sin", [rbln_config.batch_size, max_patches, head_dim], rbln_config.dtype),
        ]
        rbln_compile_config = RBLNCompileConfig(input_info=input_info)
        rbln_config.set_compile_cfgs([rbln_compile_config])
        return rbln_config

    @classmethod
    def save_torch_artifacts(
        cls,
        model: PreTrainedModel,
        save_dir_path: Path,
        subfolder: str,
        rbln_config: RBLNGemma4VisionModelConfig,
    ):
        torch.save(
            {"patch_embedder": model.patch_embedder.state_dict()},
            save_dir_path / subfolder / "torch_artifacts.pth",
        )

    def _create_patch_embedder(self) -> torch.nn.Module:
        from transformers.models.gemma4.modeling_gemma4 import Gemma4VisionPatchEmbedder

        with no_init_weights():
            patch_embedder = Gemma4VisionPatchEmbedder(self.config)
        return patch_embedder

    def _create_rotary_emb(self) -> torch.nn.Module:
        from transformers.models.gemma4.modeling_gemma4 import Gemma4VisionRotaryEmbedding

        rotary_emb = Gemma4VisionRotaryEmbedding(self.config)
        return rotary_emb

    def __post_init__(self, **kwargs):
        artifacts_path = self.model_save_dir / self.subfolder / "torch_artifacts.pth"
        artifacts = torch.load(artifacts_path, weights_only=False) if artifacts_path.exists() else {}
        if "patch_embedder" in artifacts:
            self.patch_embedder = self._create_patch_embedder()
            self.patch_embedder.load_state_dict(artifacts["patch_embedder"])
            self.patch_embedder.eval()
        else:
            self.patch_embedder = None
        # rotary_emb has no learnable weights — recreate purely from config.
        self.rotary_emb = self._create_rotary_emb().eval()
        return super().__post_init__(**kwargs)

    def forward(
        self,
        pixel_values: torch.Tensor,
        pixel_position_ids: torch.Tensor,
        **kwargs: Any,
    ) -> BaseModelOutputWithPooling:
        if self.patch_embedder is None:
            raise RuntimeError(
                "patch_embedder was not loaded from torch_artifacts.pth. "
                "Re-export the model so that patch_embedder weights are saved as a host-side artifact."
            )
        padding_positions = (pixel_position_ids == -1).all(dim=-1)  # (B, max_patches), True=pad
        with torch.no_grad():
            inputs_embeds = self.patch_embedder(pixel_values, pixel_position_ids, padding_positions)
            cos, sin = self.rotary_emb(inputs_embeds, pixel_position_ids)
        cos = cos.to(self.rbln_config.dtype)
        sin = sin.to(self.rbln_config.dtype)

        valid = (~padding_positions)[:, None, :].to(inputs_embeds.dtype)  # (B, 1, S)
        attn_mask = valid[..., None, :] * valid[..., None]  # (B, 1, S, S)
        attn_mask = (1.0 - attn_mask) * torch.finfo(inputs_embeds.dtype).min

        return super().forward(
            inputs_embeds, pixel_position_ids, attn_mask, padding_positions, cos, sin, **kwargs
        )

    def _prepare_output(self, output, return_dict):
        last_hidden_state = output[0] if isinstance(output, (tuple, list)) else output
        if not return_dict:
            return (last_hidden_state,)
        return BaseModelOutputWithPooling(last_hidden_state=last_hidden_state)


class RBLNGemma4ForCausalLM(RBLNDecoderOnlyModelForCausalLM):
    """
    Gemma4 model with a causal language modeling head optimized for RBLN NPU.

    This model inherits from [`RBLNModel`]. It implements the methods to convert and run
    pre-trained transformers based Gemma4ForCausalLM model on RBLN devices by:

    - transferring the checkpoint weights of the original into an optimized RBLN graph,
    - compiling the resulting graph using the RBLN compiler.

    Compared to the base decoder-only class, this class additionally saves and loads
    `embed_tokens_per_layer` (the auxiliary per-layer-input embedding) alongside `embed_tokens`
    as a torch artifact, and wires the resulting `per_layer_inputs` tensor through the runtime
    to match the `Gemma4ForCausalLMWrapper` argument order.
    """

    _decoder_wrapper_cls = Gemma4ForCausalLMWrapper
    _supports_non_fp32 = False

    @classmethod
    def get_input_info(
        cls,
        batch_size: int,
        query_length: int,
        rbln_config: RBLNGemma4ForCausalLMConfig,
        model_config: PretrainedConfig,
    ):
        base_info = super().get_input_info(
            batch_size=batch_size,
            query_length=query_length,
            rbln_config=rbln_config,
            model_config=model_config,
        )
        if getattr(model_config, "hidden_size_per_layer_input", 0):
            per_layer_entry = (
                "per_layer_inputs",
                [
                    batch_size,
                    query_length,
                    model_config.num_hidden_layers,
                    model_config.hidden_size_per_layer_input,
                ],
                rbln_config.dtype,
            )
            return [base_info[0], per_layer_entry, *base_info[1:]]
        return base_info

    @classmethod
    def save_torch_artifacts(
        cls,
        model: PreTrainedModel,
        save_dir_path: Path,
        subfolder: str,
        rbln_config: RBLNGemma4ForCausalLMConfig,
    ):
        save_dict = {}
        if rbln_config.use_inputs_embeds:
            save_dict["embed_tokens"] = model.get_input_embeddings().state_dict()
            inner = getattr(model, "model", None)
            embed_per_layer = getattr(inner, "embed_tokens_per_layer", None) if inner is not None else None
            if embed_per_layer is not None:
                save_dict["embed_tokens_per_layer"] = embed_per_layer.state_dict()
        if save_dict:
            torch.save(save_dict, save_dir_path / subfolder / "torch_artifacts.pth")


    def _create_per_layer_embedding_layer(self):
        from transformers.models.gemma4.modeling_gemma4 import Gemma4TextScaledWordEmbedding

        with no_init_weights():
            embed_per_layer = Gemma4TextScaledWordEmbedding(
                self.config.vocab_size_per_layer_input,
                self.config.num_hidden_layers * self.config.hidden_size_per_layer_input,
                self.config.pad_token_id,
                embed_scale=self.config.hidden_size_per_layer_input**0.5,
            )
        return embed_per_layer

    def __post_init__(self, **kwargs):
        artifacts_path = self.model_save_dir / self.subfolder / "torch_artifacts.pth"
        artifacts = torch.load(artifacts_path, weights_only=False) if artifacts_path.exists() else {}

        if self.rbln_config.use_inputs_embeds and "embed_tokens" in artifacts:
            self.embed_tokens = self._create_embedding_layer()
            self.embed_tokens.load_state_dict(artifacts["embed_tokens"])
        else:
            self.embed_tokens = None

        if getattr(self.config, "hidden_size_per_layer_input", 0) and "embed_tokens_per_layer" in artifacts:
            self.embed_tokens_per_layer = self._create_per_layer_embedding_layer()
            self.embed_tokens_per_layer.load_state_dict(artifacts["embed_tokens_per_layer"])
        else:
            self.embed_tokens_per_layer = None

        self.setup_runtime()

    def setup_runtime(self):
        dec_attn_mask = torch.zeros(self.rbln_config.batch_size, self.rbln_config.max_seq_len, dtype=torch.float32)
        page_table_manager = RBLNPageTableManager(self.rbln_config)

        common_kwargs = {
            "main_input_name": "inputs_embeds" if self.rbln_config.use_inputs_embeds else "input_ids",
            "embed_tokens": self.embed_tokens,
            "embed_tokens_per_layer": self.embed_tokens_per_layer,
            "num_hidden_layers": self.config.num_hidden_layers,
            "hidden_size_per_layer_input": getattr(self.config, "hidden_size_per_layer_input", 0),
            "dec_attn_mask": dec_attn_mask,
            "page_table_manager": page_table_manager,
            "rbln_config": self.rbln_config,
            "config": self.config,
        }

        self.prefill_decoder = RBLNGemma4RuntimeModel(
            runtime=self.model[0],
            phase="prefill",
            batch_size=self.rbln_config.batch_size,
            logits_last_dim=self.logits_last_dim,
            **common_kwargs,
        )

        self.decoders: Dict[int, RBLNGemma4RuntimeModel] = {}
        if self.can_generate():
            for i, batch_size in enumerate(self.rbln_config.decoder_batch_sizes):
                self.decoders[batch_size] = RBLNGemma4RuntimeModel(
                    runtime=self.model[i + self.rbln_config.decoder_runtime_idx],
                    phase="decode",
                    batch_size=batch_size,
                    **common_kwargs,
                )
            self.decoder = self.decoders[self.rbln_config.batch_size]

    def _create_embedding_layer(self):
        from transformers.models.gemma4.modeling_gemma4 import Gemma4TextScaledWordEmbedding

        with no_init_weights():
            embed_tokens = Gemma4TextScaledWordEmbedding(
                self.config.vocab_size,
                self.config.hidden_size,
                self.config.pad_token_id,
                embed_scale=self.config.hidden_size**0.5,
            )
        return embed_tokens


class RBLNGemma4ForConditionalGeneration(RBLNModel, RBLNDecoderOnlyGenerationMixin):
    """
    Gemma4 model for image-text-to-text generation optimized for RBLN NPU.

    This model inherits from [`RBLNModel`]. It implements the methods to convert and run
    pre-trained transformers based Gemma4ForConditionalGeneration model on RBLN devices by:

    - transferring the checkpoint weights of the original into an optimized RBLN graph,
    - compiling the resulting graph using the RBLN compiler.

    This class compiles the `embed_vision` multimodal projector (vision soft tokens -> language-model
    embedding space) as its own graph. Vision encoding and language modeling are compiled as
    submodules: `vision_tower` ([`RBLNGemma4VisionModel`], batch_size=1, looped over images at
    runtime) and `language_model` ([`RBLNGemma4ForCausalLM`]).
    """

    auto_model_class = AutoModelForImageTextToText
    _rbln_submodule_prefix = "model"
    _rbln_submodules = [
        {"name": "vision_tower"},
        {"name": "language_model"},
    ]

    def __getattr__(self, __name: str) -> Any:
        def redirect(func):
            return lambda *pargs, **kwargs: func(self, *pargs, **kwargs)

        val = getattr(Gemma4ForConditionalGeneration, __name)
        if isinstance(val, Callable) and "self" in set(inspect.signature(val).parameters):
            return redirect(val)
        return val

    def can_generate(self):
        return True

    @classmethod
    def _reconstruct_model_if_needed(cls, model: "PreTrainedModel"):
        with no_init_weights():
            model_cls_name = model.model.language_model.__class__.__name__
            causal_model_cls_name = model_cls_name.replace("TextModel", "ForCausalLM")
            causal_model_cls = getattr(importlib.import_module("transformers"), causal_model_cls_name)
            new_language_model = causal_model_cls(model.model.language_model.config)

        new_language_model.lm_head = model.lm_head
        new_language_model.model = model.model.language_model
        model.model.language_model = new_language_model
        del model.lm_head

        return model

    def __post_init__(self, **kwargs):
        self.vision_tower = LoopVisionTower(self.rbln_submodules[0])
        self.language_model = self.rbln_submodules[1]
        self.embed_vision = LoopProjector(self.model[0])

        self.vocab_size = self.config.text_config.vocab_size

        self.pad_token_id = (
            self.config.text_config.pad_token_id if self.config.text_config.pad_token_id is not None else -1
        )
        return super().__post_init__(**kwargs)

    def get_attn_impl(self) -> str:
        return self.rbln_config.language_model.attn_impl

    def get_kvcache_num_blocks(self) -> int:
        return self.rbln_config.language_model.kvcache_num_blocks

    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    @classmethod
    def _wrap_model_if_needed(cls, model: "PreTrainedModel", rbln_config: RBLNModelConfig):
        return model.model.embed_vision

    @classmethod
    def _update_rbln_config(
        cls,
        preprocessors: Optional[Union["AutoFeatureExtractor", "AutoProcessor", "AutoTokenizer"]],
        model: Optional["PreTrainedModel"] = None,
        model_config: Optional["PretrainedConfig"] = None,
        rbln_config: Optional[RBLNModelConfig] = None,
    ) -> RBLNModelConfig:
        vision_cfg = model_config.vision_config
        vt_cfg = rbln_config.vision_tower
        max_soft_tokens = getattr(vt_cfg, "max_soft_tokens", None) if vt_cfg is not None else None
        if isinstance(vt_cfg, dict):
            max_soft_tokens = vt_cfg.get("max_soft_tokens", None)
        max_soft_tokens = max_soft_tokens or 280
        vision_hidden_size = vision_cfg.hidden_size

        compile_cfgs = [
            RBLNCompileConfig(
                compiled_model_name="embed_vision",
                input_info=[
                    (
                        "image_features",
                        [1, max_soft_tokens, vision_hidden_size],
                        "float32",
                    )
                ],
            )
        ]

        rbln_config.set_compile_cfgs(compile_cfgs)
        return rbln_config

    def prepare_inputs_for_generation(
        self,
        input_ids,
        inputs_embeds=None,
        pixel_values=None,
        pixel_position_ids=None,
        attention_mask=None,
        generate_idx=None,
        padded_cache_lengths=None,
        token_type_ids=None,
        mm_token_type_ids=None,
        **kwargs,
    ):
        is_prefill_phase = generate_idx is None
        model_inputs = self.language_model.prepare_inputs_for_generation(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            generate_idx=generate_idx,
            attention_mask=attention_mask,
            padded_cache_lengths=padded_cache_lengths,
            **kwargs,
        )
        if is_prefill_phase:
            model_inputs.update(
                {
                    "pixel_values": pixel_values,
                    "pixel_position_ids": pixel_position_ids,
                    "token_type_ids": token_type_ids,
                    "mm_token_type_ids": mm_token_type_ids,
                }
            )
        model_inputs["attention_mask"] = attention_mask
        return model_inputs

    def _update_model_kwargs_for_generation(
        self,
        outputs: RBLNDecoderOnlyOutput,
        model_kwargs: Dict[str, Any],
        **kwargs,
    ) -> Dict[str, Any]:
        model_kwargs["generate_idx"] = outputs.generate_idx
        model_kwargs["padded_cache_lengths"] = outputs.padded_cache_lengths
        return model_kwargs

    def get_image_features(
        self,
        pixel_values: torch.Tensor,
        pixel_position_ids: torch.Tensor,
    ) -> torch.Tensor:
        vision_cfg = self.rbln_config.vision_tower
        text_hidden = self.config.text_config.hidden_size

        vision_out_size = [
            pixel_values.shape[0],
            vision_cfg.max_soft_tokens,
            self.config.vision_config.hidden_size,
        ]
        projector_out_size = [
            pixel_values.shape[0],
            vision_cfg.max_soft_tokens,
            text_hidden,
        ]
        vision_out_buffer = [torch.empty(size=vision_out_size, dtype=torch.float32, device="cpu")]
        projector_out_buffer = [torch.empty(size=projector_out_size, dtype=torch.float32, device="cpu")]

        vision_outputs = self.vision_tower(pixel_values, pixel_position_ids, out=vision_out_buffer).last_hidden_state
        image_features = self.embed_vision(vision_outputs, out=projector_out_buffer)
        return image_features

    def _preprocess_prefill(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        pixel_position_ids: Optional[torch.LongTensor] = None,
        **kwargs,
    ):
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        text_cfg = self.config.text_config
        image_token_id = self.config.image_token_id
        video_token_id = getattr(self.config, "video_token_id", None)

        if input_ids is not None:
            mm_mask = input_ids == image_token_id
            if video_token_id is not None:
                mm_mask = mm_mask | (input_ids == video_token_id)
            llm_input_ids = input_ids.clone()
            llm_input_ids[mm_mask] = text_cfg.pad_token_id
        else:
            mm_mask = None
            llm_input_ids = None

        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(llm_input_ids)

        if pixel_values is not None and pixel_position_ids is not None:
            if input_ids is None:
                raise ValueError(
                    "multimodal prefill requires `input_ids` for image-token mask construction; "
                    "received inputs_embeds-only"
                )
            image_features = self.get_image_features(pixel_values, pixel_position_ids)
            image_features = image_features.to(inputs_embeds.device, inputs_embeds.dtype)
            image_mask = (input_ids == image_token_id).unsqueeze(-1).expand_as(inputs_embeds)
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_features)

        return inputs_embeds

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: torch.Tensor = None,
        token_type_ids: torch.Tensor = None,
        pixel_values: torch.FloatTensor = None,
        pixel_position_ids: torch.LongTensor = None,
        cache_position: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        generate_idx: Optional[torch.Tensor] = None,
        padded_cache_lengths: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        mm_token_type_ids: Optional[torch.Tensor] = None,
        output_hidden_states: Optional[bool] = None,
        **lm_kwargs: Dict[str, Any],
    ) -> Union[Tuple, RBLNDecoderOnlyOutput]:
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.rbln_config.language_model.output_hidden_states
        )
        if output_hidden_states != self.rbln_config.language_model.output_hidden_states:
            raise ValueError(
                f"Variable output_hidden_states {output_hidden_states} is not equal to "
                f"rbln_config.language_model.output_hidden_states "
                f"{self.rbln_config.language_model.output_hidden_states}. Please compile with the correct argument."
            )

        if cache_position is None:
            logits = []
            inputs_embeds = self._preprocess_prefill(
                input_ids=input_ids,
                inputs_embeds=inputs_embeds,
                pixel_values=pixel_values,
                pixel_position_ids=pixel_position_ids,
            )
            batch_size = inputs_embeds.shape[0]

            for b_idx in range(batch_size):
                cache_position_b = torch.arange(0, generate_idx[b_idx].item(), dtype=torch.int32).unsqueeze(0)
                outputs = self.language_model.prefill_decoder(
                    input_ids=input_ids[b_idx : b_idx + 1] if input_ids is not None else None,
                    inputs_embeds=inputs_embeds[b_idx : b_idx + 1],
                    attention_mask=attention_mask[b_idx] if attention_mask is not None else None,
                    cache_position=cache_position_b,
                    batch_idx=b_idx,
                    token_type_ids=token_type_ids[b_idx : b_idx + 1] if token_type_ids is not None else None,
                )
                padded_cache_lengths[b_idx] += outputs.padded_cache_lengths
                logits.append(outputs.logits)

            logits = torch.cat(logits, dim=0)
        else:
            inputs = inputs_embeds if inputs_embeds is not None else input_ids
            batch_size = inputs.shape[0]
            if batch_size not in self.language_model.decoders:
                raise ValueError(
                    f"No decoder runtime available for batch size {batch_size}. "
                    f"Available batch sizes are: {list(self.language_model.decoders.keys())}."
                )
            outputs = self.language_model.decoders[batch_size](
                input_ids=input_ids,
                inputs_embeds=inputs_embeds,
                cache_position=cache_position,
                position_ids=position_ids if self.rbln_config.language_model.use_position_ids else None,
            )
            logits = outputs.logits

        return RBLNDecoderOnlyOutput(
            logits=logits,
            generate_idx=generate_idx,
            padded_cache_lengths=padded_cache_lengths,
        )
