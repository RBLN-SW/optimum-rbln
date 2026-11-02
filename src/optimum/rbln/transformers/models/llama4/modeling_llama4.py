# Copyright 2025 Rebellions Inc. All rights reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import inspect
from collections.abc import Callable, Sequence
from typing import Any

import torch
from transformers import (
    AutoModelForImageTextToText,
    Llama4ForConditionalGeneration,
    PretrainedConfig,
    PreTrainedModel,
)
from transformers.modeling_outputs import BaseModelOutputWithPooling

from ....configuration_utils import RBLNCompileConfig, RBLNModelConfig
from ....modeling import RBLNModel
from ....modeling_base import Preprocessor
from ....utils.logging import get_logger
from ...modeling_outputs import RBLNDecoderOnlyOutput, _validate_output_hidden_states
from ...utils.multimodal_batch_sort import RBLNImageIndexedBatchSortMixin, _placeholder_token_counts
from ..decoderonly.modeling_decoderonly import RBLNDecoderOnlyModelForCausalLM
from .configuration_llama4 import (
    RBLNLlama4ForCausalLMConfig,
    RBLNLlama4ForConditionalGenerationConfig,
    RBLNLlama4VisionModelConfig,
)
from .llama4_architecture import Llama4ForCausalLMWrapper, Llama4VisionModelWrapper, uses_chunked_attention


logger = get_logger(__name__)


# Kept small to bound device memory. Keep in sync with RBLNLlama4VisionModelConfig's docstring.
_DEFAULT_VISION_BATCH_SIZE = 4


def _tokens_per_tile(vision_config: PretrainedConfig) -> int:
    # Same arithmetic as HF's pixel_shuffle: the patch grid is shrunk by pixel_shuffle_ratio per side.
    side = vision_config.image_size // vision_config.patch_size
    return int(side * vision_config.pixel_shuffle_ratio) * int(side * vision_config.pixel_shuffle_ratio)


class RBLNLlama4ForCausalLM(RBLNDecoderOnlyModelForCausalLM):
    """Compile and run the text decoder of Llama 4 on RBLN devices.

    This model inherits from [`RBLNDecoderOnlyModelForCausalLM`] and uses
    [`RBLNLlama4ForCausalLMConfig`]. Both standalone text checkpoints and the
    language model in dense multimodal checkpoints such as Llama Guard 4 are supported.
    Mixture-of-experts (MoE) models are not supported.
    Image inputs require a conditional-generation model and are not supported here.

    Examples:
        ```python
        from optimum.rbln import RBLNAutoModelForCausalLM

        model = RBLNAutoModelForCausalLM.from_pretrained(
            "meta-llama/Llama-Guard-4-12B",
            export=True,
            rbln_batch_size=1,
            rbln_max_seq_len=8192,
            rbln_num_devices=4,
        )
        ```
    """

    _decoder_wrapper_cls = Llama4ForCausalLMWrapper

    @classmethod
    def _update_rbln_config(
        cls,
        preprocessors: Sequence[Preprocessor] | None = None,
        model: PreTrainedModel | None = None,
        model_config: PretrainedConfig | None = None,
        rbln_config: RBLNLlama4ForCausalLMConfig | None = None,
    ) -> RBLNLlama4ForCausalLMConfig:
        text_config = model_config.get_text_config()
        if text_config.moe_layers:
            raise NotImplementedError("Llama 4 MoE models are not supported yet.")
        return super()._update_rbln_config(preprocessors, model, model_config, rbln_config)

    @classmethod
    def _update_attention_config(
        cls, model: PreTrainedModel, model_config: PretrainedConfig, rbln_config: RBLNLlama4ForCausalLMConfig
    ):
        # Runs after max_seq_len is resolved and before the base class reads use_attention_mask.
        text_config = model_config.get_text_config()
        if uses_chunked_attention(text_config) and rbln_config.max_seq_len > text_config.attention_chunk_size:
            if rbln_config.use_position_ids:
                raise ValueError("Llama 4 chunked attention does not support `use_position_ids=True`.")
            if not rbln_config.use_attention_mask:
                logger.info("Llama 4 chunked attention requires an attention mask. Setting `use_attention_mask=True`.")
            rbln_config.use_attention_mask = True
        return super()._update_attention_config(model, model_config, rbln_config)


class RBLNLlama4VisionModel(RBLNModel):
    """RBLN Llama 4 vision encoder, including patch embedding and the pixel shuffle adapter."""

    auto_model_class = None
    _tp_support = True
    _output_class = BaseModelOutputWithPooling

    @classmethod
    def _wrap_model_if_needed(cls, model: PreTrainedModel, rbln_config: RBLNLlama4VisionModelConfig):
        return Llama4VisionModelWrapper(model).eval()

    @classmethod
    def _update_rbln_config(
        cls,
        preprocessors: Sequence[Preprocessor] | None = None,
        model: PreTrainedModel | None = None,
        model_config: PretrainedConfig | None = None,
        rbln_config: RBLNLlama4VisionModelConfig | None = None,
    ) -> RBLNLlama4VisionModelConfig:
        if rbln_config.batch_size is None:
            rbln_config.batch_size = 1
        rbln_config.set_compile_cfgs(
            [
                RBLNCompileConfig(
                    input_info=[
                        (
                            "pixel_values",
                            [
                                rbln_config.batch_size,
                                model_config.num_channels,
                                model_config.image_size,
                                model_config.image_size,
                            ],
                            rbln_config.dtype,
                        )
                    ]
                )
            ]
        )
        return rbln_config


class RBLNLlama4ForConditionalGeneration(RBLNModel, RBLNImageIndexedBatchSortMixin):
    """
    RBLNLlama4ForConditionalGeneration is a multi-modal model that combines vision and language processing capabilities,
    optimized for RBLN NPUs. It is designed for conditional generation tasks that involve both image and text inputs.
    Mixture-of-experts (MoE) checkpoints are not supported.

    This model inherits from [`RBLNModel`]. It implements the methods to convert and run
    pre-trained transformers based Llama4ForConditionalGeneration model on RBLN devices by:

    - transferring the checkpoint weights of the original into an optimized RBLN graph,
    - compiling the resulting graph using the RBLN compiler.

    Important Note:
        This model includes a Large Language Model (LLM) as a submodule. For optimal performance, it is highly recommended to use
        tensor parallelism for the language model. This can be achieved by using the `rbln_config` parameter in the
        `from_pretrained` method. Refer to the `from_pretrained` documentation and the RBLNLlama4ForConditionalGeneration class for details.

    Examples:
        ```python
        from optimum.rbln import RBLNLlama4ForConditionalGeneration

        model = RBLNLlama4ForConditionalGeneration.from_pretrained(
            "meta-llama/Llama-Guard-4-12B",
            export=True,
            rbln_config={
                "language_model": {
                    "max_seq_len": 8192,
                    "num_devices": 4,
                },
            },
        )
        model.save_pretrained("compiled-llama-guard-4-12b")

        # Using RBLN config instances (recommended for type checking)
        from optimum.rbln import (
            RBLNLlama4ForCausalLMConfig,
            RBLNLlama4ForConditionalGenerationConfig,
            RBLNLlama4VisionModelConfig,
        )

        rbln_config = RBLNLlama4ForConditionalGenerationConfig(
            vision_model=RBLNLlama4VisionModelConfig(),
            language_model=RBLNLlama4ForCausalLMConfig(max_seq_len=8192, num_devices=4, use_inputs_embeds=True),
        )
        model = RBLNLlama4ForConditionalGeneration.from_pretrained(
            "meta-llama/Llama-Guard-4-12B",
            export=True,
            rbln_config=rbln_config,
        )
        ```
    """

    auto_model_class = AutoModelForImageTextToText
    _rbln_submodules = [{"name": "vision_model"}, {"name": "language_model"}]
    _image_indexed_kwargs = ("pixel_values",)

    def __getattr__(self, name: str) -> Any:
        value = getattr(Llama4ForConditionalGeneration, name)
        if isinstance(value, Callable) and "self" in inspect.signature(value).parameters:
            return lambda *args, **kwargs: value(self, *args, **kwargs)
        return value

    def __post_init__(self, **kwargs):
        self.vision_model, self.language_model = self.rbln_submodules
        self.multi_modal_projector = self.model[0]
        pad_token_id = self.config.get_text_config().pad_token_id
        self.pad_token_id = pad_token_id if pad_token_id is not None else -1
        return super().__post_init__(**kwargs)

    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def get_attn_impl(self) -> str:
        return self.rbln_config.language_model.attn_impl

    def get_kvcache_num_blocks(self) -> int:
        return self.rbln_config.language_model.kvcache_num_blocks

    @classmethod
    def _update_submodule_rbln_config(
        cls,
        submodule_name: str,
        submodule_cls: type[RBLNModel],
        model: PreTrainedModel,
        submodule_config: PretrainedConfig,
        submodule_rbln_config: RBLNModelConfig,
        preprocessors: Sequence[Preprocessor] | None,
    ) -> RBLNModelConfig:
        # Reject MoE checkpoints before the vision model is compiled; the language model checks again.
        if model.config.get_text_config().moe_layers:
            raise NotImplementedError("Llama 4 MoE models are not supported yet.")
        if submodule_name == "vision_model" and submodule_rbln_config.batch_size is None:
            submodule_rbln_config.batch_size = _DEFAULT_VISION_BATCH_SIZE
        return submodule_rbln_config

    @classmethod
    def _wrap_model_if_needed(cls, model: PreTrainedModel, rbln_config: RBLNLlama4ForConditionalGenerationConfig):
        return model.multi_modal_projector

    @classmethod
    def _update_rbln_config(
        cls,
        preprocessors: Sequence[Preprocessor] | None = None,
        model: PreTrainedModel | None = None,
        model_config: PretrainedConfig | None = None,
        rbln_config: RBLNLlama4ForConditionalGenerationConfig | None = None,
    ) -> RBLNLlama4ForConditionalGenerationConfig:
        vision = model_config.vision_config
        # The projector consumes the vision adapter output; read its width from the projector itself.
        in_features = model.multi_modal_projector.linear_1.in_features
        rbln_config.set_compile_cfgs(
            [
                RBLNCompileConfig(
                    input_info=[
                        (
                            "image_features",
                            [rbln_config.vision_model.batch_size, _tokens_per_tile(vision), in_features],
                            rbln_config.dtype,
                        )
                    ]
                )
            ]
        )
        return rbln_config

    def _images_per_sample(self, input_ids: torch.LongTensor | None, kwargs: dict[str, Any]) -> list[int]:
        return _placeholder_token_counts(input_ids, self._image_token_id, _tokens_per_tile(self.config.vision_config))

    def get_image_features(self, pixel_values: torch.FloatTensor, **kwargs) -> BaseModelOutputWithPooling:
        if pixel_values.ndim != 4 or pixel_values.shape[0] == 0:
            raise ValueError(
                "pixel_values must have shape (num_tiles, channels, height, width) with at least one tile."
            )
        compiled_batch = self.rbln_config.vision_model.batch_size
        features = []
        for batch in pixel_values.split(compiled_batch):
            num_pad = compiled_batch - batch.shape[0]
            padded = torch.cat([batch, batch.new_zeros(num_pad, *batch.shape[1:])]) if num_pad else batch
            encoded = self.vision_model(padded.to(self.vision_model.dtype), return_dict=False)[0]
            projected = self.multi_modal_projector(encoded.to(self.dtype))
            features.append(projected[: compiled_batch - num_pad])
        return BaseModelOutputWithPooling(last_hidden_state=torch.cat(features, dim=0))

    def _preprocess_prefill(
        self,
        input_ids: torch.LongTensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        pixel_values: torch.FloatTensor | None = None,
    ) -> torch.FloatTensor:
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds.")
        if pixel_values is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both pixel_values and inputs_embeds.")
        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)
        if pixel_values is not None:
            features = self.get_image_features(pixel_values).last_hidden_state.to(inputs_embeds)
            mask = (input_ids == self._image_token_id).unsqueeze(-1).expand_as(inputs_embeds)
            if mask.sum().item() != features.numel():
                raise ValueError("Image features and image placeholder tokens do not match.")
            inputs_embeds = inputs_embeds.masked_scatter(mask, features)
        return inputs_embeds

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.LongTensor,
        pixel_values: torch.FloatTensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        generate_idx: torch.Tensor | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        model_inputs = super().prepare_inputs_for_generation(
            input_ids, inputs_embeds=inputs_embeds, generate_idx=generate_idx, **kwargs
        )
        model_inputs["pixel_values"] = pixel_values if generate_idx is None else None
        return model_inputs

    def _update_model_kwargs_for_generation(
        self,
        outputs: RBLNDecoderOnlyOutput,
        model_kwargs: dict[str, Any],
        is_encoder_decoder: bool = False,
        **kwargs,
    ) -> dict[str, Any]:
        model_kwargs["generate_idx"] = outputs.generate_idx
        return model_kwargs

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        pixel_values: torch.FloatTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        return_dict: bool | None = None,
        cache_position: torch.LongTensor | None = None,
        generate_idx: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        output_hidden_states: bool | None = None,
        inputs_sorted: bool = False,
        **kwargs: dict[str, Any],
    ) -> tuple | RBLNDecoderOnlyOutput:
        return_dict = return_dict if return_dict is not None else self.config.return_dict
        output_hidden_states = _validate_output_hidden_states(output_hidden_states, self.rbln_config.language_model)
        self._require_sorted_batch_inputs(inputs_embeds if inputs_embeds is not None else input_ids, inputs_sorted)
        all_hidden_states = None
        if cache_position is None:
            inputs_embeds = self._preprocess_prefill(input_ids, inputs_embeds, pixel_values)
            batch_size, seq_len = inputs_embeds.shape[:2]
            if generate_idx is None:
                lengths = attention_mask.sum(-1) if attention_mask is not None else torch.full((batch_size,), seq_len)
                generate_idx = lengths[:, None].int()
            if output_hidden_states:
                text_config = self.config.get_text_config()
                all_hidden_states = tuple(
                    torch.zeros(
                        batch_size, seq_len, text_config.hidden_size, dtype=self.rbln_config.language_model.dtype
                    )
                    for _ in range(text_config.num_hidden_layers + 1)
                )
            logits = []
            for batch_idx in range(batch_size):
                positions = torch.arange(generate_idx[batch_idx].item(), dtype=torch.int32)[None, :]
                output = self.language_model.prefill_decoder(
                    inputs_embeds=inputs_embeds[batch_idx : batch_idx + 1],
                    cache_position=positions,
                    attention_mask=attention_mask[batch_idx] if attention_mask is not None else None,
                    batch_idx=batch_idx,
                )
                logits.append(output.logits)
                if output_hidden_states:
                    # Prefill returns hidden states for unpadded tokens only; scatter them back.
                    valid = (
                        torch.nonzero(attention_mask[batch_idx], as_tuple=True)[0]
                        if attention_mask is not None
                        else torch.arange(seq_len)
                    )
                    for layer_states, layer_output in zip(all_hidden_states, output.hidden_states, strict=True):
                        layer_states[batch_idx].index_copy_(0, valid, layer_output[0])
            logits = torch.cat(logits, dim=0)
        else:
            batch_size = (inputs_embeds if inputs_embeds is not None else input_ids).shape[0]
            if batch_size not in self.language_model.decoders:
                raise ValueError(
                    f"No decoder runtime available for batch size {batch_size}. "
                    f"Available batch sizes are: {list(self.language_model.decoders.keys())}."
                )
            output = self.language_model.decoders[batch_size](
                input_ids=input_ids,
                inputs_embeds=inputs_embeds,
                cache_position=cache_position,
                position_ids=position_ids if self.rbln_config.language_model.use_position_ids else None,
            )
            logits = output.logits
            all_hidden_states = output.hidden_states
        if not return_dict:
            return (logits, generate_idx, all_hidden_states) if output_hidden_states else (logits, generate_idx)
        return RBLNDecoderOnlyOutput(logits=logits, generate_idx=generate_idx, hidden_states=all_hidden_states)


__all__ = ["RBLNLlama4ForCausalLM", "RBLNLlama4ForConditionalGeneration", "RBLNLlama4VisionModel"]
