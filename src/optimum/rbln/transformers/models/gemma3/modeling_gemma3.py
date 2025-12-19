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
import importlib
import inspect
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union

import rebel
import torch
from rebel.compile_context import CompileContext
from transformers import AutoModelForImageTextToText, Gemma3ForConditionalGeneration, PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import BaseModelOutputWithPooling
from transformers.modeling_utils import no_init_weights
from transformers.models.gemma3.modeling_gemma3 import Gemma3TextScaledWordEmbedding

from ....configuration_utils import RBLNCompileConfig, RBLNModelConfig
from ....modeling import RBLNModel
from ...modeling_outputs import RBLNDecoderOnlyOutput
from ...utils.rbln_runtime_wrapper import LoopProcessor
from ..decoderonly.decoderonly_runtime_utils import RBLNPageTableManager
from ..decoderonly.generation_decoderonly import RBLNDecoderOnlyGenerationMixin
from ..decoderonly.modeling_decoderonly import (
    RBLNDecoderOnlyModelForCausalLM,
)
from .configuration_gemma3 import RBLNGemma3ForCausalLMConfig
from .gemma3_architecture import Gemma3ForCausalLMWrapper
from .gemma3_runtime_utils import RBLNGemma3RuntimeModel


if TYPE_CHECKING:
    from transformers import AutoFeatureExtractor, AutoProcessor, AutoTokenizer, Gemma3ForConditionalGeneration


class LoopVisionTower(LoopProcessor):
    def __init__(self, vision_tower: "RBLNModel"):
        super().__init__(model=vision_tower)

    def _get_batch_size(self, pixel_values, **kwargs):
        return pixel_values.shape[0]

    def _prepare_inputs_for_iteration(self, index, common_inputs, pixel_values, **kwargs):
        pixel_values_item = pixel_values[index : index + 1]
        out_buffer = [tensor[index : index + 1] for tensor in kwargs["out"]]
        return ([pixel_values_item], {"out": out_buffer})

    def _process_outputs(self, outputs: list, **kwargs) -> "BaseModelOutputWithPooling":
        output = kwargs["out"]

        return BaseModelOutputWithPooling(
            last_hidden_state=output[0],
        )


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


class RBLNGemma3ForConditionalGeneration(RBLNModel, RBLNDecoderOnlyGenerationMixin):
    auto_model_class = AutoModelForImageTextToText
    _rbln_submodules = [
        {"name": "vision_tower"},
        {"name": "language_model"},
    ]

    def __getattr__(self, __name: str) -> Any:
        def redirect(func):
            return lambda *pargs, **kwargs: func(self, *pargs, **kwargs)

        val = getattr(Gemma3ForConditionalGeneration, __name)

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
        model.lm_head = None
        del model.lm_head
        return model

    def __post_init__(self, **kwargs):
        self.vision_tower = LoopVisionTower(self.rbln_submodules[0])
        self.language_model = self.rbln_submodules[1]
        self.multi_modal_projector = LoopProjector(self.model[0])
        self.vocab_size = self.config.text_config.vocab_size

        # Copied from the original class
        self.pad_token_id = self.config.pad_token_id if self.config.pad_token_id is not None else -1
        return super().__post_init__(**kwargs)

    def get_attn_impl(self) -> str:
        return self.rbln_config.language_model.attn_impl

    def get_kvcache_num_blocks(self) -> int:
        return self.rbln_config.language_model.kvcache_num_blocks

    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    @classmethod
    def _wrap_model_if_needed(cls, model: "PreTrainedModel", rbln_config: RBLNModelConfig):
        return model.multi_modal_projector

    @classmethod
    def _update_rbln_config(
        cls,
        preprocessors: Optional[Union["AutoFeatureExtractor", "AutoProcessor", "AutoTokenizer"]],
        model: Optional["PreTrainedModel"] = None,
        model_config: Optional["PretrainedConfig"] = None,
        rbln_config: Optional[RBLNModelConfig] = None,
    ) -> RBLNModelConfig:
        image_feature_dim = (model_config.vision_config.image_size // model_config.vision_config.patch_size) ** 2
        feature_size = model_config.vision_config.hidden_size

        input_info = [("image_features", [rbln_config.batch_size, image_feature_dim, feature_size], "float32")]
        rbln_compile_config = RBLNCompileConfig(input_info=input_info)
        rbln_config.set_compile_cfgs([rbln_compile_config])
        return rbln_config

    def prepare_inputs_for_generation(
        self,
        input_ids,
        inputs_embeds=None,
        pixel_values=None,
        image_sizes=None,
        attention_mask=None,
        generate_idx=None,
        padded_cache_lengths=None,
        token_type_ids=None,
        **kwargs,
    ):
        # Prepare HF generation
        is_prefill_phase = generate_idx is None

        model_inputs = self.language_model.prepare_inputs_for_generation(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            generate_idx=generate_idx,  # Not affect
            attention_mask=attention_mask,
            padded_cache_lengths=padded_cache_lengths,
            **kwargs,
        )

        if is_prefill_phase:
            model_inputs.update(
                {
                    "pixel_values": pixel_values,
                    "image_sizes": image_sizes,
                    "token_type_ids": token_type_ids,
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
        # update generate_idx
        model_kwargs["generate_idx"] = outputs.generate_idx
        model_kwargs["padded_cache_lengths"] = outputs.padded_cache_lengths

        return model_kwargs

    def get_image_features(self, pixel_values: torch.Tensor) -> torch.Tensor:
        # Projects the last hidden state from the vision model into language model space.

        # Args:
        #     pixel_values: (`torch.FloatTensor` of shape `(batch_size, channels, height, width)`)
        #         The tensors corresponding to the input images.

        # Returns:
        #     Image feature tensor of shape `(num_images, image_length, embed_dim)`.

        vision_out_buffer = []
        vision_out_size = [
            pixel_values.shape[0],
            (self.config.vision_config.image_size // self.config.vision_config.patch_size) ** 2,
            self.config.vision_config.hidden_size,
        ]
        projector_out_size = [
            pixel_values.shape[0],
            self.config.mm_tokens_per_image,
            self.config.text_config.hidden_size,
        ]
        vision_out_buffer.append(torch.empty(size=vision_out_size, dtype=torch.float32, device="cpu"))
        projector_out_buffer = [torch.empty(size=projector_out_size, dtype=torch.float32, device="cpu")]
        vision_outputs = self.vision_tower(pixel_values, out=vision_out_buffer).last_hidden_state
        image_features = self.multi_modal_projector(vision_outputs, out=projector_out_buffer)
        return image_features

    def _preprocess_prefill(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        **kwargs,
    ):
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        # Replace image id woth PAD if the image token if OOV, to avoid index-errors
        if input_ids is not None and self.config.image_token_index >= self.vocab_size:
            special_image_mask = input_ids == self.config.image_token_index
            llm_input_ids = input_ids.clone()
            llm_input_ids[special_image_mask] = 0
        else:
            llm_input_ids = input_ids

        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(llm_input_ids)

        # Merge text and images
        if pixel_values is not None:
            image_features = self.get_image_features(pixel_values)
            special_image_mask = (input_ids == self.config.image_token_index).unsqueeze(-1)
            special_image_mask = special_image_mask.expand_as(inputs_embeds).to(inputs_embeds.device)

            image_features = image_features.to(inputs_embeds.device, inputs_embeds.dtype)
            inputs_embeds = inputs_embeds.masked_scatter(special_image_mask, image_features)

        return inputs_embeds

    def get_padded_cache_position(
        self,
        cache_position: torch.Tensor,  # shape: [1, seq_len]
        token_type_ids: torch.Tensor,  # shape: [1, seq_len]
    ) -> torch.Tensor:
        seq_len = cache_position[0][-1].item() + 1

        # Find image start positions
        image_starts = [
            s
            for s in torch.where(token_type_ids == 1)[1]
            if torch.all(token_type_ids[:, s : s + self.rbln_config.image_prefill_chunk_size] == 1)
        ]

        # Initialize padded tensors
        padded_input_len = seq_len
        for image_start in image_starts:
            pad_needed = (
                self.rbln_config.image_prefill_chunk_size
                - (image_start + padded_input_len - seq_len) % self.rbln_config.image_prefill_chunk_size
            ) % self.rbln_config.image_prefill_chunk_size
            padded_input_len += pad_needed

        return torch.cat(
            [cache_position, torch.arange(seq_len, padded_input_len, dtype=torch.int32).unsqueeze(0)],
            dim=1,
        )

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: torch.Tensor = None,
        token_type_ids: torch.Tensor = None,
        pixel_values: torch.FloatTensor = None,
        cache_position: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        generate_idx: Optional[torch.Tensor] = None,
        padded_cache_lengths: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
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
                f"Variable output_hidden_states {output_hidden_states} is not equal to rbln_config.language_model.output_hidden_states {self.rbln_config.language_model.output_hidden_states} "
                f"Please compile again with the correct argument."
            )

        # prefill
        if cache_position is None:
            logits = []
            inputs_embeds = self._preprocess_prefill(input_ids, inputs_embeds, pixel_values)
            batch_size = inputs_embeds.shape[0]

            all_hidden_states = (
                tuple(
                    torch.zeros(
                        batch_size,
                        inputs_embeds.shape[1],
                        self.config.text_config.hidden_size,
                        dtype=self.rbln_config.dtype,
                    )
                    for _ in range(self.config.text_config.num_hidden_layers + 1)
                )
                if self.rbln_config.language_model.output_hidden_states
                else None
            )

            for b_idx in range(batch_size):
                cache_position = torch.arange(0, generate_idx[b_idx].item(), dtype=torch.int32).unsqueeze(0)
                token_type_id = token_type_ids[b_idx : b_idx + 1, attention_mask[b_idx].bool()]
                cache_position = self.get_padded_cache_position(cache_position, token_type_id)

                outputs = self.language_model.prefill_decoder(
                    inputs_embeds=inputs_embeds[b_idx : b_idx + 1],
                    attention_mask=attention_mask[b_idx],
                    cache_position=cache_position,
                    batch_idx=b_idx,
                    token_type_ids=token_type_ids[b_idx : b_idx + 1],  # do not pass token_type_id
                )
                padded_cache_lengths[b_idx] += outputs.padded_cache_lengths
                logits.append(outputs.logits)
                if self.rbln_config.language_model.output_hidden_states:
                    for l_idx in range(self.config.text_config.num_hidden_layers + 1):
                        mask_indices = torch.nonzero(attention_mask[b_idx], as_tuple=True)[0]
                        all_hidden_states[l_idx][b_idx].index_copy_(
                            dim=0, index=mask_indices, source=outputs.hidden_states[l_idx][0]
                        )

            logits = torch.cat(logits, dim=0)
        # decoder
        else:
            inputs = inputs_embeds if inputs_embeds is not None else input_ids
            batch_size = inputs.shape[0]
            if batch_size not in self.language_model.decoders:
                raise ValueError(
                    f"No decoder runtime available for batch size {batch_size}. "
                    f"Available batch sizes are: {list(self.decoders.keys())}. "
                    f"Please run your model with one of these batch sizes or add support for batch size {batch_size}."
                )

            outputs = self.language_model.decoders[batch_size](
                input_ids=input_ids,
                inputs_embeds=inputs_embeds,
                cache_position=cache_position,
                position_ids=position_ids if self.rbln_config.language_model.use_position_ids else None,
            )
            logits = outputs.logits
            all_hidden_states = outputs.hidden_states

        return RBLNDecoderOnlyOutput(
            logits=logits,
            generate_idx=generate_idx,
            padded_cache_lengths=padded_cache_lengths,
            hidden_states=all_hidden_states,
        )


class RBLNGemma3ForCausalLM(RBLNDecoderOnlyModelForCausalLM):
    """
    The Gemma3 Model transformer with a language modeling head (linear layer) on top.
    This model inherits from [`RBLNDecoderOnlyModelForCausalLM`]. Check the superclass documentation for the generic methods the library implements for all its models.

    A class to convert and run pre-trained transformers based Gemma3ForCausalLM model on RBLN devices.
    It implements the methods to convert a pre-trained transformers Gemma3ForCausalLM model into a RBLN transformer model by:
    - transferring the checkpoint weights of the original into an optimized RBLN graph,
    - compiling the resulting graph using the RBLN compiler.
    """

    _decoder_wrapper_cls = Gemma3ForCausalLMWrapper
    _supports_non_fp32 = False

    def setup_runtime(self):
        # Initialize shared resources to be used across Runtime instances (prefill and decode phases)
        dec_attn_mask = torch.zeros(self.rbln_config.batch_size, self.rbln_config.max_seq_len, dtype=torch.float32)
        page_table_manager = RBLNPageTableManager(self.rbln_config)

        common_kwargs = {
            "main_input_name": "inputs_embeds" if self.rbln_config.use_inputs_embeds else "input_ids",
            "embed_tokens": self.embed_tokens,
            "dec_attn_mask": dec_attn_mask,
            "page_table_manager": page_table_manager,
            "rbln_config": self.rbln_config,
        }

        self.prefill_decoder = RBLNGemma3RuntimeModel(
            runtime=self.model[0],
            image_prefill=self.model[1] if self.rbln_config.use_image_prefill else None,
            phase="prefill",
            batch_size=self.rbln_config.batch_size,
            **common_kwargs,
        )

        self.decoders = {}
        for i, batch_size in enumerate(self.rbln_config.decoder_batch_sizes):
            self.decoders[batch_size] = RBLNGemma3RuntimeModel(
                runtime=self.model[i + self.rbln_config.decoder_runtime_idx],
                phase="decode",
                batch_size=batch_size,
                **common_kwargs,
            )

        # NOTE(eunji): Use a decoder whose batch size matches the model's main batch size for compatibility.
        self.decoder = self.decoders[self.rbln_config.batch_size]

    def _create_embedding_layer(self):
        with no_init_weights():
            embed_tokens = Gemma3TextScaledWordEmbedding(
                self.config.vocab_size,
                self.config.hidden_size,
                self.config.pad_token_id,
                embed_scale=self.config.hidden_size**0.5,
            )
        return embed_tokens

    @classmethod
    def _update_submodule_config(
        cls,
        model: "PreTrainedModel",
        rbln_config: RBLNModelConfig,
        preprocessors: Optional[Union["AutoFeatureExtractor", "AutoProcessor", "AutoTokenizer"]],
    ):
        if rbln_config.image_prefill_chunk_size is None:
            rbln_config.image_prefill_chunk_size = model.config.mm_tokens_per_image

        if rbln_config.image_prefill_chunk_size != model.config.mm_tokens_per_image:
            raise ValueError(
                f"Image prefill chunk size is different from mm_tokens_per_image: {rbln_config.image_prefill_chunk_size} != {model.config.mm_tokens_per_image}"
            )

        return rbln_config

    @classmethod
    def _update_rbln_config(
        cls,
        preprocessors: Optional[Union["AutoFeatureExtractor", "AutoProcessor", "AutoTokenizer"]] = None,
        model: Optional["PreTrainedModel"] = None,
        model_config: Optional["PretrainedConfig"] = None,
        rbln_config: Optional[RBLNGemma3ForCausalLMConfig] = None,
    ) -> RBLNGemma3ForCausalLMConfig:
        # Update rbln_config with super class
        rbln_config = super()._update_rbln_config(preprocessors, model, model_config, rbln_config)

        if not (rbln_config.use_attention_mask and rbln_config.use_position_ids):
            raise ValueError("use_attention_mask and use_position_ids must be True for RBLNGemma3ForCausalLM")

        if rbln_config.use_image_prefill:
            if rbln_config.prefill_chunk_size != rbln_config.image_prefill_chunk_size:
                raise NotImplementedError(
                    "Not implemented for different prefill chunk sizes between text and image prefill."
                )

            # Update image prefill compile config
            img_prefill_input_info = cls.get_input_info(
                batch_size=1,
                query_length=rbln_config.image_prefill_chunk_size,
                rbln_config=rbln_config,
                model_config=model_config,
            )
            image_prefill_compile_config = RBLNCompileConfig(
                compiled_model_name="image_prefill", input_info=img_prefill_input_info
            )
            # Insert image_prefill compile config at index 1
            compile_cfgs = rbln_config.compile_cfgs
            compile_cfgs.insert(1, image_prefill_compile_config)
            rbln_config.set_compile_cfgs(compile_cfgs)

        return rbln_config

    @classmethod
    @torch.inference_mode()
    def get_compiled_model(cls, model: "PreTrainedModel", rbln_config: RBLNGemma3ForCausalLMConfig):
        wrapped_model = cls._wrap_model_if_needed(model, rbln_config)

        rbln_compile_configs = rbln_config.compile_cfgs
        prefill_compile_config = rbln_compile_configs[0]

        context = CompileContext(use_weight_sharing=True)

        # Here we use meta tensor, for the memory efficiency.
        meta_tensor_names = [name for name, _, _ in prefill_compile_config.input_info if "past_key_values" in name]
        prefill_example_inputs = prefill_compile_config.get_dummy_inputs(fill=0, meta_tensor_names=meta_tensor_names)

        # Mark static tensors (self kv states)
        static_tensors = {}
        for (name, _, _), tensor in zip(prefill_compile_config.input_info, prefill_example_inputs):
            if "past_key_values" in name:
                static_tensors[name] = tensor
                context.mark_static_address(tensor)

        def compile_model(wrapped_model, compile_config, example_inputs, compile_context, quantization):
            try:
                if quantization:
                    quantization.maybe_set_quantization_env()
                original_linear = torch.nn.functional.linear
                torch.nn.functional.linear = torch.ops.rbln_custom_ops.linear
                compiled_model = cls.compile(
                    wrapped_model,
                    compile_config,
                    create_runtimes=rbln_config.create_runtimes,
                    device=rbln_config.device,
                    example_inputs=example_inputs,
                    compile_context=compile_context,
                )
                return compiled_model
            finally:
                torch.nn.functional.linear = original_linear
                if quantization:
                    quantization.maybe_reset_quantization_env()

        wrapped_model.phase = "prefill"
        compiled_prefill = compile_model(
            wrapped_model,
            prefill_compile_config,
            prefill_example_inputs,
            context,
            rbln_config.quantization,
        )
        compiled_models = {"prefill": compiled_prefill}

        if rbln_config.use_image_prefill:
            image_prefill_compile_config = rbln_compile_configs[1]
            image_prefill_example_inputs = image_prefill_compile_config.get_dummy_inputs(
                fill=0, static_tensors=static_tensors
            )
            wrapped_model.phase = "image_prefill"
            compiled_image_prefill = compile_model(
                wrapped_model,
                image_prefill_compile_config,
                image_prefill_example_inputs,
                context,
                rbln_config.quantization,
            )
            compiled_models["image_prefill"] = compiled_image_prefill

        wrapped_model.phase = "decode"
        for batch_size, dec_compile_config in zip(
            rbln_config.decoder_batch_sizes, rbln_compile_configs[rbln_config.decoder_runtime_idx :]
        ):
            dec_example_inputs = dec_compile_config.get_dummy_inputs(fill=0, static_tensors=static_tensors)
            compiled_decoder = compile_model(
                wrapped_model,
                dec_compile_config,
                dec_example_inputs,
                context,
                rbln_config.quantization,
            )
            compiled_models[f"decoder_batch_{batch_size}"] = compiled_decoder

        return compiled_models

    @classmethod
    def _create_runtimes(
        cls,
        compiled_models: List[rebel.RBLNCompiledModel],
        rbln_config: RBLNGemma3ForCausalLMConfig,
    ) -> List[rebel.Runtime]:
        expected_model_names = [
            "prefill",
            *[f"decoder_batch_{batch_size}" for batch_size in rbln_config.decoder_batch_sizes],
        ]
        if rbln_config.use_image_prefill:
            expected_model_names.insert(1, "image_prefill")

        if any(model_name not in rbln_config.device_map for model_name in expected_model_names):
            cls._raise_missing_compiled_file_error(expected_model_names)

        ret_val = [
            rebel.Runtime(
                compiled_models[0],
                tensor_type="pt",
                device=rbln_config.device_map["prefill"],
                activate_profiler=rbln_config.activate_profiler,
                timeout=rbln_config.timeout,
            )
        ]
        if rbln_config.use_image_prefill:
            ret_val.append(
                rebel.Runtime(
                    compiled_models[1],
                    tensor_type="pt",
                    device=rbln_config.device_map["image_prefill"],
                    activate_profiler=rbln_config.activate_profiler,
                    timeout=rbln_config.timeout,
                ),
            )

        ret_val.extend(
            [
                rebel.Runtime(
                    compiled_models[i + rbln_config.decoder_runtime_idx],
                    tensor_type="pt",
                    device=rbln_config.device_map[f"decoder_batch_{batch_size}"],
                    activate_profiler=rbln_config.activate_profiler,
                    timeout=rbln_config.timeout,
                )
                for i, batch_size in enumerate(rbln_config.decoder_batch_sizes)
            ]
        )

        return ret_val
