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
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Optional, Tuple, Type, Union

import torch
from transformers import AutoModelForVision2Seq, PaliGemmaForConditionalGeneration, PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import BaseModelOutputWithPooling
from transformers.modeling_utils import no_init_weights
from transformers.models.paligemma.configuration_paligemma import PaliGemmaConfig
from transformers.models.paligemma.modeling_paligemma import PaligemmaModelOutputWithPast, PaliGemmaMultiModalProjector

from ....configuration_utils import RBLNModelConfig
from ....modeling import RBLNModel
from ....utils.logging import get_logger
from ...utils.rbln_runtime_wrapper import LoopProcessor
from ..decoderonly.generation_decoderonly import RBLNDecoderOnlyGenerationMixin
from ..decoderonly.modeling_decoderonly import RBLNDecoderOnlyOutput


logger = get_logger(__name__)

if TYPE_CHECKING:
    from transformers import AutoFeatureExtractor, AutoProcessor, AutoTokenizer, PretrainedConfig


class LoopVisionTower(LoopProcessor):
    def __init__(self, vision_tower: "RBLNModel"):
        super().__init__(model=vision_tower.model[0])

    def _get_batch_size(self, pixel_values, **kwargs):
        return pixel_values.shape[0]

    def _prepare_inputs_for_iteration(self, index, common_inputs, pixel_values, **kwargs):
        pixel_values_item = pixel_values[index : index + 1]
        out_buffer = kwargs["out"][index : index + 1]
        return ([pixel_values_item], {"out": out_buffer})

    def _process_outputs(self, outputs: list, **kwargs) -> "BaseModelOutputWithPooling":
        return BaseModelOutputWithPooling(
            last_hidden_state=kwargs["out"],
        )


class RBLNPaliGemmaForConditionalGeneration(RBLNModel, RBLNDecoderOnlyGenerationMixin):
    """
    RBLNPaliGemmaForConditionalGeneration is a multi-modal model that integrates vision and language processing capabilities,
    optimized for RBLN NPUs. It is designed for conditional generation tasks that involve both image and text inputs.

    This model inherits from [`RBLNModel`]. Check the superclass documentation for the generic methods the library implements for all its models.

    Important Note:
        This model includes a Large Language Model (LLM) as a submodule. For optimal performance, it is highly recommended to use
        tensor parallelism for the language model.  This can be achieved by using the `rbln_config` parameter in the
        `from_pretrained` method. Refer to the `from_pretrained` documentation and the RBLNPaliGemmaForConditionalGeneration class for details.

    Examples:
        ```python
        from optimum.rbln import RBLNPaliGemmaForConditionalGeneration

        model = RBLNPaliGemmaForConditionalGeneration.from_pretrained(
            "google/paligemma2-3b-mix-224",
            export=True,
            rbln_config={
                "language_model": {
                    "prefill_chunk_size": 8192,
                }
            },
            rbln_tensor_parallel_size=4,
        )

        model.save_pretrained("compiled-paligemma2-3b-mix-224")
        ```
    """

    auto_model_class = AutoModelForVision2Seq
    _rbln_submodules = [
        {"name": "vision_tower"},
        {"name": "language_model"},
    ]

    def __getattr__(self, __name: str) -> Any:
        def redirect(func):
            return lambda *pargs, **kwargs: func(self, *pargs, **kwargs)

        val = getattr(PaliGemmaForConditionalGeneration, __name)

        if isinstance(val, Callable) and "self" in set(inspect.signature(val).parameters):
            return redirect(val)
        return val

    def can_generate(self):
        return True

    @classmethod
    def _update_submodule_rbln_config(
        cls,
        submodule_name: str,
        submodule_cls: Type["RBLNModel"],
        model: "PreTrainedModel",
        submodule_config: PretrainedConfig,
        submodule_rbln_config: RBLNModelConfig,
        preprocessors: Optional[Union["AutoFeatureExtractor", "AutoProcessor", "AutoTokenizer"]],
    ):
        if submodule_name == "language_model":
            submodule_config.use_sliding_window = False
        else:
            return submodule_rbln_config

        return submodule_rbln_config

    @classmethod
    def _reconstruct_model_if_needed(cls, model: "PreTrainedModel"):
        with no_init_weights():
            model_cls_name = model.model.language_model.__class__.__name__
            causal_model_cls_name = model_cls_name.replace("Model", "ForCausalLM")
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

        artifacts = torch.load(self.model_save_dir / self.subfolder / "torch_artifacts.pth", weights_only=False)
        self.embed_tokens = self._create_embedding_layer()
        self.embed_tokens.load_state_dict(artifacts["embed_tokens"])
        self.multi_modal_projector = self._create_multi_modal_projector()
        self.multi_modal_projector.load_state_dict(artifacts["multi_modal_projector"])

        return super().__post_init__(**kwargs)

    @classmethod
    def save_torch_artifacts(
        cls,
        model: "PaliGemmaForConditionalGeneration",
        save_dir_path: Path,
        subfolder: str,
        rbln_config: RBLNModelConfig,
    ):
        save_dict = {}
        save_dict["embed_tokens"] = model.get_input_embeddings().state_dict()
        save_dict["multi_modal_projector"] = model.multi_modal_projector.state_dict()
        torch.save(save_dict, save_dir_path / subfolder / "torch_artifacts.pth")

    def get_attn_impl(self) -> str:
        return self.rbln_config.language_model.attn_impl

    def get_kvcache_num_blocks(self) -> int:
        return self.rbln_config.language_model.kvcache_num_blocks

    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def _create_embedding_layer(self):
        with no_init_weights():
            embed_tokens = torch.nn.Embedding(
                self.config.text_config.vocab_size,
                self.config.text_config.hidden_size,
                self.config.text_config.pad_token_id,
            )
        return embed_tokens

    def _create_multi_modal_projector(self):
        with no_init_weights():
            multi_modal_projector = PaliGemmaMultiModalProjector(self.config)
        return multi_modal_projector

    def prepare_inputs_for_generation(
        self,
        input_ids,
        inputs_embeds=None,
        pixel_values=None,
        image_sizes=None,
        attention_mask=None,
        generate_idx=None,
        position_ids=None,
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
            position_ids=position_ids,
            **kwargs,
        )

        if is_prefill_phase:
            model_inputs.update(
                {
                    "pixel_values": pixel_values,
                    "token_type_ids": token_type_ids,
                }
            )

        model_inputs["attention_mask"] = attention_mask

        return model_inputs

    def _update_model_kwargs_for_generation(self, outputs, model_kwargs, is_encoder_decoder, **kwargs):
        model_kwargs["generate_idx"] = outputs.generate_idx
        return model_kwargs

    def get_image_features(self, pixel_values: torch.Tensor):
        vision_output_size = [
            pixel_values.shape[0],
            self.config.vision_config.num_image_tokens,
            self.config.vision_config.hidden_size,
        ]
        vision_output = torch.empty(size=vision_output_size, dtype=torch.float32, device="cpu")
        self.vision_tower(pixel_values, out=vision_output)
        image_features = self.multi_modal_projector(vision_output)
        image_features = image_features / (self.config.text_config.hidden_size**0.5)
        return image_features

    def get_placeholder_mask(
        self, input_ids: torch.LongTensor, inputs_embeds: torch.FloatTensor, image_features: torch.FloatTensor
    ):
        if input_ids is None:
            special_image_mask = inputs_embeds == self.get_input_embeddings()(
                torch.tensor(self.config.image_token_id, dtype=torch.long, device=inputs_embeds.device)
            )
            special_image_mask = special_image_mask.all(-1)
        else:
            special_image_mask = input_ids == self.config.image_token_id

        n_image_tokens = special_image_mask.sum()
        special_image_mask = special_image_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
        n_image_features = image_features.shape[0] * image_features.shape[1]
        if inputs_embeds[special_image_mask].numel() != image_features.numel():
            raise ValueError(
                f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
            )
        return special_image_mask

    def _preprocess_prefill(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        **kwargs,
    ):
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if input_ids is not None and self.config.image_token_id >= self.config.text_config.vocab_size:
            special_image_mask = input_ids == self.config.image_token_id
            llm_input_ids = input_ids.clone()
            llm_input_ids[special_image_mask] = 0
        else:
            llm_input_ids = input_ids

        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(llm_input_ids)

        if pixel_values is not None:
            image_features = self.get_image_features(pixel_values)
            image_features = image_features.to(inputs_embeds.device, inputs_embeds.dtype)
            special_image_mask = self.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, image_features=image_features
            )
            inputs_embeds = inputs_embeds.masked_scatter(special_image_mask, image_features)

        return inputs_embeds

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        pixel_values: torch.FloatTensor = None,
        attention_mask: torch.LongTensor = None,
        position_ids: torch.LongTensor = None,
        token_type_ids: torch.LongTensor = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        cache_position: torch.Tensor = None,
        generate_idx: Optional[torch.Tensor] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Union[Tuple, RBLNDecoderOnlyOutput]:
        # Prefill
        if cache_position is None:
            inputs_embeds = self._preprocess_prefill(
                input_ids=input_ids, inputs_embeds=inputs_embeds, pixel_values=pixel_values
            )
            logits = []
            inputs = inputs_embeds if inputs_embeds is not None else input_ids
            batch_size = inputs.shape[0]

            for b_idx in range(batch_size):
                cache_position = torch.arange(0, generate_idx[b_idx].item(), dtype=torch.int32).unsqueeze(0)
                output = self.language_model.prefill_decoder(
                    input_ids=inputs[b_idx : b_idx + 1] if inputs_embeds is None else None,
                    inputs_embeds=inputs[b_idx : b_idx + 1] if inputs_embeds is not None else None,
                    attention_mask=attention_mask[b_idx] if attention_mask is not None else None,
                    position_ids=position_ids[b_idx : b_idx + 1] if position_ids is not None else None,
                    cache_position=cache_position,
                    batch_idx=b_idx,
                )
                logits.append(output.logits)

            logits = torch.cat(logits, dim=0)
        # Decoder
        else:
            logits = self.language_model.decoder(
                input_ids=input_ids,
                inputs_embeds=inputs_embeds,
                cache_position=cache_position,
                position_ids=position_ids if self.rbln_config.language_model.use_position_ids else None,
            ).logits

        if not return_dict:
            return logits, generate_idx
        else:
            return RBLNDecoderOnlyOutput(
                logits=logits,
                generate_idx=generate_idx,
            )


class RBLNPaliGemmaModel(RBLNModel):
    """
    RBLNPaliGemmaModel which consists of a vision backbone and a language model without language modeling head,
    optimized for RBLN NPUs.

    This model inherits from [`RBLNModel`]. Check the superclass documentation for the generic methods the library implements for all its models.

    Important Note:
        This model includes a Large Language Model (LLM) as a submodule. For optimal performance, it is highly recommended to use
        tensor parallelism for the language model.  This can be achieved by using the `rbln_config` parameter in the
        `from_pretrained` method. Refer to the `from_pretrained` documentation and the RBLNPaliGemmaModel class for details.

    Examples:
        ```python
        from optimum.rbln import RBLNPaliGemmaModel

        model = RBLNPaliGemmaModel.from_pretrained(
            "google/paligemma2-3b-mix-224",
            export=True,
            rbln_config={
                "language_model": {
                    "prefill_chunk_size": 8192,
                }
            },
            rbln_tensor_parallel_size=4,
        )

        model.save_pretrained("compiled-paligemma2-3b-mix-224")
        ```
    """

    _rbln_submodules = [
        {"name": "vision_tower"},
        {"name": "language_model"},
    ]

    def __post_init__(self, **kwargs):
        self.vision_tower = LoopVisionTower(self.rbln_submodules[0])
        self.language_model = self.rbln_submodules[1]

        if not isinstance(self.config.text_config, PretrainedConfig):
            cfg = self.config if isinstance(self.config, dict) else self.config.to_dict()
            text_config = cfg.pop("text_config", None)
            vision_config = cfg.pop("vision_config", None)
            self.config = PaliGemmaConfig(
                text_config=text_config,
                vision_config=vision_config,
                **cfg,
            )

        artifacts = torch.load(self.model_save_dir / self.subfolder / "torch_artifacts.pth", weights_only=False)
        self.embed_tokens = self._create_embedding_layer()
        self.embed_tokens.load_state_dict(artifacts["embed_tokens"])
        self.multi_modal_projector = self._create_multi_modal_projector()
        self.multi_modal_projector.load_state_dict(artifacts["multi_modal_projector"])

        return super().__post_init__(**kwargs)

    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    @classmethod
    def _update_submodule_rbln_config(
        cls,
        submodule_name: str,
        submodule_cls: Type["RBLNModel"],
        model: "PreTrainedModel",
        submodule_config: PretrainedConfig,
        submodule_rbln_config: RBLNModelConfig,
        preprocessors: Optional[Union["AutoFeatureExtractor", "AutoProcessor", "AutoTokenizer"]],
    ):
        if submodule_name == "language_model":
            submodule_config.use_sliding_window = False
        else:
            return submodule_rbln_config

        return submodule_rbln_config

    @classmethod
    def save_torch_artifacts(
        cls,
        model: "PaliGemmaForConditionalGeneration",
        save_dir_path: Path,
        subfolder: str,
        rbln_config: RBLNModelConfig,
    ):
        save_dict = {}
        save_dict["embed_tokens"] = model.get_input_embeddings().state_dict()
        save_dict["multi_modal_projector"] = model.multi_modal_projector.state_dict()
        torch.save(save_dict, save_dir_path / subfolder / "torch_artifacts.pth")

    def _create_embedding_layer(self):
        with no_init_weights():
            embed_tokens = torch.nn.Embedding(
                self.config.text_config.vocab_size,
                self.config.text_config.hidden_size,
                self.config.text_config.pad_token_id,
            )
        return embed_tokens

    def _create_multi_modal_projector(self):
        with no_init_weights():
            multi_modal_projector = PaliGemmaMultiModalProjector(self.config)
        return multi_modal_projector

    def get_image_features(self, pixel_values: torch.Tensor):
        vision_output_size = [
            pixel_values.shape[0],
            self.config.vision_config.num_image_tokens,
            self.config.vision_config.hidden_size,
        ]
        vision_output = torch.empty(size=vision_output_size, dtype=torch.float32, device="cpu")
        self.vision_tower(pixel_values, out=vision_output)
        image_features = self.multi_modal_projector(vision_output)
        image_features = image_features / (self.config.text_config.hidden_size**0.5)
        return image_features

    def get_placeholder_mask(
        self, input_ids: torch.LongTensor, inputs_embeds: torch.FloatTensor, image_features: torch.FloatTensor
    ):
        if input_ids is None:
            special_image_mask = inputs_embeds == self.get_input_embeddings()(
                torch.tensor(self.config.image_token_index, dtype=torch.long, device=inputs_embeds.device)
            )
            special_image_mask = special_image_mask.all(-1)
        else:
            special_image_mask = input_ids == self.config.image_token_index

        n_image_tokens = special_image_mask.sum()
        special_image_mask = special_image_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
        n_image_features = image_features.shape[0] * image_features.shape[1]
        if inputs_embeds[special_image_mask].numel() != image_features.numel():
            raise ValueError(
                f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
            )
        return special_image_mask

    def _preprocess_prefill(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        **kwargs,
    ):
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if input_ids is not None and self.config.image_token_index >= self.config.text_config.vocab_size:
            special_image_mask = input_ids == self.config.image_token_index
            llm_input_ids = input_ids.clone()
            llm_input_ids[special_image_mask] = 0
        else:
            llm_input_ids = input_ids

        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(llm_input_ids)

        image_features = None
        if pixel_values is not None:
            image_features = self.get_image_features(pixel_values)
            image_features = image_features.to(inputs_embeds.device, inputs_embeds.dtype)
            special_image_mask = self.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, image_features=image_features
            )
            inputs_embeds = inputs_embeds.masked_scatter(special_image_mask, image_features)

        return inputs_embeds, image_features

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Union[Tuple, PaligemmaModelOutputWithPast]:
        """
        Forward pass for the RBLN-optimized PaliGemmaModel model.

        Args:
            input_ids (torch.LongTensor of shape (batch_size, sequence_length)) — Indices of input sequence tokens in the vocabulary.
            pixel_values (torch.Tensor of shape (batch_size, num_channels, image_size, image_size)) — The tensors corresponding to the input images.
            attention_mask (torch.Tensor of shape (batch_size, sequence_length)) — Mask to avoid performing attention on padding token indices.
            position_ids (torch.LongTensor of shape (batch_size, sequence_length)) — Indices of positions of each input sequence tokens in the position embeddings.
            token_type_ids (torch.LongTensor of shape (batch_size, sequence_length)) — Segment token indices to indicate first and second portions of the inputs.
            output_hidden_states (bool, optional) — Whether or not to return the hidden states of all layers. See hidden_states under returned tensors for more detail.
            return_dict (bool, optional) — Whether or not to return a ModelOutput instead of a plain tuple.

        Returns:
            PaligemmaModelOutputWithPast or tuple(torch.FloatTensor)
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.rbln_config.output_hidden_states
        )
        if output_hidden_states != self.rbln_config.output_hidden_states:
            raise ValueError(
                f"Variable output_hidden_states {output_hidden_states} is not equal to rbln_config.output_hidden_states {self.rbln_config.output_hidden_states} "
                f"Please compile again with the correct argument."
            )

        inputs_embeds, image_features = self._preprocess_prefill(
            input_ids=input_ids, inputs_embeds=inputs_embeds, pixel_values=pixel_values
        )

        outputs = self.language_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_hidden_states=output_hidden_states,
        )

        return PaligemmaModelOutputWithPast(
            last_hidden_state=outputs.last_hidden_state,
            image_hidden_states=image_features if pixel_values is not None else None,
            hidden_states=outputs.hidden_states if output_hidden_states else None,
        )
