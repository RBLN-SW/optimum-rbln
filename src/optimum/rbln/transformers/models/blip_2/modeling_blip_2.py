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
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, List, Optional, Tuple, Union

import torch
from transformers import (
    AutoModelForVisualQuestionAnswering,
    Blip2ForConditionalGeneration,
    Blip2QFormerModel,
    Blip2VisionModel,
    PretrainedConfig,
    PreTrainedModel,
)
from transformers.modeling_outputs import BaseModelOutputWithPooling, BaseModelOutputWithPoolingAndCrossAttentions
from transformers.utils import logging

from ....configuration_utils import RBLNCompileConfig, RBLNModelConfig
from ....modeling import RBLNModel
from ...utils.rbln_runtime_wrapper import LoopProcessor
from ..decoderonly.generation_decoderonly import RBLNDecoderOnlyGenerationMixin


logger = logging.get_logger(__name__)

if TYPE_CHECKING:
    import rebel
    from transformers import AutoFeatureExtractor, AutoProcessor, AutoTokenizer


class LoopProjector(LoopProcessor):
    def __init__(self, language_projection: Union[RBLNModel, "rebel.Runtime"]):
        super().__init__(model=language_projection)

    def _get_batch_size(self, query_output, **kwargs):
        return query_output.shape[0]

    def _prepare_inputs_for_iteration(self, index, common_inputs, query_output, **kwargs):
        query_output_item = query_output[index : index + 1]
        return ([query_output_item], {})

    def _process_outputs(self, outputs: list, **kwargs):
        output = torch.cat(outputs, dim=0)
        return output


class RBLNBlip2VisionModel(RBLNModel):
    """
    RBLN optimized BLIP-2 vision encoder model.

    This class provides hardware-accelerated inference for BLIP-2 vision encoders
    on RBLN devices, supporting image encoding for multimodal vision-language tasks.
    """

    _tp_support = False

    def get_input_embeddings(self):
        return self.embeddings

    @classmethod
    def _wrap_model_if_needed(cls, model: torch.nn.Module, rbln_config: RBLNModelConfig) -> torch.nn.Module:
        class Blip2VisionModelWrapper(torch.nn.Module):
            def __init__(self, model: "Blip2VisionModel") -> None:
                super().__init__()
                self.model = model

            def forward(self, *args, **kwargs):
                kwargs.pop("return_dict", None)
                return self.model(*args, **kwargs, return_dict=False)

        return Blip2VisionModelWrapper(model).eval()

    @classmethod
    def _update_rbln_config(
        cls,
        preprocessors: Optional[Union["AutoFeatureExtractor", "AutoProcessor", "AutoTokenizer"]],
        model: Optional["PreTrainedModel"] = None,
        model_config: Optional["PretrainedConfig"] = None,
        rbln_config: Optional[RBLNModelConfig] = None,
    ) -> RBLNModelConfig:
        input_info = [
            (
                "pixel_values",
                [
                    rbln_config.batch_size,
                    model_config.num_channels,
                    model_config.image_size,
                    model_config.image_size,
                ],
                "float32",
            ),
        ]

        rbln_compile_config = RBLNCompileConfig(input_info=input_info)
        rbln_config.set_compile_cfgs([rbln_compile_config])
        return rbln_config

    def forward(
        self,
        pixel_values: torch.FloatTensor,
        interpolate_pos_encoding: bool = False,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        """
        Forward pass for the RBLN-optimized Blip2VisionModel model.

        Args:
            pixel_values (torch.FloatTensor of shape (batch_size, num_channels, height, width)): The tensors corresponding to the input images.
            interpolate_pos_encoding (bool, optional): Whether to interpolate the positional encoding of the image embeddings. Defaults to False.
            return_dict (bool, optional): Whether to return a ModelOutput instead of a plain tuple.

        Returns:
            BaseModelOutputWithPooling or tuple(torch.FloatTensor): The model outputs. If return_dict=False is passed, returns a tuple of tensors. Otherwise, returns a BaseModelOutputWithPooling object.
        """
        batch_size = pixel_values.shape[0]
        outputs = []
        for i in range(batch_size):
            outputs.append(self.model[0](pixel_values[i : i + 1]))

        last_hidden_state = [output[0] for output in outputs]
        pooler_output = [output[1] for output in outputs]

        last_hidden_state = torch.cat(last_hidden_state, dim=0)
        pooler_output = torch.cat(pooler_output, dim=0)

        if not return_dict:
            return (last_hidden_state, pooler_output)

        return BaseModelOutputWithPooling(
            last_hidden_state=last_hidden_state,
            pooler_output=pooler_output,
        )


class RBLNBlip2QFormerModel(RBLNModel):
    """
    RBLN optimized BLIP-2 Q-Former model.

    This class provides hardware-accelerated inference for BLIP-2 Q-Former models
    on RBLN devices, which bridge vision and language modalities through cross-attention
    mechanisms for multimodal understanding tasks.
    """

    _tp_support = False

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    @classmethod
    def _wrap_model_if_needed(cls, model: torch.nn.Module, rbln_config: RBLNModelConfig) -> torch.nn.Module:
        class Blip2QFormerModelWrapper(torch.nn.Module):
            def __init__(self, model: "Blip2QFormerModel"):
                super().__init__()
                self.model = model

            def forward(
                self,
                query_embeds: torch.FloatTensor,
                encoder_hidden_states: Optional[torch.FloatTensor] = None,
                encoder_attention_mask: Optional[torch.FloatTensor] = None,
            ) -> torch.Tensor:
                qformer_out = self.model(
                    query_embeds=query_embeds,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    return_dict=False,
                )
                return qformer_out

        return Blip2QFormerModelWrapper(model).eval()

    @classmethod
    def _update_submodule_config(
        cls,
        model: "PreTrainedModel",
        rbln_config: RBLNModelConfig,
        preprocessors: Optional[Union["AutoFeatureExtractor", "AutoProcessor", "AutoTokenizer"]],
    ):
        if rbln_config.num_query_tokens is None:
            rbln_config.num_query_tokens = model.config.num_query_tokens

        if rbln_config.image_text_hidden_size is None:
            rbln_config.image_text_hidden_size = model.config.image_text_hidden_size

        return rbln_config

    @classmethod
    def _update_rbln_config(
        cls,
        preprocessors: Optional[Union["AutoFeatureExtractor", "AutoProcessor", "AutoTokenizer"]],
        model: Optional["PreTrainedModel"] = None,
        model_config: Optional["PretrainedConfig"] = None,
        rbln_config: Optional[RBLNModelConfig] = None,
    ) -> RBLNModelConfig:
        input_info = [
            (
                "query_embeds",
                [
                    rbln_config.batch_size,
                    rbln_config.num_query_tokens,
                    model_config.hidden_size,
                ],
                "float32",
            ),
            (
                "encoder_hidden_states",
                [
                    rbln_config.batch_size,
                    # image_text_hidden_size + cls token
                    rbln_config.image_text_hidden_size + 1,
                    model_config.encoder_hidden_size,
                ],
                "float32",
            ),
            (
                "encoder_attention_mask",
                # image_text_hidden_size + cls token
                [rbln_config.batch_size, rbln_config.image_text_hidden_size + 1],
                "int64",
            ),
        ]

        rbln_compile_config = RBLNCompileConfig(input_info=input_info)
        rbln_config.set_compile_cfgs([rbln_compile_config])
        return rbln_config

    def forward(
        self,
        query_embeds: torch.FloatTensor,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], BaseModelOutputWithPoolingAndCrossAttentions]:
        """
        The forward pass for the RBLN-optimized Blip2QFormerModel model.

        Args:
            query_embeds (torch.FloatTensor): Hidden states to be used in the attention computation.
            encoder_hidden_states (torch.FloatTensor, optional): Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention if the model is configured as a decoder.
            encoder_attention_mask (torch.FloatTensor, optional): Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in the cross-attention if the model is configured as a decoder.
            return_dict (bool, optional): Whether to return a ModelOutput instead of a plain tuple.

        Returns:
            BaseModelOutputWithPoolingAndCrossAttentions or tuple(torch.FloatTensor): The model outputs. If `return_dict=False` is passed, returns a tuple of tensors. Otherwise, returns a `BaseModelOutputWithPoolingAndCrossAttentions` object.
        """
        batch_size = query_embeds.shape[0]
        outputs = []
        for i in range(batch_size):
            outputs.append(
                self.model[0](
                    query_embeds[i : i + 1], encoder_hidden_states[i : i + 1], encoder_attention_mask[i : i + 1]
                )
            )

        sequence_output = [output[0] for output in outputs]
        pooled_output = [output[1] for output in outputs]

        sequence_output = torch.cat(sequence_output, dim=0)
        pooled_output = torch.cat(pooled_output, dim=0)

        if not return_dict:
            return (sequence_output, pooled_output)

        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
        )


class RBLNBlip2ForConditionalGeneration(RBLNModel, RBLNDecoderOnlyGenerationMixin):
    """
    RBLNBlip2ForConditionalGeneration is a multi-modal model that integrates vision and language processing capabilities,
    optimized for RBLN NPUs. It is designed for conditional generation tasks that involve both image and text inputs.

    This model inherits from [`RBLNModel`]. Check the superclass documentation for the generic methods the library implements for all its models.

    Important Note:
        This model includes a Large Language Model (LLM) as a submodule. For optimal performance, it is highly recommended to use
        tensor parallelism for the language model.  This can be achieved by using the `rbln_config` parameter in the
        `from_pretrained` method. Refer to the `from_pretrained` documentation and the RBLNBlip2ForConditionalGeneration class for details.

    Examples:
        ```python
        from optimum.rbln import RBLNBlip2ForConditionalGeneration

        model = RBLNBlip2ForConditionalGeneration.from_pretrained(
            "Salesforce/blip2-opt-2.7b",
            export=True,
            rbln_config={
                "language_model": {
                    "batch_size": 1,
                    "max_seq_len": 2048,
                    "tensor_parallel_size": 1,
                    "use_inputs_embeds": True,
                },
            },
        )

        model.save_pretrained("compiled-blip2-opt-2.7b")
        ```
    """

    auto_model_class = AutoModelForVisualQuestionAnswering
    _rbln_submodules = [{"name": "vision_model"}, {"name": "qformer"}, {"name": "language_model"}]

    def __getattr__(self, __name: str) -> Any:
        def redirect(func):
            return lambda *pargs, **kwargs: func(self, *pargs, **kwargs)

        val = getattr(Blip2ForConditionalGeneration, __name)

        if isinstance(val, Callable) and "self" in set(inspect.signature(val).parameters):
            return redirect(val)
        return val

    def can_generate(self):
        return True

    @classmethod
    def save_torch_artifacts(
        cls,
        model: "Blip2ForConditionalGeneration",
        save_dir_path: Path,
        subfolder: str,
        rbln_config: RBLNModelConfig,
    ):
        # If you are unavoidably running on a CPU rather than an RBLN device,
        # store the torch tensor, weight, etc. in this function.

        save_dict = {}
        save_dict["query_tokens"] = model.query_tokens
        torch.save(save_dict, save_dir_path / subfolder / "query_tokens.pth")

    def __post_init__(self, **kwargs):
        self.vision_model = self.rbln_submodules[0]
        self.language_model = self.rbln_submodules[2]
        self.qformer = self.rbln_submodules[1]
        self.language_projection = LoopProjector(self.model[0])

        artifacts = torch.load(self.model_save_dir / self.subfolder / "query_tokens.pth", weights_only=False)
        self.query_tokens = artifacts["query_tokens"]

    def get_attn_impl(self) -> str:
        return self.rbln_config.language_model.attn_impl

    def get_kvcache_num_blocks(self) -> int:
        return self.rbln_config.language_model.kvcache_num_blocks

    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    @classmethod
    def _wrap_model_if_needed(cls, model, rbln_config):
        return model.language_projection

    @classmethod
    def _update_rbln_config(
        cls,
        preprocessors: Optional[Union["AutoFeatureExtractor", "AutoProcessor", "AutoTokenizer"]],
        model: Optional["PreTrainedModel"] = None,
        model_config: Optional["PretrainedConfig"] = None,
        rbln_config: Optional[RBLNModelConfig] = None,
    ) -> RBLNModelConfig:
        input_info = [
            (
                "query_output",
                [
                    1,
                    model_config.num_query_tokens,
                    model_config.qformer_config.hidden_size,
                ],
                "float32",
            ),
        ]

        rbln_compile_config = RBLNCompileConfig(input_info=input_info)
        rbln_config.set_compile_cfgs([rbln_compile_config])

        return rbln_config

    def _preprocess_prefill(
        self,
        pixel_values: torch.FloatTensor,
        input_ids: torch.FloatTensor,
        attention_mask: Optional[torch.LongTensor] = None,
        return_dict: Optional[bool] = None,
        interpolate_pos_encoding: bool = False,
        **kwargs,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        vision_outputs = self.vision_model(
            pixel_values=pixel_values,
            return_dict=return_dict,
            interpolate_pos_encoding=interpolate_pos_encoding,
        )
        image_embeds = vision_outputs[0]

        image_attention_mask = torch.ones(image_embeds.size()[:-1], dtype=torch.long, device=image_embeds.device)

        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)

        query_outputs = self.qformer(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_attention_mask,
            return_dict=return_dict,
        )
        query_output = query_outputs[0]

        if query_output.dtype != image_embeds.dtype:
            query_output = query_output.to(image_embeds.dtype)

        language_model_inputs = self.language_projection(query_output)
        language_model_attention_mask = torch.ones(
            language_model_inputs.size()[:-1], dtype=torch.long, device=language_model_inputs.device
        )
        inputs_embeds = self.language_model.get_input_embeddings()(input_ids)
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        if getattr(self.config, "image_token_index", None) is not None:
            special_image_mask = (input_ids == self.config.image_token_index).unsqueeze(-1).expand_as(inputs_embeds)
            language_model_inputs = language_model_inputs.to(inputs_embeds.device, inputs_embeds.dtype)
            inputs_embeds = inputs_embeds.masked_scatter(special_image_mask, language_model_inputs)
        else:
            logger.warning_once(
                "Expanding inputs for image tokens in BLIP-2 should be done in processing. "
                "Please follow instruction here (https://gist.github.com/zucchini-nlp/e9f20b054fa322f84ac9311d9ab67042) to update your BLIP-2 model. "
                "Using processors without these attributes in the config is deprecated and will throw an error in v4.50."
            )
            inputs_embeds = torch.cat([language_model_inputs, inputs_embeds.to(language_model_inputs.device)], dim=1)
            attention_mask = torch.cat(
                [language_model_attention_mask, attention_mask.to(language_model_attention_mask.device)], dim=1
            )

        return inputs_embeds

    @torch.no_grad()
    def generate(
        self,
        pixel_values: torch.FloatTensor,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        interpolate_pos_encoding: bool = False,
        **generate_kwargs,
    ) -> List[torch.LongTensor]:
        """
        The generate function is utilized in its standard form as in the HuggingFace transformers library. User can use this function to generate text from the model.
        Check the [HuggingFace transformers documentation](https://huggingface.co/docs/transformers/v4.57.1/en/model_doc/blip-2#transformers.Blip2ForConditionalGeneration.generate) for more details.

        Args:
            pixel_values (torch.FloatTensor): Input images to be processed.
            input_ids (torch.LongTensor, optional): The sequence used as a prompt for the generation.
            attention_mask (torch.LongTensor, optional): Mask to avoid performing attention on padding token indices
            inputs_embeds (torch.FloatTensor, optional): Embedded representation of the inputs. Should be float, not int tokens.
            interpolate_pos_encoding (bool, optional, defaults to False) — Whether to interpolate the positional encoding of the image embeddings.
        Returns:
            A list of strings of length batch_size * num_captions.
        """
        batch_size = pixel_values.shape[0]
        image_embeds = self.vision_model(
            pixel_values,
            return_dict=True,
            interpolate_pos_encoding=interpolate_pos_encoding,
        ).last_hidden_state
        image_attention_mask = torch.ones(image_embeds.size()[:-1], dtype=torch.long, device=image_embeds.device)

        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
        query_outputs = self.qformer(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_attention_mask,
            return_dict=True,
        )
        query_output = query_outputs.last_hidden_state

        if query_output.dtype != image_embeds.dtype:
            query_output = query_output.to(image_embeds.dtype)

        language_model_inputs = self.language_projection(query_output)

        if inputs_embeds is None:
            if input_ids is None:
                image_tokens = [self.config.image_token_index] * self.config.num_query_tokens
                start_tokens = image_tokens + [self.config.text_config.bos_token_id]
                input_ids = torch.tensor([start_tokens], dtype=torch.long, device=image_embeds.device)
                input_ids = input_ids.repeat(batch_size, 1)
            inputs_embeds = self.get_input_embeddings()(input_ids)

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        if input_ids is None:
            special_image_mask = inputs_embeds == self.get_input_embeddings()(
                torch.tensor(self.config.image_token_id, dtype=torch.long, device=inputs_embeds.device)
            )
            special_image_mask = special_image_mask.all(-1)
        else:
            special_image_mask = input_ids == self.config.image_token_id

        special_image_mask = special_image_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
        language_model_inputs = language_model_inputs.to(inputs_embeds.device, inputs_embeds.dtype)
        inputs_embeds = inputs_embeds.masked_scatter(special_image_mask, language_model_inputs)

        inputs = {"inputs_embeds": inputs_embeds, "attention_mask": attention_mask}
        if not self.language_model.config.is_encoder_decoder:
            inputs["input_ids"] = input_ids

        outputs = self.language_model.generate(**inputs, **generate_kwargs)

        return outputs
