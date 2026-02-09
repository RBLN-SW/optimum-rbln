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
from typing import TYPE_CHECKING, Any, Callable

import torch
from transformers import AutoModelForTextEncoding, T5EncoderModel, T5ForConditionalGeneration
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions

from ...modeling_generic import RBLNTransformerEncoderForFeatureExtraction
from ...models.seq2seq import RBLNModelForSeq2SeqLM
from .configuration_t5 import RBLNT5EncoderModelConfig, RBLNT5ForConditionalGenerationConfig
from .t5_architecture import T5Wrapper


if TYPE_CHECKING:
    from transformers import PreTrainedModel

    from ....diffusers.modeling_diffusers import RBLNDiffusionMixin, RBLNDiffusionMixinConfig


class T5EncoderWrapper(torch.nn.Module):
    def __init__(self, model: "T5EncoderModel") -> None:
        super().__init__()
        self.model = model

    def forward(self, *args, **kwargs):
        kwargs.pop("return_dict", None)
        return self.model(*args, **kwargs, return_dict=False)


class RBLNT5EncoderModel(RBLNTransformerEncoderForFeatureExtraction):
    """
    The T5 Model transformer with an encoder-only architecture for feature extraction.
    This model inherits from [`RBLNTransformerEncoderForFeatureExtraction`]. Check the superclass documentation for the generic methods the library implements for all its models.

    Important Note:
        This model supports various sizes of the T5EncoderModel. For optimal performance, it is highly recommended to adjust the tensor parallelism setting
        based on the model size. Please refer to the [Optimum RBLN Overview](../../../optimum_rbln.md) for guidance on choosing the appropriate tensor parallelism size for your model.

    Examples:
        ```python
        from optimum.rbln import RBLNT5EncoderModel

        model = RBLNT5EncoderModel.from_pretrained(
            "sentence-transformers/sentence-t5-xxl",
            export=True,
            rbln_tensor_parallel_size=4,
        )

        model.save_pretrained("compiled-sentence-t5-xxl")
        ```
    """

    auto_model_class = AutoModelForTextEncoding
    output_class = BaseModelOutputWithPastAndCrossAttentions

    @classmethod
    def _wrap_model_if_needed(self, model: "PreTrainedModel", rbln_config: RBLNT5EncoderModelConfig):
        return T5EncoderWrapper(model)

    @classmethod
    def update_rbln_config_using_pipe(
        cls, pipe: "RBLNDiffusionMixin", rbln_config: "RBLNDiffusionMixinConfig", submodule_name: str
    ) -> "RBLNDiffusionMixinConfig":
        return rbln_config

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        input_dict = {"input_ids": input_ids.long()}
        if attention_mask is not None:
            input_dict["attention_mask"] = attention_mask.long()

        output = super().forward(**input_dict, **kwargs)
        return output


class RBLNT5ForConditionalGeneration(RBLNModelForSeq2SeqLM):
    """
    The T5 Model transformer with a language modeling head for conditional generation.
    This model inherits from [`RBLNModelForSeq2SeqLM`]. Check the superclass documentation for the generic methods the library implements for all its models.

    Important Note:
        This model supports various sizes of the T5ForConditionalGeneration. For optimal performance, it is highly recommended to adjust the tensor parallelism setting
        based on the model size. Please refer to the [Optimum RBLN Overview](../../../optimum_rbln.md) for guidance on choosing the appropriate tensor parallelism size for your model.


    Examples:
        ```python
        from optimum.rbln import RBLNT5ForConditionalGeneration

        model = RBLNT5ForConditionalGeneration.from_pretrained(
            "google-t5/t5-11b",
            export=True,
            rbln_tensor_parallel_size=4,
        )

        model.save_pretrained("compiled-sentence-t5-xxl")
        ```
    """

    support_causal_attn = False

    @classmethod
    def _wrap_model_if_needed(self, model: "PreTrainedModel", rbln_config: RBLNT5ForConditionalGenerationConfig):
        return T5Wrapper(
            model, enc_max_seq_len=rbln_config.enc_max_seq_len, dec_max_seq_len=rbln_config.dec_max_seq_len
        )

    def __getattr__(self, __name: str) -> Any:
        def redirect(func):
            return lambda *pargs, **kwargs: func(self, *pargs, **kwargs)

        val = getattr(T5ForConditionalGeneration, __name)

        if isinstance(val, Callable) and "self" in set(inspect.signature(val).parameters):
            return redirect(val)

        return val
