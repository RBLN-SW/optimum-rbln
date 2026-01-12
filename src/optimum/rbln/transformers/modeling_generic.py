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

"""
This file defines generic base classes for various RBLN models,
such as Question Answering, Image Classification, Audio Classification,
Sequence Classification, and Masked Language Modeling. These classes
implement common functionalities and configurations to be used across
different model architectures.
"""

import inspect
from typing import TYPE_CHECKING, Optional, Union

from torch import nn
from transformers import (
    AutoModel,
    AutoModelForDepthEstimation,
    AutoModelForImageClassification,
    AutoModelForMaskedLM,
    AutoModelForQuestionAnswering,
    AutoModelForSequenceClassification,
    AutoModelForTextEncoding,
    PretrainedConfig,
)
from transformers.modeling_outputs import BaseModelOutput, QuestionAnsweringModelOutput

from ..configuration_utils import RBLNCompileConfig
from ..modeling import RBLNModel
from ..utils.logging import get_logger
from .configuration_generic import (
    RBLNImageModelConfig,
    RBLNTransformerEncoderConfig,
)


if TYPE_CHECKING:
    from transformers import AutoFeatureExtractor, AutoProcessor, AutoTokenizer, PreTrainedModel

logger = get_logger()


class RBLNTransformerEncoder(RBLNModel):
    auto_model_class = AutoModel
    rbln_model_input_names = ["input_ids", "attention_mask", "token_type_ids"]
    rbln_dtype = "int64"

    @classmethod
    def _wrap_model_if_needed(cls, model: "PreTrainedModel", rbln_config: RBLNTransformerEncoderConfig) -> nn.Module:
        class TransformerEncoderWrapper(nn.Module):
            # Parameters to disable for RBLN compilation
            DISABLED_PARAMS = {"return_dict", "use_cache"}

            def __init__(self, model: "PreTrainedModel", rbln_config: RBLNTransformerEncoderConfig):
                super().__init__()
                self.model = model
                self.rbln_config = rbln_config
                self._forward_signature = inspect.signature(model.forward)

            def forward(self, *args, **kwargs):
                # Disable parameters that are not compatible with RBLN compilation
                for param_name in self.DISABLED_PARAMS:
                    if param_name in self._forward_signature.parameters:
                        kwargs[param_name] = False

                return self.model(*args, **kwargs)

        return TransformerEncoderWrapper(model, rbln_config).eval()

    @classmethod
    def _update_rbln_config(
        cls,
        preprocessors: Optional[Union["AutoFeatureExtractor", "AutoProcessor", "AutoTokenizer"]] = None,
        model: Optional["PreTrainedModel"] = None,
        model_config: Optional["PretrainedConfig"] = None,
        rbln_config: Optional[RBLNTransformerEncoderConfig] = None,
    ) -> RBLNTransformerEncoderConfig:
        return cls.update_rbln_config_for_transformers_encoder(
            preprocessors=preprocessors,
            model=model,
            model_config=model_config,
            rbln_config=rbln_config,
        )

    @classmethod
    def update_rbln_config_for_transformers_encoder(
        cls,
        preprocessors: Optional[Union["AutoFeatureExtractor", "AutoProcessor", "AutoTokenizer"]] = None,
        model: Optional["PreTrainedModel"] = None,
        model_config: Optional["PretrainedConfig"] = None,
        rbln_config: Optional[RBLNTransformerEncoderConfig] = None,
    ) -> RBLNTransformerEncoderConfig:
        max_position_embeddings = getattr(model_config, "n_positions", None) or getattr(
            model_config, "max_position_embeddings", None
        )

        if rbln_config.max_seq_len is None:
            rbln_config.max_seq_len = max_position_embeddings
            if rbln_config.max_seq_len is None:
                for tokenizer in preprocessors:
                    if hasattr(tokenizer, "model_max_length"):
                        rbln_config.max_seq_len = tokenizer.model_max_length
                        break
                if rbln_config.max_seq_len is None:
                    raise ValueError("`max_seq_len` should be specified!")

        if max_position_embeddings is not None and rbln_config.max_seq_len > max_position_embeddings:
            raise ValueError("`max_seq_len` should be less or equal than max_position_embeddings!")

        signature_params = inspect.signature(model.forward).parameters.keys()

        if rbln_config.model_input_names is None:
            for tokenizer in preprocessors:
                if hasattr(tokenizer, "model_input_names"):
                    rbln_config.model_input_names = [
                        name for name in signature_params if name in tokenizer.model_input_names
                    ]

                    invalid_params = set(rbln_config.model_input_names) - set(signature_params)
                    if invalid_params:
                        raise ValueError(f"Invalid model input names: {invalid_params}")
                    break
            if rbln_config.model_input_names is None and cls.rbln_model_input_names is not None:
                rbln_config.model_input_names = cls.rbln_model_input_names

        else:
            invalid_params = set(rbln_config.model_input_names) - set(signature_params)
            if invalid_params:
                raise ValueError(f"Invalid model input names: {invalid_params}")
            rbln_config.model_input_names = [
                name for name in signature_params if name in rbln_config.model_input_names
            ]

        if rbln_config.model_input_names is None or len(rbln_config.model_input_names) == 0:
            raise ValueError(
                "Specify the model input names obtained by the tokenizer via `rbln_model_input_names`. "
                "This is an internal error. Please report it to the developers."
            )

        if rbln_config.model_input_shapes is None:
            input_info = [
                (model_input_name, [rbln_config.batch_size, rbln_config.max_seq_len], cls.rbln_dtype)
                for model_input_name in rbln_config.model_input_names
            ]
        else:
            input_info = [
                (model_input_name, model_input_shape, cls.rbln_dtype)
                for model_input_name, model_input_shape in zip(
                    rbln_config.model_input_names, rbln_config.model_input_shapes
                )
            ]

        rbln_config.set_compile_cfgs([RBLNCompileConfig(input_info=input_info)])
        return rbln_config


class RBLNImageModel(RBLNModel):
    auto_model_class = AutoModel
    main_input_name = "pixel_values"
    output_class = BaseModelOutput

    @classmethod
    def _update_rbln_config(
        cls,
        preprocessors: Optional[Union["AutoFeatureExtractor", "AutoProcessor", "AutoTokenizer"]] = None,
        model: Optional["PreTrainedModel"] = None,
        model_config: Optional["PretrainedConfig"] = None,
        rbln_config: Optional[RBLNImageModelConfig] = None,
    ) -> RBLNImageModelConfig:
        return cls.update_rbln_config_for_image_model(
            preprocessors=preprocessors,
            model=model,
            model_config=model_config,
            rbln_config=rbln_config,
        )

    @classmethod
    def update_rbln_config_for_image_model(
        cls,
        preprocessors: Optional[Union["AutoFeatureExtractor", "AutoProcessor", "AutoTokenizer"]] = None,
        model: Optional["PreTrainedModel"] = None,
        model_config: Optional["PretrainedConfig"] = None,
        rbln_config: Optional[RBLNImageModelConfig] = None,
    ) -> RBLNImageModelConfig:
        if rbln_config.image_size is None:
            for processor in preprocessors:
                if hasattr(processor, "size"):
                    if all(required_key in processor.size.keys() for required_key in ["height", "width"]):
                        rbln_config.image_size = (processor.size["height"], processor.size["width"])
                    elif "shortest_edge" in processor.size.keys():
                        rbln_config.image_size = (processor.size["shortest_edge"], processor.size["shortest_edge"])
                    elif "longest_edge" in processor.size.keys():
                        rbln_config.image_size = (processor.size["longest_edge"], processor.size["longest_edge"])
                    break

            if rbln_config.image_size is None:
                rbln_config.image_size = model_config.image_size

            if rbln_config.image_size is None:
                raise ValueError("`image_size` should be specified!")

        input_info = [
            (
                cls.main_input_name,
                [rbln_config.batch_size, 3, rbln_config.image_height, rbln_config.image_width],
                "float32",
            )
        ]

        rbln_config.set_compile_cfgs([RBLNCompileConfig(input_info=input_info)])
        return rbln_config


class RBLNModelForQuestionAnswering(RBLNTransformerEncoder):
    auto_model_class = AutoModelForQuestionAnswering
    rbln_model_input_names = ["input_ids", "attention_mask", "token_type_ids"]
    output_class = QuestionAnsweringModelOutput

    def _prepare_output(self, output, return_dict):
        # Prepare QuestionAnswering specific output format.
        start_logits, end_logits = output

        if not return_dict:
            return (start_logits, end_logits)
        else:
            return QuestionAnsweringModelOutput(start_logits=start_logits, end_logits=end_logits)


class RBLNModelForSequenceClassification(RBLNTransformerEncoder):
    auto_model_class = AutoModelForSequenceClassification
    rbln_model_input_names = ["input_ids", "attention_mask"]


class RBLNModelForMaskedLM(RBLNTransformerEncoder):
    auto_model_class = AutoModelForMaskedLM
    rbln_model_input_names = ["input_ids", "attention_mask"]


class RBLNModelForTextEncoding(RBLNTransformerEncoder):
    auto_model_class = AutoModelForTextEncoding
    rbln_model_input_names = ["input_ids", "attention_mask"]


class RBLNTransformerEncoderForFeatureExtraction(RBLNTransformerEncoder):
    # TODO: RBLNModel is also for feature extraction.
    auto_model_class = AutoModel
    rbln_model_input_names = ["input_ids", "attention_mask"]


class RBLNModelForImageClassification(RBLNImageModel):
    auto_model_class = AutoModelForImageClassification


class RBLNModelForDepthEstimation(RBLNImageModel):
    auto_model_class = AutoModelForDepthEstimation

    @classmethod
    def _wrap_model_if_needed(cls, model: "PreTrainedModel", rbln_config: RBLNImageModelConfig):
        class ImageModelWrapper(nn.Module):
            def __init__(self, model: "PreTrainedModel", rbln_config: RBLNImageModelConfig):
                super().__init__()
                self.model = model
                self.rbln_config = rbln_config

            def forward(self, *args, **kwargs):
                output = self.model(*args, return_dict=True, **kwargs)
                return output.predicted_depth

        return ImageModelWrapper(model, rbln_config).eval()
