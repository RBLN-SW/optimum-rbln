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

from transformers.models.auto.modeling_auto import (
    MODEL_FOR_AUDIO_CLASSIFICATION_MAPPING,
    MODEL_FOR_AUDIO_CLASSIFICATION_MAPPING_NAMES,
    MODEL_FOR_CAUSAL_LM_MAPPING,
    MODEL_FOR_CAUSAL_LM_MAPPING_NAMES,
    MODEL_FOR_CTC_MAPPING,
    MODEL_FOR_CTC_MAPPING_NAMES,
    MODEL_FOR_DEPTH_ESTIMATION_MAPPING,
    MODEL_FOR_DEPTH_ESTIMATION_MAPPING_NAMES,
    MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING,
    MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING_NAMES,
    MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING,
    MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES,
    MODEL_FOR_MASKED_LM_MAPPING,
    MODEL_FOR_MASKED_LM_MAPPING_NAMES,
    MODEL_FOR_QUESTION_ANSWERING_MAPPING,
    MODEL_FOR_QUESTION_ANSWERING_MAPPING_NAMES,
    MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING,
    MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING_NAMES,
    MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING,
    MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING_NAMES,
    MODEL_FOR_SPEECH_SEQ_2_SEQ_MAPPING,
    MODEL_FOR_SPEECH_SEQ_2_SEQ_MAPPING_NAMES,
    MODEL_FOR_TEXT_ENCODING_MAPPING,
    MODEL_FOR_TEXT_ENCODING_MAPPING_NAMES,
    MODEL_FOR_VISION_2_SEQ_MAPPING,
    MODEL_FOR_VISION_2_SEQ_MAPPING_NAMES,
    MODEL_FOR_ZERO_SHOT_OBJECT_DETECTION_MAPPING,
    MODEL_FOR_ZERO_SHOT_OBJECT_DETECTION_MAPPING_NAMES,
    MODEL_MAPPING,
    MODEL_MAPPING_NAMES,
)

from .auto_factory import _BaseAutoModelClass


MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.update(
    {
        "midm": "MidmLMHeadModel",
        "exaone": "ExaoneForCausalLM",
        "MiniMaxM2": "MiniMaxM2ForCausalLM",
    }
)


class RBLNAutoModel(_BaseAutoModelClass):
    """Automatically detect all supported transformers models."""

    _model_mapping = MODEL_MAPPING
    _model_mapping_names = MODEL_MAPPING_NAMES


class RBLNAutoModelForCTC(_BaseAutoModelClass):
    """Automatically detect Connectionist Temporal Classification (CTC) head Models."""

    _model_mapping = MODEL_FOR_CTC_MAPPING
    _model_mapping_names = MODEL_FOR_CTC_MAPPING_NAMES


class RBLNAutoModelForCausalLM(_BaseAutoModelClass):
    """Automatically detect Casual Language Models."""

    """"""
    _model_mapping = MODEL_FOR_CAUSAL_LM_MAPPING
    _model_mapping_names = MODEL_FOR_CAUSAL_LM_MAPPING_NAMES


class RBLNAutoModelForSeq2SeqLM(_BaseAutoModelClass):
    """Automatically detect Sequence to Sequence Language Models."""

    _model_mapping = MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING
    _model_mapping_names = MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING_NAMES


class RBLNAutoModelForSpeechSeq2Seq(_BaseAutoModelClass):
    """Automatically detect Sequence to Sequence Generation Models."""

    _model_mapping = MODEL_FOR_SPEECH_SEQ_2_SEQ_MAPPING
    _model_mapping_names = MODEL_FOR_SPEECH_SEQ_2_SEQ_MAPPING_NAMES


class RBLNAutoModelForDepthEstimation(_BaseAutoModelClass):
    """Automatically detect Speech Sequence to Sequence Language Models."""

    _model_mapping = MODEL_FOR_DEPTH_ESTIMATION_MAPPING
    _model_mapping_names = MODEL_FOR_DEPTH_ESTIMATION_MAPPING_NAMES


class RBLNAutoModelForSequenceClassification(_BaseAutoModelClass):
    """Automatically detect Sequence Classification Models."""

    _model_mapping = MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING
    _model_mapping_names = MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING_NAMES


class RBLNAutoModelForVision2Seq(_BaseAutoModelClass):
    """Automatically detect Vision to Sequence Generation Models."""

    _model_mapping = MODEL_FOR_VISION_2_SEQ_MAPPING
    _model_mapping_names = MODEL_FOR_VISION_2_SEQ_MAPPING_NAMES


class RBLNAutoModelForImageTextToText(_BaseAutoModelClass):
    """Automatically detect Image and Text to Text Generation Models."""

    _model_mapping = MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING
    _model_mapping_names = MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES


class RBLNAutoModelForMaskedLM(_BaseAutoModelClass):
    """Automatically detect Masked Lanuage Models."""

    _model_mapping = MODEL_FOR_MASKED_LM_MAPPING
    _model_mapping_names = MODEL_FOR_MASKED_LM_MAPPING_NAMES


class RBLNAutoModelForAudioClassification(_BaseAutoModelClass):
    """Automatically detect Audio Classification Models."""

    _model_mapping = MODEL_FOR_AUDIO_CLASSIFICATION_MAPPING
    _model_mapping_names = MODEL_FOR_AUDIO_CLASSIFICATION_MAPPING_NAMES


class RBLNAutoModelForImageClassification(_BaseAutoModelClass):
    """Automatically detect Image Classification Models."""

    _model_mapping = MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING
    _model_mapping_names = MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING_NAMES


class RBLNAutoModelForQuestionAnswering(_BaseAutoModelClass):
    """Automatically detect Question Answering Models."""

    _model_mapping = MODEL_FOR_QUESTION_ANSWERING_MAPPING
    _model_mapping_names = MODEL_FOR_QUESTION_ANSWERING_MAPPING_NAMES


class RBLNAutoModelForTextEncoding(_BaseAutoModelClass):
    """Automatically detect Text Encoding Models."""

    _model_mapping = MODEL_FOR_TEXT_ENCODING_MAPPING
    _model_mapping_names = MODEL_FOR_TEXT_ENCODING_MAPPING_NAMES


class RBLNAutoModelForZeroShotObjectDetection(_BaseAutoModelClass):
    """Automatically detect Zero Shot Object Detection Models."""

    _model_mapping = MODEL_FOR_ZERO_SHOT_OBJECT_DETECTION_MAPPING
    _model_mapping_names = MODEL_FOR_ZERO_SHOT_OBJECT_DETECTION_MAPPING_NAMES
