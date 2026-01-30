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

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

from ....configuration_utils import RBLNModelConfig
from ....transformers import RBLNQwen2_5_VLForConditionalGenerationConfig, RBLNT5EncoderModelConfig
from ....utils.logging import get_logger
from ..models import RBLNAutoencoderKLCosmosConfig, RBLNAutoencoderKLWanConfig, RBLNCosmosTransformer3DModelConfig

if TYPE_CHECKING:
    from ...pipelines.cosmos.configuration_cosmos_guardrail import (
        RBLNCosmosSafetyCheckerConfig,
        RBLNCosmosSafetyCheckerV2Config,
    )


logger = get_logger(__name__)


class RBLNCosmosPipelineBaseConfig(RBLNModelConfig):
    submodules = ["text_encoder", "transformer", "vae", "safety_checker"]
    _vae_uses_encoder = False

    def __init__(
        self,
        text_encoder: Optional[RBLNT5EncoderModelConfig] = None,
        transformer: Optional[RBLNCosmosTransformer3DModelConfig] = None,
        vae: Optional[RBLNAutoencoderKLCosmosConfig] = None,
        safety_checker: Optional[RBLNCosmosSafetyCheckerConfig] = None,
        *,
        batch_size: Optional[int] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_frames: Optional[int] = None,
        max_seq_len: Optional[int] = None,
        **kwargs: Any,
    ):
        """
        Args:
            text_encoder (Optional[RBLNT5EncoderModelConfig]): Configuration for the text encoder component.
                Initialized as RBLNT5EncoderModelConfig if not provided.
            transformer (Optional[RBLNCosmosTransformer3DModelConfig]): Configuration for the Transformer model component.
                Initialized as RBLNCosmosTransformer3DModelConfig if not provided.
            vae (Optional[RBLNAutoencoderKLCosmosConfig]): Configuration for the VAE model component.
                Initialized as RBLNAutoencoderKLCosmosConfig if not provided.
            safety_checker (Optional[RBLNCosmosSafetyCheckerConfig]): Configuration for the safety checker component.
                Initialized as RBLNCosmosSafetyCheckerConfig if not provided.
            batch_size (Optional[int]): Batch size for inference, applied to all submodules.
            height (Optional[int]): Height of the generated videos.
            width (Optional[int]): Width of the generated videos.
            num_frames (Optional[int]): The number of frames in the generated video.
            max_seq_len (Optional[int]): Maximum sequence length supported by the model.
            kwargs: Additional arguments passed to the parent RBLNModelConfig.
        """
        super().__init__(**kwargs)

        self.text_encoder = self.initialize_submodule_config(
            text_encoder,
            cls_name="RBLNT5EncoderModelConfig",
            batch_size=batch_size,
            max_seq_len=max_seq_len,
        )
        self.transformer = self.initialize_submodule_config(
            transformer,
            cls_name="RBLNCosmosTransformer3DModelConfig",
            batch_size=batch_size,
            max_seq_len=max_seq_len,
            height=height,
            width=width,
            num_frames=num_frames,
            is_v2w=self.__class__._vae_uses_encoder,
        )
        self.vae = self.initialize_submodule_config(
            vae,
            cls_name="RBLNAutoencoderKLCosmosConfig",
            batch_size=batch_size,
            uses_encoder=self.__class__._vae_uses_encoder,
            height=height,
            width=width,
            num_frames=num_frames,
        )
        self.safety_checker = self.initialize_submodule_config(
            safety_checker,
            cls_name="RBLNCosmosSafetyCheckerConfig",
            batch_size=batch_size,
            height=height,
            width=width,
        )

    @property
    def batch_size(self):
        return self.vae.batch_size

    @property
    def max_seq_len(self):
        return self.text_encoder.max_seq_len


class RBLNCosmosTextToWorldPipelineConfig(RBLNCosmosPipelineBaseConfig):
    """Config for Cosmos-Predict1 Text2World Pipeline"""

    _vae_uses_encoder = False


class RBLNCosmosVideoToWorldPipelineConfig(RBLNCosmosPipelineBaseConfig):
    """Config for Cosmos-Predict1 Video2World Pipeline"""

    _vae_uses_encoder = True


class RBLNCosmos2PipelineBaseConfig(RBLNModelConfig):
    submodules = ["text_encoder", "transformer", "vae", "safety_checker"]
    _vae_uses_encoder = False

    def __init__(
        self,
        text_encoder: Optional[RBLNT5EncoderModelConfig] = None,
        transformer: Optional[RBLNCosmosTransformer3DModelConfig] = None,
        vae: Optional[RBLNAutoencoderKLWanConfig] = None,
        safety_checker: Optional[RBLNCosmosSafetyCheckerConfig] = None,
        *,
        batch_size: Optional[int] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_frames: Optional[int] = None,
        max_seq_len: Optional[int] = None,
        **kwargs: Any,
    ):
        """
        Args:
            text_encoder (Optional[RBLNT5EncoderModelConfig]): Configuration for the text encoder component.
                Initialized as RBLNT5EncoderModelConfig if not provided.
            transformer (Optional[RBLNCosmosTransformer3DModelConfig]): Configuration for the Transformer model component.
                Initialized as RBLNCosmosTransformer3DModelConfig if not provided.
            vae (Optional[RBLNAutoencoderKLWanConfig]): Configuration for the VAE model component.
                Initialized as RBLNAutoencoderKLWanConfig if not provided.
            safety_checker (Optional[RBLNCosmosSafetyCheckerConfig]): Configuration for the safety checker component.
                Initialized as RBLNCosmosSafetyCheckerConfig if not provided.
            batch_size (Optional[int]): Batch size for inference, applied to all submodules.
            height (Optional[int]): Height of the generated videos.
            width (Optional[int]): Width of the generated videos.
            num_frames (Optional[int]): The number of frames in the generated video.
            max_seq_len (Optional[int]): Maximum sequence length supported by the model.
            **kwargs: Additional arguments passed to the parent RBLNModelConfig.
        """
        super().__init__(**kwargs)

        self.text_encoder = self.initialize_submodule_config(
            text_encoder,
            cls_name="RBLNT5EncoderModelConfig",
            batch_size=batch_size,
            max_seq_len=max_seq_len,
        )
        self.transformer = self.initialize_submodule_config(
            transformer,
            cls_name="RBLNCosmosTransformer3DModelConfig",
            batch_size=batch_size,
            max_seq_len=max_seq_len,
            height=height,
            width=width,
            num_frames=num_frames,
            is_v2w=self.__class__._vae_uses_encoder,
        )
        self.vae = self.initialize_submodule_config(
            vae,
            cls_name="RBLNAutoencoderKLWanConfig",
            batch_size=batch_size,
            uses_encoder=self.__class__._vae_uses_encoder,
            height=height,
            width=width,
            num_frames=num_frames,
        )
        self.safety_checker = self.initialize_submodule_config(
            safety_checker,
            cls_name="RBLNCosmosSafetyCheckerConfig",
            batch_size=batch_size,
            height=height,
            width=width,
        )

    @property
    def batch_size(self):
        return self.vae.batch_size

    @property
    def max_seq_len(self):
        return self.text_encoder.max_seq_len


class RBLNCosmos2TextToImagePipelineConfig(RBLNCosmos2PipelineBaseConfig):
    """Config for Cosmos-Predict2 Text2Image Pipeline"""

    _vae_uses_encoder = False


class RBLNCosmos2VideoToWorldPipelineConfig(RBLNCosmos2PipelineBaseConfig):
    """Config for Cosmos-Predict2 Video2World Pipeline"""

    _vae_uses_encoder = True


class RBLNCosmos2_5_PredictBasePipelineConfig(RBLNModelConfig):
    """Config for Cosmos-Predict2.5 Pipeline"""

    submodules = ["text_encoder", "transformer", "vae", "safety_checker"]
    _vae_uses_encoder = True

    def __init__(
        self,
        text_encoder: Optional[RBLNQwen2_5_VLForConditionalGenerationConfig] = None,
        transformer: Optional[RBLNCosmosTransformer3DModelConfig] = None,
        vae: Optional[RBLNAutoencoderKLWanConfig] = None,
        safety_checker: Optional[RBLNCosmosSafetyCheckerV2Config] = None,
        *,
        batch_size: Optional[int] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_frames: Optional[int] = None,
        max_seq_len: Optional[int] = None,
        **kwargs: Any,
    ):
        """
        Args:
            text_encoder (Optional[RBLNQwen2_5_VLForConditionalGenerationConfig]): Configuration for the text encoder component.
                Initialized as RBLNQwen2_5_VLForConditionalGenerationConfig if not provided.
            transformer (Optional[RBLNCosmosTransformer3DModelConfig]): Configuration for the Transformer model component.
                Initialized as RBLNCosmosTransformer3DModelConfig if not provided.
            vae (Optional[RBLNAutoencoderKLWanConfig]): Configuration for the VAE model component.
                Initialized as RBLNAutoencoderKLWanConfig if not provided.
            safety_checker (Optional[RBLNCosmosSafetyCheckerV2Config]): Configuration for the safety checker component.
                Initialized as RBLNCosmosSafetyCheckerV2Config if not provided.
            batch_size (Optional[int]): Batch size for inference, applied to all submodules.
            height (Optional[int]): Height of the generated videos.
            width (Optional[int]): Width of the generated videos.
            num_frames (Optional[int]): The number of frames in the generated video.
            max_seq_len (Optional[int]): Maximum sequence length supported by the model.
            **kwargs: Additional arguments passed to the parent RBLNModelConfig.
        """
        super().__init__(**kwargs)

        self.text_encoder = self.initialize_submodule_config(
            text_encoder,
            cls_name="RBLNQwen2_5_VLForConditionalGenerationConfig",
            batch_size=batch_size,
            max_seq_len=max_seq_len,
            output_hidden_states=True,
        )
        self.transformer = self.initialize_submodule_config(
            transformer,
            cls_name="RBLNCosmosTransformer3DModelConfig",
            batch_size=batch_size,
            max_seq_len=max_seq_len,
            height=height,
            width=width,
            num_frames=num_frames,
            is_v2w=self.__class__._vae_uses_encoder,
        )
        self.vae = self.initialize_submodule_config(
            vae,
            cls_name="RBLNAutoencoderKLWanConfig",
            batch_size=batch_size,
            uses_encoder=self.__class__._vae_uses_encoder,
            height=height,
            width=width,
            num_frames=num_frames,
        )
        self.safety_checker = self.initialize_submodule_config(
            safety_checker,
            cls_name="RBLNCosmosSafetyCheckerV2Config",
            batch_size=batch_size,
            height=height,
            width=width,
        )

    @property
    def batch_size(self):
        return self.vae.batch_size

    @property
    def max_seq_len(self):
        return self.text_encoder.max_seq_len
