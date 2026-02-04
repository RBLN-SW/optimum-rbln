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

from typing import Any, ClassVar, Optional

from pydantic import Field

from ....configuration_utils import RBLNModelConfig
from ....transformers import RBLNT5EncoderModelConfig
from ....utils.logging import get_logger
from ...pipelines.cosmos.cosmos_guardrail import RBLNCosmosSafetyCheckerConfig
from ..models import RBLNAutoencoderKLCosmosConfig, RBLNCosmosTransformer3DModelConfig


logger = get_logger(__name__)


class RBLNCosmosPipelineBaseConfig(RBLNModelConfig):
    """Base configuration for Cosmos pipelines."""

    submodules: ClassVar[list[str]] = ["text_encoder", "transformer", "vae", "safety_checker"]
    _vae_uses_encoder: ClassVar[bool] = False

    text_encoder: dict[str, Any] | RBLNT5EncoderModelConfig | None = Field(
        default=None, description="Configuration for the text encoder component."
    )
    transformer: dict[str, Any] | RBLNCosmosTransformer3DModelConfig | None = Field(
        default=None, description="Configuration for the Transformer model component."
    )
    vae: dict[str, Any] | RBLNAutoencoderKLCosmosConfig | None = Field(
        default=None, description="Configuration for the VAE model component."
    )
    safety_checker: dict[str, Any] | RBLNCosmosSafetyCheckerConfig | None = Field(
        default=None, description="Configuration for the safety checker component."
    )

    def __init__(
        self,
        *,
        batch_size: Optional[int] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_frames: Optional[int] = None,
        fps: Optional[int] = None,
        max_seq_len: Optional[int] = None,
        **data: Any,
    ):
        super().__init__(**data)

        self.text_encoder = self.initialize_submodule_config(
            self.text_encoder,
            cls_name="RBLNT5EncoderModelConfig",
            batch_size=batch_size,
            max_seq_len=max_seq_len,
        )
        self.transformer = self.initialize_submodule_config(
            self.transformer,
            cls_name="RBLNCosmosTransformer3DModelConfig",
            batch_size=batch_size,
            max_seq_len=max_seq_len,
            height=height,
            width=width,
            num_frames=num_frames,
            fps=fps,
        )
        self.vae = self.initialize_submodule_config(
            self.vae,
            cls_name="RBLNAutoencoderKLCosmosConfig",
            batch_size=batch_size,
            uses_encoder=self.__class__._vae_uses_encoder,
            height=height,
            width=width,
            num_frames=num_frames,
        )
        self.safety_checker = self.initialize_submodule_config(
            self.safety_checker,
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
    """Config for Cosmos Text2World Pipeline"""

    _vae_uses_encoder: ClassVar[bool] = False


class RBLNCosmosVideoToWorldPipelineConfig(RBLNCosmosPipelineBaseConfig):
    """Config for Cosmos Video2World Pipeline"""

    _vae_uses_encoder: ClassVar[bool] = True
