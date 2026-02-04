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

from typing import Any, ClassVar

from pydantic import Field, model_validator

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

    # Pass-through parameters (excluded from serialization.)
    effective_batch_size: int | None = Field(
        default=None,
        alias="batch_size",
        exclude=True,
        description="Batch size for inference. Forwarded to all submodules.",
    )
    effective_height: int | None = Field(
        default=None,
        alias="height",
        exclude=True,
        description="Height of the generated video frames. Forwarded to transformer, vae, and safety_checker.",
    )
    effective_width: int | None = Field(
        default=None,
        alias="width",
        exclude=True,
        description="Width of the generated video frames. Forwarded to transformer, vae, and safety_checker.",
    )
    effective_num_frames: int | None = Field(
        default=None,
        alias="num_frames",
        exclude=True,
        description="Number of video frames to generate. Forwarded to transformer and vae.",
    )
    effective_fps: int | None = Field(
        default=None,
        alias="fps",
        exclude=True,
        description="Frames per second for the generated video. Forwarded to transformer.",
    )
    effective_max_seq_len: int | None = Field(
        default=None,
        alias="max_seq_len",
        exclude=True,
        description="Maximum sequence length for the text encoder. Forwarded to text_encoder and transformer.",
    )

    @model_validator(mode="after")
    def initialize_submodules(self) -> "RBLNCosmosPipelineBaseConfig":
        """Initialize submodule configs with pass-through parameters."""
        # Guard against re-entry during submodule initialization
        if getattr(self, "_submodules_initialized", False):
            return self
        object.__setattr__(self, "_submodules_initialized", True)

        self.text_encoder = self.initialize_submodule_config(
            self.text_encoder,
            cls_name="RBLNT5EncoderModelConfig",
            batch_size=self.effective_batch_size,
            max_seq_len=self.effective_max_seq_len,
        )
        self.transformer = self.initialize_submodule_config(
            self.transformer,
            cls_name="RBLNCosmosTransformer3DModelConfig",
            batch_size=self.effective_batch_size,
            max_seq_len=self.effective_max_seq_len,
            height=self.effective_height,
            width=self.effective_width,
            num_frames=self.effective_num_frames,
            fps=self.effective_fps,
        )
        self.vae = self.initialize_submodule_config(
            self.vae,
            cls_name="RBLNAutoencoderKLCosmosConfig",
            batch_size=self.effective_batch_size,
            uses_encoder=self.__class__._vae_uses_encoder,
            height=self.effective_height,
            width=self.effective_width,
            num_frames=self.effective_num_frames,
        )
        self.safety_checker = self.initialize_submodule_config(
            self.safety_checker,
            cls_name="RBLNCosmosSafetyCheckerConfig",
            batch_size=self.effective_batch_size,
            height=self.effective_height,
            width=self.effective_width,
        )

        return self

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
