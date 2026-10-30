# Copyright 2026 Rebellions Inc. All rights reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
from typing import TYPE_CHECKING

import torch
from diffusers import AutoencoderKLFlux2

from ...configurations import RBLNAutoencoderKLFlux2Config
from .autoencoder_kl import RBLNAutoencoderKL


if TYPE_CHECKING:
    from pathlib import Path

    from transformers import PreTrainedModel


class RBLNAutoencoderKLFlux2(RBLNAutoencoderKL):
    """
    RBLN implementation of AutoencoderKLFlux2 (VAE) for diffusion models.

    This model is used to accelerate AutoencoderKLFlux2 (VAE) models from diffusers library on RBLN NPUs.
    The batch-norm statistics that FLUX.2 pipelines apply to the latents on the host stay available as `bn`.

    This class inherits from [`RBLNAutoencoderKL`]. Check the superclass documentation for the generic methods
    the library implements for all its models.
    """

    auto_model_class = AutoencoderKLFlux2
    _rbln_config_class = RBLNAutoencoderKLFlux2Config

    def __post_init__(self, **kwargs):
        super().__post_init__(**kwargs)
        artifacts = torch.load(self.model_save_dir / self.subfolder / "torch_artifacts.pth", weights_only=False)
        self.bn = torch.nn.BatchNorm2d(
            math.prod(self.config.patch_size) * self.config.latent_channels,
            eps=self.config.batch_norm_eps,
            momentum=self.config.batch_norm_momentum,
            affine=False,
            track_running_stats=True,
        )
        self.bn.load_state_dict(artifacts["bn"])
        self.bn.to(self.rbln_config.dtype)

    @classmethod
    def save_torch_artifacts(
        cls,
        model: "PreTrainedModel",
        save_dir_path: "Path",
        subfolder: str,
        rbln_config: RBLNAutoencoderKLFlux2Config,
    ):
        save_dict = {"bn": model.bn.state_dict()}
        torch.save(save_dict, save_dir_path / subfolder / "torch_artifacts.pth")


__all__ = [
    "RBLNAutoencoderKLFlux2",
]
