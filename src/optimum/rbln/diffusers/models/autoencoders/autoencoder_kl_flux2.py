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

import torch
from diffusers import AutoencoderKLFlux2

from ....configuration_utils import RBLNCompileConfig
from ...configurations.models.configuration_autoencoder_kl_flux2 import RBLNAutoencoderKLFlux2Config
from .autoencoder_kl import RBLNAutoencoderKL


class RBLNAutoencoderKLFlux2(RBLNAutoencoderKL):
    """Original FLUX.2 encoder/decoder with persisted nonaffine BN buffers.

    Reference images are encoded individually by the native pipeline. The
    decoder processes the complete request batch. Latent normalization stays in
    the original pipeline and reads the checkpoint's unmodified running stats.
    """

    auto_model_class = AutoencoderKLFlux2
    _rbln_config_class = RBLNAutoencoderKLFlux2Config

    def __post_init__(self, **kwargs):
        super().__post_init__(**kwargs)
        channels = self.config.latent_channels * self.config.patch_size[0] * self.config.patch_size[1]
        mean = self.rbln_config.bn_running_mean
        variance = self.rbln_config.bn_running_var
        if mean is None or variance is None or len(mean) != channels or len(variance) != channels:
            raise ValueError("The original FLUX.2 latent normalization buffers are required")
        self.bn = torch.nn.BatchNorm2d(channels, eps=self.config.batch_norm_eps, affine=False).eval()
        self.bn.running_mean.copy_(torch.tensor(mean))
        self.bn.running_var.copy_(torch.tensor(variance))

    @classmethod
    def _update_rbln_config(cls, preprocessors, model, model_config, rbln_config):
        rbln_config = super()._update_rbln_config(preprocessors, model, model_config, rbln_config)
        rbln_config.bn_running_mean = model.bn.running_mean.detach().cpu().tolist()
        rbln_config.bn_running_var = model.bn.running_var.detach().cpu().tolist()
        if rbln_config.uses_encoder:
            height, width = rbln_config.reference_image_size
            encoder = RBLNCompileConfig(
                compiled_model_name="encoder",
                input_info=[("x", [1, model_config.in_channels, height, width], rbln_config.dtype)],
            )
            rbln_config.set_compile_cfgs([encoder, rbln_config.compile_cfgs[-1]])
        return rbln_config

    @classmethod
    def update_rbln_config_using_pipe(cls, pipe, rbln_config, submodule_name):
        rbln_config.vae.vae_scale_factor = pipe.vae_scale_factor
        return rbln_config


__all__ = ["RBLNAutoencoderKLFlux2"]
