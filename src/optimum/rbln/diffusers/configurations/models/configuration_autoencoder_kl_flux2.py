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

from typing import Any

from .configuration_autoencoder_kl import RBLNAutoencoderKLConfig
from .configuration_transformer_flux2 import RBLNFlux2Transformer2DModelConfig


class RBLNAutoencoderKLFlux2Config(RBLNAutoencoderKLConfig):
    """Native FLUX.2 VAE geometry and nonaffine latent normalization statistics."""

    def __init__(
        self,
        reference_image_size: tuple[int, int] | None = None,
        bn_running_mean: list[float] | None = None,
        bn_running_var: list[float] | None = None,
        **kwargs: Any,
    ):
        """
        Args:
            reference_image_size (tuple[int, int] | None): Encoder image geometry after native preprocessing.
            bn_running_mean (list[float] | None): Original nonaffine batch-normalization running mean, saved at export.
            bn_running_var (list[float] | None): Original nonaffine batch-normalization running variance, saved at export.
            kwargs: Additional arguments passed to RBLNAutoencoderKLConfig.
        """
        kwargs.setdefault("sample_size", (1024, 1024))
        kwargs.setdefault("uses_encoder", True)
        super().__init__(**kwargs)
        self.sample_size = RBLNFlux2Transformer2DModelConfig._image_size(self.sample_size)
        self.reference_image_size = RBLNFlux2Transformer2DModelConfig._image_size(
            self.sample_size if reference_image_size is None else reference_image_size
        )
        self.bn_running_mean = bn_running_mean
        self.bn_running_var = bn_running_var


__all__ = ["RBLNAutoencoderKLFlux2Config"]
