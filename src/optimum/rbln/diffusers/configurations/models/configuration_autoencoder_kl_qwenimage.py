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

from typing import Any, Optional, Tuple

from ....configuration_utils import RBLNModelConfig


class RBLNAutoencoderKLQwenImageConfig(RBLNModelConfig):
    """
    Configuration class for RBLN AutoencoderKLQwenImage (3D causal VAE).

    This class provides configuration options for the QwenImage VAE model used in
    the Qwen-Image-Edit diffusion pipeline. The VAE uses 3D causal convolutions
    and operates on 5D tensors [batch, channels, frames, height, width].
    """

    subclass_non_save_attributes = ["_vae_scale_factor", "_z_dim", "_input_channels"]
    _vae_scale_factor = None
    _z_dim = None
    _input_channels = None

    def __init__(
        self,
        batch_size: Optional[int] = None,
        sample_size: Optional[Tuple[int, int]] = None,
        uses_encoder: Optional[bool] = None,
        **kwargs: Any,
    ):
        """
        Args:
            batch_size (Optional[int]): The batch size for inference. Defaults to 1.
            sample_size (Optional[Tuple[int, int]]): The spatial dimensions (height, width)
                of the input/output images in pixels.
            uses_encoder (Optional[bool]): Whether to include the encoder part.
                Typically set by the pipeline (True for image editing).
            kwargs: Additional arguments passed to the parent RBLNModelConfig.
        """
        super().__init__(**kwargs)
        self.batch_size = batch_size or 1
        if not isinstance(self.batch_size, int) or self.batch_size < 0:
            raise ValueError(f"batch_size must be a positive integer, got {self.batch_size}")

        self.uses_encoder = uses_encoder
        self.sample_size = sample_size
        if isinstance(sample_size, int):
            self.sample_size = (sample_size, sample_size)

    @property
    def image_size(self):
        return self.sample_size
