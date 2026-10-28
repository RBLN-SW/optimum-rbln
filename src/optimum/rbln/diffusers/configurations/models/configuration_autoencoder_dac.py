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

from ....configuration_utils import RBLNModelConfig


class RBLNDACConfig(RBLNModelConfig):
    """Configuration class for RBLN DAC audio Variational Autoencoder (VAE) models. Only the decoder is compiled."""

    def __init__(
        self,
        batch_size: int | None = None,
        num_latent_frames: int | None = None,
        max_inference_seconds: int | None = None,
        **kwargs: Any,
    ):
        """
        Args:
            batch_size (int | None): The number of clips one decode call takes. Defaults to 1.
            num_latent_frames (int | None): The number of latent frames to decode: `sample_rate * max_inference_seconds`
                divided by the hop length. A pipeline derives it from `max_inference_seconds`; a standalone export must set
                it.
            max_inference_seconds (int | None): The seconds of audio the pipeline decodes per call, from which a
                pipeline derives `num_latent_frames`. Defaults to the checkpoint's `max_inference_seconds`.
            kwargs: Additional arguments passed to the parent RBLNModelConfig.

        Raises:
            ValueError: If batch_size is not a positive integer.
        """
        super().__init__(**kwargs)
        self.batch_size = batch_size or 1
        if not isinstance(self.batch_size, int) or self.batch_size < 0:
            raise ValueError(f"batch_size must be a positive integer, got {self.batch_size}")
        self.num_latent_frames = num_latent_frames
        self.max_inference_seconds = max_inference_seconds


__all__ = [
    "RBLNDACConfig",
]
