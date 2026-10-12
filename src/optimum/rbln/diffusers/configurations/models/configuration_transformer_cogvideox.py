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


class RBLNCogVideoXTransformer3DModelConfig(RBLNModelConfig):
    """
    Configuration class for RBLN CogVideoX Transformer models.

    This class inherits from RBLNModelConfig and provides specific configuration options
    for the CogVideoX 3D transformer used in text-to-video generation.
    """

    subclass_non_save_attributes = ["_batch_size_is_specified"]

    def __init__(
        self,
        batch_size: int | None = None,
        num_frames: int | None = None,
        sample_size: tuple[int, int] | None = None,
        max_text_seq_length: int | None = None,
        vae_scale_factor_temporal: int | None = None,
        **kwargs: Any,
    ):
        """
        Args:
            batch_size (int | None): The batch size for inference. Defaults to 1. With classifier-free
                guidance the pipeline runs the conditional and unconditional inputs together, so it sets 2.
            num_frames (int | None): The number of frames in the generated video. Defaults to the
                transformer config `sample_frames`.
            sample_size (tuple[int, int] | None): The latent (height, width). Defaults to the transformer
                config (`sample_height`, `sample_width`).
            max_text_seq_length (int | None): The length of the text embeddings. Defaults to the
                transformer config `max_text_seq_length`.
            vae_scale_factor_temporal (int | None): The temporal compression of the VAE. Defaults to the
                transformer config `temporal_compression_ratio`.
            kwargs: Additional arguments passed to the parent RBLNModelConfig.

        Raises:
            ValueError: If batch_size is not a positive integer.
        """
        super().__init__(**kwargs)
        self._batch_size_is_specified = batch_size is not None

        self.batch_size = batch_size or 1
        if not isinstance(self.batch_size, int) or self.batch_size < 0:
            raise ValueError(f"batch_size must be a positive integer, got {self.batch_size}")

        self.num_frames = num_frames
        self.sample_size = tuple(sample_size) if sample_size is not None else None
        self.max_text_seq_length = max_text_seq_length
        self.vae_scale_factor_temporal = vae_scale_factor_temporal

    @property
    def batch_size_is_specified(self):
        return self._batch_size_is_specified

    @property
    def num_latent_frames(self) -> int:
        return (self.num_frames - 1) // self.vae_scale_factor_temporal + 1


__all__ = [
    "RBLNCogVideoXTransformer3DModelConfig",
]
