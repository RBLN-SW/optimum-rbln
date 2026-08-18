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

from typing import Any

from ....configuration_utils import RBLNModelConfig


class RBLNCosmosTransformer3DModelConfig(RBLNModelConfig):
    """
    Configuration class for RBLN Cosmos Transformer models.

    This class inherits from RBLNModelConfig and provides specific configuration options
    for Transformer models used in diffusion models like Cosmos.
    """

    def __init__(
        self,
        batch_size: int | None = None,
        num_frames: int | None = None,
        height: int | None = None,
        width: int | None = None,
        fps: int | None = None,
        max_seq_len: int | None = None,
        embedding_dim: int | None = None,
        num_channels_latents: int | None = None,
        num_latent_frames: int | None = None,
        latent_height: int | None = None,
        latent_width: int | None = None,
        **kwargs: Any,
    ):
        """
        Args:
            batch_size (int | None): The batch size for inference. Defaults to 1.
            num_frames (int | None): The number of frames in the generated video. Defaults to 121.
            height (int | None): The height in pixels of the generated video. Defaults to 704.
            width (int | None): The width in pixels of the generated video. Defaults to 1280.
            fps (int | None): The frames per second of the generated video.  Defaults to 30.
            max_seq_len (int | None): Maximum sequence length of prompt embeds.
            embedding_dim (int | None): Embedding vector dimension of prompt embeds.
            num_channels_latents (int | None): The number of channels in latent space.
            latent_height (int | None): The height in pixels in latent space.
            latent_width (int | None): The width in pixels in latent space.
            kwargs: Additional arguments passed to the parent RBLNModelConfig.

        Raises:
            ValueError: If batch_size is not a positive integer.
        """
        if kwargs.get("timeout") is None:
            kwargs["timeout"] = 80

        super().__init__(**kwargs)
        self.batch_size = batch_size or 1
        self.num_frames = num_frames
        self.height = height
        self.width = width

        self.max_seq_len = max_seq_len
        self.num_channels_latents = num_channels_latents
        self.num_latent_frames = num_latent_frames
        self.latent_height = latent_height
        self.latent_width = latent_width
        self.embedding_dim = embedding_dim
        self.is_v2w = is_v2w

        if not isinstance(self.batch_size, int) or self.batch_size < 0:
            raise ValueError(f"batch_size must be a positive integer, got {self.batch_size}")
