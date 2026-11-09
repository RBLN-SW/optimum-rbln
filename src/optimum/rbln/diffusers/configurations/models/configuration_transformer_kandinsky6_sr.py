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


class RBLNKandinsky6SRTransformer3DModelConfig(RBLNModelConfig):
    """
    Configuration class for RBLN Kandinsky 6 video super-resolution transformer models.

    This class inherits from RBLNModelConfig and provides specific configuration options
    for the transformer that refines the latent tiles of the Kandinsky 6 super-resolution pipeline.
    """

    def __init__(
        self,
        batch_size: int | None = None,
        num_latent_frames: int | None = None,
        sample_size: tuple[int, int] | None = None,
        **kwargs: Any,
    ):
        """
        Args:
            batch_size (int | None): The number of tiles one transformer call refines, as `tiles_batch_size`
                of the pipeline call. Defaults to 1.
            num_latent_frames (int | None): The number of latent frames of a tile. Filled by the pipeline
                config; required for a standalone compile.
            sample_size (tuple[int, int] | None): The latent (height, width) of a tile. Filled by the pipeline
                config; required for a standalone compile.
            kwargs: Additional arguments passed to the parent RBLNModelConfig.

        Raises:
            ValueError: If batch_size is not a positive integer.
        """
        super().__init__(**kwargs)
        self.batch_size = batch_size or 1
        if not isinstance(self.batch_size, int) or self.batch_size < 0:
            raise ValueError(f"batch_size must be a positive integer, got {self.batch_size}")

        self.num_latent_frames = num_latent_frames
        self.sample_size = tuple(sample_size) if sample_size is not None else None


__all__ = [
    "RBLNKandinsky6SRTransformer3DModelConfig",
]
