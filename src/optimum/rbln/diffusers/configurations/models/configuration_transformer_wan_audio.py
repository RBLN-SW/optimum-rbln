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


class RBLNWanAudioModelConfig(RBLNModelConfig):
    """
    Configuration class for RBLN WanAudio Transformer models.

    This class inherits from RBLNModelConfig and provides specific configuration options
    for the Transformer model used in text-to-audio models like MOSS-SoundEffect.
    """

    def __init__(
        self,
        batch_size: int | None = None,
        num_latent_frames: int | None = None,
        max_inference_seconds: int | None = None,
        text_len: int | None = None,
        **kwargs: Any,
    ):
        """
        Args:
            batch_size (int | None): The number of clips one forward pass denoises. The pipeline runs the positive and
                the negative prompts in separate passes, so this is the number of prompts. Defaults to 1.
            num_latent_frames (int | None): The number of latent frames: `sample_rate * max_inference_seconds` divided
                by the VAE hop length. A pipeline derives it from `max_inference_seconds`; a standalone export must set
                it.
            max_inference_seconds (int | None): The seconds of audio the pipeline denoises per call, from which a
                pipeline derives `num_latent_frames`. Defaults to the checkpoint's `max_inference_seconds`.
            text_len (int | None): The number of text tokens the graph reads. Defaults to 512, the length the
                pipeline pads every prompt to.
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
        self.text_len = text_len or 512


__all__ = [
    "RBLNWanAudioModelConfig",
]
