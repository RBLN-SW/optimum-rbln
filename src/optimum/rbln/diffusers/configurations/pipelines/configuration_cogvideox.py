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
from ....transformers import RBLNT5EncoderModelConfig
from ..models import RBLNAutoencoderKLCogVideoXConfig, RBLNCogVideoXTransformer3DModelConfig


class RBLNCogVideoXPipelineConfig(RBLNModelConfig):
    """Config for the CogVideoX text-to-video pipeline."""

    submodules = ["text_encoder", "transformer", "vae"]
    _vae_uses_encoder = False

    def __init__(
        self,
        text_encoder: RBLNT5EncoderModelConfig | None = None,
        transformer: RBLNCogVideoXTransformer3DModelConfig | None = None,
        vae: RBLNAutoencoderKLCogVideoXConfig | None = None,
        *,
        batch_size: int | None = None,
        num_frames: int | None = None,
        height: int | None = None,
        width: int | None = None,
        max_seq_len: int | None = None,
        guidance_scale: float | None = None,
        **kwargs: Any,
    ):
        """
        Args:
            text_encoder (RBLNT5EncoderModelConfig | None): Configuration for the T5 text encoder.
            transformer (RBLNCogVideoXTransformer3DModelConfig | None): Configuration for the transformer.
            vae (RBLNAutoencoderKLCogVideoXConfig | None): Configuration for the VAE (decoder only).
            batch_size (int | None): Batch size for inference (number of prompts), applied to all submodules.
            num_frames (int | None): Number of frames in the generated video. Defaults to the transformer
                config `sample_frames` (49 for CogVideoX-2b).
            height (int | None): Height of the generated video. Defaults to 480 for CogVideoX-2b.
            width (int | None): Width of the generated video. Defaults to 720 for CogVideoX-2b.
            max_seq_len (int | None): Text sequence length. Defaults to 226, the value CogVideoX uses.
            guidance_scale (float | None): Scale for classifier-free guidance. When > 1, the transformer
                batch size is doubled.
            kwargs: Additional arguments passed to the parent RBLNModelConfig.
        """
        super().__init__(**kwargs)
        if (height is None) != (width is None):
            raise ValueError("Both height and width must be provided together if used")

        max_seq_len = max_seq_len or 226
        self.text_encoder = self.initialize_submodule_config(
            text_encoder,
            cls_name="RBLNT5EncoderModelConfig",
            batch_size=batch_size,
            max_seq_len=max_seq_len,
            model_input_names=["input_ids"],
        )
        self.transformer = self.initialize_submodule_config(
            transformer,
            cls_name="RBLNCogVideoXTransformer3DModelConfig",
            num_frames=num_frames,
            max_text_seq_length=max_seq_len,
        )
        self.vae = self.initialize_submodule_config(
            vae,
            cls_name="RBLNAutoencoderKLCogVideoXConfig",
            batch_size=batch_size,
            uses_encoder=self.__class__._vae_uses_encoder,
            num_frames=num_frames,
            height=height,
            width=width,
        )

        if guidance_scale is None:
            guidance_scale = self.get_default_values_for_original_cls("__call__", ["guidance_scale"])["guidance_scale"]

        if not self.transformer.batch_size_is_specified:
            self.transformer.batch_size = self.text_encoder.batch_size * (2 if guidance_scale > 1.0 else 1)

    @property
    def batch_size(self):
        return self.text_encoder.batch_size

    @property
    def max_seq_len(self):
        return self.text_encoder.max_seq_len


__all__ = [
    "RBLNCogVideoXPipelineConfig",
]
