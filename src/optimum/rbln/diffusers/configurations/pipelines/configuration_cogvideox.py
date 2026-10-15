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
        max_sequence_length: int | None = None,
        guidance_scale: float | None = None,
        **kwargs: Any,
    ):
        """
        Args:
            text_encoder (RBLNT5EncoderModelConfig | None): Configuration for the T5 text encoder.
            transformer (RBLNCogVideoXTransformer3DModelConfig | None): Configuration for the transformer.
            vae (RBLNAutoencoderKLCogVideoXConfig | None): Configuration for the VAE (decoder only).
            batch_size (int | None): Batch size for inference (number of prompts), for the text encoder and
                the transformer. The VAE always runs one sample at a time (batch 1, slicing on) on one device.
            num_frames (int | None): Number of frames in the generated video. Defaults to the transformer
                config `sample_frames` (49 for CogVideoX-2b).
            height (int | None): Height of the generated video. Defaults to 480 for CogVideoX-2b.
            width (int | None): Width of the generated video. Defaults to 720 for CogVideoX-2b.
            max_sequence_length (int | None): Length of the encoded prompt, as `max_sequence_length` of the
                diffusers pipeline call. Sets the text encoder input length and the text length of the
                transformer graph. Defaults to 226. Like diffusers, models with positional embeddings
                (CogVideoX-2b, CogVideoX-5b-I2V) only work with the transformer config `max_text_seq_length`.
            guidance_scale (float | None): Scale for classifier-free guidance. When > 1, the transformer
                batch size is doubled.
            kwargs: Additional arguments passed to the parent RBLNModelConfig.
        """
        super().__init__(**kwargs)
        if (height is None) != (width is None):
            raise ValueError("Both height and width must be provided together if used")

        max_sequence_length = max_sequence_length or 226
        self.text_encoder = self.initialize_submodule_config(
            text_encoder,
            cls_name="RBLNT5EncoderModelConfig",
            batch_size=batch_size,
            max_seq_len=max_sequence_length,
            model_input_names=["input_ids"],
        )
        self.transformer = self.initialize_submodule_config(
            transformer,
            cls_name="RBLNCogVideoXTransformer3DModelConfig",
            num_frames=num_frames,
            max_text_seq_length=max_sequence_length,
        )
        # The pipeline runs the VAE on one device, one sample at a time: batch-1 graphs with slicing on.
        # Pipeline values left unset are not passed, so they do not conflict with values set on the VAE.
        pipeline_values = {"num_frames": num_frames, "height": height, "width": width}
        self.vae = self.initialize_submodule_config(
            vae,
            cls_name="RBLNAutoencoderKLCogVideoXConfig",
            force_kwargs=True,
            num_devices=1,
            batch_size=1,
            use_slicing=True,
            uses_encoder=self.__class__._vae_uses_encoder,
            **{key: value for key, value in pipeline_values.items() if value is not None},
        )

        if guidance_scale is None:
            guidance_scale = self.get_default_values_for_original_cls("__call__", ["guidance_scale"])["guidance_scale"]

        if not self.transformer.batch_size_is_specified:
            self.transformer.batch_size = self.text_encoder.batch_size * (2 if guidance_scale > 1.0 else 1)

    @property
    def batch_size(self):
        return self.text_encoder.batch_size

    @property
    def max_sequence_length(self):
        return self.text_encoder.max_seq_len


class RBLNCogVideoXImageToVideoPipelineConfig(RBLNCogVideoXPipelineConfig):
    """Config for the CogVideoX image-to-video pipeline. The VAE also compiles its encoder for the input image."""

    _vae_uses_encoder = True


__all__ = [
    "RBLNCogVideoXImageToVideoPipelineConfig",
    "RBLNCogVideoXPipelineConfig",
]
