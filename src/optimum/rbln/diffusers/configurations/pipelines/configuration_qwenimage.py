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
from ....transformers import RBLNQwen2_5_VLModelConfig
from ..models import RBLNAutoencoderKLQwenImageConfig, RBLNQwenImageTransformer2DModelConfig


class RBLNQwenImageEditPlusPipelineConfig(RBLNModelConfig):
    """
    Configuration for the QwenImageEditPlus pipeline.

    This pipeline uses Qwen2.5-VL as the text+image encoder, QwenImageTransformer2DModel
    as the denoising transformer, and AutoencoderKLQwenImage as the 3D causal VAE.
    Since this is an image editing pipeline, the VAE encoder is always enabled.

    The text_encoder (Qwen2.5-VL) is compiled via RBLNQwen2_5_VLModel in encoder-only
    mode (prefill only, no generation head), extracting hidden states for the diffusion
    transformer. output_hidden_states is enabled by default.
    """

    submodules = ["text_encoder", "transformer", "vae"]
    _vae_uses_encoder = True

    def __init__(
        self,
        text_encoder: Optional[RBLNQwen2_5_VLModelConfig] = None,
        transformer: Optional[RBLNQwenImageTransformer2DModelConfig] = None,
        vae: Optional[RBLNAutoencoderKLQwenImageConfig] = None,
        *,
        image_size: Optional[Tuple[int, int]] = None,
        batch_size: Optional[int] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        prompt_embed_length: Optional[int] = None,
        **kwargs: Any,
    ):
        """
        Args:
            text_encoder: Configuration for the Qwen2.5-VL text+image encoder.
            transformer: Configuration for the QwenImageTransformer2DModel.
            vae: Configuration for the AutoencoderKLQwenImage.
            image_size: Dimensions for the generated/edited images.
            batch_size: Batch size for inference, applied to all submodules.
            height: Height of the generated images.
            width: Width of the generated images.
            prompt_embed_length: Maximum length of prompt embeddings from the text encoder.
            kwargs: Additional arguments passed to the parent RBLNModelConfig.
        """
        super().__init__(**kwargs)

        if image_size is not None and (
            height is not None or width is not None
        ):
            raise ValueError("image_size cannot be provided alongside img_height/img_width or height/width")

        if height is not None and width is not None:
            image_size = (height, width)
        elif (height is not None and width is None) or (height is None and width is not None):
            raise ValueError("Both height and width must be provided together if used")

        self.text_encoder = self.initialize_submodule_config(
            text_encoder,
            cls_name="RBLNQwen2_5_VLModelConfig",
            use_inputs_embeds=True,
            output_hidden_states=True,
            max_seq_len=prompt_embed_length,
                # force_kwargs=True,
        )
        self.transformer = self.initialize_submodule_config(
            transformer,
            cls_name="RBLNQwenImageTransformer2DModelConfig",
            batch_size=batch_size,
            prompt_embed_length=prompt_embed_length,
            # force_kwargs=True,
        )
        self.vae = self.initialize_submodule_config(
            vae,
            cls_name="RBLNAutoencoderKLQwenImageConfig",
            batch_size=batch_size,
            uses_encoder=self.__class__._vae_uses_encoder,
            sample_size=image_size,
            # force_kwargs=True,
        )

    @property
    def batch_size(self):
        return self.vae.batch_size

    @property
    def image_size(self):
        return self.vae.sample_size
