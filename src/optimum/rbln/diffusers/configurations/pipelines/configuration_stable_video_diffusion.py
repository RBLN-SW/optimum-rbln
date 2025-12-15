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

from typing import Any, Optional

from ....configuration_utils import RBLNModelConfig
from ....transformers import RBLNCLIPVisionModelWithProjectionConfig
from ..models import RBLNAutoencoderKLTemporalDecoderConfig, RBLNUNetSpatioTemporalConditionModelConfig


class RBLNStableVideoDiffusionPipelineConfig(RBLNModelConfig):
    submodules = ["image_encoder", "unet", "vae"]
    _vae_uses_encoder = True

    def __init__(
        self,
        image_encoder: Optional[RBLNCLIPVisionModelWithProjectionConfig] = None,
        unet: Optional[RBLNUNetSpatioTemporalConditionModelConfig] = None,
        vae: Optional[RBLNAutoencoderKLTemporalDecoderConfig] = None,
        *,
        batch_size: Optional[int] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_frames: Optional[int] = None,
        decode_chunk_size: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        **kwargs: Any,
    ):
        """
        Args:
            image_encoder (Optional[RBLNCLIPVisionModelWithProjectionConfig]): Configuration for the image encoder component.
                Initialized as RBLNCLIPVisionModelWithProjectionConfig if not provided.
            unet (Optional[RBLNUNetSpatioTemporalConditionModelConfig]): Configuration for the UNet model component.
                Initialized as RBLNUNetSpatioTemporalConditionModelConfig if not provided.
            vae (Optional[RBLNAutoencoderKLTemporalDecoderConfig]): Configuration for the VAE model component.
                Initialized as RBLNAutoencoderKLTemporalDecoderConfig if not provided.
            batch_size (Optional[int]): Batch size for inference, applied to all submodules.
            height (Optional[int]): Height of the generated images.
            width (Optional[int]): Width of the generated images.
            num_frames (Optional[int]): The number of frames in the generated video.
            decode_chunk_size (Optional[int]): The number of frames to decode at once during VAE decoding.
                Useful for managing memory usage during video generation.
            guidance_scale (Optional[float]): Scale for classifier-free guidance.
            kwargs: Additional arguments passed to the parent RBLNModelConfig.

        Raises:
            ValueError: If both image_size and height/width are provided.

        Note:
            When guidance_scale > 1.0, the UNet batch size is automatically doubled to
            accommodate classifier-free guidance.
        """
        super().__init__(**kwargs)
        if height is not None and width is not None:
            image_size = (height, width)
        else:
            # Get default image size from original class to set UNet, VAE image size
            height = self.get_default_values_for_original_cls("__call__", ["height"])["height"]
            width = self.get_default_values_for_original_cls("__call__", ["width"])["width"]
            image_size = (height, width)

        self.image_encoder = self.initialize_submodule_config(
            image_encoder, cls_name="RBLNCLIPVisionModelWithProjectionConfig", batch_size=batch_size
        )
        self.unet = self.initialize_submodule_config(
            unet,
            cls_name="RBLNUNetSpatioTemporalConditionModelConfig",
            num_frames=num_frames,
        )
        self.vae = self.initialize_submodule_config(
            vae,
            cls_name="RBLNAutoencoderKLTemporalDecoderConfig",
            batch_size=batch_size,
            num_frames=num_frames,
            decode_chunk_size=decode_chunk_size,
            uses_encoder=self.__class__._vae_uses_encoder,
            sample_size=image_size,  # image size is equal to sample size in vae
        )

        # Get default guidance scale from original class to set UNet batch size
        if guidance_scale is None:
            guidance_scale = self.get_default_values_for_original_cls("__call__", ["max_guidance_scale"])[
                "max_guidance_scale"
            ]

        if not self.unet.batch_size_is_specified:
            do_classifier_free_guidance = guidance_scale > 1.0
            if do_classifier_free_guidance:
                self.unet.batch_size = self.image_encoder.batch_size * 2
            else:
                self.unet.batch_size = self.image_encoder.batch_size

    @property
    def batch_size(self):
        return self.vae.batch_size

    @property
    def sample_size(self):
        return self.unet.sample_size

    @property
    def image_size(self):
        return self.vae.sample_size
