# Copyright 2024 Rebellions Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from diffusers import (
    DDPMScheduler,
    KandinskyV22CombinedPipeline,
    KandinskyV22Img2ImgCombinedPipeline,
    KandinskyV22InpaintCombinedPipeline,
    PriorTransformer,
    UnCLIPScheduler,
    UNet2DConditionModel,
    VQModel,
)
from transformers import CLIPImageProcessor, CLIPTextModelWithProjection, CLIPTokenizer, CLIPVisionModelWithProjection

from ...configurations import RBLNKandinskyV22CombinedPipelineConfig
from ...modeling_diffusers import RBLNDiffusionMixin
from .pipeline_kandinsky2_2 import RBLNKandinskyV22Pipeline
from .pipeline_kandinsky2_2_img2img import RBLNKandinskyV22Img2ImgPipeline
from .pipeline_kandinsky2_2_inpaint import RBLNKandinskyV22InpaintPipeline
from .pipeline_kandinsky2_2_prior import RBLNKandinskyV22PriorPipeline


class RBLNKandinskyV22CombinedPipeline(RBLNDiffusionMixin, KandinskyV22CombinedPipeline):
    """
    RBLN-accelerated implementation of Kandinsky 2.2 combined pipeline for end-to-end text-to-image generation.

    This pipeline compiles both prior and decoder Kandinsky 2.2 models to run efficiently on RBLN NPUs, enabling
    high-performance inference for complete text-to-image generation with distinctive artistic style.
    """

    original_class = KandinskyV22CombinedPipeline
    _rbln_config_class = RBLNKandinskyV22CombinedPipelineConfig
    _connected_classes = {"prior_pipe": RBLNKandinskyV22PriorPipeline, "decoder_pipe": RBLNKandinskyV22Pipeline}
    _submodules = ["prior_image_encoder", "prior_text_encoder", "prior_prior", "unet", "movq"]
    _prefix = {"prior_pipe": "prior_"}

    def __init__(
        self,
        unet: UNet2DConditionModel,
        scheduler: DDPMScheduler,
        movq: VQModel,
        prior_prior: PriorTransformer,
        prior_image_encoder: CLIPVisionModelWithProjection,
        prior_text_encoder: CLIPTextModelWithProjection,
        prior_tokenizer: CLIPTokenizer,
        prior_scheduler: UnCLIPScheduler,
        prior_image_processor: CLIPImageProcessor,
    ):
        RBLNDiffusionMixin.__init__(self)
        super(KandinskyV22CombinedPipeline, self).__init__()

        self.register_modules(
            unet=unet,
            scheduler=scheduler,
            movq=movq,
            prior_prior=prior_prior,
            prior_image_encoder=prior_image_encoder,
            prior_text_encoder=prior_text_encoder,
            prior_tokenizer=prior_tokenizer,
            prior_scheduler=prior_scheduler,
            prior_image_processor=prior_image_processor,
        )

        self.prior_pipe = RBLNKandinskyV22PriorPipeline(
            prior=prior_prior,
            image_encoder=prior_image_encoder,
            text_encoder=prior_text_encoder,
            tokenizer=prior_tokenizer,
            scheduler=prior_scheduler,
            image_processor=prior_image_processor,
        )
        self.decoder_pipe = RBLNKandinskyV22Pipeline(
            unet=unet,
            scheduler=scheduler,
            movq=movq,
        )

    def get_compiled_image_size(self):
        return self.movq.image_size


class RBLNKandinskyV22Img2ImgCombinedPipeline(RBLNDiffusionMixin, KandinskyV22Img2ImgCombinedPipeline):
    """
    RBLN-accelerated implementation of Kandinsky 2.2 combined pipeline for end-to-end image-to-image generation.

    This pipeline compiles both prior and decoder Kandinsky 2.2 models to run efficiently on RBLN NPUs, enabling
    high-performance inference for complete image-to-image transformation with distinctive artistic style.
    """

    original_class = KandinskyV22Img2ImgCombinedPipeline
    _connected_classes = {"prior_pipe": RBLNKandinskyV22PriorPipeline, "decoder_pipe": RBLNKandinskyV22Img2ImgPipeline}
    _submodules = ["prior_image_encoder", "prior_text_encoder", "prior_prior", "unet", "movq"]
    _prefix = {"prior_pipe": "prior_"}

    def __init__(
        self,
        unet: UNet2DConditionModel,
        scheduler: DDPMScheduler,
        movq: VQModel,
        prior_prior: PriorTransformer,
        prior_image_encoder: CLIPVisionModelWithProjection,
        prior_text_encoder: CLIPTextModelWithProjection,
        prior_tokenizer: CLIPTokenizer,
        prior_scheduler: UnCLIPScheduler,
        prior_image_processor: CLIPImageProcessor,
    ):
        RBLNDiffusionMixin.__init__(self)
        super(KandinskyV22Img2ImgCombinedPipeline, self).__init__()

        self.register_modules(
            unet=unet,
            scheduler=scheduler,
            movq=movq,
            prior_prior=prior_prior,
            prior_image_encoder=prior_image_encoder,
            prior_text_encoder=prior_text_encoder,
            prior_tokenizer=prior_tokenizer,
            prior_scheduler=prior_scheduler,
            prior_image_processor=prior_image_processor,
        )

        self.prior_pipe = RBLNKandinskyV22PriorPipeline(
            prior=prior_prior,
            image_encoder=prior_image_encoder,
            text_encoder=prior_text_encoder,
            tokenizer=prior_tokenizer,
            scheduler=prior_scheduler,
            image_processor=prior_image_processor,
        )
        self.decoder_pipe = RBLNKandinskyV22Img2ImgPipeline(
            unet=unet,
            scheduler=scheduler,
            movq=movq,
        )

    def get_compiled_image_size(self):
        return self.movq.image_size


class RBLNKandinskyV22InpaintCombinedPipeline(RBLNDiffusionMixin, KandinskyV22InpaintCombinedPipeline):
    """
    RBLN-accelerated implementation of Kandinsky 2.2 combined pipeline for end-to-end image inpainting.

    This pipeline compiles both prior and decoder Kandinsky 2.2 models to run efficiently on RBLN NPUs, enabling
    high-performance inference for complete image inpainting with distinctive artistic style and seamless integration.
    """

    original_class = KandinskyV22InpaintCombinedPipeline
    _connected_classes = {"prior_pipe": RBLNKandinskyV22PriorPipeline, "decoder_pipe": RBLNKandinskyV22InpaintPipeline}
    _submodules = ["prior_image_encoder", "prior_text_encoder", "prior_prior", "unet", "movq"]
    _prefix = {"prior_pipe": "prior_"}

    def __init__(
        self,
        unet: UNet2DConditionModel,
        scheduler: DDPMScheduler,
        movq: VQModel,
        prior_prior: PriorTransformer,
        prior_image_encoder: CLIPVisionModelWithProjection,
        prior_text_encoder: CLIPTextModelWithProjection,
        prior_tokenizer: CLIPTokenizer,
        prior_scheduler: UnCLIPScheduler,
        prior_image_processor: CLIPImageProcessor,
    ):
        RBLNDiffusionMixin.__init__(self)
        super(KandinskyV22InpaintCombinedPipeline, self).__init__()

        self.register_modules(
            unet=unet,
            scheduler=scheduler,
            movq=movq,
            prior_prior=prior_prior,
            prior_image_encoder=prior_image_encoder,
            prior_text_encoder=prior_text_encoder,
            prior_tokenizer=prior_tokenizer,
            prior_scheduler=prior_scheduler,
            prior_image_processor=prior_image_processor,
        )

        self.prior_pipe = RBLNKandinskyV22PriorPipeline(
            prior=prior_prior,
            image_encoder=prior_image_encoder,
            text_encoder=prior_text_encoder,
            tokenizer=prior_tokenizer,
            scheduler=prior_scheduler,
            image_processor=prior_image_processor,
        )
        self.decoder_pipe = RBLNKandinskyV22InpaintPipeline(
            unet=unet,
            scheduler=scheduler,
            movq=movq,
        )

    def get_compiled_image_size(self):
        return self.movq.image_size


__all__ = [
    "RBLNKandinskyV22CombinedPipeline",
    "RBLNKandinskyV22Img2ImgCombinedPipeline",
    "RBLNKandinskyV22InpaintCombinedPipeline",
]
