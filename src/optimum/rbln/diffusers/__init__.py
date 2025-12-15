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

from typing import TYPE_CHECKING

from diffusers.pipelines.pipeline_utils import ALL_IMPORTABLE_CLASSES, LOADABLE_CLASSES
from transformers.utils import _LazyModule


LOADABLE_CLASSES["optimum.rbln"] = {
    "RBLNBaseModel": ["save_pretrained", "from_pretrained"],
    "RBLNCosmosSafetyChecker": ["save_pretrained", "from_pretrained"],
}
ALL_IMPORTABLE_CLASSES.update(LOADABLE_CLASSES["optimum.rbln"])


_import_structure = {
    "configurations": [
        "RBLNAutoencoderKLConfig",
        "RBLNAutoencoderKLCosmosConfig",
        "RBLNControlNetModelConfig",
        "RBLNCosmosTextToWorldPipelineConfig",
        "RBLNCosmosVideoToWorldPipelineConfig",
        "RBLNCosmosTransformer3DModelConfig",
        "RBLNKandinskyV22CombinedPipelineConfig",
        "RBLNKandinskyV22Img2ImgCombinedPipelineConfig",
        "RBLNKandinskyV22Img2ImgPipelineConfig",
        "RBLNKandinskyV22InpaintCombinedPipelineConfig",
        "RBLNKandinskyV22InpaintPipelineConfig",
        "RBLNKandinskyV22PipelineConfig",
        "RBLNKandinskyV22PriorPipelineConfig",
        "RBLNPriorTransformerConfig",
        "RBLNStableDiffusionControlNetPipelineConfig",
        "RBLNStableDiffusionControlNetImg2ImgPipelineConfig",
        "RBLNStableDiffusionImg2ImgPipelineConfig",
        "RBLNStableDiffusionInpaintPipelineConfig",
        "RBLNStableDiffusionPipelineConfig",
        "RBLNStableDiffusionXLControlNetPipelineConfig",
        "RBLNStableDiffusionXLControlNetImg2ImgPipelineConfig",
        "RBLNStableDiffusionXLImg2ImgPipelineConfig",
        "RBLNStableDiffusionXLInpaintPipelineConfig",
        "RBLNStableDiffusionXLPipelineConfig",
        "RBLNStableDiffusion3PipelineConfig",
        "RBLNStableDiffusion3Img2ImgPipelineConfig",
        "RBLNStableDiffusion3InpaintPipelineConfig",
        "RBLNSD3Transformer2DModelConfig",
        "RBLNUNet2DConditionModelConfig",
        "RBLNVQModelConfig",
        "RBLNUNetSpatioTemporalConditionModelConfig",
        "RBLNStableVideoDiffusionPipelineConfig",
        "RBLNAutoencoderKLTemporalDecoderConfig",
    ],
    "pipelines": [
        "RBLNAutoPipelineForImage2Image",
        "RBLNAutoPipelineForInpainting",
        "RBLNAutoPipelineForText2Image",
        "RBLNCosmosTextToWorldPipeline",
        "RBLNCosmosVideoToWorldPipeline",
        "RBLNCosmosSafetyChecker",
        "RBLNKandinskyV22CombinedPipeline",
        "RBLNKandinskyV22Img2ImgCombinedPipeline",
        "RBLNKandinskyV22InpaintCombinedPipeline",
        "RBLNKandinskyV22InpaintPipeline",
        "RBLNKandinskyV22Img2ImgPipeline",
        "RBLNKandinskyV22PriorPipeline",
        "RBLNKandinskyV22Pipeline",
        "RBLNStableDiffusionPipeline",
        "RBLNStableDiffusionXLPipeline",
        "RBLNStableDiffusionImg2ImgPipeline",
        "RBLNStableDiffusionInpaintPipeline",
        "RBLNStableDiffusionControlNetImg2ImgPipeline",
        "RBLNMultiControlNetModel",
        "RBLNStableDiffusionXLImg2ImgPipeline",
        "RBLNStableDiffusionXLInpaintPipeline",
        "RBLNStableDiffusionControlNetPipeline",
        "RBLNStableDiffusionXLControlNetPipeline",
        "RBLNStableDiffusionXLControlNetImg2ImgPipeline",
        "RBLNStableDiffusion3Pipeline",
        "RBLNStableDiffusion3Img2ImgPipeline",
        "RBLNStableDiffusion3InpaintPipeline",
        "RBLNStableVideoDiffusionPipeline",
    ],
    "models": [
        "RBLNAutoencoderKL",
        "RBLNAutoencoderKLCosmos",
        "RBLNUNet2DConditionModel",
        "RBLNUNetSpatioTemporalConditionModel",
        "RBLNControlNetModel",
        "RBLNCosmosTransformer3DModel",
        "RBLNSD3Transformer2DModel",
        "RBLNAutoencoderKLTemporalDecoder",
        "RBLNPriorTransformer",
        "RBLNVQModel",
    ],
    "modeling_diffusers": [
        "RBLNDiffusionMixin",
    ],
}

if TYPE_CHECKING:
    from .configurations import (
        RBLNAutoencoderKLConfig,
        RBLNAutoencoderKLCosmosConfig,
        RBLNAutoencoderKLTemporalDecoderConfig,
        RBLNControlNetModelConfig,
        RBLNCosmosTextToWorldPipelineConfig,
        RBLNCosmosTransformer3DModelConfig,
        RBLNCosmosVideoToWorldPipelineConfig,
        RBLNKandinskyV22CombinedPipelineConfig,
        RBLNKandinskyV22Img2ImgCombinedPipelineConfig,
        RBLNKandinskyV22Img2ImgPipelineConfig,
        RBLNKandinskyV22InpaintCombinedPipelineConfig,
        RBLNKandinskyV22InpaintPipelineConfig,
        RBLNKandinskyV22PipelineConfig,
        RBLNKandinskyV22PriorPipelineConfig,
        RBLNPriorTransformerConfig,
        RBLNSD3Transformer2DModelConfig,
        RBLNStableDiffusion3Img2ImgPipelineConfig,
        RBLNStableDiffusion3InpaintPipelineConfig,
        RBLNStableDiffusion3PipelineConfig,
        RBLNStableDiffusionControlNetImg2ImgPipelineConfig,
        RBLNStableDiffusionControlNetPipelineConfig,
        RBLNStableDiffusionImg2ImgPipelineConfig,
        RBLNStableDiffusionInpaintPipelineConfig,
        RBLNStableDiffusionPipelineConfig,
        RBLNStableDiffusionXLControlNetImg2ImgPipelineConfig,
        RBLNStableDiffusionXLControlNetPipelineConfig,
        RBLNStableDiffusionXLImg2ImgPipelineConfig,
        RBLNStableDiffusionXLInpaintPipelineConfig,
        RBLNStableDiffusionXLPipelineConfig,
        RBLNStableVideoDiffusionPipelineConfig,
        RBLNUNet2DConditionModelConfig,
        RBLNUNetSpatioTemporalConditionModelConfig,
        RBLNVQModelConfig,
    )
    from .modeling_diffusers import RBLNDiffusionMixin
    from .models import (
        RBLNAutoencoderKL,
        RBLNAutoencoderKLCosmos,
        RBLNAutoencoderKLTemporalDecoder,
        RBLNControlNetModel,
        RBLNCosmosTransformer3DModel,
        RBLNPriorTransformer,
        RBLNSD3Transformer2DModel,
        RBLNUNet2DConditionModel,
        RBLNUNetSpatioTemporalConditionModel,
        RBLNVQModel,
    )
    from .pipelines import (
        RBLNAutoPipelineForImage2Image,
        RBLNAutoPipelineForInpainting,
        RBLNAutoPipelineForText2Image,
        RBLNCosmosSafetyChecker,
        RBLNCosmosTextToWorldPipeline,
        RBLNCosmosVideoToWorldPipeline,
        RBLNKandinskyV22CombinedPipeline,
        RBLNKandinskyV22Img2ImgCombinedPipeline,
        RBLNKandinskyV22Img2ImgPipeline,
        RBLNKandinskyV22InpaintCombinedPipeline,
        RBLNKandinskyV22InpaintPipeline,
        RBLNKandinskyV22Pipeline,
        RBLNKandinskyV22PriorPipeline,
        RBLNMultiControlNetModel,
        RBLNStableDiffusion3Img2ImgPipeline,
        RBLNStableDiffusion3InpaintPipeline,
        RBLNStableDiffusion3Pipeline,
        RBLNStableDiffusionControlNetImg2ImgPipeline,
        RBLNStableDiffusionControlNetPipeline,
        RBLNStableDiffusionImg2ImgPipeline,
        RBLNStableDiffusionInpaintPipeline,
        RBLNStableDiffusionPipeline,
        RBLNStableDiffusionXLControlNetImg2ImgPipeline,
        RBLNStableDiffusionXLControlNetPipeline,
        RBLNStableDiffusionXLImg2ImgPipeline,
        RBLNStableDiffusionXLInpaintPipeline,
        RBLNStableDiffusionXLPipeline,
        RBLNStableVideoDiffusionPipeline,
    )
else:
    import sys

    sys.modules[__name__] = _LazyModule(
        __name__,
        globals()["__file__"],
        _import_structure,
        module_spec=__spec__,
    )
