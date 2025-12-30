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

from transformers.utils import _LazyModule


_import_structure = {
    "auto_pipeline": [
        "RBLNAutoPipelineForImage2Image",
        "RBLNAutoPipelineForInpainting",
        "RBLNAutoPipelineForText2Image",
    ],
    "controlnet": [
        "RBLNMultiControlNetModel",
        "RBLNStableDiffusionControlNetImg2ImgPipeline",
        "RBLNStableDiffusionControlNetPipeline",
        "RBLNStableDiffusionXLControlNetImg2ImgPipeline",
        "RBLNStableDiffusionXLControlNetPipeline",
    ],
    "cosmos": [
        "RBLNCosmosTextToWorldPipeline",
        "RBLNCosmosVideoToWorldPipeline",
        "RBLNCosmosSafetyChecker",
    ],
    "kandinsky2_2": [
        "RBLNKandinskyV22CombinedPipeline",
        "RBLNKandinskyV22Img2ImgCombinedPipeline",
        "RBLNKandinskyV22InpaintCombinedPipeline",
        "RBLNKandinskyV22InpaintPipeline",
        "RBLNKandinskyV22Img2ImgPipeline",
        "RBLNKandinskyV22PriorPipeline",
        "RBLNKandinskyV22Pipeline",
    ],
    "stable_diffusion": [
        "RBLNStableDiffusionImg2ImgPipeline",
        "RBLNStableDiffusionPipeline",
        "RBLNStableDiffusionInpaintPipeline",
    ],
    "stable_diffusion_xl": [
        "RBLNStableDiffusionXLImg2ImgPipeline",
        "RBLNStableDiffusionXLPipeline",
        "RBLNStableDiffusionXLInpaintPipeline",
    ],
    "stable_diffusion_3": [
        "RBLNStableDiffusion3Pipeline",
        "RBLNStableDiffusion3Img2ImgPipeline",
        "RBLNStableDiffusion3InpaintPipeline",
    ],
    "stable_video_diffusion": [
        "RBLNStableVideoDiffusionPipeline",
    ],
}
if TYPE_CHECKING:
    from .auto_pipeline import (
        RBLNAutoPipelineForImage2Image,
        RBLNAutoPipelineForInpainting,
        RBLNAutoPipelineForText2Image,
    )
    from .controlnet import (
        RBLNMultiControlNetModel,
        RBLNStableDiffusionControlNetImg2ImgPipeline,
        RBLNStableDiffusionControlNetPipeline,
        RBLNStableDiffusionXLControlNetImg2ImgPipeline,
        RBLNStableDiffusionXLControlNetPipeline,
    )
    from .cosmos import RBLNCosmosSafetyChecker, RBLNCosmosTextToWorldPipeline, RBLNCosmosVideoToWorldPipeline
    from .kandinsky2_2 import (
        RBLNKandinskyV22CombinedPipeline,
        RBLNKandinskyV22Img2ImgCombinedPipeline,
        RBLNKandinskyV22Img2ImgPipeline,
        RBLNKandinskyV22InpaintCombinedPipeline,
        RBLNKandinskyV22InpaintPipeline,
        RBLNKandinskyV22Pipeline,
        RBLNKandinskyV22PriorPipeline,
    )
    from .stable_diffusion import (
        RBLNStableDiffusionImg2ImgPipeline,
        RBLNStableDiffusionInpaintPipeline,
        RBLNStableDiffusionPipeline,
    )
    from .stable_diffusion_3 import (
        RBLNStableDiffusion3Img2ImgPipeline,
        RBLNStableDiffusion3InpaintPipeline,
        RBLNStableDiffusion3Pipeline,
    )
    from .stable_diffusion_xl import (
        RBLNStableDiffusionXLImg2ImgPipeline,
        RBLNStableDiffusionXLInpaintPipeline,
        RBLNStableDiffusionXLPipeline,
    )
    from .stable_video_diffusion import RBLNStableVideoDiffusionPipeline
else:
    import sys

    sys.modules[__name__] = _LazyModule(
        __name__,
        globals()["__file__"],
        _import_structure,
        module_spec=__spec__,
    )
