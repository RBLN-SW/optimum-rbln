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

import torch
from diffusers.models.attention import BasicTransformerBlock, TemporalBasicTransformerBlock
from diffusers.models.resnet import ResnetBlock2D, SpatioTemporalResBlock


class NestedCompileRegionMixin:
    @torch.compiler.nested_compile_region
    def forward(self, *args, **kwargs):
        return super().forward(*args, **kwargs)


class BasicTransformerBlockRegion(NestedCompileRegionMixin, BasicTransformerBlock):
    pass


class TemporalBasicTransformerBlockRegion(NestedCompileRegionMixin, TemporalBasicTransformerBlock):
    pass


class ResnetBlock2DRegion(NestedCompileRegionMixin, ResnetBlock2D):
    pass


class SpatioTemporalResBlockRegion(NestedCompileRegionMixin, SpatioTemporalResBlock):
    pass


_REGION_CLASSES = {
    BasicTransformerBlock: BasicTransformerBlockRegion,
    TemporalBasicTransformerBlock: TemporalBasicTransformerBlockRegion,
    ResnetBlock2D: ResnetBlock2DRegion,
    SpatioTemporalResBlock: SpatioTemporalResBlockRegion,
}


def set_nested_compile_regions(model: torch.nn.Module, enabled: bool) -> None:
    # Static classes keep pickle/deepcopy working without adding a module path to weights.
    classes = _REGION_CLASSES if enabled else {region: original for original, region in _REGION_CLASSES.items()}
    for module in model.modules():
        if type(module) in classes:
            module.__class__ = classes[type(module)]
