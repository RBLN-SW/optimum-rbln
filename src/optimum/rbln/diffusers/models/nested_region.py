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

import copy

import torch
from diffusers.models.attention import BasicTransformerBlock, TemporalBasicTransformerBlock
from diffusers.models.resnet import ResnetBlock2D, SpatioTemporalResBlock

from ...utils.nested_compile_region import NestedCompileRegionMixin


class DiffusersModuleWrapper:
    """Reuse upstream module state and forward methods without rerunning weight initialization."""

    def __init__(self, module: torch.nn.Module):
        self.__dict__ = module.__dict__.copy()


class BasicTransformerBlockRegion(NestedCompileRegionMixin, DiffusersModuleWrapper, BasicTransformerBlock):
    pass


class TemporalBasicTransformerBlockRegion(
    NestedCompileRegionMixin, DiffusersModuleWrapper, TemporalBasicTransformerBlock
):
    pass


class ResnetBlock2DRegion(NestedCompileRegionMixin, DiffusersModuleWrapper, ResnetBlock2D):
    pass


class SpatioTemporalResBlockRegion(NestedCompileRegionMixin, DiffusersModuleWrapper, SpatioTemporalResBlock):
    pass


DIFFUSERS_REGION_CLASSES = {
    BasicTransformerBlock: BasicTransformerBlockRegion,
    TemporalBasicTransformerBlock: TemporalBasicTransformerBlockRegion,
    ResnetBlock2D: ResnetBlock2DRegion,
    SpatioTemporalResBlock: SpatioTemporalResBlockRegion,
}


def wrap_diffusers_modules(
    model: torch.nn.Module, region_classes: dict[type[torch.nn.Module], type[torch.nn.Module]]
) -> torch.nn.Module:
    """Build a compile-time module tree with shared weights and independent module registrations.

    Upstream Diffusers forwards retain their configuration and attribute names. Only the
    selected block types gain region entry points; the caller's modules remain untouched.
    """
    wrapped_modules = {}

    def wrap(module: torch.nn.Module) -> torch.nn.Module:
        if module in wrapped_modules:
            return wrapped_modules[module]
        wrapper_cls = region_classes.get(type(module))
        wrapped = wrapper_cls(module) if wrapper_cls is not None else copy.copy(module)
        wrapped_modules[module] = wrapped
        wrapped._parameters = module._parameters.copy()
        wrapped._buffers = module._buffers.copy()
        wrapped._modules = {
            name: wrap(child) if child is not None else None for name, child in module._modules.items()
        }
        return wrapped

    return wrap(model)
