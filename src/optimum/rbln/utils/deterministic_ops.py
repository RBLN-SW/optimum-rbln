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

import numpy as np
import torch


"""
Deterministic host-side trigonometric ops.

torch's CPU cos/sin kernels are not bit-reproducible: the result can differ in the
last bits depending on the OMP/MKL thread configuration and the tensor size.
numpy's cos/sin are reproducible across thread counts, so any cos/sin that is
computed on the host (e.g. rotary tables fed to a compiled graph as inputs) must
go through these helpers instead of `torch.cos`/`torch.sin`.
"""


def deterministic_cos(x: torch.Tensor) -> torch.Tensor:
    """Compute cos(x) on the host via numpy so the result is bit-reproducible."""
    return torch.from_numpy(np.cos(x.detach().cpu().numpy()))


def deterministic_sin(x: torch.Tensor) -> torch.Tensor:
    """Compute sin(x) on the host via numpy so the result is bit-reproducible."""
    return torch.from_numpy(np.sin(x.detach().cpu().numpy()))
