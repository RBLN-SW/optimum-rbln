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

import torch
from torch import Tensor


@torch.library.custom_op("rbln_custom_ops::rbln_cache_update", mutates_args=(["cache"]))
def rbln_cache_update(cache: Tensor, state: Tensor, position: Tensor, axis: Tensor) -> Tensor:
    # Define the RBLN custom operation "rbln_cache_update" which updates a cache tensor with a given state tensor.
    # This operation is designed to perform in-place updates directly on the device without needing to transfer the cache back to the host.
    # The `position` parameter specifies the start index for the update along the specified axis, allowing flexible updates to any part of the cache tensor.
    return torch.empty_like(cache)


@rbln_cache_update.register_fake
def rbln_cache_update_fake(cache: Tensor, state: Tensor, position: Tensor, axis: Tensor) -> Tensor:
    return torch.empty_like(cache)
