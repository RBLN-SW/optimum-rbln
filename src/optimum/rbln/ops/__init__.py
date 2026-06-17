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

# All ``rbln_custom_ops::`` operators are registered by rebel-compiler as the single
# source of truth. Importing rebel triggers that registration as an import side effect,
# so optimum-rbln no longer defines its own duplicate ``torch.library.custom_op``s.
import rebel  # noqa: F401

from .moe import compute_masked_routing_weight_softmax_first, compute_masked_routing_weight_topk_first
