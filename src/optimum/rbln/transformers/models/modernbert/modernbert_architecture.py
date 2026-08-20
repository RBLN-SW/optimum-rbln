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


class ModernBertModelWrapper(torch.nn.Module):
    """Compilation wrapper for ModernBERT encoder models.

    Unlike the generic `TransformerEncoderWrapper`, this wrapper passes the 2D
    padding `attention_mask` through unchanged instead of expanding it with
    `_prepare_4d_attention_mask`. ModernBERT uses alternating full / local
    (sliding-window) attention and builds its two 4D masks internally via
    `create_bidirectional_mask` and `create_bidirectional_sliding_window_mask`.
    Pre-expanding the mask here would feed the same global mask to the sliding
    layers and drop the local windowing, producing incorrect outputs.

    `return_dict` is forced off so the traced graph exposes a single `logits`
    tensor; the RBLN runtime rebuilds the `MaskedLMOutput` from it.
    """

    def __init__(self, model, rbln_config):
        super().__init__()
        self.model = model
        self.rbln_config = rbln_config

    def forward(self, *args, **kwargs):
        output = self.model(*args, return_dict=False, **kwargs)
        if isinstance(output, torch.Tensor):
            return output
        elif isinstance(output, (tuple, list)):
            return tuple(x for x in output if x is not None)
        return output
