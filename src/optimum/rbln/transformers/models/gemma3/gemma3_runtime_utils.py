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

import rebel
import torch

from ...modeling_outputs import RBLNGemma3ForCausalLMOutput
from ..decoderonly.decoderonly_runtime_utils import (
    RBLNDecoderOnlyChunkedMultimodalPrefillMixin,
    RBLNPytorchRuntime,
)
from ..decoderonly.modeling_decoderonly import RBLNRuntimeModel


class RBLNGemma3RuntimeModel(RBLNDecoderOnlyChunkedMultimodalPrefillMixin, RBLNRuntimeModel):
    # Gemma3 (text + image) chunked prefill. The tight-pack planning, partition alignment, and the
    # prefill/forward loop all come from RBLNDecoderOnlyChunkedMultimodalPrefillMixin; Gemma3 only
    # supplies the runtime registry and the per-chunk runtime argument order. Gemma3 compiles a
    # single image-prefill bucket and has no per-layer inputs, so it relies on the mixin defaults
    # (single `image_prefill` runtime, single-bucket `_resolve_image_chunk`, inherited `forward`).

    _prefill_output_cls = RBLNGemma3ForCausalLMOutput

    def __init__(self, *args: Any, image_prefill: Optional[rebel.Runtime] = None, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.image_prefill = RBLNPytorchRuntime(image_prefill)
        self.prefill = RBLNPytorchRuntime(self.runtime) if self.phase == "prefill" else None

    def _invoke_prefill_chunk(
        self,
        runtime,
        input_chunk: torch.Tensor,
        per_layer_chunk: Optional[torch.Tensor],
        cache_pos_chunk: torch.Tensor,
        block_tables: torch.Tensor,
        local_block_tables: Optional[torch.Tensor],
        query_position: torch.Tensor,
        chunked_attention_mask: torch.Tensor,
        position_ids_chunk: Optional[torch.Tensor],
        lora_int_ids: Optional[torch.Tensor],
    ):
        # Gemma3 has no per-layer inputs, so `per_layer_chunk` is ignored. Argument order mirrors
        # Gemma3ForCausalLMWrapper.prepare_forward_args. The callee owns the `use_lora` gate.
        return runtime(
            input_chunk,
            cache_pos_chunk,
            block_tables,
            local_block_tables,
            query_position,
            chunked_attention_mask,
            position_ids_chunk,
            lora_int_ids if self.rbln_config.use_lora else None,
        )
