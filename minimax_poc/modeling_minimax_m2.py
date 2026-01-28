# Copyright 2026 Rebellions Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

from typing import Optional

import torch
from minimax_m2_architecture import MiniMaxM2Wrapper
from transformers import AutoModelForCausalLM, PreTrainedModel

from optimum.rbln.transformers.models.decoderonly import (
    RBLNDecoderOnlyModelForCausalLM,
)


def _w8a16_block_fp8_matmul(self, input: torch.Tensor) -> torch.Tensor:
    _input_dtype = input.dtype
    out_features, in_features = self.weight.shape

    weight_scale = self.weight_scale_inv.repeat_interleave(self.block_size[0], 0).unsqueeze(-1)
    weight = self.weight.view(out_features, in_features // self.block_size[1], self.block_size[1])
    weight = (weight.to(_input_dtype) * weight_scale).view(out_features, in_features)
    output = torch.nn.functional.linear(input, weight, self.bias)

    return output


class RBLNMiniMaxM2ForCausalLM(RBLNDecoderOnlyModelForCausalLM):
    """
    MiniMax-M2.* decoder-only MoE model with a language modeling head, optimized for RBLN.

    Notes:
    - MiniMaxAI/MiniMax-M2.1 on the Hub is FP8-quantized and requires GPU/XPU in vanilla Transformers.
      For RBLN compilation, prefer a BF16/FP16 (dequantized) checkpoint, or ensure your environment
      can load FP8 weights.
    """

    _decoder_wrapper_cls = MiniMaxM2Wrapper

    @classmethod
    def get_pytorch_model(
        cls,
        model_id: str,
        *args,
        rbln_config=None,
        trust_remote_code: Optional[bool] = None,
        **kwargs,
    ) -> PreTrainedModel:
        # NOTE: MiniMaxAI/MiniMax-M2.1 is `custom_code` on the Hub, so it does NOT have a static
        # `transformers.MiniMaxM2ForCausalLM` symbol. Therefore, the default `get_hf_class()` path in
        # optimum-rbln would return None. We must load via AutoModel.
        #
        # The user intentionally monkeypatches the FP8 quantizer env-check (CPU-only environments).
        from transformers.integrations.finegrained_fp8 import FP8Linear
        from transformers.quantizers.quantizer_finegrained_fp8 import (
            FineGrainedFP8HfQuantizer,
        )

        FineGrainedFP8HfQuantizer.validate_environment = lambda self, *a, **k: None
        FP8Linear.forward = _w8a16_block_fp8_matmul

        if trust_remote_code is not None:
            kwargs["trust_remote_code"] = trust_remote_code
        else:
            kwargs.setdefault("trust_remote_code", True)

        model = AutoModelForCausalLM.from_pretrained(model_id, *args, **kwargs)
        return model
