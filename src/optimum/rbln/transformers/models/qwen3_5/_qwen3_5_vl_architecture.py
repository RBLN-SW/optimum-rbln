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

"""RBLN graph rewrite for Qwen3.5-VL (vision-language).

Qwen3.5-VL = a Qwen3-VL-style vision encoder whose merged image embeddings are scattered into
the text ``inputs_embeds``, on top of the hybrid Qwen3.5 text backbone (``linear_attention``
GatedDeltaNet + ``full_attention`` gated layers).

Two things make this LIGHTER than the Qwen3-VL wrapper:

1. **No deepstack.** Qwen3.5 deletes deepstack from the Qwen3-VL vision model
   (`Qwen3_5VisionModel.__init__` does `del self.deepstack_visual_indexes / deepstack_merger_list`),
   so the vision encoder returns only the merged embeddings and there are NO
   ``visual_pos_mask`` / ``deepstack_visual_embeds`` graph inputs and no per-layer injection.
2. **The text backbone is unchanged.** It is exactly the text-only hybrid graph from
   ``Qwen3_5Wrapper``; the VL wrapper only adapts the I/O for the VL runtime: mRoPE
   ``position_embeds`` are precomputed on the host and passed in as a stacked tensor (so the
   model uses them directly instead of an inline ``RotaryEmbedding``), and ``inputs_embeds``
   (image-merged) is always the input.
"""

import torch
import torch.nn as nn
from transformers import PreTrainedModel

from ..qwen3_vl.qwen3_vl_architecture import Qwen3VLVisionBlock
from .qwen3_5_architecture import Qwen3_5Wrapper


class Qwen3_5VisionModelWrapper(nn.Module):
    """Qwen3.5 vision encoder for RBLN: transformer blocks + merger, NO deepstack.

    Patch embedding, position-embed interpolation and rotary computation run on the host; the
    compiled graph takes the patch ``hidden_states`` plus the precomputed attention mask and
    rotary ``cos``/``sin``, and returns the merged image embeddings.
    """

    def __init__(self, model: nn.Module, rbln_config):
        super().__init__()
        self.merger = model.merger
        self.rbln_config = rbln_config
        self.blocks = nn.ModuleList([Qwen3VLVisionBlock(block, rbln_config) for block in model.blocks])

    def forward(
        self,
        hidden_states: torch.Tensor,
        attn_mask: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        attn_mask = (1.0 - attn_mask) * torch.finfo(hidden_states.dtype).min
        for block in self.blocks:
            hidden_states = block(hidden_states, attn_mask, [cos, sin])
        return self.merger(hidden_states)


class Qwen3_5VL_LanguageModelWrapper(Qwen3_5Wrapper):
    """The hybrid Qwen3.5 text backbone wired for the VL runtime.

    Reuses ``Qwen3_5Wrapper``'s hybrid graph rewrite (``convert_to_rbln_class``, the
    ``get_rbln_*`` factories that emit GatedDeltaNet linear layers + gated full-attention
    layers, and the linear-state threading in ``Qwen3_5Model``). The only changes vs the
    text-only wrapper:

    - ``model.config`` is a ``Qwen3_5Config`` (vision + text); swap it to ``text_config`` for
      the parent ``DecoderOnlyWrapper`` initialization (which expects text attributes).
    - the language model is reached via ``model.get_decoder()`` (it is nested under the VL model).
    - ``position_embeds`` (precomputed mRoPE cos/sin) is an explicit graph input, passed to the
      model as the ``rotary_emb`` tensor; there is no inline ``RotaryEmbedding`` and no deepstack.
    """

    _use_rotary_emb = False

    def __init__(self, model: PreTrainedModel, rbln_config, use_rotary_emb: bool):
        original_config = model.config
        model.config = model.config.text_config
        super().__init__(model, rbln_config, use_rotary_emb)
        model.config = original_config

    def get_decoder_layers(self, model: PreTrainedModel):
        return model.get_decoder().layers

    def get_model_layer(self, model: PreTrainedModel):
        return model.get_decoder()

    def prepare_forward_args(self, *args):
        args = list(args)
        input_ids = None if self.rbln_config.use_inputs_embeds else args.pop(0)
        inputs_embeds = args.pop(0) if self.rbln_config.use_inputs_embeds else None
        cache_position = args.pop(0)
        global_block_tables = args.pop(0)
        local_block_tables = None
        position_embeds = args.pop(0)
        query_position = args.pop(0) if self.phase == "prefill" and self.rbln_config.logits_to_keep > 0 else None
        position_ids = None
        attention_mask = args.pop(0) if self.rbln_config.use_attention_mask else None
        lora_int_id = args.pop(0) if self.rbln_config.lora_config else None

        # 2 state slots per layer: (conv_state, recurrent_state) for linear_attention layers,
        # (key, value) for full_attention layers. The model dispatches per layer_idx.
        past_key_values = args
        if len(past_key_values) != 2 * self.num_hidden_layers:
            raise ValueError(
                f"Different past_key_values to model's config. {len(past_key_values)} != {2 * self.num_hidden_layers}"
            )
        _past_key_values = []
        for i in range(self.num_hidden_layers):
            _past_key_values.append([past_key_values[i * 2], past_key_values[i * 2 + 1]])
        past_key_values = _past_key_values

        return (
            input_ids,
            inputs_embeds,
            cache_position,
            global_block_tables,
            local_block_tables,
            query_position,
            attention_mask,
            position_ids,
            lora_int_id,
            past_key_values,
            position_embeds,
        )

    def forward(self, *args):
        (
            input_ids,
            inputs_embeds,
            cache_position,
            global_block_tables,
            local_block_tables,
            query_position,
            attention_mask,
            position_ids,
            lora_int_id,
            past_key_values,
            position_embeds,
        ) = self.prepare_forward_args(*args)

        logits, all_hidden_states, new_states = self.model(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            position_ids=position_ids,
            query_position=query_position,
            past_key_values=past_key_values,
            rotary_emb=position_embeds,  # precomputed mRoPE (cos, sin); see Qwen3_5Model.forward
            global_block_tables=global_block_tables,
            local_block_tables=local_block_tables,
            lora_int_id=lora_int_id,
            output_hidden_states=self.rbln_config.output_hidden_states,
        )

        # Linear-attention state updates are returned so the runtime can persist them.
        if self.rbln_config.output_hidden_states:
            return (logits, *new_states, *all_hidden_states)
        return (logits, *new_states)
