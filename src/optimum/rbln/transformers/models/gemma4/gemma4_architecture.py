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
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers import PreTrainedModel
from transformers.activations import ACT2FN

from ..decoderonly.configuration_decoderonly import RBLNLoRAConfig
from ..decoderonly.decoderonly_architecture import (
    DecoderOnlyAttention,
    DecoderOnlyForCausalLM,
    DecoderOnlyLayer,
    DecoderOnlyModel,
    DecoderOnlyWrapper,
    RotaryEmbedding,
    slice_and_unsqueeze_cos_sin,
)


class Gemma4ForCausalLMWrapper(DecoderOnlyWrapper):
    """A wrapper for the text decoder component of a Gemma4 model, designed for RBLN optimization.

    Extends `DecoderOnlyWrapper` with two Gemma4-specific behaviors:

    1. Two RoPE caches are provided: one for full-attention layers (using `global_head_dim` and
       `proportional` rope_type) and one for sliding-attention layers (using `head_dim` and
       `default` rope_type). Mirrors the Gemma3 wrapper.
    2. A `per_layer_inputs` positional argument is extracted from the wrapper inputs and forwarded
       to `Gemma4TextModel`.
    """

    def get_rotary_emb(self, max_seq_len):
        config = self.config
        rope_params_full = (
            config.rope_parameters.get("full_attention", {}) if isinstance(config.rope_parameters, dict) else {}
        )
        rope_params_sliding = (
            config.rope_parameters.get("sliding_attention", {}) if isinstance(config.rope_parameters, dict) else {}
        )

        config_full = copy.deepcopy(config)
        config_full.rope_scaling = {
            "rope_type": rope_params_full.get("rope_type", "default"),
            "rope_theta": rope_params_full.get("rope_theta", getattr(config, "rope_theta", 10000.0)),
            "partial_rotary_factor": rope_params_full.get("partial_rotary_factor", 1.0),
            "factor": rope_params_full.get("factor", 1.0),
        }
        config_full.rope_theta = rope_params_full.get("rope_theta", getattr(config, "rope_theta", 10000.0))
        config_full.head_dim = getattr(config, "global_head_dim", None) or config.head_dim
        rotary_emb_global = RotaryEmbedding(config=config_full, max_seq_len_cached=max_seq_len)

        config_sliding = copy.deepcopy(config)
        config_sliding.rope_scaling = {
            "rope_type": rope_params_sliding.get("rope_type", "default"),
            "rope_theta": rope_params_sliding.get("rope_theta", getattr(config, "rope_theta", 10000.0)),
        }
        config_sliding.rope_theta = rope_params_sliding.get("rope_theta", getattr(config, "rope_theta", 10000.0))
        config_sliding.head_dim = config.head_dim
        rotary_emb_local = RotaryEmbedding(config=config_sliding, max_seq_len_cached=max_seq_len)

        return (rotary_emb_global, rotary_emb_local)

    def get_rbln_attn_class(self):
        return Gemma4Attention

    def get_rbln_layer_class(self):
        return Gemma4DecoderLayer

    def get_rbln_model_class(self):
        return Gemma4TextModel

    def get_rbln_causal_lm_class(self):
        return Gemma4ForCausalLM

    def prepare_forward_args(self, *args):
        """Override to extract ``per_layer_inputs`` after the leading inputs/cache_position."""
        args = list(args)
        input_ids = None if self.rbln_config.use_inputs_embeds else args.pop(0)
        inputs_embeds = args.pop(0) if self.rbln_config.use_inputs_embeds else None
        per_layer_inputs = args.pop(0) if getattr(self.config, "hidden_size_per_layer_input", 0) else None
        cache_position = args.pop(0)
        global_block_tables = args.pop(0) if self.rbln_config.use_global_attention else None
        local_block_tables = args.pop(0) if self.rbln_config.use_local_attention else None
        query_position = (
            args.pop(0)
            if (
                "prefill" in self.phase
                and (self.rbln_config.logits_to_keep == 1 or self.rbln_config.use_local_attention)
            )
            else None
        )
        attention_mask = args.pop(0) if self.rbln_config.use_attention_mask else None
        position_ids = args.pop(0) if self.rbln_config.use_position_ids else None
        lora_int_id = args.pop(0) if self.rbln_config.lora_config else None
        past_key_values = args

        if len(past_key_values) != 2 * self.num_hidden_layers:
            raise ValueError(
                f"Different past_key_values to model's config. {len(past_key_values)} != {2 * self.num_hidden_layers}"
            )

        _past_key_values = []
        for i in range(self.config.num_hidden_layers):
            key_states = past_key_values[i * 2]
            value_states = past_key_values[i * 2 + 1]
            _past_key_values.append([key_states, value_states])
        past_key_values = _past_key_values

        rotary_emb = (self.rotary_emb_global, self.rotary_emb_local)

        return (
            input_ids,
            inputs_embeds,
            per_layer_inputs,
            cache_position,
            global_block_tables,
            local_block_tables,
            query_position,
            attention_mask,
            position_ids,
            lora_int_id,
            past_key_values,
            rotary_emb,
        )

    def forward(self, *args):
        (
            input_ids,
            inputs_embeds,
            per_layer_inputs,
            cache_position,
            global_block_tables,
            local_block_tables,
            query_position,
            attention_mask,
            position_ids,
            lora_int_id,
            past_key_values,
            rotary_emb,
        ) = self.prepare_forward_args(*args)

        logits, all_hidden_states = self.model(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            per_layer_inputs=per_layer_inputs,
            attention_mask=attention_mask,
            cache_position=cache_position,
            position_ids=position_ids,
            query_position=query_position,
            past_key_values=past_key_values,
            rotary_emb=rotary_emb,
            global_block_tables=global_block_tables,
            local_block_tables=local_block_tables,
            lora_int_id=lora_int_id,
            output_hidden_states=self.rbln_config.output_hidden_states,
        )

        if self.rbln_config.output_hidden_states:
            return logits, all_hidden_states
        else:
            return logits


class Gemma4TextModel(DecoderOnlyModel):
    """A wrapper for the text model component of a Gemma4 model, designed for RBLN optimization.

    Extends `DecoderOnlyModel` with Gemma4-specific behaviors:

    - Two rotary embedding caches (full vs sliding), like Gemma3.
    - Holds `per_layer_model_projection` and `per_layer_projection_norm` from the original model.
      The host-side `embed_tokens_per_layer` lookup is done outside the wrapper; the result is passed
      in as `per_layer_inputs` and the full `project_per_layer_inputs` math runs here on NPU.
    - Forwards the per-layer slice `per_layer_inputs[:, :, i, :]` to each decoder layer.
    """

    def __init__(self, model, layers, rbln_config, use_learned_pos_emb=None, use_rotary_emb=True):
        super().__init__(model, layers, rbln_config, use_learned_pos_emb, use_rotary_emb)
        self.hidden_size_per_layer_input = getattr(self.config, "hidden_size_per_layer_input", 0)
        if self.hidden_size_per_layer_input:
            self.per_layer_model_projection = model.per_layer_model_projection
            self.per_layer_projection_norm = model.per_layer_projection_norm
            self.per_layer_model_projection_scale = self.config.hidden_size**-0.5
            self.per_layer_input_scale = 2.0**-0.5
        else:
            self.per_layer_model_projection = None
            self.per_layer_projection_norm = None

    def _project_per_layer_inputs(
        self,
        inputs_embeds: torch.Tensor,
        per_layer_inputs: Optional[torch.Tensor],
    ) -> Optional[torch.Tensor]:
        if not self.hidden_size_per_layer_input:
            return None

        per_layer_projection = self.per_layer_model_projection(inputs_embeds) * self.per_layer_model_projection_scale
        per_layer_projection = per_layer_projection.reshape(
            *inputs_embeds.shape[:-1],
            self.config.num_hidden_layers,
            self.hidden_size_per_layer_input,
        )
        per_layer_projection = self.per_layer_projection_norm(per_layer_projection)

        if per_layer_inputs is None:
            return per_layer_projection
        return (per_layer_projection + per_layer_inputs) * self.per_layer_input_scale

    def forward(
        self,
        input_ids: torch.Tensor = None,
        inputs_embeds: torch.Tensor = None,
        per_layer_inputs: Optional[torch.Tensor] = None,
        attention_mask: torch.Tensor = None,
        cache_position: torch.Tensor = None,
        position_ids: torch.Tensor = None,
        query_position: torch.Tensor = None,
        past_key_values: Tuple[Tuple[torch.Tensor]] = None,
        rotary_emb: torch.nn.Module = None,
        global_block_tables: Optional[torch.Tensor] = None,
        local_block_tables: Optional[torch.Tensor] = None,
        lora_int_id: Optional[torch.Tensor] = None,
        output_hidden_states: Optional[bool] = None,
    ):
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.get_embedding()(input_ids)
        hidden_states = inputs_embeds

        # Merge per-layer inputs (host-supplied per-layer residual stream) with the inputs_embeds projection.
        per_layer_inputs_projected = self._project_per_layer_inputs(inputs_embeds, per_layer_inputs)

        # Global Position Embeddings
        cos_global, sin_global = rotary_emb[0](hidden_states, self.max_seq_len)
        cos_global, sin_global = slice_and_unsqueeze_cos_sin(cos_global, sin_global, position_ids)

        # Local Position Embeddings
        cos_local, sin_local = rotary_emb[1](hidden_states, self.max_seq_len)
        cos_local, sin_local = slice_and_unsqueeze_cos_sin(cos_local, sin_local, position_ids)

        # (batch, seq_len) -> (batch,)
        if self.attn_impl == "flash_attn":
            seq_positions = cache_position[:, 0]
            seq_positions = self.convert_sequence_positions_for_flash_attn(
                seq_positions=seq_positions, max_seq_len=self.max_seq_len
            )
        else:
            seq_positions = cache_position[:, :1]

        cache_seq_len, cache_offset, swa_attn_mask = self.get_swa_custom_op_args(position_ids, query_position)
        sliding_cache_pos = (cache_seq_len, cache_offset)

        all_hidden_states = () if output_hidden_states else None
        for layer_idx, layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            is_sliding = layer_idx in self.sliding_window_layers
            is_sliding_decode = is_sliding and self.phase == "decode"
            per_layer_input_slice = (
                per_layer_inputs_projected[:, :, layer_idx, :] if per_layer_inputs_projected is not None else None
            )
            hidden_states = layer(
                hidden_states=hidden_states,
                attention_mask=swa_attn_mask if is_sliding_decode else attention_mask,
                seq_positions=sliding_cache_pos if is_sliding else seq_positions,
                past_key_values=past_key_values,
                cos=cos_local if is_sliding else cos_global,
                sin=sin_local if is_sliding else sin_global,
                block_tables=local_block_tables if is_sliding else global_block_tables,
                lora_int_id=lora_int_id,
                per_layer_input=per_layer_input_slice,
            )

        hidden_states = self.get_last_layernorm()(hidden_states)
        if output_hidden_states:
            all_hidden_states += (hidden_states,)
        return hidden_states, all_hidden_states


class Gemma4DecoderLayer(DecoderOnlyLayer):
    """A wrapper for a single transformer block of a Gemma4 model, designed for RBLN optimization.

    Extends the standard pre/post-attn + pre/post-ff layernorm structure with three Gemma4-specific features:

    1. An optional Mixture-of-Experts (MoE) branch run in parallel to the regular MLP. The MoE branch
       uses pre/post layernorms (`pre_feedforward_layernorm_2` / `post_feedforward_layernorm_2`) and
       its output is summed with the MLP output before the regular `post_feedforward_layernorm`.
    2. An optional per-layer-input merge after the FF residual block: a learned gate + projection
       that receives the layer-specific embedding slice.
    3. A learned `layer_scalar` multiplier applied to the final output.
    """

    _PRE_FF_LAYERNORM_ATTRS = ["pre_feedforward_layernorm"]
    _POST_FF_LAYERNORM_ATTRS = ["post_feedforward_layernorm"]

    def __init__(self, layer, self_attn: DecoderOnlyAttention, lora_config: Optional[RBLNLoRAConfig] = None):
        super().__init__(layer, self_attn, lora_config)

        # MoE branch (optional, when enable_moe_block=True)
        self.enable_moe_block = getattr(layer, "enable_moe_block", False)
        if self.enable_moe_block:
            self.router = Gemma4Router(layer.router)
            self.experts = Gemma4Experts(layer.experts, layer.router)
            self.pre_feedforward_layernorm_2 = layer.pre_feedforward_layernorm_2
            self.post_feedforward_layernorm_1 = layer.post_feedforward_layernorm_1
            self.post_feedforward_layernorm_2 = layer.post_feedforward_layernorm_2

        # Per-layer-input branch (optional, when hidden_size_per_layer_input > 0)
        self.hidden_size_per_layer_input = getattr(layer, "hidden_size_per_layer_input", 0)
        if self.hidden_size_per_layer_input:
            self.per_layer_input_gate = layer.per_layer_input_gate
            self.per_layer_projection = layer.per_layer_projection
            self.post_per_layer_input_norm = layer.post_per_layer_input_norm
            self.layer_act_fn = ACT2FN[layer.config.hidden_activation]

        self.layer_scalar = getattr(layer, "layer_scalar", None)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        seq_positions: Union[torch.LongTensor, Tuple[torch.LongTensor]],
        past_key_values: Tuple[Tuple[torch.Tensor]],
        cos: Optional[torch.Tensor] = None,
        sin: Optional[torch.Tensor] = None,
        block_tables: Optional[torch.Tensor] = None,
        lora_int_id: Optional[torch.Tensor] = None,
        per_layer_input: Optional[torch.Tensor] = None,
    ):
        residual = hidden_states
        hidden_states = self.get_pre_attention_layernorm()(hidden_states)
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            seq_positions=seq_positions,
            past_key_values=past_key_values,
            cos=cos,
            sin=sin,
            block_tables=block_tables,
            lora_int_id=lora_int_id,
        )
        hidden_states = self.get_post_attention_layernorm()(hidden_states)
        hidden_states = residual + hidden_states

        # Feed-forward block (and optional MoE branch in parallel)
        residual = hidden_states
        ff_input = self.get_pre_feedforward_layernorm()(hidden_states)
        mlp_out = self.forward_mlp(ff_input, lora_int_id)

        if self.enable_moe_block:
            mlp_out_normed = self.post_feedforward_layernorm_1(mlp_out)

            hidden_flat = residual.reshape(-1, residual.shape[-1])
            router_logits = self.router(hidden_flat)
            moe_in = self.pre_feedforward_layernorm_2(hidden_flat)
            moe_out = self.experts(moe_in, router_logits)
            moe_out = moe_out.reshape(residual.shape)
            moe_out = self.post_feedforward_layernorm_2(moe_out)

            mlp_out = mlp_out_normed + moe_out

        hidden_states = self.get_post_feedforward_layernorm()(mlp_out)
        hidden_states = residual + hidden_states

        if self.hidden_size_per_layer_input and per_layer_input is not None:
            residual = hidden_states
            gated = self.per_layer_input_gate(hidden_states)
            gated = self.layer_act_fn(gated)
            gated = gated * per_layer_input
            projected = self.per_layer_projection(gated)
            projected = self.post_per_layer_input_norm(projected)
            hidden_states = residual + projected

        if self.layer_scalar is not None:
            hidden_states = hidden_states * self.layer_scalar
        return hidden_states


class Gemma4Attention(DecoderOnlyAttention):
    """A wrapper for the attention component of a Gemma4 model, designed for RBLN optimization.

    Extends `DecoderOnlyAttention` with Gemma4-specific behaviors:

    - `q_norm` and `k_norm` are taken from the original module (Gemma4 applies them per-head
      pre-RoPE, the same way Gemma3 does).
    - `head_dim` differs between sliding-attention (`config.head_dim`) and full-attention layers
      (`config.global_head_dim`); the original `self_attn.head_dim` already encodes this.
    - `num_key_value_heads` is recomputed from the projection shape to account for Gemma4-specific
      knobs (`num_global_key_value_heads`, `attention_k_eq_v`) not exposed as attributes.
    """

    def __init__(self, self_attn, rbln_config, is_sliding=False):
        if hasattr(self_attn, "k_proj") and self_attn.k_proj is not None:
            num_kv_heads = self_attn.k_proj.out_features // self_attn.head_dim
            self_attn.num_key_value_heads = num_kv_heads
        super().__init__(self_attn, rbln_config, is_sliding=is_sliding)

    def get_attn_scale(self, self_attn):
        return 1.0 / (self_attn.head_dim**0.5)

    def projection(
        self, hidden_states, lora_int_id: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Projects input hidden states into query, key, and value representations.

        Args:
            hidden_states: Input tensor of shape [batch_size, seq_len, hidden_dim]
            lora_int_id: Adapter ID tensor for LoRA selection [batch_size]

        Returns:
            Tuple of (query_states, key_states, value_states)
        """
        if self.lora_config:
            query_states = self.q_proj(hidden_states, lora_int_id)
            key_states = self.k_proj(hidden_states, lora_int_id)
            value_states = self.v_proj(hidden_states, lora_int_id) if self.v_proj is not None else key_states
        else:
            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states) if self.v_proj is not None else key_states

        return query_states, key_states, value_states


class Gemma4ForCausalLM(DecoderOnlyForCausalLM):
    def forward(
        self,
        input_ids: torch.Tensor = None,
        inputs_embeds: torch.Tensor = None,
        per_layer_inputs: torch.Tensor = None,
        attention_mask: torch.Tensor = None,
        cache_position: torch.Tensor = None,
        position_ids: torch.Tensor = None,
        query_position: torch.Tensor = None,
        past_key_values: Tuple[Tuple[torch.Tensor]] = None,
        rotary_emb: nn.Module = None,
        global_block_tables: Optional[torch.Tensor] = None,
        local_block_tables: Optional[torch.Tensor] = None,
        lora_int_id: Optional[torch.Tensor] = None,
        output_hidden_states: Optional[bool] = None,
    ):
        # outputs
        hidden_states, all_hidden_states = self.model(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            per_layer_inputs=per_layer_inputs,
            attention_mask=attention_mask,
            cache_position=cache_position,
            position_ids=position_ids,
            query_position=query_position,
            past_key_values=past_key_values,
            rotary_emb=rotary_emb,
            global_block_tables=global_block_tables,
            local_block_tables=local_block_tables,
            lora_int_id=lora_int_id,
            output_hidden_states=output_hidden_states,
        )

        if "prefill" in self.phase and query_position is not None:
            hidden_states = hidden_states[:, query_position.to(torch.int).unsqueeze(0)]

        logits = self.lm_head(hidden_states)

        if getattr(self.config, "final_logit_softcapping", None) is not None:
            logits = logits / self.config.final_logit_softcapping
            logits = torch.tanh(logits)
            logits = logits * self.config.final_logit_softcapping

        return logits, all_hidden_states


class Gemma4Router(nn.Module):
    """Router for Gemma4 MoE blocks.

    Replicates `Gemma4TextRouter`'s logit computation: RMSNorm (no-scale) -> per-token scale -> linear.
    The downstream `Gemma4Experts` module handles softmax + topk + top_k_norm via the fused
    `custom_moe_glu` op; this module emits raw logits only.

    Per-expert scale (`per_expert_scale` in HF) is folded into the `down_proj` weights of
    `Gemma4Experts` at construction time, so it is not applied here.
    """

    def __init__(self, router: nn.Module):
        super().__init__()
        self.norm = router.norm
        self.scale = router.scale
        self.scalar_root_size = float(router.scalar_root_size)
        self.proj = router.proj

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.norm(hidden_states)
        hidden_states = hidden_states * self.scale * self.scalar_root_size
        return self.proj(hidden_states)


class Gemma4Experts(nn.Module):
    """Fused MoE expert block for Gemma4, dispatching to `rbln_custom_ops.custom_moe_glu`.

    The HF `Gemma4TextExperts` stores packed weight tensors `gate_up_proj` (E, 2*I, H) and
    `down_proj` (E, H, I). This class splits and transposes them at construction time to match the
    shape contract of `custom_moe_glu`: `gate_proj_weight` (E, H, I), `up_proj_weight` (E, H, I),
    `down_proj_weight` (E, I, H).

    Per-expert routing scale (`per_expert_scale` in HF) is folded into `down_proj` weights
    at construction time because the op does not accept an external per-expert constant.
    """

    def __init__(self, experts: nn.Module, router: nn.Module):
        super().__init__()
        self.num_experts = experts.num_experts
        self.hidden_size = experts.hidden_dim
        self.intermediate_size = experts.intermediate_dim
        self.top_k = int(router.config.top_k_experts)
        self.norm_topk_prob = True

        gate_up = experts.gate_up_proj
        gate_w = gate_up[:, : self.intermediate_size, :]
        up_w = gate_up[:, self.intermediate_size :, :]
        down_w = experts.down_proj

        per_expert_scale = router.per_expert_scale.detach()
        down_w = down_w * per_expert_scale[:, None, None]

        gate_w_op = gate_w.transpose(1, 2).contiguous()
        up_w_op = up_w.transpose(1, 2).contiguous()
        down_w_op = down_w.transpose(1, 2).contiguous()

        self.gate_proj = nn.Linear(self.hidden_size, self.num_experts * self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.num_experts * self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.num_experts * self.intermediate_size, self.hidden_size, bias=False)
        self.gate_proj.weight.data = gate_w_op
        self.up_proj.weight.data = up_w_op
        self.down_proj.weight.data = down_w_op

    def forward(self, hidden_states: torch.Tensor, router_logits: torch.Tensor) -> torch.Tensor:
        return torch.ops.rbln_custom_ops.custom_moe_glu(
            hidden_states,
            gate_proj_weight=self.gate_proj.weight,
            up_proj_weight=self.up_proj.weight,
            down_proj_weight=self.down_proj.weight,
            router_logits=router_logits,
            scoring_func="softmax",
            topk=self.top_k,
            norm_topk_prob=self.norm_topk_prob,
        )


class Gemma4VisionModelWrapper(nn.Module):
    """A wrapper for the vision encoder component of a Gemma4 model, designed for RBLN optimization.

    The compiled graph covers (in order):

    1. The encoder layer chain — `model.encoder.layers` is registered directly so
       `Gemma4VisionEncoder.forward` is **bypassed**; only `Gemma4VisionEncoderLayer.forward`
       runs per layer. This intentionally excludes `Gemma4VisionEncoder.forward`'s internal
       `create_bidirectional_mask` and `rotary_emb` calls from the traced graph.
    2. `pooler` (spatial 2D average pool by patch positions) producing `num_soft_tokens` outputs.
    3. Optional `standardize` affine.

    Host-side responsibilities (NOT in the compiled graph):

    - `patch_embedder` — produces `inputs_embeds` from `pixel_values`.
    - `Gemma4VisionRotaryEmbedding` — produces `(cos, sin)` from `pixel_position_ids`.
    - `padding_positions` (`(pixel_position_ids == -1).all(dim=-1)`) and the 4D additive
      `attn_mask` (`(1 - mask_2d * mask_2d.T) * finfo.min`) — both derived from
      `pixel_position_ids` on the host. The compiled graph receives them as inputs.

    Inputs (compiled):
        inputs_embeds: (batch, max_patches, hidden_size)
        pixel_position_ids: (batch, max_patches, 2)
        attn_mask: (batch, 1, max_patches, max_patches) — additive, finfo.min for masked-out
        padding_positions: (batch, max_patches) — bool, True for padded patches
        cos / sin: (batch, max_patches, head_dim) — rotary tables from host

    Output:
        hidden_states: (batch, num_soft_tokens, hidden_size) — post-pool, post-standardize.

    The dynamic per-image padding strip after pooling is left to the host
    (`RBLNGemma4ForConditionalGeneration.get_image_features`).

    Args:
        model (nn.Module): The `Gemma4VisionModel` instance.
        num_soft_tokens (int): Number of soft tokens per image after pooling (equals `max_soft_tokens`).
    """

    def __init__(self, model: PreTrainedModel, num_soft_tokens: int):
        super().__init__()
        self.encoder_layers = model.encoder.layers
        self.num_layers = int(model.config.num_hidden_layers)
        self.pooler = model.pooler
        self.standardize = getattr(model.config, "standardize", False)
        if self.standardize:
            self.std_bias = model.std_bias
            self.std_scale = model.std_scale
        self.num_soft_tokens = int(num_soft_tokens)

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        pixel_position_ids: torch.Tensor,
        attn_mask: torch.Tensor,
        padding_positions: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states = inputs_embeds
        position_embeddings = (cos, sin)
        for layer in self.encoder_layers[: self.num_layers]:
            hidden_states = layer(
                hidden_states,
                attention_mask=attn_mask,
                position_embeddings=position_embeddings,
                position_ids=pixel_position_ids,
            )

        hidden_states, _ = self.pooler(
            hidden_states=hidden_states,
            pixel_position_ids=pixel_position_ids,
            padding_positions=padding_positions,
            output_length=self.num_soft_tokens,
        )

        if self.standardize:
            hidden_states = (hidden_states - self.std_bias) * self.std_scale

        return hidden_states
