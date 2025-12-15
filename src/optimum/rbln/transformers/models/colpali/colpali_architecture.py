from typing import List, Optional, Tuple, Union

import torch
from torch import nn
from transformers import GemmaForCausalLM, GemmaModel

from ..decoderonly.decoderonly_architecture import RotaryEmbedding, apply_rotary_pos_emb


def slice_and_unsqueeze_cos_sin(cos, sin, position_ids):
    """Slice cos[cache_position], sin[cache_position] vector for the query."""
    cos = cos[position_ids[0]][None, None, None, :, :]
    sin = sin[position_ids[0]][None, None, None, :, :]

    return cos, sin


class RBLNColPaliForRetrievalWrapper(nn.Module):
    def __init__(
        self,
        causal_lm: GemmaForCausalLM,
        embedding_proj_layer: nn.Module,
        max_seq_len: int,
        output_hidden_states: bool = False,
    ):
        super().__init__()
        self.text_config = causal_lm.config.text_config
        self.rotary_emb = self.get_rotary_emb(max_seq_len=max_seq_len)

        self.output_hidden_states = output_hidden_states
        self.language_model = self.convert_to_rbln_language_model(causal_lm.model.language_model, max_seq_len)

        self.num_hidden_layers = getattr(self.text_config, "num_hidden_layers", None)
        self.embedding_proj_layer = embedding_proj_layer

    def get_rotary_emb(self, max_seq_len):
        return RotaryEmbedding(config=self.text_config, max_seq_len_cached=max_seq_len)

    def convert_to_rbln_language_model(self, gemma_model: GemmaModel, max_seq_len: int):
        new_layers = []
        for layer in gemma_model.layers:
            new_self_attn = ColPaliAttention(
                layer.self_attn,
            )
            new_layer = ColPaliLayer(layer, new_self_attn)
            new_layers.append(new_layer)

        new_model = ColPaliModel(
            gemma_model,
            new_layers,
            output_hidden_states=self.output_hidden_states,
            max_seq_len=max_seq_len,
        )

        return new_model

    def forward(self, inputs_embeds: torch.Tensor, attention_mask: torch.Tensor, position_ids: torch.Tensor):
        attention_mask = (1.0 - attention_mask) * torch.finfo(torch.float32).min
        attention_mask = attention_mask[:, None, None, None, :]

        hidden_states, all_hidden_states = self.language_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            rotary_emb=self.rotary_emb,
            position_ids=position_ids,
        )
        embeddings = self.embedding_proj_layer(hidden_states)

        if self.output_hidden_states:
            return embeddings, all_hidden_states
        else:
            return embeddings


class ColPaliModel(nn.Module):
    def __init__(
        self, model, layers: List["ColPaliLayer"], output_hidden_states: bool = False, max_seq_len: int = 2048
    ):
        super().__init__()
        self._original_mod = model
        self.layers = nn.ModuleList(layers)
        self.output_hidden_states = output_hidden_states
        self.norm = self._original_mod.norm
        self.hidden_size = self._original_mod.config.hidden_size
        self.max_seq_len = max_seq_len

    def forward(
        self,
        inputs_embeds: Optional[torch.Tensor] = None,
        attention_mask: torch.Tensor = None,
        rotary_emb: Optional[Union[nn.Module, torch.Tensor]] = None,
        position_ids: Optional[torch.Tensor] = None,
    ):
        hidden_states = inputs_embeds * self.hidden_size**0.5

        cos, sin = rotary_emb(hidden_states, self.max_seq_len)  # dtype carrier, max_seq_len
        cos, sin = slice_and_unsqueeze_cos_sin(cos, sin, position_ids)

        all_hidden_states = () if self.output_hidden_states else None
        for layer in self.layers:
            if self.output_hidden_states:
                all_hidden_states += (hidden_states,)

            hidden_states = layer(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                cos=cos,
                sin=sin,
            )
        hidden_states = self.norm(hidden_states)

        if self.output_hidden_states:
            all_hidden_states += (hidden_states,)

        return hidden_states, all_hidden_states


class ColPaliLayer(nn.Module):
    def __init__(self, layer, self_attn: "ColPaliAttention"):
        super().__init__()
        self._original_mod = layer
        self.self_attn = self_attn
        self.mlp = layer.mlp
        self.input_layernorm = layer.input_layernorm
        self.post_attention_layernorm = layer.post_attention_layernorm

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        cos: Optional[torch.Tensor] = None,
        sin: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.FloatTensor]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            cos=cos,
            sin=sin,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class ColPaliAttention(nn.Module):
    def __init__(self, self_attn):
        super().__init__()
        self._original_mod = self_attn
        self.num_heads = (
            getattr(self._original_mod, "num_heads", None) or self._original_mod.config.num_attention_heads
        )
        self.head_dim = self._original_mod.head_dim
        self.scaling = self.head_dim**-0.5

        if hasattr(self._original_mod, "num_key_value_heads"):
            self.num_key_value_heads = self._original_mod.num_key_value_heads
        elif hasattr(self._original_mod, "config") and hasattr(self._original_mod.config, "num_key_value_heads"):
            self.num_key_value_heads = self._original_mod.config.num_key_value_heads
        else:
            self.num_key_value_heads = self.num_heads

        self.__post_init__()

    def __post_init__(self):
        self.q_proj = self._original_mod.q_proj
        self.k_proj = self._original_mod.k_proj
        self.v_proj = self._original_mod.v_proj
        self.o_proj = self._original_mod.o_proj

    def projection(self, hidden_states) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        return query_states, key_states, value_states

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        cos: Optional[torch.Tensor] = None,
        sin: Optional[torch.Tensor] = None,
    ):
        batch_size, query_length, _ = hidden_states.size()

        query_states, key_states, value_states = self.projection(hidden_states=hidden_states)

        query_states = query_states.view(batch_size, query_length, 1, self.num_heads, self.head_dim).transpose(1, 3)
        key_states = key_states.view(batch_size, query_length, 1, self.num_key_value_heads, self.head_dim).transpose(
            1, 3
        )
        value_states = value_states.view(
            batch_size, query_length, 1, self.num_key_value_heads, self.head_dim
        ).transpose(1, 3)

        if cos is not None and sin is not None:
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        attn_weights = torch.matmul(query_states, key_states.transpose(3, 4)) * self.scaling
        attn_weights = attn_weights + attention_mask
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32)
        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 3)

        attn_output = attn_output.reshape(batch_size, query_length, -1)
        attn_output = self.o_proj(attn_output)

        return attn_output
