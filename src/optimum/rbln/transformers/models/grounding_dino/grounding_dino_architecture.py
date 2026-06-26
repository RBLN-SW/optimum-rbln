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
import math
from functools import wraps
from typing import TYPE_CHECKING, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor
from transformers.models.grounding_dino.modeling_grounding_dino import (
    GroundingDinoDecoder,
    GroundingDinoEncoder,
)


if TYPE_CHECKING:
    from .configuration_grounding_dino import RBLNGroundingDinoDecoderConfig, RBLNGroundingDinoEncoderConfig


def monkey_patch():
    from transformers.models.grounding_dino.modeling_grounding_dino import (
        GroundingDinoBiMultiHeadAttention,
        GroundingDinoEncoderLayer,
        GroundingDinoMultiscaleDeformableAttention,
        MultiScaleDeformableAttention,
    )

    original_forward = GroundingDinoMultiscaleDeformableAttention.forward
    original_bi_multihead_attention_forward = GroundingDinoBiMultiHeadAttention.forward
    original_encoder_layer_forward = GroundingDinoEncoderLayer.forward
    original_multiscale_deform_attn = MultiScaleDeformableAttention.forward

    # Patch the methods with the custom implementations
    GroundingDinoMultiscaleDeformableAttention.forward = _GroundingDinoMultiscaleDeformableAttention.forward
    GroundingDinoBiMultiHeadAttention.forward = _GroundingDinoBiMultiHeadAttention.forward
    GroundingDinoEncoderLayer.forward = _GroundingDinoEncoderLayer.forward
    MultiScaleDeformableAttention.forward = _MultiScaleDeformableAttention.forward

    return (
        original_forward,
        original_bi_multihead_attention_forward,
        original_encoder_layer_forward,
        original_multiscale_deform_attn,
    )


def restore_monkey_patch(
    original_forward,
    original_bi_multihead_attention_forward,
    original_encoder_layer_forward,
    original_multiscale_deform_attn,
):
    from transformers.models.grounding_dino.modeling_grounding_dino import (
        GroundingDinoBiMultiHeadAttention,
        GroundingDinoEncoderLayer,
        GroundingDinoMultiscaleDeformableAttention,
        MultiScaleDeformableAttention,
    )

    # Restore the original methods
    GroundingDinoMultiscaleDeformableAttention.forward = original_forward
    GroundingDinoBiMultiHeadAttention.forward = original_bi_multihead_attention_forward
    GroundingDinoEncoderLayer.forward = original_encoder_layer_forward
    MultiScaleDeformableAttention.forward = original_multiscale_deform_attn


def monkey_patch_decorator(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Apply monkey patch and capture original methods
        original_functions = monkey_patch()
        try:
            # Call the original function
            result = func(*args, **kwargs)
        finally:
            # Restore original methods
            restore_monkey_patch(*original_functions)
        return result

    return wrapper


def get_sine_pos_embed(
    pos_tensor: torch.Tensor, num_pos_feats: int = 128, temperature: int = 10000, exchange_xy: bool = True
) -> Tensor:
    scale = 2 * math.pi
    dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=pos_tensor.device)
    dim_t = temperature ** (2 * torch.div(dim_t, 2, rounding_mode="floor") / num_pos_feats)

    scaled_pos = pos_tensor.unsqueeze(-1) * scale / dim_t
    reshaped_pos = scaled_pos.view(*scaled_pos.shape[:-1], -1, 2)
    sin_chunk, cos_chunk = torch.split(reshaped_pos, 1, dim=-1)
    sin_embed = sin_chunk.squeeze(-1).sin()
    cos_embed = cos_chunk.squeeze(-1).cos()

    pos_embed = torch.stack((sin_embed, cos_embed), dim=-1).flatten(-2)

    if exchange_xy and pos_tensor.shape[-1] >= 2:
        swapped_embeds = torch.cat([pos_embed[..., 1:2, :], pos_embed[..., 0:1, :], pos_embed[..., 2:, :]], dim=-2)
        pos_embed = swapped_embeds

    position_embeddings = pos_embed.flatten(start_dim=-2)

    return position_embeddings


class _GroundingDinoEncoder(torch.nn.Module):
    def __init__(self, model: "GroundingDinoEncoder", rbln_config: "RBLNGroundingDinoEncoderConfig"):
        super().__init__()
        self.layers = model.layers
        self.config = model.config
        self.rbln_config = rbln_config
        self.spatial_shapes = self.rbln_config.spatial_shapes
        self.spatial_shapes_list = self.rbln_config.spatial_shapes_list
        self.text_position_embedding = model.layers[0].get_text_position_embeddings(
            torch.zeros(1, model.config.max_text_len, model.config.d_model),
            None,
            torch.arange(model.config.max_text_len, dtype=torch.int32).unsqueeze(0),
        )

    @monkey_patch_decorator
    def forward(
        self,
        vision_features: torch.Tensor,
        vision_attention_mask: torch.Tensor,
        vision_position_embedding: torch.Tensor,
        text_features: Optional[torch.Tensor] = None,
        text_attention_mask: Optional[torch.Tensor] = None,
        text_self_attention_masks: Optional[torch.Tensor] = None,
        reference_points: Optional[torch.Tensor] = None,
    ):
        output_attentions = self.rbln_config.output_attentions
        output_hidden_states = self.rbln_config.output_hidden_states

        encoder_vision_states = () if output_hidden_states else None
        encoder_text_states = () if output_hidden_states else None
        all_attns = () if output_attentions else None
        all_attn_fused_text = () if output_attentions else None
        all_attn_fused_vision = () if output_attentions else None
        all_attn_enhanced_text = () if output_attentions else None
        all_attn_deformable = () if output_attentions else None
        for _, encoder_layer in enumerate(self.layers):
            if output_hidden_states:
                encoder_vision_states += (vision_features,)
                encoder_text_states += (text_features,)

            (vision_features, text_features), attentions = encoder_layer(
                vision_features=vision_features,
                vision_position_embedding=vision_position_embedding,
                spatial_shapes=self.spatial_shapes,
                spatial_shapes_list=self.spatial_shapes_list,
                level_start_index=None,
                key_padding_mask=vision_attention_mask,
                reference_points=reference_points,
                text_features=text_features,
                text_attention_mask=text_attention_mask,
                text_position_embedding=self.text_position_embedding,
                text_self_attention_masks=text_self_attention_masks,
            )
            if output_attentions:
                all_attn_fused_vision += (attentions[0],)
                all_attn_fused_text += (attentions[1],)
                all_attn_enhanced_text += (attentions[2],)
                all_attn_deformable += (attentions[3],)

        if output_hidden_states:
            encoder_vision_states += (vision_features,)
            encoder_text_states += (text_features,)

        if output_attentions:
            all_attns = (all_attn_fused_vision, all_attn_fused_text, all_attn_enhanced_text, all_attn_deformable)

        enc_outputs = [vision_features, text_features, encoder_vision_states, encoder_text_states, all_attns]

        return tuple(v for v in enc_outputs if v is not None)


class _GroundingDinoDecoder(torch.nn.Module):
    def __init__(self, model: "GroundingDinoDecoder", rbln_config: "RBLNGroundingDinoDecoderConfig"):
        super().__init__()
        self.layers = model.layers
        self.config = model.config
        self.spatial_shapes = rbln_config.spatial_shapes
        self.spatial_shapes_list = rbln_config.spatial_shapes_list
        self.rbln_config = rbln_config
        self.reference_points_head = model.reference_points_head
        self.bbox_embed = model.bbox_embed
        self.layer_norm = model.layer_norm

    @monkey_patch_decorator
    def forward(
        self,
        inputs_embeds,
        vision_encoder_hidden_states,
        vision_encoder_attention_mask=None,
        text_encoder_hidden_states=None,
        text_encoder_attention_mask=None,
        reference_points=None,
        valid_ratios=None,
    ):
        output_attentions = self.rbln_config.output_attentions
        output_hidden_states = self.rbln_config.output_hidden_states

        if inputs_embeds is not None:
            hidden_states = inputs_embeds

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        all_attns = () if output_attentions else None
        all_cross_attns_vision = () if (output_attentions and vision_encoder_hidden_states is not None) else None
        all_cross_attns_text = () if (output_attentions and text_encoder_hidden_states is not None) else None
        intermediate = ()
        intermediate_reference_points = ()

        if text_encoder_attention_mask is not None:
            text_encoder_attention_mask = text_encoder_attention_mask[:, None, None, :]
            text_encoder_attention_mask = text_encoder_attention_mask.repeat(
                1, self.config.decoder_attention_heads, self.config.num_queries, 1
            )
            text_encoder_attention_mask = text_encoder_attention_mask
            text_encoder_attention_mask = text_encoder_attention_mask * torch.finfo(torch.float16).min

        for idx, decoder_layer in enumerate(self.layers):
            num_coordinates = reference_points.shape[-1]
            if num_coordinates == 4:
                reference_points_input = (
                    reference_points[:, :, None] * torch.cat([valid_ratios, valid_ratios], -1)[:, None]
                )
            elif num_coordinates == 2:
                reference_points_input = reference_points[:, :, None] * valid_ratios[:, None]
            else:
                raise ValueError("Last dim of reference_points must be 2 or 4, but got {reference_points.shape[-1]}")
            _query_pos = get_sine_pos_embed(reference_points_input[:, :, 0, :], num_pos_feats=self.config.d_model // 2)
            query_pos = self.reference_points_head(_query_pos)

            # In original implementation they apply layer norm before outputting intermediate hidden states
            # Though that's not through between layers so the layers use as input the output of the previous layer
            # withtout layer norm
            if output_hidden_states:
                all_hidden_states += (self.layer_norm(hidden_states),)

            layer_outputs = decoder_layer(
                hidden_states=hidden_states,
                position_embeddings=query_pos,
                reference_points=reference_points_input,
                spatial_shapes=self.spatial_shapes,
                spatial_shapes_list=self.spatial_shapes_list,
                level_start_index=None,
                vision_encoder_hidden_states=vision_encoder_hidden_states,
                vision_encoder_attention_mask=vision_encoder_attention_mask,
                text_encoder_hidden_states=text_encoder_hidden_states,
                text_encoder_attention_mask=text_encoder_attention_mask,
                self_attn_mask=None,
                output_attentions=output_attentions,
            )

            hidden_states = layer_outputs[0]

            # hack implementation for iterative bounding box refinement
            if self.bbox_embed is not None:
                tmp = self.bbox_embed[idx](hidden_states)
                num_coordinates = reference_points.shape[-1]
                if num_coordinates == 4:
                    new_reference_points = tmp + torch.special.logit(reference_points, eps=1e-5)
                    new_reference_points = new_reference_points.sigmoid()
                elif num_coordinates == 2:
                    new_reference_points = tmp
                    new_reference_points[..., :2] = tmp[..., :2] + torch.special.logit(reference_points, eps=1e-5)
                    new_reference_points = new_reference_points.sigmoid()
                else:
                    raise ValueError(
                        f"Last dim of reference_points must be 2 or 4, but got {reference_points.shape[-1]}"
                    )
                reference_points = new_reference_points.detach()

            intermediate += (self.layer_norm(hidden_states),)
            intermediate_reference_points += (reference_points,)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

                if text_encoder_hidden_states is not None:
                    all_cross_attns_text += (layer_outputs[2],)

                if vision_encoder_hidden_states is not None:
                    all_cross_attns_vision += (layer_outputs[3],)

        # Keep batch_size as first dimension
        intermediate = torch.stack(intermediate, dim=1)
        intermediate_reference_points = torch.stack(intermediate_reference_points, dim=1)
        hidden_states = self.layer_norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if output_attentions:
            all_attns += (all_self_attns, all_cross_attns_text, all_cross_attns_vision)

        return tuple(
            v
            for v in [
                hidden_states,
                intermediate,
                intermediate_reference_points,
                all_hidden_states,
                all_attns,
            ]
            if v is not None
        )


class _GroundingDinoEncoderLayer(torch.nn.Module):
    def forward(
        self,
        vision_features: Tensor,
        vision_position_embedding: Tensor,
        spatial_shapes: Tensor,
        spatial_shapes_list: List[Tuple[int, int]],
        level_start_index: Tensor,
        key_padding_mask: Tensor,
        reference_points: Tensor,
        text_features: Optional[Tensor] = None,
        text_attention_mask: Optional[Tensor] = None,
        text_position_embedding: Optional[Tensor] = None,
        text_self_attention_masks: Optional[Tensor] = None,
        text_position_ids: Optional[Tensor] = None,
    ):
        text_position_embedding = self.get_text_position_embeddings(
            text_features, text_position_embedding, text_position_ids
        )

        (vision_features, vision_fused_attn), (text_features, text_fused_attn) = self.fusion_layer(
            vision_features=vision_features,
            text_features=text_features,
            attention_mask_vision=key_padding_mask,
            attention_mask_text=text_attention_mask,
        )

        (text_features, text_enhanced_attn) = self.text_enhancer_layer(
            hidden_states=text_features,
            attention_masks=(1.0 - text_self_attention_masks),  # RBLN FIX, change from ~ to 1.0 -
            position_embeddings=(text_position_embedding if text_position_embedding is not None else None),
        )

        (vision_features, vision_deformable_attn) = self.deformable_layer(
            hidden_states=vision_features,
            attention_mask=(1.0 - key_padding_mask),  # RBLN FIX, change from ~ to 1.0 -
            position_embeddings=vision_position_embedding,
            reference_points=reference_points,
            spatial_shapes=spatial_shapes,
            spatial_shapes_list=spatial_shapes_list,
            level_start_index=level_start_index,
        )

        return (
            (vision_features, text_features),
            (vision_fused_attn, text_fused_attn, text_enhanced_attn, vision_deformable_attn),
        )


class _GroundingDinoMultiscaleDeformableAttention(torch.nn.Module):
    """
    Multiscale deformable attention as proposed in Deformable DETR.
    """

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        position_embeddings: Optional[torch.Tensor] = None,
        reference_points=None,
        spatial_shapes=None,
        spatial_shapes_list=None,
        level_start_index=None,
        output_attentions: bool = False,
    ):
        # add position embeddings to the hidden states before projecting to queries and keys
        if position_embeddings is not None:
            hidden_states = self.with_pos_embed(hidden_states, position_embeddings)

        batch_size, num_queries, _ = hidden_states.shape
        batch_size, sequence_length, _ = encoder_hidden_states.shape
        # Ignore copy
        if torch.compiler.is_exporting():
            torch._check(
                (spatial_shapes[:, 0] * spatial_shapes[:, 1]).sum().item() == sequence_length,
                "Make sure to align the spatial shapes with the sequence length of the encoder hidden states",
            )
        else:
            if (spatial_shapes[:, 0] * spatial_shapes[:, 1]).sum() != sequence_length:
                raise ValueError(
                    "Make sure to align the spatial shapes with the sequence length of the encoder hidden states"
                )

        value = self.value_proj(encoder_hidden_states)
        if attention_mask is not None:
            # RBLN FIX: bool tensor to float tensor
            value = attention_mask * value

        value = value.view(batch_size, sequence_length, self.n_heads, self.d_model // self.n_heads)
        sampling_offsets = self.sampling_offsets(hidden_states).view(
            batch_size, num_queries, self.n_heads, self.n_levels, self.n_points, 2
        )
        attention_weights = self.attention_weights(hidden_states).view(
            batch_size, num_queries, self.n_heads, self.n_levels * self.n_points
        )
        attention_weights = F.softmax(attention_weights, -1).view(
            batch_size, num_queries, self.n_heads, self.n_levels, self.n_points
        )
        # batch_size, num_queries, n_heads, n_levels, n_points, 2
        num_coordinates = reference_points.shape[-1]
        if num_coordinates == 2:
            offset_normalizer = 0.5 * torch.stack([spatial_shapes[..., 1], spatial_shapes[..., 0]], -1)
            sampling_grids = (
                2 * reference_points[:, :, None, :, None, :]
                - 1
                + sampling_offsets / offset_normalizer[None, None, None, :, None, :]
            )
        elif num_coordinates == 4:
            ref_points_xy, ref_points_wh = torch.split(reference_points, 2, dim=-1)
            ref_points_xy = ref_points_xy[:, :, None, :, None, :]
            ref_points_wh = ref_points_wh[:, :, None, :, None, :]
            ref_points_grids = 2 * ref_points_xy - 1
            offset_grids = sampling_offsets / self.n_points * ref_points_wh
            sampling_grids = ref_points_grids + offset_grids

        else:
            raise ValueError(f"Last dim of reference_points must be 2 or 4, but got {reference_points.shape[-1]}")

        output = self.attn(
            value,
            spatial_shapes,
            spatial_shapes_list,
            level_start_index,
            sampling_grids,
            attention_weights,
            self.im2col_step,
        )

        output = self.output_proj(output)

        return output, attention_weights


class _GroundingDinoBiMultiHeadAttention(torch.nn.Module):
    def forward(
        self,
        vision_features: torch.FloatTensor,
        text_features: torch.FloatTensor,
        vision_attention_mask: Optional[torch.BoolTensor] = None,
        text_attention_mask: Optional[torch.BoolTensor] = None,
    ) -> Tuple[Tuple[torch.FloatTensor, torch.FloatTensor], Tuple[torch.FloatTensor, torch.FloatTensor]]:
        batch_size, tgt_len, _ = vision_features.size()

        vision_query_states = self.vision_proj(vision_features) * self.scale
        vision_query_states = self._reshape(vision_query_states, tgt_len, batch_size)

        text_key_states = self.text_proj(text_features)
        text_key_states = self._reshape(text_key_states, -1, batch_size)

        vision_value_states = self.values_vision_proj(vision_features)
        vision_value_states = self._reshape(vision_value_states, -1, batch_size)

        text_value_states = self.values_text_proj(text_features)
        text_value_states = self._reshape(text_value_states, -1, batch_size)

        proj_shape = (batch_size * self.num_heads, -1, self.head_dim)

        vision_query_states = vision_query_states.view(*proj_shape)
        text_key_states = text_key_states.view(*proj_shape)
        vision_value_states = vision_value_states.view(*proj_shape)
        text_value_states = text_value_states.view(*proj_shape)

        src_len = text_key_states.size(1)
        attn_weights = torch.bmm(vision_query_states, text_key_states.transpose(1, 2))  # bs*nhead, nimg, ntxt

        if attn_weights.size() != (batch_size * self.num_heads, tgt_len, src_len):
            raise ValueError(
                f"Attention weights should be of size {(batch_size * self.num_heads, tgt_len, src_len)}, but is {attn_weights.size()}"
            )

        # RBLN FIX: max_values from scalar to vector
        attn_weights = attn_weights - torch.max(attn_weights).reshape(1).repeat(src_len)
        # # Do not increase -50000/50000, data type half has quite limited range
        attn_weights = torch.clamp(attn_weights, min=-50000, max=50000)

        # RBLN FIX: max_values from scalar to vector
        text_attn_weights = attn_weights - torch.max(attn_weights, dim=1, keepdim=True)[0].repeat(1, tgt_len, 1)

        # # Do not increase -50000/50000, data type half has quite limited range
        text_attn_weights = torch.clamp(text_attn_weights, min=-50000, max=50000)

        text_attn_weights = text_attn_weights.transpose(1, 2)

        # mask vision for language
        if vision_attention_mask is not None:
            # RBLN FIX: bool tensor to float tensor, broadcast across heads and src_len
            mask = vision_attention_mask
            if mask.dim() == 3:
                mask = mask[..., 0]
            mask = mask[:, None, None, :].repeat(1, self.num_heads, 1, 1).flatten(0, 1)
            text_attn_weights = text_attn_weights + mask * torch.finfo(text_attn_weights.dtype).min

        text_attn_weights = text_attn_weights.softmax(dim=-1)

        # mask language for vision
        if text_attention_mask is not None:
            text_attention_mask = text_attention_mask[:, None, None, :].repeat(1, self.num_heads, 1, 1).flatten(0, 1)
            # RBLN FIX: bool tensor to float tensor
            mask = text_attention_mask * torch.finfo(torch.float16).min
            attn_weights = attn_weights + mask

        vision_attn_weights = attn_weights.softmax(dim=-1)

        vision_attn_probs = F.dropout(vision_attn_weights, p=self.dropout, training=self.training)
        text_attn_probs = F.dropout(text_attn_weights, p=self.dropout, training=self.training)

        vision_attn_output = torch.bmm(vision_attn_probs, text_value_states)
        text_attn_output = torch.bmm(text_attn_probs, vision_value_states)

        if vision_attn_output.size() != (batch_size * self.num_heads, tgt_len, self.head_dim):
            raise ValueError(
                f"`vision_attn_output` should be of size {(batch_size, self.num_heads, tgt_len, self.head_dim)}, but is {vision_attn_output.size()}"
            )

        if text_attn_output.size() != (batch_size * self.num_heads, src_len, self.head_dim):
            raise ValueError(
                f"`text_attn_output` should be of size {(batch_size, self.num_heads, src_len, self.head_dim)}, but is {text_attn_output.size()}"
            )

        vision_attn_output = vision_attn_output.view(batch_size, self.num_heads, tgt_len, self.head_dim)
        vision_attn_output = vision_attn_output.transpose(1, 2)
        vision_attn_output = vision_attn_output.reshape(batch_size, tgt_len, self.embed_dim)

        text_attn_output = text_attn_output.view(batch_size, self.num_heads, src_len, self.head_dim)
        text_attn_output = text_attn_output.transpose(1, 2)
        text_attn_output = text_attn_output.reshape(batch_size, src_len, self.embed_dim)

        vision_attn_output = self.out_vision_proj(vision_attn_output)
        text_attn_output = self.out_text_proj(text_attn_output)

        return (vision_attn_output, vision_attn_weights), (text_attn_output, text_attn_weights)


class _MultiScaleDeformableAttention(torch.nn.Module):
    def forward(
        self,
        value: Tensor,
        value_spatial_shapes: Tensor,
        value_spatial_shapes_list: List[Tuple],
        level_start_index: Tensor,
        sampling_grids: Tensor,
        attention_weights: Tensor,
        im2col_step: int,
    ):
        batch_size, _, num_heads, hidden_dim = value.shape
        _, num_queries, num_heads, num_levels, num_points, _ = sampling_grids.shape
        value_list = value.split([height * width for height, width in value_spatial_shapes_list], dim=1)
        sampling_value_list = []
        sampling_grids_list = [t.squeeze(3) for t in torch.split(sampling_grids, 1, dim=3)]
        for level_id, (height, width) in enumerate(value_spatial_shapes_list):
            value_l_ = (
                value_list[level_id].permute(0, 2, 3, 1).reshape(batch_size * num_heads, hidden_dim, height, width)
            )
            sampling_grid_l_ = sampling_grids_list[level_id].transpose(1, 2).flatten(0, 1)
            sampling_value_l_ = torch.nn.functional.grid_sample(
                value_l_,
                sampling_grid_l_,
                mode="bilinear",
                padding_mode="zeros",
                align_corners=False,
            )
            sampling_value_list.append(sampling_value_l_)

        sampling_values = torch.cat(sampling_value_list, dim=-1)
        attention_weights_prep = attention_weights.transpose(1, 2)
        values_permuted = sampling_values.permute(0, 2, 3, 1)

        weights_for_matmul = attention_weights_prep.reshape(
            batch_size * num_heads, num_queries, 1, num_levels * num_points
        )
        output_before_permute = torch.matmul(weights_for_matmul, values_permuted)
        output_before_view = output_before_permute.squeeze(2).permute(0, 2, 1)
        output = output_before_view.reshape(batch_size, num_heads * hidden_dim, num_queries)

        return output.transpose(1, 2).contiguous()
