# Copyright 2025 Rebellions Inc. All rights reserved.
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

"""SAM3 monkey patches for RBLN static-shape torch.jit.trace compilation."""

from contextlib import contextmanager
from typing import Any, Union

import torch

EAGER_MASK_FP16_SAFE_MIN: float = float(torch.finfo(torch.float16).min)


def _to_static_int(x: Any) -> int:
    """Convert tensor scalar to Python int (avoids Sym in torch.arange during trace)."""
    if isinstance(x, torch.Tensor):
        return int(x.item())
    return int(x)


def _get_coords_rbln(
    self: Any,
    height: Union[int, torch.Tensor],
    width: Union[int, torch.Tensor],
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Use Python int for torch.arange to avoid Sym in trace."""
    h = _to_static_int(height)
    w = _to_static_int(width)
    coords_h = torch.arange(0, h, device=device, dtype=dtype) / h
    coords_w = torch.arange(0, w, device=device, dtype=dtype) / w
    return coords_h, coords_w


def _sine_position_embedding_forward_rbln(
    self: Any,
    shape: torch.Tensor,
    device: torch.device | str,
    dtype: torch.dtype,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Use (1.0 - mask) instead of ~mask for NPU trace compatibility."""
    if mask is None:
        mask = torch.zeros((shape[0], shape[2], shape[3]), device=device, dtype=torch.bool)
    not_mask = (1.0 - mask.float()).to(dtype)
    y_embed = not_mask.cumsum(1)
    x_embed = not_mask.cumsum(2)
    if self.normalize:
        eps = 1e-6
        y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
        x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

    dim_t = torch.arange(self.num_pos_feats, dtype=torch.int64, device=device).to(dtype)
    dim_t = self.temperature ** (2 * torch.div(dim_t, 2, rounding_mode="floor") / self.num_pos_feats)

    pos_x = x_embed[:, :, :, None] / dim_t
    pos_y = y_embed[:, :, :, None] / dim_t
    pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
    pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
    pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
    return pos


def monkey_patch_sam3_detr_decoder() -> Any:
    """Patch Sam3DetrDecoder._get_coords for static-shape trace."""
    from transformers.models.sam3.modeling_sam3 import Sam3DetrDecoder

    original = Sam3DetrDecoder._get_coords
    Sam3DetrDecoder._get_coords = _get_coords_rbln
    return original


def monkey_patch_sam3_sine_position_embedding() -> Any:
    """Patch Sam3SinePositionEmbedding.forward for NPU trace compatibility."""
    from transformers.models.sam3.modeling_sam3 import Sam3SinePositionEmbedding

    original = Sam3SinePositionEmbedding.forward
    Sam3SinePositionEmbedding.forward = _sine_position_embedding_forward_rbln
    return original


@contextmanager
def _patch_eager_mask_fp16_safe():
    """Patch eager_mask to use fp16-safe min value for rebel compile."""
    from transformers import masking_utils

    _orig = masking_utils.eager_mask

    def _patched(
        batch_size,
        cache_position,
        kv_length,
        kv_offset=0,
        mask_function=None,
        attention_mask=None,
        dtype=torch.float32,
        allow_is_bidirectional_skip=False,
        use_vmap=False,
        **kwargs,
    ):
        _ = kwargs.pop("allow_is_causal_skip", None)
        _ = kwargs.pop("allow_torch_fix", None)
        mask = masking_utils.sdpa_mask(
            batch_size=batch_size,
            cache_position=cache_position,
            kv_length=kv_length,
            kv_offset=kv_offset,
            mask_function=mask_function,
            attention_mask=attention_mask,
            allow_is_causal_skip=False,
            allow_is_bidirectional_skip=allow_is_bidirectional_skip,
            allow_torch_fix=False,
            use_vmap=use_vmap,
            **kwargs,
        )
        if mask is not None:
            min_val = torch.tensor(EAGER_MASK_FP16_SAFE_MIN, device=mask.device, dtype=dtype)
            return torch.where(mask, torch.tensor(0.0, device=mask.device, dtype=dtype), min_val)
        return None

    masking_utils.eager_mask = _patched
    try:
        yield
    finally:
        masking_utils.eager_mask = _orig


def setup_decoder_and_patch(
    model: torch.nn.Module, spatial_shapes_list: list[tuple[int, int]]
) -> tuple[Any, ...]:
    """Set decoder spatial shapes and patch _get_rpb_matrix for static trace."""
    from transformers.models.sam3.modeling_sam3 import Sam3DetrDecoder

    decoder = model.detr_decoder
    decoder._rbln_spatial_shape_ints = spatial_shapes_list[0] if spatial_shapes_list else None
    original_get_rpb_matrix = Sam3DetrDecoder._get_rpb_matrix

    def _get_rpb_matrix_rbln(
        self: Any,
        reference_boxes: torch.Tensor,
        spatial_shape: tuple[Any, Any],
    ) -> torch.Tensor:
        """Use config-based spatial_shape for static trace."""
        if hasattr(self, "_rbln_spatial_shape_ints") and self._rbln_spatial_shape_ints is not None:
            spatial_shape = self._rbln_spatial_shape_ints
        return original_get_rpb_matrix(self, reference_boxes, spatial_shape)

    Sam3DetrDecoder._get_rpb_matrix = _get_rpb_matrix_rbln
    return (original_get_rpb_matrix,)
