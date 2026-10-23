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

from copy import deepcopy
from typing import Any

import torch
from torch import nn
from transformers.modeling_outputs import BaseModelOutputWithPooling
from transformers.models.siglip2.modeling_siglip2 import Siglip2Output

from ....configuration_utils import RBLNCompileConfig
from ....modeling import RBLNModel


def _grid_offsets(max_patches):
    if not isinstance(max_patches, int) or max_patches < 1:
        raise ValueError("max_patches must be a positive integer")
    offsets = {}
    offset = 0
    for height in range(1, max_patches + 1):
        for width in range(1, max_patches // height + 1):
            offsets[height, width] = offset
            offset += height * width
    return offsets


def patch_position_ids(spatial_shapes, max_patches):
    if spatial_shapes.ndim != 2 or spatial_shapes.shape[1] != 2 or spatial_shapes.shape[0] < 1:
        raise ValueError("spatial_shapes must have shape (image_batch_size, 2)")
    if spatial_shapes.dtype not in (torch.int32, torch.int64):
        raise ValueError("spatial_shapes must have an integer dtype")
    offsets = _grid_offsets(max_patches)
    positions = torch.arange(max_patches, dtype=torch.int64)
    rows = []
    for height, width in spatial_shapes.cpu().tolist():
        if (height, width) not in offsets:
            raise ValueError("Image grids must be positive and fit within max_patches")
        rows.append(offsets[height, width] + torch.where(positions < height * width, positions, 0))
    return torch.stack(rows)


class Siglip2VisionEmbeddingsWrapper(nn.Module):
    def __init__(self, embeddings, max_patches):
        super().__init__()
        self.patch_embedding = embeddings.patch_embedding
        grid_size = embeddings.position_embedding_size
        position = embeddings.position_embedding.weight.reshape(grid_size, grid_size, -1)
        rows = []
        with torch.no_grad():
            for height, width in _grid_offsets(max_patches):
                resized = embeddings.resize_positional_embeddings(
                    position,
                    torch.tensor([[height, width]], device=position.device),
                    max_length=height * width,
                )
                rows.append(resized[0])
            table = torch.cat(rows)
        self.position_embedding = nn.Embedding.from_pretrained(table, freeze=True)

    def forward(self, pixel_values, position_ids):
        patches = self.patch_embedding(pixel_values.to(dtype=self.patch_embedding.weight.dtype))
        return patches + self.position_embedding(position_ids)


class Siglip2Wrapper(nn.Module):
    def __init__(self, model, max_patches):
        super().__init__()
        self.model = deepcopy(model)
        self.model.vision_model.embeddings = Siglip2VisionEmbeddingsWrapper(
            self.model.vision_model.embeddings, max_patches
        )

    def forward(
        self, input_ids, pixel_values, attention_mask, pixel_attention_mask, patch_position_ids, position_ids=None
    ):
        output = self.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
            pixel_attention_mask=pixel_attention_mask,
            spatial_shapes=patch_position_ids,
            position_ids=position_ids,
            return_loss=False,
            output_hidden_states=False,
            output_attentions=False,
            return_dict=True,
        )
        return (
            output.logits_per_image,
            output.logits_per_text,
            output.text_embeds,
            output.image_embeds,
            output.text_model_output.last_hidden_state,
            output.text_model_output.pooler_output,
            output.vision_model_output.last_hidden_state,
            output.vision_model_output.pooler_output,
        )


class RBLNSiglip2Model(RBLNModel):
    """Run native NaFlex image/text similarity with dynamic per-image patch grids."""

    _tp_support = False

    @classmethod
    def _wrap_model_if_needed(cls, model, rbln_config):
        return Siglip2Wrapper(model, rbln_config.max_patches).eval()

    @classmethod
    def _update_rbln_config(cls, preprocessors, model=None, model_config=None, rbln_config=None):
        if rbln_config.max_seq_len is None:
            rbln_config.max_seq_len = model_config.text_config.max_position_embeddings
        if rbln_config.max_seq_len > model_config.text_config.max_position_embeddings:
            raise ValueError("max_seq_len exceeds the native text positional limit")
        if rbln_config.max_patches is None:
            rbln_config.max_patches = model_config.vision_config.num_patches
        patch_size = model_config.vision_config.patch_size
        patch_dim = model_config.vision_config.num_channels * patch_size * patch_size
        image_shape = [rbln_config.batch_size, rbln_config.max_patches]
        text_shape = [rbln_config.text_batch_size, rbln_config.max_seq_len]
        rbln_config.set_compile_cfgs(
            [
                RBLNCompileConfig(
                    input_info=[
                        ("input_ids", text_shape, "int64"),
                        ("pixel_values", [*image_shape, patch_dim], rbln_config.dtype),
                        ("attention_mask", text_shape, "int64"),
                        ("pixel_attention_mask", image_shape, "int64"),
                        ("patch_position_ids", image_shape, "int64"),
                        ("position_ids", text_shape, "int64"),
                    ]
                )
            ]
        )
        return rbln_config

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        pixel_values: torch.Tensor | None = None,
        pixel_attention_mask: torch.Tensor | None = None,
        spatial_shapes: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        return_loss: bool | None = None,
        return_dict: bool | None = None,
        output_hidden_states: bool | None = None,
        output_attentions: bool | None = None,
        **kwargs: Any,
    ) -> Siglip2Output | tuple:
        """Compute normalized embeddings and calibrated image/text similarity.

        Args:
            input_ids: Text token IDs at the compiled batch and sequence length.
            pixel_values: Native patchified images padded to the compiled patch budget.
            pixel_attention_mask: Optional binary mask over image patches.
            spatial_shapes: Positive per-image patch grids fitting the patch budget.
            attention_mask: Optional binary text attention mask.
            position_ids: Optional text position IDs, with batch one or the compiled text batch.
            return_loss: Training losses are not supported.
            return_dict: Return a native Siglip2Output when true, otherwise its native tuple.
            output_hidden_states: Additional hidden-state diagnostics are not compiled.
            output_attentions: Attention diagnostics are not compiled.
            kwargs: Additional nonempty arguments are rejected.

        Returns:
            Native similarity, embedding and pooled encoder outputs.
        """
        if return_loss or output_hidden_states or output_attentions:
            raise ValueError("Training loss and diagnostic tensors are not compiled")
        if any(value is not None for value in kwargs.values()):
            raise TypeError(f"Unexpected arguments: {tuple(kwargs)}")
        config = self.rbln_config
        image_shape = (config.batch_size, config.max_patches)
        text_shape = (config.text_batch_size, config.max_seq_len)
        patch_size = self.config.vision_config.patch_size
        patch_dim = self.config.vision_config.num_channels * patch_size * patch_size
        if input_ids is None or tuple(input_ids.shape) != text_shape:
            raise ValueError(f"input_ids must match the compiled shape {text_shape}; use padding=max_length")
        if input_ids.dtype not in (torch.int32, torch.int64):
            raise ValueError("input_ids must have an integer dtype")
        if (input_ids < 0).any() or (input_ids >= self.config.text_config.vocab_size).any():
            raise ValueError("input_ids are outside the checkpoint vocabulary")
        if pixel_values is None or tuple(pixel_values.shape) != (*image_shape, patch_dim):
            raise ValueError(f"pixel_values must match the compiled patch shape {(*image_shape, patch_dim)}")
        if spatial_shapes is None or spatial_shapes.ndim != 2 or spatial_shapes.shape[0] != config.batch_size:
            raise ValueError("spatial_shapes must match the image batch")
        patch_ids = patch_position_ids(spatial_shapes, config.max_patches)
        if attention_mask is None:
            attention_mask = torch.ones(text_shape, dtype=torch.int64)
        if pixel_attention_mask is None:
            pixel_attention_mask = torch.ones(image_shape, dtype=torch.int64)
        for name, value, shape in [
            ("attention_mask", attention_mask, text_shape),
            ("pixel_attention_mask", pixel_attention_mask, image_shape),
        ]:
            if tuple(value.shape) != shape or ((value != 0) & (value != 1)).any():
                raise ValueError(f"{name} must be a binary mask with shape {shape}")
        if position_ids is None:
            position_ids = torch.arange(config.max_seq_len, dtype=torch.int64).unsqueeze(0)
        if position_ids.dtype not in (torch.int32, torch.int64):
            raise ValueError("position_ids must have an integer dtype")
        if tuple(position_ids.shape) not in ((1, config.max_seq_len), text_shape):
            raise ValueError("position_ids must match the text length and broadcast over the text batch")
        if (position_ids < 0).any() or (position_ids >= self.config.text_config.max_position_embeddings).any():
            raise ValueError("position_ids exceed the native positional range")
        output = self.model[0](
            input_ids.to(device="cpu", dtype=torch.int64),
            pixel_values,
            attention_mask.to(device="cpu", dtype=torch.int64),
            pixel_attention_mask.to(device="cpu", dtype=torch.int64),
            patch_ids,
            position_ids.to(device="cpu", dtype=torch.int64).expand(text_shape).contiguous(),
        )
        result = Siglip2Output(
            logits_per_image=output[0],
            logits_per_text=output[1],
            text_embeds=output[2],
            image_embeds=output[3],
            text_model_output=BaseModelOutputWithPooling(last_hidden_state=output[4], pooler_output=output[5]),
            vision_model_output=BaseModelOutputWithPooling(last_hidden_state=output[6], pooler_output=output[7]),
        )
        return_dict = self.config.use_return_dict if return_dict is None else return_dict
        return result if return_dict else result.to_tuple()


__all__ = ["RBLNSiglip2Model"]
