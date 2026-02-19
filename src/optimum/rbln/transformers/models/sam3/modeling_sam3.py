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

from typing import TYPE_CHECKING, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoModel
from transformers.models.sam3.modeling_sam3 import Sam3ImageSegmentationOutput

from ...modeling_generic import RBLNImageModel
from .configuration_sam3 import RBLNSam3ModelConfig, _compute_sam3_spatial_shapes
from .sam3_architecture import (
    monkey_patch_sam3_detr_decoder,
    monkey_patch_sam3_sine_position_embedding,
    setup_decoder_and_patch,
    _patch_eager_mask_fp16_safe,
)

if TYPE_CHECKING:
    from transformers import (
        AutoFeatureExtractor,
        AutoProcessor,
        AutoTokenizer,
        PretrainedConfig,
        PreTrainedModel,
    )


class RBLNSam3Model(RBLNImageModel):
    """
    RBLN optimized SAM3 model for image segmentation and object detection.

    SAM3 combines vision (ViT + FPN) and text (CLIP) encoders with DETR-style
    encoder-decoder and mask decoder for text-conditioned segmentation and detection.
    """

    auto_model_class = AutoModel

    @classmethod
    def _wrap_model_if_needed(cls, model: nn.Module, rbln_config: RBLNSam3ModelConfig) -> nn.Module:
        """Apply monkey patches and wrap model for static-shape torch.jit.trace compatibility."""
        monkey_patch_sam3_detr_decoder()
        monkey_patch_sam3_sine_position_embedding()
        if rbln_config.spatial_shapes_list is None:
            raise ValueError(
                "spatial_shapes_list must be set. Ensure _update_rbln_config runs before compile."
            )
        setup_decoder_and_patch(model, rbln_config.spatial_shapes_list)

        class Sam3TraceWrapper(nn.Module):
            def __init__(self, model: nn.Module):
                super().__init__()
                self.model = model

            def forward(
                self,
                pixel_values: torch.FloatTensor,
                input_ids: torch.LongTensor,
                attention_mask: Optional[torch.Tensor] = None,
                **kwargs,
            ):
                out = self.model(
                    pixel_values=pixel_values,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    return_dict=True,
                    **kwargs,
                )
                return (out.pred_masks, out.pred_boxes, out.pred_logits, out.presence_logits, out.semantic_seg)

        return Sam3TraceWrapper(model).eval()

    @classmethod
    def get_compiled_model(cls, model: "PreTrainedModel", rbln_config: "RBLNSam3ModelConfig"):
        """Build example_inputs and compile with trace-compatible wrapper."""
        if rbln_config._allow_no_compile_cfgs:
            return {}

        example_inputs = rbln_config.example_inputs or cls._build_example_inputs_from_input_info(
            rbln_config.compile_cfgs[0].input_info
        )
        model = cls._wrap_model_if_needed(model, rbln_config)
        rbln_compile_config = rbln_config.compile_cfgs[0]

        with _patch_eager_mask_fp16_safe():
            compiled_model = cls.compile(
                model,
                rbln_compile_config=rbln_compile_config,
                create_runtimes=rbln_config.create_runtimes,
                device=rbln_config.device,
                example_inputs=example_inputs,
            )
        return compiled_model

    @classmethod
    def _build_example_inputs_from_input_info(
        cls, input_info: list
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build dummy example_inputs from input_info when preprocessor is unavailable."""
        example_inputs = []
        for name, shape, dtype in input_info:
            if dtype == "float32":
                example_inputs.append(torch.randn(*shape, dtype=torch.float32))
            elif name == "attention_mask":
                example_inputs.append(torch.ones(*shape, dtype=torch.int64))
            else:
                example_inputs.append(torch.zeros(*shape, dtype=torch.int64))
        return tuple(example_inputs)

    def forward(
        self,
        pixel_values: torch.FloatTensor,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        input_boxes: Optional[torch.FloatTensor] = None,
        input_boxes_labels: Optional[torch.LongTensor] = None,
        return_dict: bool = True,
        **kwargs,
    ) -> Union[Tuple, Sam3ImageSegmentationOutput]:
        """
        Forward pass. Returns Sam3ImageSegmentationOutput.

        input_boxes/input_boxes_labels not supported in compiled mode.
        Smaller images are padded; use post_process_instance_segmentation with
        target_sizes=inputs["original_sizes"] to resize outputs.
        """
        pixel_values = self._pad_image_to_compiled_size(pixel_values)
        input_ids, attention_mask = self._pad_text_to_compiled_size(input_ids, attention_mask)

        output = self.model[0](
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        if isinstance(output, (tuple, list)):
            pred_masks, pred_boxes, pred_logits, presence_logits, semantic_seg = output
            if return_dict:
                return Sam3ImageSegmentationOutput(
                    pred_masks=pred_masks,
                    pred_boxes=pred_boxes,
                    pred_logits=pred_logits,
                    presence_logits=presence_logits,
                    semantic_seg=semantic_seg,
                )
            return output
        return output

    @classmethod
    def _update_rbln_config(
        cls,
        preprocessors: Optional[Union["AutoFeatureExtractor", "AutoProcessor", "AutoTokenizer"]] = None,
        model: Optional["PreTrainedModel"] = None,
        model_config: Optional["PretrainedConfig"] = None,
        rbln_config: Optional[RBLNSam3ModelConfig] = None,
    ) -> RBLNSam3ModelConfig:
        rbln_config.image_size = cls._resolve_image_size(preprocessors, model_config, rbln_config)
        patch_size = cls._get_patch_size_from_config(model_config)
        max_text_len = cls._get_max_text_len_from_config(model_config, rbln_config)
        rbln_config.max_text_len = max_text_len

        if rbln_config.spatial_shapes_list is None:
            rbln_config.spatial_shapes_list = _compute_sam3_spatial_shapes(
                rbln_config.image_height, rbln_config.image_width, patch_size
            )

        from ....configuration_utils import RBLNCompileConfig

        input_info = [
            (
                "pixel_values",
                [rbln_config.batch_size, 3, rbln_config.image_height, rbln_config.image_width],
                "float32",
            ),
            ("input_ids", [rbln_config.batch_size, max_text_len], "int64"),
            ("attention_mask", [rbln_config.batch_size, max_text_len], "int64"),
        ]
        rbln_config.set_compile_cfgs([RBLNCompileConfig(input_info=input_info)])
        return rbln_config

    @classmethod
    def _resolve_image_size(
        cls,
        preprocessors: Optional[list],
        model_config: Optional["PretrainedConfig"],
        rbln_config: RBLNSam3ModelConfig,
    ) -> Tuple[int, int]:
        """Resolve (height, width) from rbln_config, preprocessors, or model_config."""
        if rbln_config.image_size is not None:
            return rbln_config.image_size
        for processor in preprocessors or []:
            if hasattr(processor, "image_processor") and hasattr(processor.image_processor, "size"):
                size = processor.image_processor.size
                if isinstance(size, dict):
                    if "height" in size and "width" in size:
                        return (size["height"], size["width"])
                    if "shortest_edge" in size:
                        return (size["shortest_edge"], size["shortest_edge"])
                break
        if model_config is not None:
            img_size = getattr(model_config, "image_size", None)
            if img_size is None and hasattr(model_config, "vision_config"):
                vc = model_config.vision_config
                if hasattr(vc, "backbone_config"):
                    img_size = getattr(vc.backbone_config, "image_size", None)
            if img_size is not None:
                if isinstance(img_size, (list, tuple)) and len(img_size) >= 2:
                    return (int(img_size[0]), int(img_size[1]))
                val = int(img_size[0] if isinstance(img_size, (list, tuple)) else img_size)
                return (val, val)
        raise ValueError(
            "image_size could not be resolved. Provide rbln_image_size, a processor with "
            "image_processor.size, or load a model with config.json (vision_config.backbone_config.image_size)."
        )

    @classmethod
    def _get_patch_size_from_config(cls, model_config: Optional["PretrainedConfig"]) -> int:
        """Get patch_size from model_config (vision_config.backbone_config.patch_size)."""
        vc = getattr(model_config, "vision_config", None) if model_config else None
        patch_size = getattr(getattr(vc, "backbone_config", None), "patch_size", None)
        if patch_size is not None:
            return patch_size
        raise ValueError(
            "patch_size not found. Load model from config.json "
            "(vision_config.backbone_config.patch_size)."
        )

    @classmethod
    def _get_max_text_len_from_config(
        cls, model_config: Optional["PretrainedConfig"], rbln_config: RBLNSam3ModelConfig
    ) -> int:
        """Get max_text_len from model_config.text_config or rbln_config."""
        if model_config is not None and hasattr(model_config, "text_config"):
            max_len = getattr(model_config.text_config, "max_position_embeddings", None)
            if max_len is not None:
                return max_len
        if rbln_config.max_text_len is not None:
            return rbln_config.max_text_len
        raise ValueError(
            "max_text_len could not be resolved. Provide rbln_max_text_len or load a model "
            "with config.json (text_config.max_position_embeddings)."
        )

    def _pad_image_to_compiled_size(self, pixel_values: torch.FloatTensor) -> torch.FloatTensor:
        """Pad pixel_values to compiled (image_height, image_width)."""
        target_h = self.rbln_config.image_height
        target_w = self.rbln_config.image_width
        _, _, orig_h, orig_w = pixel_values.shape

        if orig_h > target_h or orig_w > target_w:
            raise ValueError(
                f"Input image size ({orig_h}x{orig_w}) exceeds the compiled size ({target_h}x{target_w}). "
                "Use images up to the compiled size or recompile with a larger rbln_image_size."
            )
        if orig_h == target_h and orig_w == target_w:
            return pixel_values

        pad_h = target_h - orig_h
        pad_w = target_w - orig_w
        return F.pad(pixel_values, (0, pad_w, 0, pad_h), mode="constant", value=0)

    def _get_compiled_max_text_len(self) -> int:
        """Return max_text_len from compile input_info or rbln_config."""
        if self.rbln_config.compile_cfgs:
            for name, shape, _ in self.rbln_config.compile_cfgs[0].input_info:
                if name == "input_ids" and len(shape) >= 2:
                    return shape[1]
        if self.rbln_config.max_text_len is not None:
            return self.rbln_config.max_text_len
        raise ValueError("max_text_len not set. Ensure _update_rbln_config runs before compile.")

    def _pad_text_to_compiled_size(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.LongTensor, torch.Tensor]:
        """Pad or truncate input_ids and attention_mask to compiled max_text_len."""
        max_text_len = self._get_compiled_max_text_len()
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        if attention_mask is None:
            attention_mask = torch.ones(batch_size, seq_len, dtype=torch.int64, device=device)

        if seq_len > max_text_len:
            input_ids = input_ids[:, :max_text_len].contiguous()
            attention_mask = attention_mask[:, :max_text_len].contiguous()
        elif seq_len < max_text_len:
            pad_len = max_text_len - seq_len
            input_ids = F.pad(input_ids, (0, pad_len), value=0)
            attention_mask = F.pad(attention_mask, (0, pad_len), value=0)

        return input_ids, attention_mask
