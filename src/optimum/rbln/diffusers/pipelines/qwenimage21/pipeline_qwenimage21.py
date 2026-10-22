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

from collections.abc import Mapping
from os import PathLike
from typing import TYPE_CHECKING

import torch
from diffusers import QwenImage21Pipeline
from PIL import Image as PILImage

from ....configuration_utils import RBLNModelConfig
from ....modeling import RBLNModel
from ...configurations.pipelines.configuration_qwenimage21 import RBLNQwenImage21PipelineConfig
from ...modeling_diffusers import RBLNDiffusionMixin


if TYPE_CHECKING:
    from diffusers import DiffusionPipeline


class RBLNQwenImage21Pipeline(RBLNDiffusionMixin, QwenImage21Pipeline):
    """
    RBLN-accelerated implementation of the Qwen-Image 2.1 pipeline for text-to-image generation and image editing.

    The pipeline compiles the Qwen3-VL text encoder, the block-causal transformer and the VAE for the image sizes in
    its configuration. It also compiles for image editing, for a 1024x1024 condition image unless
    `condition_image_size` lists other sizes, so one compiled pipeline serves both tasks.

    Examples:
        ```python
        import torch
        from optimum.rbln import RBLNQwenImage21Pipeline

        pipe = RBLNQwenImage21Pipeline.from_pretrained(
            "Qwen/Qwen-Image-2.1-Turbo",
            export=True,
            torch_dtype=torch.bfloat16,
            rbln_config={
                "image_size": (1024, 1024),
                "text_encoder": {"num_devices": 4, "device": [0, 1, 2, 3]},
                "transformer": {"num_devices": 4, "device": [4, 5, 6, 7]},
                "vae": {"device": 8},
            },
        )
        image = pipe("A boat on a calm lake at sunrise").images[0]
        ```
    """

    original_class = QwenImage21Pipeline
    _rbln_config_class = RBLNQwenImage21PipelineConfig
    _submodules = ["text_encoder", "transformer", "vae"]

    @classmethod
    def _compile_submodules(
        cls,
        model: "DiffusionPipeline",
        passed_submodules: Mapping[str, RBLNModel | None],
        model_save_dir: str | PathLike[str] | None,
        rbln_config: RBLNModelConfig,
        prefix: str = "",
    ) -> dict[str, RBLNModel]:
        # The transformer reads the text encoder's last decoder layer before the final norm. Upstream hooks the norm
        # out for each call; a compiled graph takes no hooks, so the hook is in place while the text encoder compiles.
        text_encoder = passed_submodules.get("text_encoder") or model.text_encoder
        handle = None
        if not isinstance(text_encoder, RBLNModel):
            text_model = getattr(text_encoder.model, "language_model", text_encoder.model)
            handle = text_model.norm.register_forward_hook(lambda module, args, output: args[0])
        try:
            return super()._compile_submodules(model, passed_submodules, model_save_dir, rbln_config, prefix)
        finally:
            if handle is not None:
                handle.remove()

    def _get_qwen_prompt_embeds(
        self,
        prompt: str | list[str] = None,
        image: list | None = None,
        device: torch.device | None = None,
    ):
        # Same as upstream, without the forward hook on the final norm: the compiled text encoder already skips it.
        # TODO: drop this override and the hook in `_compile_submodules` once upstream drops its hook for
        # `tie_last_hidden_states=False` (huggingface/transformers#48087).
        device = device or self._execution_device
        prompt = [prompt] if isinstance(prompt, str) else prompt
        prompt = [" " if not p else p for p in prompt]
        is_t2i = image is None

        if is_t2i:
            prompts = [self.prompt_template_t2i.format(t) for t in prompt]
        else:
            prompts = []
            condition_pil_list = []
            for t in prompt:
                n_imgs = len(image)
                replace = "<image1><|vision_start|><|image_pad|><|vision_end|>"
                for i in range(2, n_imgs + 1):
                    replace += f" <image{i}><|vision_start|><|image_pad|><|vision_end|>"
                template = self.prompt_template_ti2i.replace(
                    "<image1><|vision_start|><|image_pad|><|vision_end|>", replace
                )
                prompts.append(template.format(t))
            for _ in prompt:
                for img in image:
                    if not isinstance(img, PILImage.Image):
                        img = PILImage.fromarray(img)
                    if img.mode == "RGBA":
                        white = PILImage.new("RGB", img.size, (255, 255, 255))
                        white.paste(img, mask=img.getchannel("A"))
                        img = white
                    condition_pil_list.append(img)

        processor_kwargs = {
            "text": prompts,
            "padding": True,
            "padding_side": "left",
            "return_tensors": "pt",
        }
        if not is_t2i:
            processor_kwargs["images"] = condition_pil_list

        model_inputs = self.processor(**processor_kwargs).to(device)

        forward_kwargs = {
            "input_ids": model_inputs.input_ids,
            "attention_mask": model_inputs.attention_mask,
            "output_hidden_states": True,
        }
        if not is_t2i and hasattr(model_inputs, "pixel_values"):
            forward_kwargs.update(pixel_values=model_inputs.pixel_values, image_grid_thw=model_inputs.image_grid_thw)
        if hasattr(model_inputs, "mm_token_type_ids"):
            forward_kwargs["mm_token_type_ids"] = model_inputs.mm_token_type_ids

        outputs = self.text_encoder(**forward_kwargs)
        hidden_states = outputs.hidden_states[-1]

        split_hidden_states = list(self._extract_masked_hidden(hidden_states, model_inputs.attention_mask))
        split_hidden_states = [e[self._drop_idx :] for e in split_hidden_states]

        image_pad_mask = [
            (sample_ids[sample_mask.bool()] == self._img_token_id)
            for sample_ids, sample_mask in zip(model_inputs.input_ids, model_inputs.attention_mask, strict=True)
        ]
        image_pad_mask = [e[self._drop_idx :] for e in image_pad_mask]

        attn_mask_list = [torch.ones(e.size(0), dtype=torch.long, device=e.device) for e in split_hidden_states]
        max_seq_len = max(e.size(0) for e in split_hidden_states)
        prompt_embeds = torch.stack(
            [torch.cat([u, u.new_zeros(max_seq_len - u.size(0), u.size(1))]) for u in split_hidden_states]
        )
        encoder_attention_mask = torch.stack(
            [torch.cat([u, u.new_zeros(max_seq_len - u.size(0))]) for u in attn_mask_list]
        )
        image_pad_mask = torch.stack([torch.cat([u, u.new_zeros(max_seq_len - u.size(0))]) for u in image_pad_mask])

        return prompt_embeds, encoder_attention_mask, image_pad_mask


__all__ = [
    "RBLNQwenImage21Pipeline",
]
