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

import itertools
from typing import Any

from ....configuration_utils import RBLNModelConfig
from ....transformers import RBLNQwen3VLForConditionalGenerationConfig
from ....transformers.modeling_attention_utils import align
from ..models import RBLNAutoencoderKLQwenImage21Config, RBLNQwenImage21Transformer2DModelConfig


# One latent token covers a 16x16 pixel tile.
_LATENT_TILE = 16
# The pipeline rounds image sizes down to multiples of 32.
_SIZE_MULTIPLE = 32


class RBLNQwenImage21PipelineConfig(RBLNModelConfig):
    """Config for Qwen-Image 2.1 Pipeline"""

    submodules = ["text_encoder", "transformer", "vae"]

    def __init__(
        self,
        text_encoder: RBLNQwen3VLForConditionalGenerationConfig | None = None,
        transformer: RBLNQwenImage21Transformer2DModelConfig | None = None,
        vae: RBLNAutoencoderKLQwenImage21Config | None = None,
        *,
        batch_size: int | None = None,
        image_size: tuple[int, int] | list[tuple[int, int]] | None = None,
        condition_image_size: tuple[int, int] | list[tuple[int, int]] | None = None,
        num_condition_images: int | None = None,
        output_resolution: int | None = None,
        max_seq_len: int | None = None,
        **kwargs: Any,
    ):
        """
        Args:
            text_encoder (RBLNQwen3VLForConditionalGenerationConfig | None): Configuration for the text encoder.
                Initialized as RBLNQwen3VLForConditionalGenerationConfig if not provided.
            transformer (RBLNQwenImage21Transformer2DModelConfig | None): Configuration for the transformer.
                Initialized as RBLNQwenImage21Transformer2DModelConfig if not provided.
            vae (RBLNAutoencoderKLQwenImage21Config | None): Configuration for the VAE.
                Initialized as RBLNAutoencoderKLQwenImage21Config if not provided.
            batch_size (int | None): Batch size for inference, applied to all submodules.
            image_size (tuple[int, int] | list[tuple[int, int]] | None): The (height, width) of the generated image,
                in pixels, or a list of them. Defaults to (1024, 1024).
            condition_image_size (tuple[int, int] | list[tuple[int, int]] | None): The (height, width) of the
                VAE encoder canvases for image editing, in pixels, or a list of them. A condition image, after the
                pipeline resizes it, runs in the smallest canvas that holds it. Each canvas also sets token budgets
                for the largest image the pipeline can resize to that fits it, or for the canvas itself if none does.
                Defaults to (1408, 1408), which holds every condition image from 9:16 to 16:9 at the default
                `output_resolution`.
            num_condition_images (int | None): The maximum number of condition images in one call. Defaults to 1.
            output_resolution (int | None): The `output_resolution` the pipeline is called with, which sets the
                size condition images are resized to. Defaults to 1024, the pipeline's default.
            max_seq_len (int | None): The maximum number of text tokens the text encoder reads, without the tokens
                of the condition images. Defaults to 512.
            kwargs: Additional arguments passed to the parent RBLNModelConfig.

        Raises:
            ValueError: If an image size is not a multiple of 32, or if `num_condition_images` is less than 1.
        """
        super().__init__(**kwargs)

        max_seq_len = max_seq_len or 512
        output_resolution = output_resolution or 1024
        if image_size is not None and isinstance(image_size[0], int):
            image_size = [image_size]
        image_size = [(1024, 1024)] if image_size is None else [(size[0], size[1]) for size in image_size]
        if condition_image_size is not None and isinstance(condition_image_size[0], int):
            condition_image_size = [condition_image_size]
        # 1376 would also hold every condition image from 9:16 to 16:9, but the compiler splits the VAE encoder into
        # several device graphs unless the canvas width is a multiple of 128.
        condition_image_size = (
            [(1408, 1408)] if condition_image_size is None else [(size[0], size[1]) for size in condition_image_size]
        )
        if num_condition_images is None:
            num_condition_images = 1
        if num_condition_images < 1:
            raise ValueError(f"num_condition_images must be 1 or more, not {num_condition_images}.")

        not_aligned = [
            size for size in image_size + condition_image_size if size[0] % _SIZE_MULTIPLE or size[1] % _SIZE_MULTIPLE
        ]
        if not_aligned:
            raise ValueError(
                f"Image sizes {not_aligned} are not multiples of {_SIZE_MULTIPLE}. The pipeline rounds the height and "
                f"width down to multiples of {_SIZE_MULTIPLE}, so it never runs these sizes."
            )

        def count_latent_tokens(size: tuple[int, int]) -> int:
            return (size[0] // _LATENT_TILE) * (size[1] // _LATENT_TILE)

        # The largest condition image each canvas takes; smaller ones pad into its token budgets.
        condition_sizes = [
            self._get_condition_sizes(output_resolution, *canvas) or [canvas] for canvas in condition_image_size
        ]
        largest_condition_sizes = [max(sizes, key=count_latent_tokens) for sizes in condition_sizes]

        def is_in_condition_range(size: tuple[int, int]) -> bool:
            # The transformer runs an image block in the smallest image graph that holds it. Condition images already
            # pad into the graph of their canvas from its smallest condition size up, so an output size in that range
            # runs there too instead of compiling its own graph.
            return any(
                min(map(count_latent_tokens, sizes)) <= count_latent_tokens(size) <= count_latent_tokens(largest)
                for sizes, largest in zip(condition_sizes, largest_condition_sizes, strict=True)
            )

        condition_tokens = num_condition_images * max(map(count_latent_tokens, largest_condition_sizes))
        # The vision encoder runs once per condition image, on one patch per latent token, padded to the smallest of
        # its sequence lengths that holds them. The compiler takes sequence lengths of the text and vision encoders
        # only in multiples of 64.
        visual = {"max_seq_len": sorted({align(count_latent_tokens(size), 64) for size in largest_condition_sizes})}

        self.text_encoder = self.initialize_submodule_config(
            text_encoder,
            cls_name="RBLNQwen3VLForConditionalGenerationConfig",
            batch_size=batch_size,
            # The vision encoder merges 2x2 of its 16x16 patches into one image token of the prompt.
            max_seq_len=align(max_seq_len + condition_tokens // 4, 64),
            phases=["prefill"],  # The pipeline reads the hidden states of one forward pass and never generates.
            output_hidden_states=True,
            visual=visual,
            force_kwargs=True,
        )
        self.transformer = self.initialize_submodule_config(
            transformer,
            cls_name="RBLNQwenImage21Transformer2DModelConfig",
            batch_size=batch_size,
            sample_size=[
                (height // _LATENT_TILE, width // _LATENT_TILE)
                for height, width in dict.fromkeys(
                    [size for size in image_size if not is_in_condition_range(size)] + largest_condition_sizes
                )
            ],
            # In the transformer every image token of the prompt becomes the latent tokens it covers.
            max_seq_len=max_seq_len + condition_tokens,
        )
        self.vae = self.initialize_submodule_config(
            vae,
            cls_name="RBLNAutoencoderKLQwenImage21Config",
            batch_size=batch_size,
            sample_size=image_size,
            uses_encoder=True,
            encoder_sample_size=condition_image_size,
        )

    @property
    def batch_size(self):
        return self.vae.batch_size

    @property
    def image_size(self):
        return self.vae.sample_size

    @staticmethod
    def _get_condition_sizes(output_resolution: int, height: int, width: int) -> list[tuple[int, int]]:
        # The sizes `calculate_dimensions` gives condition images that fit a (height, width) canvas. A size only
        # changes where a side crosses a multiple of 32, so one aspect ratio between each pair of boundaries finds all.
        from diffusers.pipelines.qwenimage21.pipeline_qwenimage21 import calculate_dimensions

        area = output_resolution**2
        half = _SIZE_MULTIPLE / 2
        # Aspect ratios (width / height) whose height and width round to at most the canvas.
        low, high = area / (height + half) ** 2, (width + half) ** 2 / area
        bounds = {low, high}
        for side in range(_SIZE_MULTIPLE, max(height, width) + 2 * _SIZE_MULTIPLE, _SIZE_MULTIPLE):
            bounds.update(((side - half) ** 2 / area, area / (side - half) ** 2))
        sizes = set()
        for lower, upper in itertools.pairwise(sorted(bound for bound in bounds if low <= bound <= high)):
            resized_width, resized_height, _ = calculate_dimensions(area, (lower + upper) / 2)
            if 0 < resized_height <= height and 0 < resized_width <= width:
                sizes.add((resized_height, resized_width))
        return sorted(sizes)


__all__ = [
    "RBLNQwenImage21PipelineConfig",
]
