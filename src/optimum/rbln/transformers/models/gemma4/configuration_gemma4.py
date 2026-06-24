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

from typing import Any, List, Optional, Union

from ....configuration_utils import RBLNModelConfig
from ....utils.logging import get_logger
from ..decoderonly.configuration_decoderonly import RBLNDecoderOnlyModelForCausalLMConfig


logger = get_logger(__name__)


DEFAULT_MAX_SOFT_TOKENS = [70, 140, 280, 560, 1120]


def ceil_to_multiple_of_128(value: int) -> int:
    # Smallest multiple of 128 greater than or equal to value.
    return ((int(value) + 127) // 128) * 128


class RBLNGemma4ForCausalLMConfig(RBLNDecoderOnlyModelForCausalLMConfig):
    # Both use_position_ids and use_attention_mask are forced to True for Gemma4.
    # The text decoder uses two RoPE flavors (proportional for full-attention layers and
    # default for sliding-attention layers) with potentially different head_dim values.

    def __init__(
        self,
        use_position_ids: Optional[bool] = None,
        use_attention_mask: Optional[bool] = None,
        prefill_chunk_size: Optional[int] = None,
        image_prefill_chunk_size: Optional[int] = None,
        image_prefill_chunk_sizes: Optional[Union[int, List[int]]] = None,
        **kwargs: Any,
    ):
        """
        Args:
            use_position_ids (Optional[bool]): Whether to use `position_ids`. Forced to `True` for Gemma4.
            use_attention_mask (Optional[bool]): Whether to use `attention_mask`. Forced to `True` for Gemma4.
            prefill_chunk_size (Optional[int]): Chunk size used during the prefill phase. Defaults to 128.
            image_prefill_chunk_size (Optional[int]): Single chunk size used for image-prefill (multimodal
                Gemma4). Mutually exclusive with `image_prefill_chunk_sizes`. When neither is given, the
                image-prefill buckets are decided later in `_update_rbln_config`.
            image_prefill_chunk_sizes (Optional[Union[int, List[int]]]): Desired bucketing shapes for the
                image-prefill phase. When given, a separate `image_prefill` graph is compiled for every
                value (sorted in descending order), allowing the runtime to pick the smallest bucket that
                fits an image run. Mutually exclusive with `image_prefill_chunk_size`; when this is set,
                `image_prefill_chunk_size` is derived as `max(image_prefill_chunk_sizes)`.
            kwargs: Additional arguments passed to the parent `RBLNDecoderOnlyModelForCausalLMConfig`.

        Raises:
            ValueError: If `use_attention_mask` or `use_position_ids` are False, if both
                `image_prefill_chunk_size` and `image_prefill_chunk_sizes` are specified, or if any
                image-prefill chunk size is not a positive integer divisible by 128.
        """
        if use_attention_mask is None:
            use_attention_mask = True
        if use_position_ids is None:
            use_position_ids = True

        if prefill_chunk_size is None:
            prefill_chunk_size = 128

        if image_prefill_chunk_size is not None and image_prefill_chunk_sizes is not None:
            raise ValueError(
                "Specify only one of `image_prefill_chunk_size` or `image_prefill_chunk_sizes`, not both."
            )

        if image_prefill_chunk_sizes is None and image_prefill_chunk_size is not None:
            image_prefill_chunk_sizes = image_prefill_chunk_size
        if image_prefill_chunk_sizes is not None:
            image_prefill_chunk_sizes = self._normalize_image_prefill_chunk_sizes(image_prefill_chunk_sizes)

        super().__init__(
            prefill_chunk_size=prefill_chunk_size,
            use_attention_mask=use_attention_mask,
            use_position_ids=use_position_ids,
            **kwargs,
        )
        self.image_prefill_chunk_sizes = image_prefill_chunk_sizes

        if not (self.use_attention_mask and self.use_position_ids):
            raise ValueError("use_attention_mask and use_position_ids must be True for RBLNGemma4ForCausalLM")

    @property
    def image_prefill_chunk_size(self) -> Optional[int]:
        # Largest image-prefill bucket, derived from image_prefill_chunk_sizes.
        # Read-only derived value (not stored) so it is never persisted alongside image_prefill_chunk_sizes;
        # a reloaded config carrying both would trip the "specify only one" guard in __init__.
        if not self.image_prefill_chunk_sizes:
            return None
        return max(self.image_prefill_chunk_sizes)

    @staticmethod
    def _normalize_image_prefill_chunk_sizes(chunk_sizes: Union[int, List[int]]) -> List[int]:
        # Single enforcement point: accepts int or list, returns a de-duplicated descending list.
        # Each chunk must be a positive multiple of 128 (stricter than text prefill_chunk_size's
        # 64-alignment, and independent of it).
        if isinstance(chunk_sizes, int):
            chunk_sizes = [chunk_sizes]
        chunk_sizes = sorted(set(chunk_sizes), reverse=True)
        for chunk_size in chunk_sizes:
            if chunk_size <= 0 or chunk_size % 128 != 0:
                raise ValueError(
                    "Every image-prefill chunk size must be a positive integer divisible by 128, "
                    f"but got image_prefill_chunk_sizes={chunk_sizes}."
                )
        return chunk_sizes

    @property
    def num_image_prefill_buckets(self) -> int:
        # Number of separate image_prefill_{chunk} graphs (0 when image-prefill is disabled).
        if not self.use_image_prefill or not self.image_prefill_chunk_sizes:
            return 0
        return len(self.image_prefill_chunk_sizes)

    @property
    def expected_compiled_model_names(self):
        names = ["prefill"]
        if self.use_image_prefill and self.image_prefill_chunk_sizes:
            names += [f"image_prefill_{chunk_size}" for chunk_size in self.image_prefill_chunk_sizes]
        if self.can_generate:
            names += [f"decoder_batch_{batch_size}" for batch_size in self.decoder_batch_sizes]
        return names

    @property
    def decoder_runtime_idx(self):
        if not self.can_generate:
            raise ValueError("`decode` phase is not in the phases.")
        return 1 + self.num_image_prefill_buckets


class RBLNGemma4VisionModelConfig(RBLNModelConfig):
    # The vision encoder takes pre-patched pixel_values of shape (batch, max_patches, 3*patch_size**2)
    # and pixel_position_ids of shape (batch, max_patches, 2).
    # max_patches is derived from max_soft_tokens and pooling_kernel_size via
    # max_patches = max_soft_tokens * pooling_kernel_size**2.

    def __init__(
        self,
        batch_size: Optional[int] = None,
        max_soft_tokens: Optional[int] = None,
        pooling_kernel_size: Optional[int] = None,
        patch_size: Optional[int] = None,
        output_hidden_states: Optional[bool] = None,
        **kwargs: Any,
    ):
        """
        Args:
            batch_size (Optional[int]): The batch size of images (number of images, not patches). Defaults to 1.
            max_soft_tokens (Optional[int]): The number of soft tokens emitted per image after pooling.
                Defaults to 280 (the upstream default in `Gemma4ImageProcessor`). Must be a value
                supported by the image processor (e.g. 70/140/280/560/1120).
            pooling_kernel_size (Optional[int]): Spatial pooling kernel size applied after patchification.
                Defaults to `model_config.pooling_kernel_size` (3 by default).
            patch_size (Optional[int]): Patch height/width in pixels. Defaults to `model_config.patch_size`.
            output_hidden_states (Optional[bool]): Whether to return per-layer hidden states.
            kwargs: Additional arguments passed to the parent `RBLNModelConfig`.

        Raises:
            ValueError: If `batch_size` is not a positive integer.
        """
        super().__init__(**kwargs)
        self.batch_size = batch_size or 1
        if not isinstance(self.batch_size, int) or self.batch_size <= 0:
            raise ValueError(f"batch_size must be a positive integer, got {self.batch_size}")

        if max_soft_tokens is not None:
            if isinstance(max_soft_tokens, int):
                max_soft_tokens = [max_soft_tokens]
            elif isinstance(max_soft_tokens, list):
                max_soft_tokens.sort(reverse=True)

        self.max_soft_tokens = max_soft_tokens
        self.pooling_kernel_size = pooling_kernel_size
        self.patch_size = patch_size
        self.output_hidden_states = output_hidden_states or False

    def get_max_patches(self) -> int:
        if self.max_soft_tokens is None or self.pooling_kernel_size is None:
            raise ValueError(
                "`max_patches` cannot be computed until both `max_soft_tokens` and `pooling_kernel_size` are set."
            )
        return [
            max_soft_tokens * self.pooling_kernel_size * self.pooling_kernel_size
            for max_soft_tokens in self.max_soft_tokens
        ]


class RBLNGemma4ForConditionalGenerationConfig(RBLNModelConfig):
    # Holds nested configs for the vision encoder and the language model. The vision tower
    # is always compiled with batch_size=1 and looped over samples at runtime.

    submodules = ["vision_tower", "language_model"]

    def __init__(
        self,
        batch_size: Optional[int] = None,
        vision_tower: Optional[RBLNModelConfig] = None,
        language_model: Optional[RBLNModelConfig] = None,
        **kwargs: Any,
    ):
        """
        Args:
            batch_size (Optional[int]): The batch size for inference. Defaults to 1.
            vision_tower (Optional[RBLNModelConfig]): Configuration for the vision encoder component.
            language_model (Optional[RBLNModelConfig]): Configuration for the language model component.
            kwargs: Additional arguments passed to the parent `RBLNModelConfig`.

        Raises:
            ValueError: If `batch_size` is not a positive integer.
        """
        super().__init__(**kwargs)
        self.batch_size = batch_size or 1
        if not isinstance(self.batch_size, int) or self.batch_size <= 0:
            raise ValueError(f"batch_size must be a positive integer, got {self.batch_size}")

        if self.batch_size != 1:
            logger.warning("Ignore batch_size for Gemma4 vision tower. It will be set to 1.")

        self.vision_tower = self.initialize_submodule_config(
            submodule_config=vision_tower, batch_size=1, force_kwargs=True
        )
        self.language_model = self.initialize_submodule_config(
            submodule_config=language_model,
            batch_size=self.batch_size,
            force_kwargs=True,
            use_inputs_embeds=True,
        )

        self._update_image_prefill_chunk_sizes()

    def _get_vision_max_soft_tokens(self) -> List[int]:
        # Per-image soft-token counts the vision tower emits, sorted in descending order.
        # Reads max_soft_tokens from the vision sub-config (may still be a raw dict at this point)
        # and falls back to DEFAULT_MAX_SOFT_TOKENS when unset, matching the default applied in
        # RBLNGemma4VisionModel._update_rbln_config.
        vt_cfg = self.vision_tower
        if isinstance(vt_cfg, dict):
            max_soft_tokens = vt_cfg.get("max_soft_tokens", None)
        else:
            max_soft_tokens = getattr(vt_cfg, "max_soft_tokens", None)
        if max_soft_tokens is None:
            max_soft_tokens = DEFAULT_MAX_SOFT_TOKENS
        if isinstance(max_soft_tokens, int):
            max_soft_tokens = [max_soft_tokens]
        return sorted(max_soft_tokens, reverse=True)

    def _update_image_prefill_chunk_sizes(self) -> None:
        # Derives or validates the language model's image-prefill buckets against the vision tower's
        # soft-token counts.
        # When neither image_prefill_chunk_size nor image_prefill_chunk_sizes is pinned: map each
        # max_soft_tokens to the smallest multiple of 128 that holds it, then keep only the largest
        # (n + 1) // 2 buckets — dropped smallest-first so the largest count keeps its own bucket and
        # smaller images route to the smallest kept bucket >= their length at runtime.
        # When the user did pin the buckets: verify the largest can hold the vision tower's largest
        # max_soft_tokens; otherwise some image run would have no bucket (detected here, not at runtime).
        # Note: these buckets are independent of the language model's text prefill_chunk_size.
        lm_cfg = self.language_model
        soft_tokens = self._get_vision_max_soft_tokens()
        max_soft_token = max(soft_tokens)

        if isinstance(lm_cfg, dict):
            pinned = lm_cfg.get("image_prefill_chunk_sizes", lm_cfg.get("image_prefill_chunk_size", None))
        else:
            pinned = getattr(lm_cfg, "image_prefill_chunk_sizes", None)

        if pinned is not None:
            pinned_buckets = RBLNGemma4ForCausalLMConfig._normalize_image_prefill_chunk_sizes(pinned)
            if max(pinned_buckets) < max_soft_token:
                raise ValueError(
                    f"The largest image-prefill chunk size ({max(pinned_buckets)}) is smaller than the "
                    f"vision tower's largest max_soft_tokens ({max_soft_token}); some images would have no "
                    f"bucket able to hold them. Provide an image-prefill chunk size of at least "
                    f"{ceil_to_multiple_of_128(max_soft_token)}."
                )
            return

        buckets = RBLNGemma4ForCausalLMConfig._normalize_image_prefill_chunk_sizes(
            [ceil_to_multiple_of_128(t) for t in soft_tokens]
        )
        # Keep only the largest `(n + 1) // 2` buckets (descending list -> drop smallest first).
        num_keep = (len(soft_tokens) + 1) // 2
        buckets = buckets[:num_keep]

        if isinstance(lm_cfg, dict):
            lm_cfg["image_prefill_chunk_sizes"] = buckets
        else:
            lm_cfg.image_prefill_chunk_sizes = buckets

    @property
    def image_prefill_chunk_size(self):
        return self.language_model.image_prefill_chunk_size

    @property
    def image_prefill_chunk_sizes(self):
        return self.language_model.image_prefill_chunk_sizes

    @property
    def prefill_chunk_size(self):
        return self.language_model.prefill_chunk_size
