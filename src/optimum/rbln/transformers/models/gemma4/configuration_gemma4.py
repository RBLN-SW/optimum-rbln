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


DEFAULT_MAX_SOFT_TOKENS = 280


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
        image_prefill_chunk_size: Optional[Union[int, List[int]]] = None,
        **kwargs: Any,
    ):
        """
        Args:
            use_position_ids (Optional[bool]): Whether to use `position_ids`. Forced to `True` for Gemma4.
            use_attention_mask (Optional[bool]): Whether to use `attention_mask`. Forced to `True` for Gemma4.
            prefill_chunk_size (Optional[int]): Chunk size used during the prefill phase. Defaults to 128.
            image_prefill_chunk_size (Optional[Union[int, List[int]]]): Chunk size(s) used for image-prefill
                (multimodal Gemma4). A single int compiles one `image_prefill` graph; a list compiles one
                graph per value (sorted in descending order) and the runtime picks the smallest bucket that
                fits an image run. When not given, it is derived from the vision tower's `max_soft_tokens`
                (the default single `max_soft_tokens=280` yields a single chunk size of 384).
            kwargs: Additional arguments passed to the parent `RBLNDecoderOnlyModelForCausalLMConfig`.

        Raises:
            ValueError: If `use_attention_mask` or `use_position_ids` are False, or if any image-prefill
                chunk size is not a positive integer divisible by 128.
        """
        if use_attention_mask is None:
            use_attention_mask = True
        if use_position_ids is None:
            use_position_ids = True

        if prefill_chunk_size is None:
            prefill_chunk_size = 128

        if image_prefill_chunk_size is not None:
            image_prefill_chunk_size = self._validate_image_prefill_chunk_size(image_prefill_chunk_size)

        super().__init__(
            prefill_chunk_size=prefill_chunk_size,
            use_attention_mask=use_attention_mask,
            use_position_ids=use_position_ids,
            **kwargs,
        )
        # Always stored as a de-duplicated descending list of buckets (or None until derived from
        # the vision tower's max_soft_tokens in RBLNGemma4ForConditionalGenerationConfig).
        self.image_prefill_chunk_size = image_prefill_chunk_size

        if not (self.use_attention_mask and self.use_position_ids):
            raise ValueError("use_attention_mask and use_position_ids must be True for RBLNGemma4ForCausalLM")

    @staticmethod
    def _validate_image_prefill_chunk_size(chunk_size: Union[int, List[int]]) -> List[int]:
        # Single enforcement point: validates that every image_prefill_chunk_size (int or list) is a
        # positive multiple of 128 and returns it in canonical form — de-duplicated and sorted descending.
        if isinstance(chunk_size, int):
            chunk_size = [chunk_size]
        chunk_size = sorted(set(chunk_size), reverse=True)
        for size in chunk_size:
            if size <= 0 or size % 128 != 0:
                raise ValueError(
                    "Every image-prefill chunk size must be a positive integer divisible by 128, "
                    f"but got image_prefill_chunk_size={chunk_size}."
                )
        return chunk_size

    @property
    def num_image_prefill_buckets(self) -> int:
        # Number of separate image_prefill_{chunk} graphs (0 when image-prefill is disabled).
        if not self.use_image_prefill or not self.image_prefill_chunk_size:
            return 0
        return len(self.image_prefill_chunk_size)

    @property
    def expected_compiled_model_names(self):
        names = ["prefill"]
        if self.use_image_prefill and self.image_prefill_chunk_size:
            names += [f"image_prefill_{chunk_size}" for chunk_size in self.image_prefill_chunk_size]
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
        max_soft_tokens: Optional[Union[int, List[int]]] = None,
        pooling_kernel_size: Optional[int] = None,
        patch_size: Optional[int] = None,
        output_hidden_states: Optional[bool] = None,
        **kwargs: Any,
    ):
        """
        Args:
            batch_size (Optional[int]): The batch size of images (number of images, not patches). Defaults to 1.
            max_soft_tokens (Optional[Union[int, List[int]]]): The number of soft tokens emitted per image
                after pooling. Defaults to 280 (the upstream default in `Gemma4ImageProcessor`). A single int
                compiles one vision graph; a list compiles one graph per value (sorted descending) so the
                runtime can serve images at multiple soft-token counts. Must be a value supported by the image
                processor (e.g. 70/140/280/560/1120).
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

        if max_soft_tokens is None:
            max_soft_tokens = DEFAULT_MAX_SOFT_TOKENS
        if isinstance(max_soft_tokens, int):
            max_soft_tokens = [max_soft_tokens]
        else:
            max_soft_tokens = sorted(max_soft_tokens, reverse=True)

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

        self._update_image_prefill_chunk_size()

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

    def _update_image_prefill_chunk_size(self) -> None:
        # Resolves the language model's image-prefill chunk size(s) against the vision tower's
        # max_soft_tokens.
        # When image_prefill_chunk_size is not pinned: derive it from max_soft_tokens by mapping each
        # count to the smallest multiple of 128 that holds it — one bucket per distinct soft-token count.
        # With the default single max_soft_tokens (280) this derives a single chunk size of 384.
        # When the user did pin it: use it as-is.
        # Either way the resolved buckets are validated against max_soft_tokens below (asserted here, not
        # at runtime); these buckets are independent of the text prefill_chunk_size.
        lm_cfg = self.language_model
        # max_soft_tokens follows HF's single-int convention but is normalized to a descending list here
        # to support multiple vision buckets; min/max are computed inline where needed.
        max_soft_tokens = self._get_vision_max_soft_tokens()

        if isinstance(lm_cfg, dict):
            pinned = lm_cfg.get("image_prefill_chunk_size", None)
        else:
            pinned = getattr(lm_cfg, "image_prefill_chunk_size", None)

        if pinned is not None:
            buckets = RBLNGemma4ForCausalLMConfig._validate_image_prefill_chunk_size(pinned)
        else:
            buckets = RBLNGemma4ForCausalLMConfig._validate_image_prefill_chunk_size(
                [ceil_to_multiple_of_128(t) for t in max_soft_tokens]
            )
            if isinstance(lm_cfg, dict):
                lm_cfg["image_prefill_chunk_size"] = buckets
            else:
                lm_cfg.image_prefill_chunk_size = buckets

        # (1) The largest image-prefill bucket must hold the largest max_soft_tokens; otherwise those
        # images cannot be image-prefilled by any bucket.
        if max(buckets) < max(max_soft_tokens):
            raise ValueError(
                f"The largest image-prefill chunk size ({max(buckets)}) is smaller than the vision "
                f"tower's largest max_soft_tokens ({max(max_soft_tokens)}); those images cannot be "
                f"image-prefilled by any bucket. Provide an image-prefill chunk size of at least "
                f"{ceil_to_multiple_of_128(max(max_soft_tokens))}."
            )

        # (2) Every compiled bucket must serve at least one max_soft_tokens count. At runtime an image
        # of t soft tokens is dispatched to the smallest bucket >= t; buckets that no count routes to
        # (too small to fit any, or shadowed by a smaller bucket that already covers every count) are
        # compiled but never selected. `max(buckets) >= max(max_soft_tokens)` (checked above) guarantees
        # every count has at least one fitting bucket here.
        used_buckets = {min(b for b in buckets if b >= t) for t in max_soft_tokens}
        unused_buckets = [b for b in buckets if b not in used_buckets]
        if unused_buckets:
            raise ValueError(
                f"Image-prefill chunk sizes {unused_buckets} are never used at runtime: no vision tower "
                f"max_soft_tokens ({max_soft_tokens}) routes to them (each image is dispatched to the "
                f"smallest image-prefill chunk size >= its soft-token count). Remove them."
            )

    @property
    def image_prefill_chunk_size(self):
        return self.language_model.image_prefill_chunk_size

    @property
    def prefill_chunk_size(self):
        return self.language_model.prefill_chunk_size
