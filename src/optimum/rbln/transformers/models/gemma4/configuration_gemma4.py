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
from typing import Any, Optional

from ....configuration_utils import RBLNModelConfig
from ....utils.logging import get_logger
from ..decoderonly.configuration_decoderonly import RBLNDecoderOnlyModelForCausalLMConfig


logger = get_logger(__name__)


class RBLNGemma4ForCausalLMConfig(RBLNDecoderOnlyModelForCausalLMConfig):
    """
    Configuration class for `RBLNGemma4ForCausalLM`.

    This configuration class stores the configuration parameters specific to
    RBLN-optimized Gemma4 causal language model (text decoder).

    Both `use_position_ids` and `use_attention_mask` are forced to `True` for Gemma4.
    The text decoder uses two RoPE flavors (`proportional` for full-attention layers and `default`
    for sliding-attention layers) with potentially different `head_dim` values.
    """

    def __init__(
        self,
        use_position_ids: Optional[bool] = None,
        use_attention_mask: Optional[bool] = None,
        prefill_chunk_size: Optional[int] = None,
        image_prefill_chunk_size: Optional[int] = None,
        **kwargs: Any,
    ):
        """
        Args:
            use_position_ids (Optional[bool]): Whether to use `position_ids`. Forced to `True` for Gemma4.
            use_attention_mask (Optional[bool]): Whether to use `attention_mask`. Forced to `True` for Gemma4.
            prefill_chunk_size (Optional[int]): Chunk size used during the prefill phase. Defaults to 256.
            image_prefill_chunk_size (Optional[int]): Chunk size used for image-prefill (multimodal Gemma4).
                Currently must equal `prefill_chunk_size`.
            kwargs: Additional arguments passed to the parent `RBLNDecoderOnlyModelForCausalLMConfig`.

        Raises:
            ValueError: If `use_attention_mask` or `use_position_ids` are False.
        """
        if use_attention_mask is None:
            use_attention_mask = True
        if use_position_ids is None:
            use_position_ids = True
        prefill_chunk_size = prefill_chunk_size or 256

        super().__init__(
            prefill_chunk_size=prefill_chunk_size,
            use_attention_mask=use_attention_mask,
            use_position_ids=use_position_ids,
            **kwargs,
        )
        self.image_prefill_chunk_size = image_prefill_chunk_size

        if not (self.use_attention_mask and self.use_position_ids):
            raise ValueError("use_attention_mask and use_position_ids must be True for RBLNGemma4ForCausalLM")


class RBLNGemma4VisionModelConfig(RBLNModelConfig):
    """
    Configuration class for `RBLNGemma4VisionModel`.

    This configuration class stores the configuration parameters specific to
    RBLN-optimized Gemma4 vision encoder models.

    The vision encoder takes pre-patched `pixel_values` of shape
    `(batch, max_patches, 3 * patch_size**2)` and `pixel_position_ids` of shape
    `(batch, max_patches, 2)`. `max_patches` is derived from `max_soft_tokens` and
    `pooling_kernel_size` via `max_patches = max_soft_tokens * pooling_kernel_size**2`.
    """

    def __init__(
        self,
        batch_size: Optional[int] = None,
        max_soft_tokens: Optional[int] = None,
        pooling_kernel_size: Optional[int] = None,
        patch_size: Optional[int] = None,
        output_hidden_states: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
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
            output_attentions (Optional[bool]): Whether to return per-layer attention weights.
            kwargs: Additional arguments passed to the parent `RBLNModelConfig`.

        Raises:
            ValueError: If `batch_size` is not a positive integer.
        """
        super().__init__(**kwargs)
        self.batch_size = batch_size or 1
        if not isinstance(self.batch_size, int) or self.batch_size <= 0:
            raise ValueError(f"batch_size must be a positive integer, got {self.batch_size}")

        self.max_soft_tokens = max_soft_tokens
        self.pooling_kernel_size = pooling_kernel_size
        self.patch_size = patch_size
        self.output_hidden_states = output_hidden_states or False
        self.output_attentions = output_attentions or False

    @property
    def max_patches(self) -> int:
        if self.max_soft_tokens is None or self.pooling_kernel_size is None:
            raise ValueError(
                "`max_patches` cannot be computed until both `max_soft_tokens` and `pooling_kernel_size` are set."
            )
        return self.max_soft_tokens * self.pooling_kernel_size * self.pooling_kernel_size


class RBLNGemma4ForConditionalGenerationConfig(RBLNModelConfig):
    """
    Configuration class for `RBLNGemma4ForConditionalGeneration`.

    This configuration class stores the configuration parameters specific to
    RBLN-optimized multimodal Gemma4 (text + vision) models.

    Holds nested configs for the vision encoder and the language model. The vision tower
    is always compiled with `batch_size=1` and looped over samples at runtime.
    """

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
            submodule_config=vision_tower, batch_size=batch_size, force_kwargs=True
        )
        self.language_model = self.initialize_submodule_config(
            submodule_config=language_model,
            batch_size=batch_size,
            force_kwargs=True,
            use_inputs_embeds=True,
        )

    @property
    def image_prefill_chunk_size(self):
        return self.language_model.image_prefill_chunk_size

    @property
    def prefill_chunk_size(self):
        return self.language_model.prefill_chunk_size
