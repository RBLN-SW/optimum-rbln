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
from typing import Any, Optional

from ....configuration_utils import RBLNModelConfig
from ....utils.logging import get_logger
from ..decoderonly.configuration_decoderonly import RBLNDecoderOnlyModelForCausalLMConfig


logger = get_logger(__name__)


class RBLNGemma3ForCausalLMConfig(RBLNDecoderOnlyModelForCausalLMConfig):
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
            use_position_ids (Optional[bool]): Whether or not to use `position_ids`, which is indices of positions of each input sequence tokens in the position embeddings.
            use_attention_mask (Optional[bool]): Whether or not to use `attention_mask` to to avoid performing attention on padding token indices.
            prefill_chunk_size (Optional[int]): The chunk size used during the prefill phase for
                processing input sequences. Defaults to 256. Must be a positive integer
                divisible by 64. Affects prefill performance and memory usage.
            image_prefill_chunk_size (Optional[int]): The chunk size used during the prefill phase for
                processing images. This config is used when `use_image_prefill` is True.
                Currently, the `prefill_chunk_size` and `image_prefill_chunk_size` should be the same value.
            kwargs: Additional arguments passed to the parent `RBLNDecoderOnlyModelForCausalLMConfig`.

        Raises:
            ValueError: If `use_attention_mask` or `use_position_ids` are False.
        """
        # use_attention_mask and use_position_ids are always True for Gemma3
        use_attention_mask = use_attention_mask or True
        use_position_ids = use_position_ids or True
        prefill_chunk_size = prefill_chunk_size or 256
        phases = kwargs.pop("phases", None)
        if phases is None:
            if image_prefill_chunk_size is not None:
                phases = ["prefill", "image_prefill", "decode"]
            else:
                phases = ["prefill", "decode"]

        super().__init__(
            prefill_chunk_size=prefill_chunk_size,
            use_attention_mask=use_attention_mask,
            use_position_ids=use_position_ids,
            phases=phases,
            **kwargs,
        )
        self.image_prefill_chunk_size = image_prefill_chunk_size

        if not (self.use_attention_mask and self.use_position_ids):
            raise ValueError("use_attention_mask and use_position_ids must be True for RBLNGemma3ForCausalLM")


class RBLNGemma3ForConditionalGenerationConfig(RBLNModelConfig):
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
            kwargs: Additional arguments passed to the parent RBLNModelConfig.

        Raises:
            ValueError: If `batch_size` is not a positive integer.
        """
        super().__init__(**kwargs)
        self.batch_size = batch_size or 1
        if not isinstance(self.batch_size, int) or self.batch_size < 0:
            raise ValueError(f"batch_size must be a positive integer, got {self.batch_size}")

        if self.batch_size != 1:
            logger.warning("Ignore batch_size for Gemma3 vision tower. It will be set to 1.")

        self.vision_tower = self.initialize_submodule_config(
            submodule_config=vision_tower, batch_size=1, force_kwargs=True
        )
        self.language_model = self.initialize_submodule_config(submodule_config=language_model)

    @property
    def image_prefill_chunk_size(self):
        return self.language_model.image_prefill_chunk_size

    @property
    def prefill_chunk_size(self):
        return self.language_model.prefill_chunk_size
