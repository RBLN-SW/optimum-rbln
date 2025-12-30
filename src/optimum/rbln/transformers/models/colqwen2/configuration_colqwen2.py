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

from typing import Optional

from optimum.rbln.configuration_utils import RBLNModelConfig

from ..decoderonly.configuration_decoderonly import RBLNDecoderOnlyModelConfig


class RBLNColQwen2ForRetrievalConfig(RBLNDecoderOnlyModelConfig):
    """
    Configuration class for RBLN ColQwen2 models for document retrieval.

    This class extends RBLNModelConfig with specific configurations for ColQwen2 models,
    including vision tower settings and multi-sequence length support.

    Example usage:
        ```python
        from optimum.rbln import RBLNColQwen2ForRetrievalConfig, RBLNColQwen2ForRetrievalConfig

        # Create a configuration object
        config = RBLNColQwen2ForRetrievalConfig(
            vlm = {
                "visual": {
                    "max_seq_lens": 6400,
                    "device": 0,
                },
                "max_seq_len": 32_768,
                "tensor_parallel_size": 4,
                "device": [0, 1, 2, 3],
                "output_hidden_states": False,
                }
        )

        # Use the configuration with from_pretrained
        model = RBLNColQwen2ForRetrieval.from_pretrained(
            "vidore/colqwen2-v1.0-hf",
            export=True,
            rbln_config=config
        )
        ```
    """

    submodules = ["vlm"]
    _allow_no_compile_cfgs = True

    def __init__(
        self,
        batch_size: Optional[int] = None,
        output_hidden_states: Optional[bool] = None,
        vlm: Optional[RBLNModelConfig] = None,
        **kwargs,
    ):
        """
        Args:
            batch_size (Optional[int]): The batch size for the model.
            output_hidden_states (Optional[bool]): Whether to output the hidden states of the VLM model.
            vlm (Optional[RBLNModelConfig]): Configuration for the VLM component.
            kwargs: Additional arguments passed to the parent RBLNModelConfig.
        Raises:
            ValueError: If batch_size is not a positive integer.
        """
        super().__init__(**kwargs)
        self.batch_size = batch_size or 1
        self.output_hidden_states = output_hidden_states or False

        if not isinstance(self.batch_size, int) or self.batch_size < 0:
            raise ValueError(f"batch_size must be a positive integer, got {self.batch_size}")

        self.vlm = self.initialize_submodule_config(
            submodule_config=vlm,
            batch_size=batch_size,
            output_hidden_states=output_hidden_states,
            force_kwargs=True,
            logits_to_keep=0,
            use_inputs_embeds=True,
        )
