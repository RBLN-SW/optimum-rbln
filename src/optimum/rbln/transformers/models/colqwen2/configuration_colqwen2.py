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

from __future__ import annotations

from typing import Any, ClassVar

from pydantic import Field

from ....configuration_utils import RBLNModelConfig
from ..decoderonly.configuration_decoderonly import RBLNDecoderOnlyModelConfig


class RBLNColQwen2ForRetrievalConfig(RBLNDecoderOnlyModelConfig):
    """
    Configuration class for RBLN ColQwen2 models for document retrieval.

    This class extends RBLNDecoderOnlyModelConfig with specific configurations for ColQwen2 models,
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

    submodules: ClassVar[list[str]] = ["vlm"]
    submodule_config_classes: ClassVar[dict[str, str]] = {"vlm": "RBLNQwen2VLModelConfig"}
    _allow_no_compile_cfgs = True

    vlm: RBLNModelConfig | None = Field(default=None, description="Configuration for the VLM component.")

    def __init__(self, **data: Any):
        super().__init__(**data)
        # initialize_submodule_config handles None, dict, and RBLNModelConfig.
        # kwargs always take priority.
        self.vlm = self.initialize_submodule_config(
            submodule_name="vlm",
            submodule_config=self.vlm,
            batch_size=self.batch_size,
            output_hidden_states=self.output_hidden_states,
            logits_to_keep=0,
            use_inputs_embeds=True,
        )
