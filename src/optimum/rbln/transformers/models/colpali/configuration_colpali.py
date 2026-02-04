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

from ....configuration_utils import PositiveIntDefaultOne, RBLNModelConfig
from ....utils.logging import get_logger


logger = get_logger(__name__)


class RBLNColPaliForRetrievalConfig(RBLNModelConfig):
    """
    Configuration class for RBLN ColPali models for document retrieval.

    This class extends RBLNModelConfig with specific configurations for ColPali models,
    including vision tower settings and multi-sequence length support.

    Example usage:
        ```python
        from optimum.rbln import RBLNColPaliForRetrieval, RBLNColPaliForRetrievalConfig

        # Create a configuration object
        config = RBLNColPaliForRetrievalConfig(
            vlm={
                "language_model": {"prefill_chunk_size": 8192},
            }
            output_hidden_states=False,
            tensor_parallel_size=4
        )

        # Use the configuration with from_pretrained
        model = RBLNColPaliForRetrieval.from_pretrained(
            "vidore/colpali-v1.3-hf",
            export=True,
            rbln_config=config
        )
        ```
    """

    _allow_no_compile_cfgs = True
    submodules: ClassVar[list[str]] = ["vlm"]
    submodule_config_classes: ClassVar[dict[str, str]] = {
        "vlm": "RBLNPaliGemmaForConditionalGenerationConfig",
    }

    batch_size: PositiveIntDefaultOne = Field(default=1, description="The batch size for the model.")
    vlm: RBLNModelConfig | None = Field(default=None, description="Configuration for the VLM component.")
    output_hidden_states: bool = Field(
        default=False, description="Whether to output the hidden states of the decoder."
    )

    def __init__(self, **data: Any):
        super().__init__(**data)
        # vlm is converted by @model_validator, but we still handle None case
        if self.vlm is None:
            self.vlm = self.initialize_submodule_config(
                submodule_name="vlm",
                submodule_config=None,
                batch_size=self.batch_size,
                output_hidden_states=self.output_hidden_states,
            )
