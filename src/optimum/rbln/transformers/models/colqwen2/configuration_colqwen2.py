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
            visual={
                "max_seq_lens": 6400,
                "device": 0,
            },
            max_seq_len=32_768,
            tensor_parallel_size=4,
            device=[0, 1, 2, 3],
            output_hidden_states=False,
        )

        # Use the configuration with from_pretrained
        model = RBLNColQwen2ForRetrieval.from_pretrained(
            "vidore/colqwen2-v1.0-hf",
            export=True,
            rbln_config=config
        )
        ```
    """

    submodules = ["visual"]

    def __init__(
        self,
        visual: Optional[RBLNModelConfig] = None,
        batch_size: Optional[int] = None,
        use_inputs_embeds: bool = True,
        **kwargs,
    ):
        super().__init__(use_inputs_embeds=use_inputs_embeds, **kwargs)
        if not self.use_inputs_embeds:
            raise ValueError(
                "RBLNColQwen2ForRetrievalConfig does not allow `use_inputs_embeds` to be set to False, "
                "as RBLNColQwen2ForRetrieval accepts only `inputs_embeds` as input."
            )
        if batch_size is not None and batch_size != 1:
            raise ValueError("batch_size is not supported for RBLNColQwen2ForRetrievalConfig")

        self.visual = visual
