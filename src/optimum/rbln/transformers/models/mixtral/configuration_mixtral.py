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

from ..decoderonly.configuration_decoderonly import RBLNDecoderOnlyModelForCausalLMConfig


class RBLNMixtralForCausalLMConfig(RBLNDecoderOnlyModelForCausalLMConfig):
    """
    Configuration class for RBLN Mixtral models.
    This class is an alias of RBLNDecoderOnlyModelForCausalLMConfig.
    Example usage:
    ```python
    from optimum.rbln import RBLNMixtralForCausalLM, RBLNMixtralForCausalLMConfig
    # Create a configuration object
    config = RBLNMixtralForCausalLMConfig(
        batch_size=1,
        max_seq_len=262144,
        tensor_parallel_size=4
    )
    # Use the configuration with from_pretrained
    model = RBLNMixtralForCausalLM.from_pretrained(
        "mistralai/Mixtral-8x7B-Instruct-v0.1",
        export=True,
        rbln_config=config
    )
    ```
    """
