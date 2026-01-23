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

from ...models.decoderonly import RBLNDecoderOnlyModelForCausalLM
from .mixtral_architecture import MixtralWrapper


class RBLNMixtralForCausalLM(RBLNDecoderOnlyModelForCausalLM):
    """
    The Mixtral is a Mixture-of-Experts (MoE) variant of Mixtral, available as a base model and an aligned chat model.
    This model inherits from [`RBLNDecoderOnlyModelForCausalLM`]. Check the superclass documentation for the generic methods the library implements for all its models.
    A class to convert and run pre-trained transformers based MixtralForCausalLM model on RBLN devices.
    It implements the methods to convert a pre-trained transformers MixtralForCausalLM model into a RBLN transformer model by:
    - transferring the checkpoint weights of the original into an optimized RBLN graph,
    - compiling the resulting graph using the RBLN compiler.
    **Configuration:**
    This model uses [`RBLNMixtralForCausalLMConfig`] for configuration. When calling methods like `from_pretrained` or `from_model`,
    the `rbln_config` parameter should be an instance of [`RBLNMixtralForCausalLMConfig`] or a dictionary conforming to its structure.
    See the [`RBLNMixtralForCausalLMConfig`] class for all available configuration options.
    Examples:
        ```python
        from optimum.rbln import RBLNMixtralForCausalLM
        # Simple usage using rbln_* arguments
        # `max_seq_len` is automatically inferred from the model config
        model = RBLNMixtralForCausalLM.from_pretrained(
            "mistralai/Mixtral-8x7B-Instruct-v0.1",
            export=True,
            rbln_batch_size=1,
            rbln_tensor_parallel_size=4,
        )
        # Using a config dictionary
        rbln_config = {
            "batch_size": 1,
            "max_seq_len": 32768,
            "tensor_parallel_size": 4,
        }
        model = RBLNMixtralForCausalLM.from_pretrained(
            "mistralai/Mixtral-8x7B-Instruct-v0.1",
            export=True,
            rbln_config=rbln_config
        )
        # Using a RBLNMixtralForCausalLMConfig instance (recommended for type checking)
        from optimum.rbln import RBLNMixtralForCausalLMConfig
        config = RBLNMixtralForCausalLMConfig(
            batch_size=1,
            max_seq_len=32768,
            tensor_parallel_size=4
        )
        model = RBLNMixtralForCausalLM.from_pretrained(
            "mistralai/Mixtral-8x7B-Instruct-v0.1",
            export=True,
            rbln_config=config
        )
        ```
    """

    _decoder_wrapper_cls = MixtralWrapper
