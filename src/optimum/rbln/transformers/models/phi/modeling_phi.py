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

from ....utils import logging
from ...models.decoderonly import RBLNDecoderOnlyModel, RBLNDecoderOnlyModelForCausalLM
from .phi_architecture import PhiWrapper


logger = logging.get_logger(__name__)


class RBLNPhiForCausalLM(RBLNDecoderOnlyModelForCausalLM):
    """
    The Phi Model transformer with a language modeling head (linear layer) on top.
    This model inherits from [`RBLNDecoderOnlyModelForCausalLM`]. Check the superclass documentation for the generic methods the library implements for all its models.

    A class to convert and run pre-trained transformers based PhiForCausalLM model on RBLN devices.
    It implements the methods to convert a pre-trained transformers PhiForCausalLM model into a RBLN transformer model by:

    - transferring the checkpoint weights of the original into an optimized RBLN graph,
    - compiling the resulting graph using the RBLN compiler.

    **Configuration:**
    This model uses [`RBLNPhiForCausalLMConfig`] for configuration. When calling methods like `from_pretrained` or `from_model`,
    the `rbln_config` parameter should be an instance of [`RBLNPhiForCausalLMConfig`] or a dictionary conforming to its structure.

    See the [`RBLNPhiForCausalLMConfig`] class for all available configuration options.

    Examples:
        ```python
        from optimum.rbln import RBLNPhiForCausalLM

        # Simple usage using rbln_* arguments
        # `max_seq_len` is automatically inferred from the model config
        model = RBLNPhiForCausalLM.from_pretrained(
            "microsoft/phi-2",
            export=True,
            rbln_batch_size=1,
            rbln_tensor_parallel_size=4,
        )


        # Using a config dictionary
        rbln_config = {
            "batch_size": 1,
            "max_seq_len": 4096,
            "tensor_parallel_size": 4,
        }
        model = RBLNPhiForCausalLM.from_pretrained(
            "microsoft/phi-2",
            export=True,
            rbln_config=rbln_config
        )


        # Using a RBLNPhiForCausalLMConfig instance (recommended for type checking)
        from optimum.rbln import RBLNPhiForCausalLMConfig

        config = RBLNPhiForCausalLMConfig(
            batch_size=1,
            max_seq_len=4096,
            tensor_parallel_size=4
        )
        model = RBLNPhiForCausalLM.from_pretrained(
            "microsoft/phi-2",
            export=True,
            rbln_config=config
        )
        ```
    """

    _decoder_wrapper_cls = PhiWrapper


class RBLNPhiModel(RBLNDecoderOnlyModel):
    """
    The Phi Model transformer without a language modeling head.
    This model inherits from [`RBLNDecoderOnlyModel`]. Check the superclass documentation for the generic methods the library implements for all its models.
    """

    _decoder_wrapper_cls = PhiWrapper
