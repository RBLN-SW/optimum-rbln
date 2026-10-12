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
from ...models.decoderonly import (
    RBLNDecoderOnlyModel,
    RBLNDecoderOnlyModelForCausalLM,
)
from .granite_architecture import GraniteWrapper


logger = logging.get_logger(__name__)


class RBLNGraniteForCausalLM(RBLNDecoderOnlyModelForCausalLM):
    """
    The Granite Model transformer with a language modeling head (linear layer) on top.
    This model inherits from [`RBLNDecoderOnlyModelForCausalLM`]. Check the superclass documentation for the generic methods the library implements for all its models.

    A class to convert and run pre-trained transformers based GraniteForCausalLM model on RBLN devices.
    It implements the methods to convert a pre-trained transformers GraniteForCausalLM model into a RBLN transformer model by:
    - transferring the checkpoint weights of the original into an optimized RBLN graph,
    - compiling the resulting graph using the RBLN compiler.

    **Configuration:**
    This model uses [`RBLNGraniteForCausalLMConfig`] for configuration. When calling methods like `from_pretrained` or `from_model`,
    the `rbln_config` parameter should be an instance of [`RBLNGraniteForCausalLMConfig`] or a dictionary conforming to its structure.

    See the [`RBLNGraniteForCausalLMConfig`] class for all available configuration options.

    Examples:
        ```python
        from optimum.rbln import RBLNGraniteForCausalLM

        # Simple usage using rbln_* arguments
        # `max_seq_len` is automatically inferred from the model config
        model = RBLNGraniteForCausalLM.from_pretrained(
            "ibm-granite/granite-3.3-8b-instruct",
            export=True,
            rbln_batch_size=1,
            rbln_num_devices=4,
        )

        # Using a config dictionary
        rbln_config = {
            "batch_size": 1,
            "max_seq_len": 4096,
            "num_devices": 4,
        }
        model = RBLNGraniteForCausalLM.from_pretrained(
            "ibm-granite/granite-3.3-8b-instruct",
            export=True,
            rbln_config=rbln_config
        )

        # Using a RBLNGraniteForCausalLMConfig instance (recommended for type checking)
        from optimum.rbln import RBLNGraniteForCausalLMConfig

        config = RBLNGraniteForCausalLMConfig(
            batch_size=1,
            max_seq_len=4096,
            num_devices=4
        )
        model = RBLNGraniteForCausalLM.from_pretrained(
            "ibm-granite/granite-3.3-8b-instruct",
            export=True,
            rbln_config=config
        )
        ```
    """

    _decoder_wrapper_cls = GraniteWrapper


class RBLNGraniteModel(RBLNDecoderOnlyModel):
    """
    The Granite Model transformer without a language modeling head.
    This model inherits from [`RBLNDecoderOnlyModel`]. Check the superclass documentation for the generic methods the library implements for all its models.
    """

    _decoder_wrapper_cls = GraniteWrapper


__all__ = [
    "RBLNGraniteForCausalLM",
    "RBLNGraniteModel",
]
