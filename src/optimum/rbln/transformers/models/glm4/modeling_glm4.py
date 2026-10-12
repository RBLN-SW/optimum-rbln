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
from .glm4_architecture import Glm4Wrapper


logger = logging.get_logger(__name__)


class RBLNGlm4ForCausalLM(RBLNDecoderOnlyModelForCausalLM):
    """
    The Glm4 Model transformer with a language modeling head (linear layer) on top.
    This model inherits from [`RBLNDecoderOnlyModelForCausalLM`]. Check the superclass documentation for the generic methods the library implements for all its models.

    A class to convert and run pre-trained transformers based Glm4ForCausalLM model on RBLN devices.
    It implements the methods to convert a pre-trained transformers Glm4ForCausalLM model into a RBLN transformer model by:
    - transferring the checkpoint weights of the original into an optimized RBLN graph,
    - compiling the resulting graph using the RBLN compiler.

    **Configuration:**
    This model uses [`RBLNGlm4ForCausalLMConfig`] for configuration. When calling methods like `from_pretrained` or `from_model`,
    the `rbln_config` parameter should be an instance of [`RBLNGlm4ForCausalLMConfig`] or a dictionary conforming to its structure.

    See the [`RBLNGlm4ForCausalLMConfig`] class for all available configuration options.

    Examples:
        ```python
        from optimum.rbln import RBLNGlm4ForCausalLM

        # Simple usage using rbln_* arguments
        # `max_seq_len` is automatically inferred from the model config
        model = RBLNGlm4ForCausalLM.from_pretrained(
            "zai-org/GLM-4-9B-0414",
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
        model = RBLNGlm4ForCausalLM.from_pretrained(
            "zai-org/GLM-4-9B-0414",
            export=True,
            rbln_config=rbln_config
        )

        # Using a RBLNGlm4ForCausalLMConfig instance (recommended for type checking)
        from optimum.rbln import RBLNGlm4ForCausalLMConfig

        config = RBLNGlm4ForCausalLMConfig(
            batch_size=1,
            max_seq_len=4096,
            num_devices=4
        )
        model = RBLNGlm4ForCausalLM.from_pretrained(
            "zai-org/GLM-4-9B-0414",
            export=True,
            rbln_config=config
        )
        ```
    """

    _decoder_wrapper_cls = Glm4Wrapper


class RBLNGlm4Model(RBLNDecoderOnlyModel):
    """
    The Glm4 Model transformer without a language modeling head.
    This model inherits from [`RBLNDecoderOnlyModel`]. Check the superclass documentation for the generic methods the library implements for all its models.
    """

    _decoder_wrapper_cls = Glm4Wrapper


__all__ = [
    "RBLNGlm4ForCausalLM",
    "RBLNGlm4Model",
]
