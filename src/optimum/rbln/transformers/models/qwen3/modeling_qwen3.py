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
from .qwen3_architecture import Qwen3Wrapper


logger = logging.get_logger(__name__)


class RBLNQwen3ForCausalLM(RBLNDecoderOnlyModelForCausalLM):
    """
    The Qwen3 Model transformer with a language modeling head (linear layer) on top.
    This model inherits from [`RBLNDecoderOnlyModelForCausalLM`]. Check the superclass documentation for the generic methods the library implements for all its models.
    A class to convert and run pre-trained transformers based Qwen3ForCausalLM model on RBLN devices.
    It implements the methods to convert a pre-trained transformers Qwen3ForCausalLM model into a RBLN transformer model by:
    - transferring the checkpoint weights of the original into an optimized RBLN graph,
    - compiling the resulting graph using the RBLN compiler.
    **Configuration:**
    This model uses [`RBLNQwen3ForCausalLMConfig`] for configuration. When calling methods like `from_pretrained` or `from_model`,
    the `rbln_config` parameter should be an instance of [`RBLNQwen3ForCausalLMConfig`] or a dictionary conforming to its structure.
    See the [`RBLNQwen3ForCausalLMConfig`] class for all available configuration options.
    Examples:
        ```python
        from optimum.rbln import RBLNQwen3ForCausalLM
        # Simple usage using rbln_* arguments
        # `max_seq_len` is automatically inferred from the model config
        model = RBLNQwen3ForCausalLM.from_pretrained(
            "Qwen/Qwen3-4B",
            export=True,
            rbln_batch_size=1,
            rbln_tensor_parallel_size=4,
        )
        # Using a config dictionary
        rbln_config = {
            "batch_size": 1,
            "max_seq_len": 40_960,
            "tensor_parallel_size": 4,
            "kvcache_partition_len": 8192,
        }
        model = RBLNQwen3ForCausalLM.from_pretrained(
            "Qwen/Qwen3-4B",
            export=True,
            rbln_config=rbln_config
        )
        # Using a RBLNQwen3ForCausalLMConfig instance (recommended for type checking)
        from optimum.rbln import RBLNQwen3ForCausalLMConfig
        config = RBLNQwen3ForCausalLMConfig(
            batch_size=1,
            max_seq_len=40_960,
            tensor_parallel_size=4,
            kvcache_partition_len=8192,
        )
        model = RBLNQwen3ForCausalLM.from_pretrained(
            "Qwen/Qwen3-4B",
            export=True,
            rbln_config=config
        )
        ```
    """

    _decoder_wrapper_cls = Qwen3Wrapper

    def forward(self, *args, **kwargs):
        kwargs["return_dict"] = True
        return super().forward(*args, **kwargs)


class RBLNQwen3Model(RBLNDecoderOnlyModel):
    """
    The bare Qwen3 Model outputting raw hidden-states without any specific head on top.
    This model inherits from [`RBLNDecoderOnlyModel`]. Check the superclass documentation for the generic methods the library implements for all its models.
    A class to convert and run pre-trained transformers based Qwen3Model on RBLN devices.
    It implements the methods to convert a pre-trained transformers Qwen3Model into a RBLN transformer model by:
    - transferring the checkpoint weights of the original into an optimized RBLN graph,
    - compiling the resulting graph using the RBLN compiler.
    **Configuration:**
    This model uses [`RBLNQwen3ModelConfig`] for configuration. When calling methods like `from_pretrained` or `from_model`,
    the `rbln_config` parameter should be an instance of [`RBLNQwen3ModelConfig`] or a dictionary conforming to its structure.
    See the [`RBLNQwen3ModelConfig`] class for all available configuration options.
    Examples:
        ```python
        from optimum.rbln import RBLNQwen3Model
        # Simple usage using rbln_* arguments
        # `max_seq_len` is automatically inferred from the model config
        model = RBLNQwen3Model.from_pretrained(
            "Qwen/Qwen3-Embedding-4B",
            export=True,
            rbln_batch_size=1,
            rbln_max_seq_len=40_960,
            rbln_tensor_parallel_size=4,
            rbln_kvcache_partition_len=8192,
        )
    """

    _decoder_wrapper_cls = Qwen3Wrapper
    _use_rotary_emb = True
