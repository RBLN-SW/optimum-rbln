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

from typing import TYPE_CHECKING, Optional, Type, Union

import torch
from safetensors.torch import load_file
from transformers import AutoConfig, AutoModelForCausalLM, PretrainedConfig
from transformers.integrations.mxfp4 import Mxfp4GptOssExperts
from transformers.modeling_utils import PreTrainedModel, no_init_weights

from ....utils.logging import get_logger
from ...models.decoderonly import (
    RBLNDecoderOnlyModelConfig,
    RBLNDecoderOnlyModelForCausalLM,
    RBLNDecoderOnlyModelForCausalLMConfig,
)
from ...utils.rbln_quantization import RBLNQuantizerMixin, load_weight_files
from .gpt_oss_architecture import RBLNGptOssWrapper


if TYPE_CHECKING:
    from transformers.models.auto.modeling_auto import _BaseAutoModelClass
    from transformers import AutoFeatureExtractor, AutoProcessor, AutoTokenizer, PreTrainedModel

logger = get_logger(__name__)


# =============================================================================
# RBLNMXFP4Quantizer - Mxfp4HfQuantizer + RBLNQuantizerMixin
# =============================================================================

try:
    from transformers.quantizers.quantizer_mxfp4 import Mxfp4HfQuantizer

    _MXFP4_QUANTIZER_AVAILABLE = True
except ImportError:
    _MXFP4_QUANTIZER_AVAILABLE = False
    Mxfp4HfQuantizer = object  # Fallback


class RBLNMXFP4Quantizer(RBLNQuantizerMixin, Mxfp4HfQuantizer if _MXFP4_QUANTIZER_AVAILABLE else object):
    """
    RBLN-compatible MXFP4 Quantizer that inherits from both:
    - RBLNQuantizerMixin: Common loading logic (get_quantized_model pattern)
    - Mxfp4HfQuantizer: HuggingFace MXFP4 interface

    Example:
        model = RBLNMXFP4Quantizer.get_quantized_model(
            AutoModelForCausalLM,
            "openai/gpt-oss-20b",
        )
    """

    # =========================================================================
    # RBLNQuantizerMixin override points
    # =========================================================================

    @classmethod
    def _get_default_dtype(cls) -> torch.dtype:
        """MXFP4 uses bfloat16 by default."""
        return torch.bfloat16

    @classmethod
    def _get_exception_keywords(cls) -> list:
        """Exclude 'original' files for MXFP4."""
        return ["original"]

    @classmethod
    def _replace_linear_layers_cls(
        cls,
        quantizer_cls,
        model: torch.nn.Module,
        config: "PretrainedConfig",
        dtype: torch.dtype,
        kwargs: dict,
    ) -> None:
        """Replace GptOssExperts with Mxfp4GptOssExperts."""
        if not _MXFP4_QUANTIZER_AVAILABLE:
            raise ImportError(
                "Mxfp4HfQuantizer is not available. "
                "Please upgrade transformers to a version that supports MXFP4 quantization."
            )
        _replace_with_mxfp4_linear(model, config)


class RBLNGptOssForCausalLM(RBLNDecoderOnlyModelForCausalLM):
    """
    The GPT-OSS Model transformer with a language modeling head (linear layer) on top.
    This model inherits from [`RBLNDecoderOnlyModelForCausalLM`]. Check the superclass documentation for the generic methods the library implements for all its models.

    A class to convert and run pre-trained transformers based GptOssForCausalLM model on RBLN devices.
    It implements the methods to convert a pre-trained transformers GptOssForCausalLM model into a RBLN transformer model by:
    - transferring the checkpoint weights of the original into an optimized RBLN graph,
    - compiling the resulting graph using the RBLN compiler.

    **Configuration:**
    This model uses [`RBLNGptOssForCausalLMConfig`] for configuration. When calling methods like `from_pretrained` or `from_model`,
    the `rbln_config` parameter should be an instance of [`RBLNGptOssForCausalLMConfig`] or a dictionary conforming to its structure.

    See the [`RBLNGptOssForCausalLMConfig`] class for all available configuration options.

    Examples:
        ```python
        from optimum.rbln import RBLNGptOssForCausalLM

        # Simple usage using rbln_* arguments
        # `max_seq_len` is automatically inferred from the model config
        model = RBLNGptOssForCausalLM.from_pretrained(
            "openai/gpt-oss-20b",
            export=True,
            rbln_batch_size=1,
            rbln_tensor_parallel_size=8,
            rbln_kvcache_partition_len=8192,
        )


        # Using a config dictionary
        rbln_config = {
            "batch_size": 1,
            "tensor_parallel_size": 8,
            "kvcache_partition_len": 8192,
        }
        model = RBLNGptOssForCausalLM.from_pretrained(
            "openai/gpt-oss-20b",
            export=True,
            rbln_config=rbln_config,
        )


        # Using a RBLNGptOssForCausalLMConfig instance (recommended for type checking)
        from optimum.rbln import RBLNGptOssForCausalLMConfig

        config = RBLNGptOssForCausalLMConfig(
            batch_size=1,
            tensor_parallel_size=8,
            kvcache_partition_len=8192,
        )
        model = RBLNGptOssForCausalLM.from_pretrained(
            "openai/gpt-oss-20b",
            export=True,
            rbln_config=config,
        )
        ```
    """

    _decoder_wrapper_cls = RBLNGptOssWrapper

    @staticmethod
    def _get_dtype(dtype: Union[str, torch.dtype] = None, torch_dtype: Union[str, torch.dtype] = None):
        # For BC on torch_dtype argument
        if torch_dtype is not None:
            logger.warning_once("`torch_dtype` is deprecated! Use `dtype` instead!")
            # If both kwargs are provided, use `dtype`
            dtype = dtype if dtype is not None else torch_dtype

        # As mxfp4_quantizer's default dtype
        if dtype is None or dtype == "auto":
            dtype = torch.bfloat16

        return dtype

    @classmethod
    def get_pytorch_model(
        cls,
        model_id: str,
        *args,
        rbln_config: Optional[RBLNDecoderOnlyModelConfig] = None,
        dtype: Union[str, torch.dtype] = None,
        torch_dtype: Union[str, torch.dtype] = None,
        config: Optional[PretrainedConfig] = None,
        **kwargs,
    ) -> PreTrainedModel:
        """
        Load a MXFP4 quantized PyTorch model.

        Delegates to RBLNMXFP4Quantizer.get_quantized_model() for the actual implementation.
        """
        dtype = cls._get_dtype(dtype, torch_dtype)

        return RBLNMXFP4Quantizer.get_quantized_model(
            hf_auto_model_class=AutoModelForCausalLM,
            model_id=model_id,
            dtype=dtype,
            config=config,
            **kwargs,
        )

    @classmethod
    def _update_rbln_config(
        cls,
        preprocessors: Optional[Union["AutoFeatureExtractor", "AutoProcessor", "AutoTokenizer"]] = None,
        model: Optional["PreTrainedModel"] = None,
        model_config: Optional["PretrainedConfig"] = None,
        rbln_config: Optional[RBLNDecoderOnlyModelForCausalLMConfig] = None,
    ) -> RBLNDecoderOnlyModelForCausalLMConfig:
        rbln_config = super()._update_rbln_config(preprocessors, model, model_config, rbln_config)

        if rbln_config.use_attention_mask:
            raise ValueError(
                "use_attention_mask is not supported for GPT-OSS because custom attention does not support attention sink for masked attention"
            )

        return rbln_config


def _replace_with_mxfp4_linear(
    model,
    config,
):
    for name, module in model.named_children():
        if module.__class__.__name__ == "GptOssExperts":
            model._modules[name] = Mxfp4GptOssExperts(config)
        if len(list(module.children())) > 0:
            _replace_with_mxfp4_linear(module, config)
