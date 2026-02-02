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

import glob
import os
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional, Tuple, Type, Union

import torch
from huggingface_hub import hf_hub_download, list_repo_files
from safetensors.torch import load_file
from torch.nn import Linear, Parameter
from transformers import AutoConfig
from transformers.modeling_utils import get_state_dict_dtype, no_init_weights

from ...configuration_utils import RBLNSerializableConfigProtocol
from ...utils.logging import get_logger
from .qlinear import QFloatLinear, QIntLinear


if TYPE_CHECKING:
    from transformers.models.auto.modeling_auto import _BaseAutoModelClass

logger = get_logger()


# Constants - Default target modules for quantization (Llama-style)
DEFAULT_TARGET_MODULES: List[str] = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]

# For backward compatibility
QUANTIZED_WEIGHTS = set(DEFAULT_TARGET_MODULES)

# Common alias sets seen in community checkpoints
DEFAULT_SCALE_ALIASES: Dict[str, List[str]] = {
    "weight_scale": ["weight_scale", "scales", "w_scale", "scale"],
    "input_scale": ["input_scale", "act_scale", "activation_scale", "a_scale"],
    "kv_scale": ["kv_scale", "kv_scales"],
    "k_scale": ["k_scale", "k_scales"],
    "v_scale": ["v_scale", "v_scales"],
}

# For backward compatibility
VARIANT_ALIASES = DEFAULT_SCALE_ALIASES


class RBLNQuantizationConfig(RBLNSerializableConfigProtocol):
    """
    Configuration for RBLN quantization.

    Args:
        format: Quantization format. Currently only "rbln" is supported.
        weights: Weight quantization type. One of "int4", "int8", "fp8", "fp16".
        activations: Activation quantization type. One of "int8", "fp8", "fp16".
        kv_caches: KV cache quantization type. One of "fp8", "fp16".
        dynamic: Whether to use dynamic quantization.
        precision: Deprecated. Use `weights` and `activations` instead.
        target_modules: Optional list of module names to quantize. If None, uses default
            Llama-style modules (q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj).
        scale_aliases: Optional dict mapping canonical scale names to their aliases.
            Used for loading weights from different checkpoint formats.
    """

    SUPPORTED_FORMATS = ["rbln"]
    SUPPORTED_WEIGHTS = ["int4", "int8", "fp8", "fp16"]
    SUPPORTED_ACTIVATIONS = ["int8", "fp8", "fp16"]
    SUPPORTED_KVCACHES = ["fp8", "fp16"]
    RBLN_QUANT_BITS_ENV = "RBLN_QUANT_BITS"

    def __init__(
        self,
        format: Optional[str] = None,
        weights: Optional[str] = None,
        activations: Optional[str] = None,
        kv_caches: Optional[str] = None,
        dynamic: Optional[bool] = None,
        *,
        precision: Optional[str] = None,
        # New optional extension fields (backward compatible)
        target_modules: Optional[List[str]] = None,
        scale_aliases: Optional[Dict[str, List[str]]] = None,
    ):
        self.format = format or "rbln"
        if self.format not in self.SUPPORTED_FORMATS:
            raise ValueError(f"Invalid format: {self.format}, supported formats are: {self.SUPPORTED_FORMATS}")

        if precision is not None:
            logger.warning("The `precision` argument is deprecated. Use `weights` and `activations` instead.")
            if any(precision_arg is not None for precision_arg in (weights, activations)):
                raise ValueError("`precision` and `weights` or `activations` cannot be set at the same time.")

            if precision == "w4a16":
                weights = "int4"
                activations = "fp16"
            else:
                raise ValueError(f"Invalid precision: {precision}")

        self.weights = weights or "fp16"
        self.activations = activations or "fp16"
        self.kv_caches = kv_caches or "fp16"
        self.dynamic = dynamic if dynamic is not None else False

        # Store optional extension fields (None means use defaults)
        self._target_modules = target_modules
        self._scale_aliases = scale_aliases

        self._validate()

    @property
    def target_modules(self) -> List[str]:
        """Get target modules for quantization. Returns default if not specified."""
        return self._target_modules if self._target_modules is not None else DEFAULT_TARGET_MODULES

    @property
    def scale_aliases(self) -> Dict[str, List[str]]:
        """Get scale aliases for checkpoint loading. Returns default if not specified."""
        return self._scale_aliases if self._scale_aliases is not None else DEFAULT_SCALE_ALIASES

    def _validate(self):
        if self.format not in self.SUPPORTED_FORMATS:
            raise ValueError(f"Invalid format: {self.format}, supported formats are: {self.SUPPORTED_FORMATS}")
        if self.weights not in self.SUPPORTED_WEIGHTS:
            raise ValueError(f"Invalid weights: {self.weights}, supported weights are: {self.SUPPORTED_WEIGHTS}")
        if self.activations not in self.SUPPORTED_ACTIVATIONS:
            raise ValueError(
                f"Invalid activations: {self.activations}, supported activations are: {self.SUPPORTED_ACTIVATIONS}"
            )
        if self.kv_caches not in self.SUPPORTED_KVCACHES:
            raise ValueError(
                f"Invalid kv_caches: {self.kv_caches}, supported kv_caches are: {self.SUPPORTED_KVCACHES}"
            )
        if self.weights == "fp16" and self.activations == "fp16":
            raise ValueError("weights and activations of QuantizationConfig cannot be both fp16. It is meaningless.")

    def _prepare_for_serialization(self) -> Dict[str, Any]:
        return {
            "format": self.format,
            "weights": self.weights,
            "activations": self.activations,
            "kv_caches": self.kv_caches,
        }

    def maybe_set_quantization_env(self):
        if self.weights == "int4":
            os.environ[self.RBLN_QUANT_BITS_ENV] = "4"

    def maybe_reset_quantization_env(self):
        if self.RBLN_QUANT_BITS_ENV in os.environ:
            os.environ.pop(self.RBLN_QUANT_BITS_ENV)

    @property
    def nbits_per_param(self) -> int:
        if self.weights in ["int4", "fp4"]:
            return 4
        elif self.weights in ["int8", "fp8"]:
            return 8
        else:
            raise ValueError(f"Invalid weights: {self.weights}")


class QuantizedLayerFactory:
    def __init__(self, quantization_config: RBLNQuantizationConfig):
        self.quantization_config = quantization_config

    def create_linear(self, layer: Linear, scale_dtype: torch.dtype) -> Linear:
        if self.quantization_config.weights in ["int4", "int8"]:
            return self.convert_to_qint_linear(layer, scale_dtype)
        elif self.quantization_config.weights == "fp8":
            return self.convert_to_qfloat_linear(layer, scale_dtype)
        else:
            raise ValueError(f"Invalid quantization weights: {self.quantization_config.weights}")

    def convert_to_qint_linear(self, layer: Linear, scale_dtype: torch.dtype) -> Linear:
        return convert_to_qint_linear(layer, self.quantization_config, scale_dtype)

    def convert_to_qfloat_linear(self, layer: Linear, scale_dtype: torch.dtype) -> Linear:
        return convert_to_qfloat_linear(layer, self.quantization_config, scale_dtype)


def get_quantized_model(
    hf_auto_model_class: Type["_BaseAutoModelClass"],
    model_id: str,
    use_auth_token: Optional[Union[bool, str]] = None,
    revision: Optional[str] = None,
    cache_dir: Optional[str] = None,
    force_download: bool = False,
    local_files_only: bool = False,
    rbln_quantization: Optional[RBLNQuantizationConfig] = None,
    **kwargs,
):
    """
    Get a quantized model from a model class and model id.

    This function delegates to RBLNQuantizer.get_quantized_model() for the actual implementation.
    Maintained for backward compatibility.

    Args:
        hf_auto_model_class: HuggingFace AutoModel class
        model_id: Model ID or path
        use_auth_token: HuggingFace auth token
        revision: Model revision
        cache_dir: Cache directory
        force_download: Force download
        local_files_only: Use only local files
        rbln_quantization: RBLN quantization configuration
        **kwargs: Additional arguments

    Returns:
        Quantized model
    """
    # Delegate to RBLNQuantizer (imported at module level after class definition)
    return RBLNQuantizer.get_quantized_model(
        hf_auto_model_class=hf_auto_model_class,
        model_id=model_id,
        use_auth_token=use_auth_token,
        revision=revision,
        cache_dir=cache_dir,
        force_download=force_download,
        local_files_only=local_files_only,
        rbln_quantization=rbln_quantization,
        **kwargs,
    )


def load_weight_files(
    model_id: str,
    use_auth_token: Optional[Union[bool, str]] = None,
    revision: Optional[str] = None,
    cache_dir: Optional[str] = None,
    force_download: bool = False,
    local_files_only: bool = False,
    exception_keywords: Optional[List[str]] = None,
) -> list[str]:
    """
    Discover and download safetensors files for the given model id.
    """
    exception_keywords = exception_keywords or []
    if os.path.isdir(model_id):
        safetensor_files = glob.glob(f"{model_id}/*.safetensors")
    else:
        try:
            # List all files in the repository
            repo_files = list_repo_files(model_id, revision=revision, token=use_auth_token)
            # Filter for safetensors files
            safetensor_files = []

            for file in repo_files:
                if file.endswith(".safetensors"):
                    exculde = False
                    for except_key in exception_keywords:
                        if except_key in file:
                            exculde = True
                            break

                    if not exculde:
                        # Download the safetensors file
                        downloaded_file = hf_hub_download(
                            repo_id=model_id,
                            filename=file,
                            revision=revision,
                            token=use_auth_token,
                            cache_dir=cache_dir,
                            force_download=force_download,
                            local_files_only=local_files_only,
                        )
                        safetensor_files.append(downloaded_file)
        except Exception as e:
            logger.error(f"Failed to download safetensors files from Hugging Face Hub: {e}")
            raise e

    if not safetensor_files:
        raise FileNotFoundError(f"No safetensors files found for model_id: {model_id}")

    return safetensor_files


# DEPRECATED: Use RBLNQuantizer._update_layers_to_quantize instead
# Kept for backward compatibility - delegates to RBLNQuantizer
def update_layers_to_quantize(
    module: torch.nn.Module,
    scale_dtype: torch.dtype,
    rbln_quantization: Optional[RBLNQuantizationConfig] = None,
) -> None:
    """
    DEPRECATED: Use RBLNQuantizer instead.

    Updates specified linear layers to quantized (qlinear) layers in the given module.
    """
    quantizer = RBLNQuantizer(rbln_quantization, pre_quantized=True)
    quantizer._update_layers_to_quantize(module, scale_dtype)


def _last_segment(key: str) -> str:
    parts = key.split(".")
    return parts[-1]


def _replace_last_with(key: str, new_tail: str) -> str:
    parts = key.split(".")
    return ".".join(parts[:-1] + new_tail.split("."))


def _matches_any_alias(
    key: str,
    kind: str,
    scale_aliases: Optional[Dict[str, List[str]]] = None,
) -> bool:
    """Check if the key's last segment matches any alias for the given kind."""
    if scale_aliases is None:
        scale_aliases = DEFAULT_SCALE_ALIASES
    tail = _last_segment(key)
    return tail in scale_aliases.get(kind, [])


def _reduce_to_scalar(t: torch.Tensor) -> torch.Tensor:
    if t.ndim == 0:
        return t
    return t.reshape(-1).amax()


def _scalar_value_as_1d(scale: torch.Tensor) -> torch.Tensor:
    v = _reduce_to_scalar(scale)
    return v.reshape(1).contiguous()


def _coerce_per_out_channel_scale(scale: torch.Tensor, out_features: int) -> torch.Tensor:
    s = scale
    if s.ndim == 0:
        # scalar -> expand to [out_features, 1]
        return s.reshape(1, 1).expand(out_features, 1).contiguous()
    if s.ndim == 1:
        if s.numel() == 1:
            return s.reshape(1, 1).expand(out_features, 1).contiguous()
        if s.numel() == out_features:
            return s.reshape(out_features, 1).contiguous()
        # fallback: reduce to scalar then expand
        v = _reduce_to_scalar(s)
        return v.reshape(1, 1).expand(out_features, 1).contiguous()
    if s.ndim == 2:
        if s.shape == (out_features, 1):
            return s.contiguous()
        if s.shape == (1, out_features):
            return s.transpose(0, 1).contiguous()
        # fallback: reduce to [out_features] on non-out dims if possible
        if s.shape[0] == out_features:
            v = s
            while v.ndim > 2:
                v = v.amax(dim=-1)
            if v.shape[-1] != 1:
                v = v.amax(dim=-1, keepdim=True)
            return v.contiguous()
        # otherwise reduce to scalar then expand
        v = _reduce_to_scalar(s)
        return v.reshape(1, 1).expand(out_features, 1).contiguous()
    # high-rank: reduce to scalar then expand
    v = _reduce_to_scalar(s)
    return v.reshape(1, 1).expand(out_features, 1).contiguous()


def _kv_split_items(base_key: str, tensor: torch.Tensor) -> List[Tuple[str, torch.Tensor]]:
    # base_key is the original key whose last token was 'kv_scale'
    # We produce keys with 'k_proj.k_scale' and 'v_proj.v_scale'
    if tensor.ndim == 1 and tensor.numel() >= 2:
        tk, tv = tensor[0], tensor[1]
    elif tensor.ndim == 2 and tensor.shape[0] >= 2 and tensor.shape[1] == 1:
        tk, tv = tensor[0, 0], tensor[1, 0]
    else:
        tk = tv = tensor
    k_key = _replace_last_with(base_key, "k_proj.k_scale")
    v_key = _replace_last_with(base_key, "v_proj.v_scale")
    return [(k_key, tk), (v_key, tv)]


def canonicalize_checkpoint_items(
    model: torch.nn.Module,
    items: Iterable[Tuple[str, torch.Tensor]],
    rbln_quantization: Optional[RBLNQuantizationConfig],
) -> List[Tuple[str, torch.Tensor]]:
    """
    Canonicalize checkpoint items (key-value pairs) to match model parameter names.

    Uses `rbln_quantization.scale_aliases` to map various checkpoint key formats
    to the canonical format expected by the model.
    """
    params = dict(model.named_parameters(recurse=True))
    results: List[Tuple[str, torch.Tensor]] = []

    # Get scale aliases from config, or use default
    scale_aliases = (
        rbln_quantization.scale_aliases
        if rbln_quantization is not None
        else DEFAULT_SCALE_ALIASES
    )

    for key, value in items:
        t = value
        # Normalize weight scale variants
        if _matches_any_alias(key, "weight_scale", scale_aliases):
            # rename last token to the canonical weight scale key
            target_key = _replace_last_with(key, "weight_scale")

            # Determine associated weight param to infer shape
            weight_key = _replace_last_with(target_key, "weight")
            out_features = None
            if weight_key in params:
                wshape = params[weight_key].shape
                if len(wshape) == 2:
                    out_features = int(wshape[0])

            if out_features is not None:
                t = _coerce_per_out_channel_scale(t, out_features)
            else:
                t = _scalar_value_as_1d(t)

            results.append((target_key, t))
            continue

        # Normalize input/activation scale variants
        if _matches_any_alias(key, "input_scale", scale_aliases):
            target_key = _replace_last_with(key, "input_scale")
            t = _scalar_value_as_1d(t)
            results.append((target_key, t))
            continue

        # KV scale handling
        if _matches_any_alias(key, "kv_scale", scale_aliases):
            # For quark-like formats, expand to k/v
            kv_items = _kv_split_items(key, t)
            for k2, v2 in kv_items:
                results.append((k2, v2))
            continue

        if _matches_any_alias(key, "k_scale", scale_aliases) or _matches_any_alias(key, "v_scale", scale_aliases):
            results.append((key, t))
            continue

        # Default: passthrough
        results.append((key, t))

    return results


# DEPRECATED: Use RBLNQuantizer._load_weights_from_files instead
# Kept for backward compatibility - delegates to RBLNQuantizer
def load_weights_from_files(
    model: torch.nn.Module,
    safetensors: List[Dict[str, torch.Tensor]],
    rbln_quantization: Optional[RBLNQuantizationConfig] = None,
):
    """
    DEPRECATED: Use RBLNQuantizer instead.

    Load safetensor file data directly into the model from provided safetensor files.
    """
    quantizer = RBLNQuantizer(rbln_quantization, pre_quantized=True)
    quantizer._load_weights_from_files(model, safetensors)


def is_target_for_adding_kv_scales(layer_name: str) -> bool:
    return layer_name.split(".")[-1] in ["self_attn"]


def get_parent_and_child(module: torch.nn.Module, full_name: str) -> tuple:
    """
    Splits the full layer name to retrieve the parent module and the child layer.
    """
    *parent_address, child_name = full_name.split(".")
    parent_module = access_attribute(module, parent_address)
    return parent_module, child_name


def access_attribute(obj: Any, attributes: list[str]) -> Any:
    """
    Recursively accesses a nested attribute from an object using a list of attribute names.
    """
    for attr in attributes:
        obj = getattr(obj, attr)
    return obj


def convert_to_qint_linear(
    layer: Linear, rbln_quantization: RBLNQuantizationConfig, scale_dtype: torch.dtype
) -> Linear:
    """
    Converts a standard linear layer to a quantized linear (qlinear) layer with a custom forward pass.
    """

    layer.weight = Parameter(layer.weight.to(torch.int8), requires_grad=False)
    weight_scale = Parameter(torch.ones(layer.out_features, 1, dtype=scale_dtype), requires_grad=False)
    input_scale = None

    if rbln_quantization.activations == "int8" and not rbln_quantization.dynamic:
        # Keep non-scalar shape for consistency with fp path
        input_scale = Parameter(torch.ones(1, dtype=scale_dtype), requires_grad=False)

    return QIntLinear(
        weight=layer.weight,
        bias=layer.bias,
        weight_scale=weight_scale,
        input_scale=input_scale,
        dynamic=rbln_quantization.dynamic,
    )


def convert_to_qfloat_linear(
    layer: Linear, rbln_quantization: RBLNQuantizationConfig, scale_dtype: torch.dtype
) -> Linear:
    """
    Converts a standard linear layer to a fp8 linear layer with a custom forward pass.
    """
    # assign here to free weight from the original layer
    layer.weight = Parameter(layer.weight.to(torch.float8_e4m3fn), requires_grad=False)
    weight_scale = Parameter(torch.ones(layer.out_features, 1, dtype=scale_dtype), requires_grad=False)
    input_scale = None

    if rbln_quantization.activations == "fp8":
        # Keep a non-scalar shape for input scale as well ([1]) for consistency
        input_scale = Parameter(torch.ones(1, dtype=scale_dtype), requires_grad=False)

    k_scale, v_scale = None, None
    if rbln_quantization.kv_caches == "fp8":
        k_scale = Parameter(torch.tensor(1, dtype=scale_dtype), requires_grad=False)
        v_scale = Parameter(torch.tensor(1, dtype=scale_dtype), requires_grad=False)

    return QFloatLinear(
        weight=layer.weight,
        bias=layer.bias,
        weight_scale=weight_scale,
        input_scale=input_scale,
        k_scale=k_scale,
        v_scale=v_scale,
    )


# =============================================================================
# RBLNQuantizerMixin - Common logic for RBLN quantizers
# =============================================================================


class RBLNQuantizerMixin:
    """
    Mixin class providing common logic for RBLN quantizers.

    This mixin can be combined with HuggingFace quantizer classes to create
    RBLN-compatible quantizers with the `get_quantized_model()` pattern.

    Override Points:
        - `_replace_linear_layers()`: Implement custom linear layer replacement
        - `_get_default_dtype()`: Return default dtype for the quantizer
        - `_get_exception_keywords()`: Return keywords to exclude when loading weights
    """

    # =========================================================================
    # Override points for subclasses
    # =========================================================================

    def _replace_linear_layers(
        self,
        model: torch.nn.Module,
        config: "PretrainedConfig",
        dtype: torch.dtype,
    ) -> None:
        """
        Replace linear layers with quantized versions.

        Override this method in subclasses to implement custom replacement logic.

        Args:
            model: Model to modify
            config: Model configuration
            dtype: Target dtype for scales
        """
        raise NotImplementedError("Subclasses must implement _replace_linear_layers()")

    @classmethod
    def _get_default_dtype(cls) -> torch.dtype:
        """Return default dtype for this quantizer. Override in subclasses."""
        return None  # Will use dtype from weights

    @classmethod
    def _get_exception_keywords(cls) -> List[str]:
        """Return keywords to exclude when loading weight files. Override in subclasses."""
        return []

    # =========================================================================
    # Common loading methods
    # =========================================================================

    @classmethod
    def _load_safetensor_files(
        cls,
        model_id: str,
        use_auth_token: Optional[Union[bool, str]] = None,
        revision: Optional[str] = None,
        cache_dir: Optional[str] = None,
        force_download: bool = False,
        local_files_only: bool = False,
        exception_keywords: Optional[List[str]] = None,
    ) -> List[str]:
        """Load safetensor file paths."""
        if exception_keywords is None:
            exception_keywords = cls._get_exception_keywords()

        return load_weight_files(
            model_id,
            use_auth_token=use_auth_token,
            revision=revision,
            cache_dir=cache_dir,
            force_download=force_download,
            local_files_only=local_files_only,
            exception_keywords=exception_keywords,
        )

    @classmethod
    def _build_state_dict(cls, safetensor_files: List[str]) -> Dict[str, torch.Tensor]:
        """Build state dict from safetensor files."""
        return {k: v for f in safetensor_files for k, v in load_file(f).items()}

    @classmethod
    def _load_model_config(
        cls,
        model_id: str,
        use_auth_token: Optional[Union[bool, str]] = None,
        revision: Optional[str] = None,
        cache_dir: Optional[str] = None,
        force_download: bool = False,
        local_files_only: bool = False,
        **kwargs,
    ) -> Tuple["PretrainedConfig", Dict[str, Any]]:
        """Load model config with unused kwargs."""
        return AutoConfig.from_pretrained(
            model_id,
            use_auth_token=use_auth_token,
            revision=revision,
            cache_dir=cache_dir,
            force_download=force_download,
            local_files_only=local_files_only,
            return_unused_kwargs=True,
            **kwargs,
        )

    @classmethod
    def _create_model_skeleton(
        cls,
        hf_auto_model_class: Type["_BaseAutoModelClass"],
        config: "PretrainedConfig",
        dtype: torch.dtype,
    ) -> torch.nn.Module:
        """Create empty model skeleton."""
        with no_init_weights():
            return hf_auto_model_class.from_config(config, torch_dtype=dtype)

    @classmethod
    def _load_state_dict_to_model(
        cls,
        model: torch.nn.Module,
        state_dict: Dict[str, torch.Tensor],
        strict: bool = False,
        **kwargs,
    ) -> None:
        """
        Load state dict into model.

        Override this method in subclasses to add custom pre-processing
        (e.g., key canonicalization, scale normalization).
        """
        model.load_state_dict(state_dict, strict=strict)

    # =========================================================================
    # Main entry point
    # =========================================================================

    @classmethod
    def get_quantized_model(
        cls,
        hf_auto_model_class: Type["_BaseAutoModelClass"],
        model_id: str,
        use_auth_token: Optional[Union[bool, str]] = None,
        revision: Optional[str] = None,
        cache_dir: Optional[str] = None,
        force_download: bool = False,
        local_files_only: bool = False,
        dtype: Optional[torch.dtype] = None,
        config: Optional["PretrainedConfig"] = None,
        exception_keywords: Optional[List[str]] = None,
        **kwargs,
    ) -> torch.nn.Module:
        """
        Load a quantized model.

        This is the main entry point for standalone model loading.
        The method orchestrates the loading process by calling individual methods
        that can be overridden in subclasses.

        Args:
            hf_auto_model_class: HuggingFace AutoModel class
            model_id: Model ID or path
            use_auth_token: HuggingFace auth token
            revision: Model revision
            cache_dir: Cache directory
            force_download: Force download
            local_files_only: Use only local files
            dtype: Model dtype (if None, determined from weights or _get_default_dtype)
            config: Pre-loaded config (optional)
            exception_keywords: Keywords to exclude when loading weight files
            **kwargs: Additional arguments

        Returns:
            Quantized model
        """
        # Handle deprecated torch_dtype
        torch_dtype = kwargs.pop("torch_dtype", None)
        if torch_dtype is not None:
            logger.warning(
                "torch_dtype is deprecated. Use dtype instead. "
                "The value will be ignored."
            )

        # Step 1: Load weight files
        safetensor_files = cls._load_safetensor_files(
            model_id,
            use_auth_token=use_auth_token,
            revision=revision,
            cache_dir=cache_dir,
            force_download=force_download,
            local_files_only=local_files_only,
            exception_keywords=exception_keywords,
        )

        # Step 2: Build state dict
        state_dict = cls._build_state_dict(safetensor_files)

        # Step 3: Determine dtype
        if dtype is None:
            default_dtype = cls._get_default_dtype()
            if default_dtype is not None:
                dtype = default_dtype
            else:
                # Get dtype from weights
                dtype = get_state_dict_dtype(state_dict)

        # Step 4: Load config
        if config is None:
            # Handle n_layer_keys
            n_layer_keys = ["num_hidden_layers", "n_layers"]
            for n_layer_key in n_layer_keys:
                if n_layer_key in kwargs and kwargs[n_layer_key] is None:
                    kwargs.pop(n_layer_key)

            config, kwargs = cls._load_model_config(
                model_id,
                use_auth_token=use_auth_token,
                revision=revision,
                cache_dir=cache_dir,
                force_download=force_download,
                local_files_only=local_files_only,
                **kwargs,
            )

        # Step 5: Create model skeleton
        model = cls._create_model_skeleton(hf_auto_model_class, config, dtype)

        # Step 6: Replace linear layers (override point)
        cls._replace_linear_layers_cls(cls, model, config, dtype, kwargs)

        # Step 7: Load weights (subclasses can override for custom processing)
        cls._load_state_dict_to_model(model, state_dict, strict=False, config=config, **kwargs)

        return model

    @classmethod
    def _replace_linear_layers_cls(
        cls,
        quantizer_cls,
        model: torch.nn.Module,
        config: "PretrainedConfig",
        dtype: torch.dtype,
        kwargs: Dict[str, Any],
    ) -> None:
        """
        Class method wrapper for _replace_linear_layers.
        Override in subclasses that need class-level access.
        """
        # Default: create instance and call instance method
        # Subclasses may override this for different behavior
        pass  # Will be implemented by subclasses


# =============================================================================
# RBLNQuantizer - HuggingFace HfQuantizer compatible class
# =============================================================================

try:
    from transformers.quantizers.base import HfQuantizer

    _HF_QUANTIZER_AVAILABLE = True
except ImportError:
    _HF_QUANTIZER_AVAILABLE = False
    HfQuantizer = object  # Fallback for older transformers versions


class RBLNQuantizer(RBLNQuantizerMixin, HfQuantizer if _HF_QUANTIZER_AVAILABLE else object):
    """
    RBLN Quantizer compatible with HuggingFace's HfQuantizer interface.

    This class provides:
    1. HuggingFace pipeline integration (when HfQuantizer is available)
    2. Standalone model loading via `get_quantized_model()` class method
    3. Backward compatibility with existing RBLN quantization code

    Example:
        # Standalone usage (works without HfQuantizer)
        model = RBLNQuantizer.get_quantized_model(
            AutoModelForCausalLM,
            "model_id",
            rbln_quantization=RBLNQuantizationConfig(weights="int8", activations="int8"),
        )
    """

    requires_calibration = True  # Requires pre-quantized checkpoint

    def __init__(
        self,
        quantization_config: RBLNQuantizationConfig,
        **kwargs,
    ):
        """
        Initialize the RBLNQuantizer.

        Args:
            quantization_config: RBLN quantization configuration
            **kwargs: Additional arguments (pre_quantized, etc.)
        """
        self.quantization_config = quantization_config
        self.pre_quantized = kwargs.pop("pre_quantized", True)
        self._layer_factory = QuantizedLayerFactory(quantization_config)

        if _HF_QUANTIZER_AVAILABLE and not self.pre_quantized and self.requires_calibration:
            raise ValueError(
                "RBLN quantization requires pre-quantized checkpoints. "
                "Please use `pre_quantized=True` or provide a quantized checkpoint."
            )

    # =========================================================================
    # HfQuantizer interface implementation
    # =========================================================================

    def is_serializable(self):
        """Whether the quantized model can be serialized."""
        return True

    @property
    def is_trainable(self):
        """Whether the quantized model can be trained."""
        return False  # RBLN quantization is inference-only

    def validate_environment(self, *args, **kwargs):
        """Validate the environment for RBLN quantization."""
        try:
            import rebel  # noqa: F401
        except ImportError:
            logger.warning(
                "RBLN SDK (rebel) is not installed. "
                "Some features may not work correctly."
            )

    def _process_model_before_weight_loading(
        self,
        model: torch.nn.Module,
        **kwargs,
    ) -> torch.nn.Module:
        """
        Process model before weight loading: convert Linear -> QLinear.

        This method is called by HuggingFace's from_pretrained pipeline.
        """
        scale_dtype = kwargs.get("scale_dtype", model.dtype)
        self._update_layers_to_quantize(model, scale_dtype)
        return model

    def _process_model_after_weight_loading(
        self,
        model: torch.nn.Module,
        **kwargs,
    ) -> torch.nn.Module:
        """
        Process model after weight loading: validation.

        This method is called by HuggingFace's from_pretrained pipeline.
        """
        # Validation is done during weight loading in _load_weights_from_files
        return model

    # =========================================================================
    # Core quantization methods
    # =========================================================================

    def _update_layers_to_quantize(
        self,
        module: torch.nn.Module,
        scale_dtype: torch.dtype,
    ) -> None:
        """
        Update linear layers to quantized layers.

        Uses `quantization_config.target_modules` to determine which layers to quantize.
        """
        processed_layers = []

        for name, layer in module.named_modules():
            if self._is_target_for_replacement(name, layer):
                parent_module, layer_name = get_parent_and_child(module, name)
                setattr(
                    parent_module,
                    layer_name,
                    self._layer_factory.create_linear(layer, scale_dtype),
                )
                processed_layers.append(name)

        if processed_layers:
            logger.debug(
                f"Updated the following linear layers to quantized layers:\n"
                f" {{{', '.join(processed_layers)}}}"
            )

    def _is_target_for_replacement(
        self,
        layer_name: str,
        layer: torch.nn.Module,
    ) -> bool:
        """
        Check if a layer should be replaced with quantized version.

        Uses `quantization_config.target_modules` instead of hardcoded constant.
        """
        module_name = layer_name.split(".")[-1]
        return (
            module_name in self.quantization_config.target_modules
            and isinstance(layer, torch.nn.Linear)
        )

    def _load_weights_from_files(
        self,
        model: torch.nn.Module,
        safetensors: List[Dict[str, torch.Tensor]],
    ) -> None:
        """
        Load weights from safetensor files into the model.

        This method handles key canonicalization and scale shape normalization.
        """
        config = self.quantization_config
        model_params = dict(model.named_parameters(recurse=True))
        model_buffers = dict(model.named_buffers(recurse=True))

        unloaded_keys = []
        loaded_input_scale = False
        loaded_kv_scale = False
        loaded_weight_scale = False

        for safetensor in safetensors:
            # Normalize all (key, tensor) pairs
            normalized_items = canonicalize_checkpoint_items(
                model=model,
                items=safetensor.items(),
                rbln_quantization=config,
            )

            for key, value in normalized_items:
                # Track which types of scales were observed
                if key.endswith("input_scale"):
                    loaded_input_scale = True
                if key.endswith("weight_scale"):
                    loaded_weight_scale = True
                if key.endswith("k_scale") or key.endswith("v_scale"):
                    loaded_kv_scale = True

                # Copy into parameters or buffers
                if key in model_params:
                    if model_params[key].dtype != value.dtype:
                        value = value.to(model_params[key].dtype)
                    model_params[key].data.copy_(value)
                elif key in model_buffers:
                    if model_buffers[key].dtype != value.dtype:
                        value = value.to(model_buffers[key].dtype)
                    model_buffers[key].data.copy_(value)
                else:
                    unloaded_keys.append(key)

        # Warnings and errors
        if unloaded_keys:
            logger.warning(
                f"There are unexpected parameters/buffers on the checkpoint: {unloaded_keys}"
            )

        self._validate_loaded_scales(
            loaded_input_scale,
            loaded_weight_scale,
            loaded_kv_scale,
        )

    def _validate_loaded_scales(
        self,
        loaded_input_scale: bool,
        loaded_weight_scale: bool,
        loaded_kv_scale: bool,
    ) -> None:
        """Validate that required scales were loaded based on config."""
        config = self.quantization_config

        if not loaded_input_scale and config.activations == "fp8":
            raise ValueError(
                "No input_scale found in the checkpoint. Did you use the correct quantization config? "
                "If you are using fp8 quantization, you need to use the correct quantization config."
            )
        if not loaded_weight_scale and config.weights == "fp8":
            raise ValueError(
                "No weight_scale found in the checkpoint. Did you use the correct quantization config? "
                "If you are using fp8 quantization, you need to use the correct quantization config."
            )
        if not loaded_kv_scale and config.kv_caches == "fp8":
            raise ValueError(
                "No kv_scale found in the checkpoint. Did you use the correct quantization config? "
                "If you are using fp8 quantization, you need to use the correct quantization config."
            )
        if loaded_kv_scale and config.kv_caches != "fp8":
            logger.warning(
                "kv_scale found in the checkpoint, but kv_caches of quantization config is not fp8. "
                "Ignoring kv_scale."
            )

    # =========================================================================
    # RBLNQuantizerMixin override points
    # =========================================================================

    @classmethod
    def _replace_linear_layers_cls(
        cls,
        quantizer_cls,
        model: torch.nn.Module,
        config: "PretrainedConfig",
        dtype: torch.dtype,
        kwargs: Dict[str, Any],
    ) -> None:
        """Replace Linear layers with QIntLinear/QFloatLinear."""
        rbln_quantization = kwargs.get("rbln_quantization")
        if rbln_quantization is None:
            raise ValueError("rbln_quantization is required for RBLNQuantizer")

        quantizer = cls(rbln_quantization, pre_quantized=True)
        quantizer._update_layers_to_quantize(model, dtype)

    @classmethod
    def _load_state_dict_to_model(
        cls,
        model: torch.nn.Module,
        state_dict: Dict[str, torch.Tensor],
        strict: bool = False,
        **kwargs,
    ) -> None:
        """
        Load state dict with RBLN-specific canonicalization.

        This method handles key canonicalization and scale normalization
        for various checkpoint formats.
        """
        rbln_quantization = kwargs.get("rbln_quantization")
        if rbln_quantization is None:
            # Fallback to simple loading
            model.load_state_dict(state_dict, strict=strict)
            return

        quantizer = cls(rbln_quantization, pre_quantized=True)

        # Convert state_dict to safetensors-like format for _load_weights_from_files
        safetensors = [state_dict]
        quantizer._load_weights_from_files(model, safetensors)

    # =========================================================================
    # Standalone model loading (backward compatible)
    # =========================================================================

    @classmethod
    def get_quantized_model(
        cls,
        hf_auto_model_class: Type["_BaseAutoModelClass"],
        model_id: str,
        use_auth_token: Optional[Union[bool, str]] = None,
        revision: Optional[str] = None,
        cache_dir: Optional[str] = None,
        force_download: bool = False,
        local_files_only: bool = False,
        rbln_quantization: Optional[RBLNQuantizationConfig] = None,
        **kwargs,
    ):
        """
        Load a quantized model using RBLN quantization.

        This is the main entry point for standalone model loading (outside HuggingFace pipeline).
        Maintains backward compatibility with the old `get_quantized_model()` function.

        Args:
            hf_auto_model_class: HuggingFace AutoModel class
            model_id: Model ID or path
            use_auth_token: HuggingFace auth token
            revision: Model revision
            cache_dir: Cache directory
            force_download: Force download
            local_files_only: Use only local files
            rbln_quantization: RBLN quantization configuration
            **kwargs: Additional arguments passed to AutoConfig.from_pretrained

        Returns:
            Quantized model
        """
        # Pass rbln_quantization through kwargs for _replace_linear_layers_cls
        kwargs["rbln_quantization"] = rbln_quantization

        # Use parent's get_quantized_model (calls our overridden methods via MRO)
        return super().get_quantized_model(
            hf_auto_model_class=hf_auto_model_class,
            model_id=model_id,
            use_auth_token=use_auth_token,
            revision=revision,
            cache_dir=cache_dir,
            force_download=force_download,
            local_files_only=local_files_only,
            **kwargs,
        )
