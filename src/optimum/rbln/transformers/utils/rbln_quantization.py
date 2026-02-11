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
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Iterable,
    List,
    Optional,
    Tuple,
    Type,
    Union,
)

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


# Constants
QUANTIZED_WEIGHTS = {
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
}

# Common alias sets seen in community checkpoints
VARIANT_ALIASES: Dict[str, List[str]] = {
    "weight_scale": ["weight_scale", "scales", "w_scale", "scale"],
    "input_scale": ["input_scale", "act_scale", "activation_scale", "a_scale"],
    "kv_scale": ["kv_scale", "kv_scales"],
    "k_scale": ["k_scale", "k_scales"],
    "v_scale": ["v_scale", "v_scales"],
}


class RBLNQuantizationConfig(RBLNSerializableConfigProtocol):
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

        self._validate()

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
    """
    # torch_dtype should not be passed to AutoConfig.from_pretrained
    # since it doesn't support 'auto'
    if "torch_dtype" in kwargs:
        if "dtype" in kwargs:
            raise ValueError("torch_dtype and dtype cannot be set at the same time.")
        logger.warning_once("torch_dtype is deprecated. Use dtype instead.")
        dtype = kwargs.pop("torch_dtype")
    else:
        dtype = kwargs.pop("torch_dtype", None)
    if dtype is not None:
        logger.warning_once(
            "dtype is not supported for quantized models. "
            "It will be ignored and the dtype of the model will be determined by the weights."
        )
        dtype = None

    # get paths of safetensors files in the model repo
    safetensor_files = load_weight_files(
        model_id,
        use_auth_token=use_auth_token,
        revision=revision,
        cache_dir=cache_dir,
        force_download=force_download,
        local_files_only=local_files_only,
    )

    # load safetensors files into memory
    safetensors = [load_file(safetensor_file) for safetensor_file in safetensor_files]

    # get the dtype of the model from the first safetensor file
    dtype = get_state_dict_dtype(safetensors[0])

    # FIXME :: Currently rebel compiler does not support bfloat16 for quantized models.
    # So we use float32 instead.
    dtype = torch.float32

    # remove n_layer_keys from kwargs if they are None.
    # otherwise AutoConfig.from_pretrained will raise an error.
    n_layer_keys = ["num_hidden_layers", "n_layers"]
    for n_layer_key in n_layer_keys:
        if n_layer_key in kwargs:
            if kwargs[n_layer_key] is None:
                kwargs.pop(n_layer_key)

    config = AutoConfig.from_pretrained(
        model_id,
        use_auth_token=use_auth_token,
        revision=revision,
        cache_dir=cache_dir,
        force_download=force_download,
        local_files_only=local_files_only,
        **kwargs,
    )

    with no_init_weights():
        model = hf_auto_model_class.from_config(config, dtype=dtype)

    # Quantize the model
    update_layers_to_quantize(model, model.dtype, rbln_quantization)

    # Load weights into the model
    load_weights_from_files(model, safetensors, rbln_quantization)

    return model


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


def update_layers_to_quantize(
    module: torch.nn.Module,
    scale_dtype: torch.dtype,
    rbln_quantization: Optional[RBLNQuantizationConfig] = None,
) -> None:
    """
    Updates specified linear layers to quantized (qlinear) layers in the given module.
    """

    processed_layers = []
    quantized_layer_factory = QuantizedLayerFactory(rbln_quantization)
    for name, layer in module.named_modules():
        if is_target_for_qlinear_replacement(name, layer):
            parent_module, layer_name = get_parent_and_child(module, name)
            setattr(
                parent_module,
                layer_name,
                quantized_layer_factory.create_linear(layer, scale_dtype),
            )
            processed_layers.append(name)

    # Register k_scale/v_scale on self_attn modules for fp8 kv caches
    if rbln_quantization.kv_caches == "fp8":
        for name, layer in module.named_modules():
            if is_target_for_adding_kv_scales(name):
                layer.k_scale = Parameter(torch.tensor(1, dtype=scale_dtype), requires_grad=False)
                layer.v_scale = Parameter(torch.tensor(1, dtype=scale_dtype), requires_grad=False)

    if processed_layers:
        logger.debug(f"Updated the following linear layers to quantized layers:\n {{{', '.join(processed_layers)}}}")


def _last_segment(key: str) -> str:
    parts = key.split(".")
    return parts[-1]


def _replace_last_with(key: str, new_tail: str) -> str:
    parts = key.split(".")
    return ".".join(parts[:-1] + new_tail.split("."))


def _matches_any_alias(key: str, kind: str) -> bool:
    tail = _last_segment(key)
    return tail in VARIANT_ALIASES.get(kind, [])


def _reduce_to_scalar(t: torch.Tensor) -> torch.Tensor:
    if t.ndim == 0:
        return t
    return t.reshape(-1).amax()


def _1d_value_as_scalar(value: torch.Tensor) -> torch.Tensor:
    if value.ndim == 1:
        return value.reshape(-1).amax()
    else:
        raise ValueError(f"Expected 1D tensor, got {value.ndim}D tensor")


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
    # We produce keys with 'k_scale' and 'v_scale' at the self_attn level
    if tensor.ndim == 1 and tensor.numel() >= 2:
        tk, tv = tensor[0], tensor[1]
    elif tensor.ndim == 2 and tensor.shape[0] >= 2 and tensor.shape[1] == 1:
        tk, tv = tensor[0, 0], tensor[1, 0]
    else:
        tk = tv = tensor
    k_key = _replace_last_with(base_key, "k_scale")
    v_key = _replace_last_with(base_key, "v_scale")
    return [(k_key, tk), (v_key, tv)]


def canonicalize_checkpoint_items(
    model: torch.nn.Module,
    items: Iterable[Tuple[str, torch.Tensor]],
    rbln_quantization: Optional[RBLNQuantizationConfig],
) -> List[Tuple[str, torch.Tensor]]:
    params = dict(model.named_parameters(recurse=True))
    results: List[Tuple[str, torch.Tensor]] = []

    for key, value in items:
        t = value
        # Normalize weight scale variants
        if _matches_any_alias(key, "weight_scale"):
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
        if _matches_any_alias(key, "input_scale"):
            target_key = _replace_last_with(key, "input_scale")
            t = _scalar_value_as_1d(t)
            results.append((target_key, t))
            continue

        # KV scale handling
        if _matches_any_alias(key, "kv_scale"):
            # For quark-like formats, expand to k/v
            kv_items = _kv_split_items(key, t)
            for k2, v2 in kv_items:
                if v2.ndim == 0:
                    pass
                else:
                    v2 = _1d_value_as_scalar(v2)
                results.append((k2, v2))
            continue

        if _matches_any_alias(key, "k_scale") or _matches_any_alias(key, "v_scale"):
            canonical_name = "k_scale" if _matches_any_alias(key, "k_scale") else "v_scale"
            parts = key.split(".")
            # If parent is a projection layer (e.g., k_proj, v_proj), move scale up to self_attn level
            if len(parts) >= 2 and parts[-2] in ("k_proj", "v_proj"):
                target_key = ".".join(parts[:-2] + [canonical_name])
            else:
                target_key = _replace_last_with(key, canonical_name)

            if t.ndim == 0:
                pass
            else:
                t = _1d_value_as_scalar(t)
            results.append((target_key, t))
            continue

        # Default: passthrough
        results.append((key, t))

    return results


def load_weights_from_files(
    model: torch.nn.Module,
    safetensors: List[Dict[str, torch.Tensor]],
    rbln_quantization: Optional[RBLNQuantizationConfig] = None,
):
    """
    Load safetensor file data directly into the model from provided safetensor files.
    """

    model_params = dict(model.named_parameters(recurse=True))
    model_buffers = dict(model.named_buffers(recurse=True))

    unloaded_keys = []
    loaded_input_scale = False
    loaded_kv_scale = False
    loaded_weight_scale = False

    for safetensor in safetensors:
        # Normalize all (key, tensor) pairs to the internal schema
        normalized_items = canonicalize_checkpoint_items(
            model=model,
            items=safetensor.items(),
            rbln_quantization=rbln_quantization,
        )

        for key, value in normalized_items:
            # Track which types of scales were observed (post-normalization)
            if key.endswith("input_scale"):
                loaded_input_scale = True
            if key.endswith("weight_scale"):
                loaded_weight_scale = True
            if key.endswith("k_scale") or key.endswith("v_scale"):
                loaded_kv_scale = True

            # Copy into parameters or buffers
            if key in model_params:
                # Ensure dtype compatibility
                if model_params[key].dtype != value.dtype:
                    value = value.to(model_params[key].dtype)
                model_params[key].data.copy_(value)
            elif key in model_buffers:
                if model_buffers[key].dtype != value.dtype:
                    value = value.to(model_buffers[key].dtype)
                model_buffers[key].data.copy_(value)
            else:
                unloaded_keys.append(key)

    if len(unloaded_keys) > 0:
        logger.warning(f"There are unexpected parameters/buffers on the checkpoint: {unloaded_keys}")
    if not loaded_input_scale and rbln_quantization.activations == "fp8":
        raise ValueError(
            "No input_scale found in the checkpoint. Did you use the correct quantization config? "
            "If you are using fp8 quantization, you need to use the correct quantization config."
        )
    if not loaded_weight_scale and rbln_quantization.weights == "fp8":
        raise ValueError(
            "No weight_scale found in the checkpoint. Did you use the correct quantization config? "
            "If you are using fp8 quantization, you need to use the correct quantization config."
        )
    if not loaded_kv_scale and rbln_quantization.kv_caches == "fp8":
        raise ValueError(
            "No kv_scale found in the checkpoint. Did you use the correct quantization config? "
            "If you are using fp8 quantization, you need to use the correct quantization config."
        )
    if loaded_kv_scale and rbln_quantization.kv_caches != "fp8":
        logger.warning(
            "kv_scale found in the checkpoint, but kv_caches of quantization config is not fp8. Ignoring kv_scale."
        )


def is_target_for_qlinear_replacement(layer_name: str, layer: torch.nn.Module) -> bool:
    """
    Checks if a layer is a target for qlinear replacement.
    """
    return layer_name.split(".")[-1] in QUANTIZED_WEIGHTS and isinstance(layer, torch.nn.Linear)


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

    return QFloatLinear(
        weight=layer.weight,
        bias=layer.bias,
        weight_scale=weight_scale,
        input_scale=input_scale,
    )
