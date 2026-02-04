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

from __future__ import annotations

import importlib
import inspect
import json
from pathlib import Path
from typing import Any, ClassVar, get_args, get_origin

import numpy as np
import torch
from pydantic import (
    AliasChoices,
    BaseModel,
    ConfigDict,
    Field,
    PrivateAttr,
    field_validator,
    model_validator,
)

from .__version__ import __version__
from .utils.deprecation import warn_deprecated_npu
from .utils.logging import get_logger
from .utils.runtime_utils import ContextRblnConfig


logger = get_logger(__name__)


DEFAULT_COMPILED_MODEL_NAME = "compiled_model"
RUNTIME_KEYWORDS = ["create_runtimes", "device", "device_map", "activate_profiler", "timeout"]
CONFIG_MAPPING: dict[str, type["RBLNModelConfig"]] = {}


def normalize_dtype(dtype: str | torch.dtype | np.dtype) -> str:
    """
    Convert framework-specific dtype to string representation.
    e.g., torch.float32 -> "float32"

    Args:
        dtype: The input dtype (can be string, torch dtype, or numpy dtype).

    Returns:
        The normalized string representation of the dtype.
    """
    if isinstance(dtype, str):
        return dtype
    else:
        dtype_str: str = repr(dtype).split(".")[-1]
        if dtype_str.endswith("'>"):  # numpy
            dtype_str = dtype_str[:-2]
        return dtype_str


def nested_update(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """
    Recursively merge override dict into base dict.
    For nested dicts, values are merged recursively instead of being replaced.
    For non-dict values, override takes precedence.
    Args:
        base: The base dictionary to merge into (modified in-place).
        override: The dictionary with values to merge.
    Returns:
        The merged base dictionary.
    Example:
        >>> base = {"a": 1, "nested": {"x": 10, "y": 20}}
        >>> override = {"b": 2, "nested": {"y": 30, "z": 40}}
        >>> nested_update(base, override)
        {"a": 1, "b": 2, "nested": {"x": 10, "y": 30, "z": 40}}
    """
    for key, value in override.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            nested_update(base[key], value)
        else:
            base[key] = value
    return base


def _is_rbln_config_type(annotation: Any) -> bool:
    """Check if the annotation is or contains an RBLNModelConfig type."""
    if annotation is None:
        return False

    origin = get_origin(annotation)
    if origin is not None:
        # Handle Union, Optional, etc.
        args = get_args(annotation)
        return any(_is_rbln_config_type(arg) for arg in args)

    # Check if it's a class and subclass of RBLNModelConfig
    try:
        if isinstance(annotation, type) and issubclass(annotation, RBLNModelConfig):
            return True
    except TypeError:
        pass

    # Check string annotation
    if isinstance(annotation, str) and "RBLNModelConfig" in annotation:
        return True

    return False


def get_rbln_config_class(rbln_config_class_name: str) -> type["RBLNModelConfig"]:
    """Get config class by name from registry or optimum.rbln module."""
    cls = getattr(importlib.import_module("optimum.rbln"), rbln_config_class_name, None)
    if cls is None:
        if rbln_config_class_name in CONFIG_MAPPING:
            cls = CONFIG_MAPPING[rbln_config_class_name]
        else:
            raise ValueError(f"Configuration for {rbln_config_class_name} not found.")
    return cls


def load_config(path: str) -> tuple[type["RBLNModelConfig"], dict[str, Any]]:
    """Load config JSON file and return the config class and data."""
    path = Path(path)
    if path.is_dir():
        path = path / "rbln_config.json"

    with open(path, "r") as jsonf:
        config_file = json.load(jsonf)

    if "_meta" in config_file:
        raise RuntimeError(f"`{path}` is an old version. Please recompile the model to get the latest config file.")

    cls_name = config_file["cls_name"]
    cls = get_rbln_config_class(cls_name)
    return cls, config_file


class RBLNCompileConfig(BaseModel):
    """
    Configuration for RBLN compilation.

    Attributes:
        compiled_model_name (str): Name of the compiled model.
        input_info (list): Information about input tensors.
        npu (str | None): NPU configuration.
        tensor_parallel_size (int | None): Size for tensor parallelism.
    """

    model_config = ConfigDict(frozen=False, extra="forbid", validate_assignment=True)

    compiled_model_name: str = DEFAULT_COMPILED_MODEL_NAME
    input_info: list[tuple[str, tuple[int, ...], str]] | list[list[tuple[str, tuple[int, ...], str]]] | None = None
    npu: str | None = None
    tensor_parallel_size: int | None = None

    @field_validator("input_info", mode="before")
    @classmethod
    def normalize_input_info(cls, v: Any) -> Any:
        """Normalize input_info by converting dtypes to strings."""
        if v is None:
            return v

        def normalize_single_info(info: list) -> list[tuple[str, tuple[int, ...], str]]:
            return [(i[0], tuple(i[1]), normalize_dtype(i[2]) or "float32") for i in info]

        def is_valid_input_info(input_info: Any) -> bool:
            if not isinstance(input_info, list):
                return False
            return all(
                isinstance(item, (tuple, list))
                and len(item) == 3
                and isinstance(item[0], str)
                and isinstance(item[1], (tuple, list))
                and all(isinstance(x, int) for x in item[1])
                and (isinstance(item[2], (str, torch.dtype)))
                for item in input_info
            )

        # Check if this is multiple input_info (list of lists)
        if isinstance(v, list) and all(is_valid_input_info(info) for info in v):
            return [normalize_single_info(info) for info in v]
        else:
            return normalize_single_info(v)

    @property
    def is_multiple_input_info(self) -> bool:
        """Check if input_info contains multiple input specifications."""

        def is_valid_input_info(input_info: Any) -> bool:
            if not isinstance(input_info, list):
                return False
            return all(
                isinstance(item, (tuple, list))
                and len(item) == 3
                and isinstance(item[0], str)
                and isinstance(item[1], (tuple, list))
                and all(isinstance(x, int) for x in item[1])
                and isinstance(item[2], str)
                for item in input_info
            )

        if isinstance(self.input_info, list):
            return all(is_valid_input_info(info) for info in self.input_info)
        return False

    def get_dummy_inputs(
        self,
        fill: int = 0,
        static_tensors: dict[str, torch.Tensor] | None = None,
        meta_tensor_names: list[str] | None = None,
    ) -> tuple[torch.Tensor, ...]:
        """Generate dummy inputs for compilation."""
        dummy = []
        static_tensors = static_tensors if static_tensors is not None else {}
        meta_tensor_names = meta_tensor_names if meta_tensor_names is not None else []
        for name, shape, dtype in self.input_info:
            if name in static_tensors:
                tensor = static_tensors[name]
                if list(shape) != list(tensor.shape):
                    raise RuntimeError(f"Different shape for dummy inputs. ({shape} != {list(tensor.shape)})")
                if getattr(torch, dtype) != tensor.dtype:
                    raise RuntimeError(f"Different dtype for dummy inputs ({dtype} != {tensor.dtype})")
                dummy.append(tensor)
            else:
                device = "meta" if name in meta_tensor_names else "cpu"
                if len(shape) > 0:
                    dummy.append(
                        torch.fill(torch.empty(*shape, dtype=getattr(torch, dtype), device=torch.device(device)), fill)
                    )
                else:
                    dummy.append(torch.tensor(fill, dtype=getattr(torch, dtype), device=torch.device(device)))
        return tuple(dummy)


class RBLNModelConfig(BaseModel):
    """Base configuration class for RBLN models that handles compilation settings, runtime options, and submodules.

    This class provides functionality for:

    1. Managing compilation configurations for RBLN devices
    2. Configuring runtime behavior such as device placement
    3. Handling nested configuration objects for complex model architectures
    4. Serializing and deserializing configurations

    Examples:
        Using with RBLNModel.from_pretrained():
        ```python
        from optimum.rbln import RBLNResNetForImageClassification

        # Method 1: Using rbln_ prefixed arguments (recommended for simple cases)
        model = RBLNResNetForImageClassification.from_pretrained(
            "model_id",
            export=True,  # Compile the model
            rbln_image_size=224,
            rbln_batch_size=16,
            rbln_create_runtimes=True,
            rbln_device=0
        )

        # Method 2: Using a config dictionary
        rbln_config_dict = {
            "image_size": 224,
            "batch_size": 16,
            "create_runtimes": True
        }
        model = RBLNResNetForImageClassification.from_pretrained(
            "model_id",
            export=True,
            rbln_config=rbln_config_dict
        )

        # Method 3: Using a RBLNModelConfig instance
        from optimum.rbln import RBLNResNetForImageClassificationConfig

        config = RBLNResNetForImageClassificationConfig(
            image_size=224,
            batch_size=16,
            create_runtimes=True
        )

        model = RBLNResNetForImageClassification.from_pretrained(
            "model_id",
            export=True,
            rbln_config=config
        )
        ```
    """

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        arbitrary_types_allowed=True,
        populate_by_name=True,
    )

    # Class-level attributes (not instance fields)
    supports_tp: ClassVar[bool] = False
    supports_quantization: ClassVar[bool] = False
    submodules: ClassVar[list[str]] = []
    submodule_config_classes: ClassVar[dict[str, str]] = {}
    _allow_no_compile_cfgs: ClassVar[bool] = False

    # Serialized fields
    cls_name: str = Field(default="")
    optimum_rbln_version: str = Field(default=__version__)
    rbln_dtype: str = Field(
        default="float32",
        serialization_alias="dtype",
        validation_alias=AliasChoices("dtype", "rbln_dtype", "_torch_dtype"),
    )
    rbln_compile_cfgs: list[RBLNCompileConfig] = Field(
        default_factory=list,
        serialization_alias="compile_cfgs",
        validation_alias=AliasChoices("compile_cfgs", "rbln_compile_cfgs", "_compile_cfgs"),
    )

    # Non-serialized runtime fields (excluded from serialization)
    _create_runtimes: bool | None = PrivateAttr(default=None)
    _device: int | list[int] | None = PrivateAttr(default=None)
    _device_map: dict[str, int | list[int]] | None = PrivateAttr(default=None)
    _activate_profiler: bool | None = PrivateAttr(default=None)
    _timeout: int | None = PrivateAttr(default=None)
    npu: str | None = Field(default=None, exclude=True)
    tensor_parallel_size: int | None = Field(default=None, exclude=True)

    # Internal state
    _frozen: bool = PrivateAttr(default=False)

    def __init__(self, **data: Any):
        # Handle deprecated _torch_dtype
        if "_torch_dtype" in data:
            logger.warning_once("`_torch_dtype` is deprecated. Use `dtype` instead.")
            if isinstance(data["_torch_dtype"], torch.dtype):
                data["dtype"] = normalize_dtype(data["_torch_dtype"])
            else:
                data["dtype"] = data.pop("_torch_dtype")

        # Handle torch.dtype for dtype field
        if "dtype" in data and isinstance(data["dtype"], torch.dtype):
            data["dtype"] = normalize_dtype(data["dtype"])

        # Extract runtime options before validation
        create_runtimes = data.pop("create_runtimes", None)
        device = data.pop("device", None)
        device_map = data.pop("device_map", None)
        activate_profiler = data.pop("activate_profiler", None)
        timeout = data.pop("timeout", None)

        # Handle deprecated optimize_host_memory
        if "optimize_host_memory" in data:
            data.pop("optimize_host_memory")
            logger.warning("`optimize_host_memory` is deprecated and will be removed in future versions.")

        # Set default cls_name if not provided
        if not data.get("cls_name"):
            data["cls_name"] = self.__class__.__name__

        # Convert compile_cfgs dicts to RBLNCompileConfig objects
        compile_cfgs = data.get("rbln_compile_cfgs") or data.get("compile_cfgs") or data.get("_compile_cfgs") or []
        if compile_cfgs and isinstance(compile_cfgs, list) and len(compile_cfgs) > 0:
            if isinstance(compile_cfgs[0], dict):
                compile_cfgs = [RBLNCompileConfig(**cfg) for cfg in compile_cfgs]
            data["rbln_compile_cfgs"] = compile_cfgs
            data.pop("compile_cfgs", None)
            data.pop("_compile_cfgs", None)

        super().__init__(**data)

        # Set runtime options as private attributes
        self._create_runtimes = create_runtimes
        self._device = device
        self._device_map = device_map
        self._activate_profiler = activate_profiler
        self._timeout = timeout

    @model_validator(mode="before")
    @classmethod
    def handle_submodule_dicts(cls, data: dict[str, Any]) -> dict[str, Any]:
        """Convert submodule dicts to config instances.

        This validator handles the conversion of submodule configuration dictionaries
        to their corresponding RBLNModelConfig instances. It determines the config class
        from cls_name or submodule_config_classes mapping.

        Runtime options (npu, tensor_parallel_size, optimum_rbln_version) are inherited
        from the parent config to the submodule config.
        """
        if not isinstance(data, dict):
            return data

        for submodule_name in cls.submodules:
            if submodule_name not in data:
                continue

            submodule_data = data[submodule_name]

            # Skip if already a config instance
            if isinstance(submodule_data, RBLNModelConfig):
                continue

            # Skip if not a dict
            if not isinstance(submodule_data, dict):
                continue

            # Determine config class name with priority: cls_name > submodule_config_classes
            if "cls_name" in submodule_data:
                cls_name = submodule_data["cls_name"]
            elif submodule_name in cls.submodule_config_classes:
                cls_name = cls.submodule_config_classes[submodule_name]
            else:
                # Cannot determine class, leave as dict for initialize_submodule_config to handle
                continue

            # Inherit runtime options from parent
            for key in ["npu", "tensor_parallel_size", "optimum_rbln_version"]:
                if key in data and key not in submodule_data:
                    submodule_data[key] = data[key]

            # Inherit runtime options that are stored differently
            for key in ["create_runtimes", "device", "device_map", "activate_profiler", "timeout"]:
                if key in data and key not in submodule_data:
                    submodule_data[key] = data[key]

            config_cls = get_rbln_config_class(cls_name)
            data[submodule_name] = config_cls(**submodule_data)

        return data

    @model_validator(mode="after")
    def check_version_and_extra_kwargs(self) -> "RBLNModelConfig":
        """Check for version mismatch if optimum_rbln_version differs from current."""
        return self

    def __setattr__(self, name: str, value: Any) -> None:
        """Override setattr to prevent modifications when frozen."""
        if name.startswith("_"):
            # Private attributes can always be set
            super().__setattr__(name, value)
            return

        if hasattr(self, "_frozen") and self._frozen:
            current_value = getattr(self, name, None)
            if current_value != value:
                raise RuntimeError(
                    f"`{self.__class__.__name__}` is frozen. Cannot update or set attribute after freezing."
                )
        super().__setattr__(name, value)

    @classmethod
    def _get_submodule_names(cls) -> list[str]:
        """Get list of submodule field names by checking the class's submodules class variable."""
        return cls.submodules

    def _get_runtime_options(self) -> dict[str, Any]:
        """Get all runtime options as a dictionary."""
        return {
            "create_runtimes": self._create_runtimes,
            "device": self._device,
            "device_map": self._device_map,
            "activate_profiler": self._activate_profiler,
            "timeout": self._timeout,
        }

    @property
    def _runtime_options(self) -> dict[str, Any]:
        """Get all runtime options as a dictionary (for backward compatibility)."""
        return self._get_runtime_options()

    @property
    def torch_dtype(self) -> torch.dtype:
        """Get dtype as torch.dtype (deprecated, use dtype property instead)."""
        logger.warning_once("`torch_dtype` is deprecated. Use `dtype` instead.")
        return self.dtype

    @torch_dtype.setter
    def torch_dtype(self, value: str | torch.dtype) -> None:
        """Set dtype (deprecated, use dtype property instead)."""
        logger.warning_once("`torch_dtype` is deprecated. Use `dtype` instead.")
        if isinstance(value, torch.dtype):
            self.rbln_dtype = normalize_dtype(value)
        else:
            self.rbln_dtype = value

    @property
    def dtype(self) -> torch.dtype:
        """Get dtype as torch.dtype."""
        return getattr(torch, self.rbln_dtype)

    @dtype.setter
    def dtype(self, value: str | torch.dtype) -> None:
        """Set dtype from string or torch.dtype."""
        if isinstance(value, torch.dtype):
            self.rbln_dtype = normalize_dtype(value)
        else:
            self.rbln_dtype = value

    @property
    def rbln_model_cls_name(self) -> str:
        """Get the corresponding RBLN model class name."""
        return self.__class__.__name__[:-6]  # Remove 'Config' suffix

    @property
    def rbln_model_cls(self) -> type:
        """Get the corresponding RBLN model class."""
        rbln_model_cls = getattr(importlib.import_module("optimum.rbln"), self.rbln_model_cls_name, None)
        if rbln_model_cls is None:
            raise ValueError(
                f"RBLN model class {self.rbln_model_cls_name} not found. This is an internal error. "
                "Please report it to the developers."
            )
        return rbln_model_cls

    @property
    def compile_cfgs(self) -> list[RBLNCompileConfig]:
        """Get compile configurations."""
        return self.rbln_compile_cfgs

    @compile_cfgs.setter
    def compile_cfgs(self, value: list[RBLNCompileConfig]) -> None:
        """Compile configs should be set via set_compile_cfgs method."""
        raise RuntimeError("`compile_cfgs` cannot be set directly. Please use `set_compile_cfgs` instead.")

    def set_compile_cfgs(self, compile_cfgs: list[RBLNCompileConfig]) -> None:
        """Set compile configurations and propagate npu/tensor_parallel_size."""
        if not isinstance(compile_cfgs, list):
            raise ValueError("`compile_cfgs` must be a list of `RBLNCompileConfig`.")
        if len(compile_cfgs) == 0:
            raise ValueError("`compile_cfgs` must contain at least one `RBLNCompileConfig`.")
        if not isinstance(compile_cfgs[0], RBLNCompileConfig):
            raise ValueError("`compile_cfgs` must contain only `RBLNCompileConfig`.")

        self.rbln_compile_cfgs = compile_cfgs
        for compile_cfg in self.rbln_compile_cfgs:
            compile_cfg.npu = self.npu
            compile_cfg.tensor_parallel_size = self.tensor_parallel_size

        target_npu = self.npu or next((cfg.npu for cfg in self.rbln_compile_cfgs if cfg.npu is not None), None)
        warn_deprecated_npu(target_npu)

    # Runtime property accessors with context override support
    @property
    def create_runtimes(self) -> bool:
        """Get create_runtimes considering context overrides."""
        context = ContextRblnConfig.get_current_context()["create_runtimes"]
        if context is not None:
            return context
        elif self._create_runtimes is None:
            return True
        return self._create_runtimes

    @create_runtimes.setter
    def create_runtimes(self, value: bool) -> None:
        self._create_runtimes = value

    @property
    def device(self) -> int | list[int] | None:
        """Get device considering context overrides."""
        context = ContextRblnConfig.get_current_context()["device"]
        if context is not None:
            return context
        return self._device

    @device.setter
    def device(self, value: int | list[int] | None) -> None:
        self._device = value

    @property
    def device_map(self) -> dict[str, int | list[int]] | None:
        """Get device_map considering context overrides."""
        context = ContextRblnConfig.get_current_context()["device_map"]
        if context:
            return context
        elif self._device_map is None:
            rbln_device_map = {}
            device_val = self.device
            for cfg in self.compile_cfgs:
                rbln_device_map[cfg.compiled_model_name] = device_val
            return rbln_device_map
        return self._device_map

    @device_map.setter
    def device_map(self, value: dict[str, int | list[int]] | None) -> None:
        self._device_map = value

    @property
    def activate_profiler(self) -> bool | None:
        """Get activate_profiler considering context overrides."""
        context = ContextRblnConfig.get_current_context()["activate_profiler"]
        if context is not None:
            return context
        return self._activate_profiler

    @activate_profiler.setter
    def activate_profiler(self, value: bool | None) -> None:
        self._activate_profiler = value

    @property
    def timeout(self) -> int | None:
        """Get timeout considering context overrides."""
        context = ContextRblnConfig.get_current_context()["timeout"]
        if context is not None:
            return context
        return self._timeout

    @timeout.setter
    def timeout(self, value: int | None) -> None:
        self._timeout = value

    def initialize_submodule_config(
        self,
        submodule_config: dict[str, Any] | "RBLNModelConfig" | None = None,
        submodule_name: str | None = None,
        force_kwargs: bool = False,  # Deprecated: kwargs now always take priority
        **kwargs: Any,
    ) -> "RBLNModelConfig" | dict[str, Any]:
        """Initialize a submodule configuration with inherited runtime options.

        Args:
            submodule_config: The submodule configuration dict or RBLNModelConfig instance.
            submodule_name: The name of the submodule. Used to look up the default config class
                from submodule_config_classes if cls_name is not provided.
            force_kwargs: Deprecated. kwargs now always take priority over submodule_config.
            **kwargs: Additional keyword arguments to pass to the config. These always take
                priority over submodule_config values.

        Returns:
            The initialized submodule config (RBLNModelConfig instance) or dict if class cannot be determined.

        Priority order (highest to lowest):
            1. kwargs (always wins)
            2. submodule_config
            3. inherited runtime options from parent
        """
        if submodule_config is None:
            submodule_config = {}

        if isinstance(submodule_config, RBLNModelConfig):
            # Apply kwargs to existing config instance
            for key, value in kwargs.items():
                if hasattr(submodule_config, key) and getattr(submodule_config, key) != value:
                    setattr(submodule_config, key, value)
            return submodule_config

        if isinstance(submodule_config, dict):
            # Build init_kwargs with priority: runtime_options < submodule_config < kwargs
            init_kwargs = self._get_runtime_options().copy()
            init_kwargs.update(
                {
                    "npu": self.npu,
                    "tensor_parallel_size": self.tensor_parallel_size,
                    "optimum_rbln_version": self.optimum_rbln_version,
                }
            )
            init_kwargs.update(submodule_config)
            init_kwargs.update(kwargs)  # kwargs always win

            if "cls_name" in init_kwargs:
                config_cls = get_rbln_config_class(init_kwargs["cls_name"])
            elif submodule_name and submodule_name in self.submodule_config_classes:
                cls_name = self.submodule_config_classes[submodule_name]
                init_kwargs["cls_name"] = cls_name
                config_cls = get_rbln_config_class(cls_name)
            else:
                return init_kwargs

            submodule_config = config_cls(**init_kwargs)

        if not isinstance(submodule_config, RBLNModelConfig):
            raise TypeError(f"Invalid submodule config type: {type(submodule_config)}")

        return submodule_config

    def filter_parameters(self, config_cls: type["RBLNModelConfig"], parameters: dict[str, Any]) -> dict[str, Any]:
        """Filter out parameters not supported by the config class."""
        model_cls_name = config_cls.__name__.replace("Config", "")
        modeling_module_name = config_cls.__module__.replace("configuration_", "modeling_")

        model_cls = None
        try:
            modeling_module = importlib.import_module(modeling_module_name)
            if hasattr(modeling_module, model_cls_name):
                model_cls = getattr(modeling_module, model_cls_name)
        except ImportError:
            logger.debug(f"Could not import modeling module: {modeling_module_name}")

        filtered_out_params = set()

        if model_cls is not None:
            if not getattr(model_cls, "_tp_support", False):
                filtered_out_params.add("tensor_parallel_size")

        filtered_params = {}
        for key, value in parameters.items():
            if key in filtered_out_params:
                logger.debug(
                    f"Parameter '{key}' filtered out for {config_cls.__name__} (not supported by model flags)."
                )
            else:
                filtered_params[key] = value

        return filtered_params

    def freeze(self) -> None:
        """Freeze config to prevent modification."""
        if self._frozen:
            raise RuntimeError(f"`{self.__class__.__name__}` is already frozen.")

        if (
            not isinstance(self.rbln_compile_cfgs, list)
            or len(self.rbln_compile_cfgs) == 0
            or not all(isinstance(cfg, RBLNCompileConfig) for cfg in self.rbln_compile_cfgs)
        ):
            if not self._allow_no_compile_cfgs:
                raise RuntimeError("`compile_cfgs` must contain at least one `RBLNCompileConfig` before freezing.")

        for submodule_name in self.submodules:
            submodule_config = getattr(self, submodule_name, None)
            if not isinstance(submodule_config, RBLNModelConfig):
                raise ValueError(f"`{submodule_name}` must be an instance of `RBLNModelConfig` before freezing.")

        self._frozen = True

    def is_frozen(self) -> bool:
        """Check if config is frozen."""
        return self._frozen

    def __repr__(self) -> str:
        # Use serialize_as_any=True to include all fields from submodule config subclasses
        return json.dumps(self.model_dump(by_alias=True, exclude_none=True, serialize_as_any=True), indent=2)

    def save(self, path: str | Path) -> None:
        """Save config to JSON file."""
        path = Path(path)
        if path.is_dir():
            path = path / "rbln_config.json"

        with open(path, "w") as jsonf:
            # Use serialize_as_any=True to ensure submodule configs are serialized with their actual type's fields
            json.dump(self.model_dump(by_alias=True, exclude_none=True, serialize_as_any=True), jsonf, indent=2)

    @classmethod
    def from_pretrained(
        cls,
        path: str,
        rbln_config: dict[str, Any] | "RBLNModelConfig" | None = None,
        return_unused_kwargs: bool = False,
        **kwargs: Any,
    ) -> "RBLNModelConfig" | tuple["RBLNModelConfig", dict[str, Any]]:
        """
        Load a RBLNModelConfig from a path.

        Args:
            path (str): Path to the RBLNModelConfig file or directory.
            rbln_config (Optional[Dict[str, Any]]): Additional configuration to override.
            return_unused_kwargs (bool): Whether to return unused kwargs.
            kwargs: Additional keyword arguments to override configuration values.

        Returns:
            RBLNModelConfig: The loaded configuration instance.
        """
        cls_reserved, config_file = load_config(path)
        if cls_reserved != cls:
            logger.warning(f"Expected {cls.__name__}, but got {cls_reserved.__name__}.")

        if isinstance(rbln_config, dict):
            for key, value in rbln_config.items():
                if key not in kwargs:
                    kwargs[f"rbln_{key}"] = value

        rbln_keys = [key for key in kwargs.keys() if key.startswith("rbln_")]
        rbln_runtime_kwargs = {key[5:]: kwargs.pop(key) for key in rbln_keys if key[5:] in RUNTIME_KEYWORDS}
        rbln_submodule_kwargs = {key[5:]: kwargs.pop(key) for key in rbln_keys if key[5:] in cls.submodules}

        rbln_kwargs = {
            key[5:]: kwargs.pop(key)
            for key in rbln_keys
            if key[5:] not in RUNTIME_KEYWORDS and key[5:] not in cls.submodules
        }

        # Process submodule's rbln_config
        for submodule in cls.submodules:
            if submodule not in config_file:
                raise ValueError(f"Submodule {submodule} not found in rbln_config.json.")
            submodule_config = config_file[submodule]
            submodule_config.update(rbln_runtime_kwargs)

            update_dict = rbln_submodule_kwargs.pop(submodule, {})
            if update_dict:
                nested_update(submodule_config, update_dict)
            config_file[submodule] = RBLNAutoConfig.load_from_dict(submodule_config)

        if isinstance(rbln_config, RBLNModelConfig):
            config_file.update(rbln_config._get_runtime_options())

            # update submodule runtime
            for submodule in rbln_config.submodules:
                if str(config_file[submodule]) != str(getattr(rbln_config, submodule)):
                    raise ValueError(
                        f"Passed rbln_config has different attributes for submodule {submodule} than the config_file"
                    )
                config_file[submodule] = getattr(rbln_config, submodule)

        config_file.update(rbln_runtime_kwargs)
        loaded_config = cls(**config_file)
        if len(rbln_kwargs) > 0:
            for key, value in rbln_kwargs.items():
                if getattr(loaded_config, key) != value:
                    raise ValueError(
                        f"Cannot set the following arguments: {list(rbln_kwargs.keys())} "
                        f"Since the value is already set to {getattr(loaded_config, key)}"
                    )
        if return_unused_kwargs:
            return loaded_config, kwargs
        else:
            return loaded_config

    @classmethod
    def load(
        cls,
        path: str,
        rbln_config: dict[str, Any] | "RBLNModelConfig" | None = None,
        return_unused_kwargs: bool = False,
        **kwargs: Any,
    ) -> "RBLNModelConfig" | tuple["RBLNModelConfig", dict[str, Any]]:
        """Alias for from_pretrained for backward compatibility."""
        return cls.from_pretrained(path, rbln_config=rbln_config, return_unused_kwargs=return_unused_kwargs, **kwargs)

    @classmethod
    def initialize_from_kwargs(
        cls: type["RBLNModelConfig"],
        rbln_config: dict[str, Any] | "RBLNModelConfig" | None = None,
        **kwargs: Any,
    ) -> tuple["RBLNModelConfig", dict[str, Any]]:
        """Initialize RBLNModelConfig from kwargs."""
        kwargs_keys = list(kwargs.keys())
        rbln_kwargs = {key[5:]: kwargs.pop(key) for key in kwargs_keys if key.startswith("rbln_")}

        if isinstance(rbln_config, dict):
            rbln_config.update(rbln_kwargs)
            rbln_config = cls(**rbln_config)

        elif rbln_config is None:
            rbln_config = cls(**rbln_kwargs)

        elif isinstance(rbln_config, RBLNModelConfig):
            for key, value in rbln_kwargs.items():
                setattr(rbln_config, key, value)

        return rbln_config, kwargs

    def get_default_values_for_original_cls(self, func_name: str, keys: list[str]) -> dict[str, Any]:
        """Get default values for original class attributes from RBLNModelConfig."""
        model_cls = self.rbln_model_cls.get_hf_class()
        func = getattr(model_cls, func_name)
        func_signature = inspect.signature(func)
        default_values = {}
        for key in keys:
            if key in func_signature.parameters:
                default_values[key] = func_signature.parameters[key].default
            else:
                raise ValueError(f"Default value for `{key}` is not set for the model class.")
        return default_values


class RBLNAutoConfig:
    """
    Factory class for loading RBLN configurations.

    This class cannot be instantiated. Use its static methods to load configurations.
    """

    def __new__(cls, **kwargs: Any) -> RBLNModelConfig:
        """Create a config instance based on cls_name in kwargs."""
        cls_name = kwargs.get("cls_name")
        if cls_name is None:
            raise ValueError("`cls_name` is required.")
        config_cls = get_rbln_config_class(cls_name)
        return config_cls(**kwargs)

    @staticmethod
    def load_from_dict(config_dict: dict[str, Any]) -> RBLNModelConfig:
        """
        Build a `RBLNModelConfig` from a plain dictionary.

        The dictionary must contain `cls_name`, which identifies the concrete
        configuration class to instantiate.

        Args:
            config_dict: Mapping typically created by `json.load` or `yaml.safe_load`.

        Returns:
            RBLNModelConfig: A configuration instance.

        Raises:
            ValueError: If `cls_name` is missing.
        """
        cls_name = config_dict.get("cls_name")
        if cls_name is None:
            raise ValueError("`cls_name` is required.")
        config_cls = get_rbln_config_class(cls_name)
        return config_cls(**config_dict)

    @staticmethod
    def register(config_cls: type[RBLNModelConfig], exist_ok: bool = False) -> None:
        """
        Register a new configuration class.

        Args:
            config_cls (type[RBLNModelConfig]): The config class to register.
            exist_ok (bool): Whether to allow registering an already registered model.
        """
        if not issubclass(config_cls, RBLNModelConfig):
            raise ValueError("`config_cls` must be a subclass of RBLNModelConfig.")

        native_cls = getattr(importlib.import_module("optimum.rbln"), config_cls.__name__, None)
        if config_cls.__name__ in CONFIG_MAPPING or native_cls is not None:
            if not exist_ok:
                raise ValueError(f"Configuration for {config_cls.__name__} already registered.")

        CONFIG_MAPPING[config_cls.__name__] = config_cls

    @classmethod
    def from_pretrained(
        cls,
        path: str,
        rbln_config: dict[str, Any] | RBLNModelConfig | None = None,
        return_unused_kwargs: bool = False,
        **kwargs: Any,
    ) -> RBLNModelConfig | tuple[RBLNModelConfig, dict[str, Any]]:
        """
        Load RBLNModelConfig from a path.
        Class name is automatically inferred from the `rbln_config.json` file.

        Args:
            path (str): Path to the RBLNModelConfig.
            rbln_config (Optional[Dict[str, Any]]): Additional configuration to override.
            return_unused_kwargs (bool): Whether to return unused kwargs.
            kwargs: Additional keyword arguments.

        Returns:
            RBLNModelConfig: The loaded RBLNModelConfig.
        """
        target_cls, _ = load_config(path)
        return target_cls.from_pretrained(
            path, rbln_config=rbln_config, return_unused_kwargs=return_unused_kwargs, **kwargs
        )

    @classmethod
    def load(
        cls,
        path: str,
        rbln_config: dict[str, Any] | RBLNModelConfig | None = None,
        return_unused_kwargs: bool = False,
        **kwargs: Any,
    ) -> RBLNModelConfig | tuple[RBLNModelConfig, dict[str, Any]]:
        """Alias for from_pretrained for backward compatibility."""
        return cls.from_pretrained(path, rbln_config=rbln_config, return_unused_kwargs=return_unused_kwargs, **kwargs)


def convert_rbln_config_dict(
    rbln_config: dict[str, Any] | RBLNModelConfig | None = None, **kwargs: Any
) -> tuple[dict[str, Any] | RBLNModelConfig | None, dict[str, Any]]:
    """Validate and merge rbln_ prefixed kwargs into rbln_config."""
    kwargs_keys = list(kwargs.keys())
    rbln_kwargs = {key[5:]: kwargs.pop(key) for key in kwargs_keys if key.startswith("rbln_")}

    rbln_config = {} if rbln_config is None else rbln_config

    if isinstance(rbln_config, dict) and len(rbln_kwargs) > 0:
        rbln_config.update(rbln_kwargs)

    return rbln_config, kwargs
