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

import importlib
import inspect
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Tuple, Type, Union, runtime_checkable

import numpy as np
import torch
from packaging.version import Version

from .__version__ import __version__
from .utils.deprecation import deprecate_kwarg, warn_deprecated_npu
from .utils.logging import get_logger
from .utils.runtime_utils import ContextRblnConfig


logger = get_logger(__name__)


DEFAULT_COMPILED_MODEL_NAME = "compiled_model"
TypeInputInfo = List[Tuple[str, Tuple[int], str]]


@runtime_checkable
class RBLNSerializableConfigProtocol(Protocol):
    def _prepare_for_serialization(self) -> Dict[str, Any]: ...

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self._prepare_for_serialization()})"


@dataclass
class RBLNCompileConfig:
    """
    Configuration for RBLN compilation.

    Attributes:
        compiled_model_name (str): Name of the compiled model.
        input_info (Union[List[TypeInputInfo], TypeInputInfo]): Information about input tensors.
        npu (Optional[str]): NPU configuration.
        tensor_parallel_size (Optional[int]): Size for tensor parallelism.
    """

    compiled_model_name: str = DEFAULT_COMPILED_MODEL_NAME
    input_info: Union[List[TypeInputInfo], TypeInputInfo] = None
    npu: Optional[str] = None
    tensor_parallel_size: Optional[int] = None

    @staticmethod
    def normalize_dtype(dtype: Union[str, torch.dtype, np.dtype]) -> str:
        """
        Convert framework-specific dtype to string representation.
        i.e. torch.float32 -> "float32"

        Args:
            dtype: The input dtype (can be string, torch dtype, or numpy dtype).

        Returns:
            The normalized string representation of the dtype.
        """
        if isinstance(dtype, str):
            return dtype
        else:
            dtype: str = repr(dtype).split(".")[-1]
            if dtype.endswith("'>"):  # numpy
                dtype = dtype[:-2]
            return dtype

    @property
    def is_multiple_input_info(self) -> bool:
        def is_valid_input_info(input_info):
            if not isinstance(input_info, list):
                return False
            return all(
                isinstance(item, (tuple, list))
                and len(item) == 3
                and isinstance(item[0], str)  # name
                and isinstance(item[1], (tuple, list))  # shape
                and all(isinstance(x, int) for x in item[1])
                and (isinstance(item[2], str) or isinstance(item[2], torch.dtype))  # dtype
                for item in input_info
            )

        if isinstance(self.input_info, list):
            return all(is_valid_input_info(info) for info in self.input_info)
        return False

    def __post_init__(self):
        def normalize_input_info(input_info):
            return [(i[0], i[1], RBLNCompileConfig.normalize_dtype(i[2]) or "float32") for i in input_info]

        if self.is_multiple_input_info:
            self.input_info = [normalize_input_info(info) for info in self.input_info]
        else:
            self.input_info = normalize_input_info(self.input_info)

    def update(self, kwargs: Dict[str, Any]):
        self.compiled_model_name = kwargs.get("compiled_model_name", self.compiled_model_name)
        self.input_info = kwargs.get("input_info", self.input_info)
        self.npu = kwargs.get("npu", self.npu)
        self.tensor_parallel_size = kwargs.get("tensor_parallel_size", self.tensor_parallel_size)
        return self

    def get_dummy_inputs(
        self,
        fill=0,
        static_tensors: Optional[Dict[str, torch.Tensor]] = None,
        meta_tensor_names: Optional[List[str]] = None,
    ):
        dummy = []
        static_tensors = static_tensors if static_tensors is not None else {}
        meta_tensor_names = meta_tensor_names if meta_tensor_names is not None else []
        for name, shape, dtype in self.input_info:
            if name in static_tensors:
                tensor = static_tensors[name]
                if shape != list(tensor.shape):
                    raise RuntimeError(f"Different shape for dummy inputs. ({shape} != {list(tensor.shape)})")
                if getattr(torch, dtype) != tensor.dtype:
                    raise RuntimeError(f"Different dtype for dummy inputs ({dtype} != {tensor.dtype})")
                dummy.append(tensor)
            else:
                if name in meta_tensor_names:
                    device = "meta"
                else:
                    device = "cpu"

                dummy.append(
                    torch.fill(torch.empty(*shape, dtype=getattr(torch, dtype), device=torch.device(device)), fill)
                    if len(shape) > 0
                    else torch.tensor(fill, dtype=getattr(torch, dtype), device=torch.device(device))
                )
        return tuple(dummy)

    def asdict(self):
        return asdict(self)


RUNTIME_KEYWORDS = ["create_runtimes", "device", "device_map", "activate_profiler", "timeout"]
CONFIG_MAPPING: Dict[str, Type["RBLNModelConfig"]] = {}


def get_rbln_config_class(rbln_config_class_name: str) -> Type["RBLNModelConfig"]:
    cls = getattr(importlib.import_module("optimum.rbln"), rbln_config_class_name, None)
    if cls is None:
        if rbln_config_class_name in CONFIG_MAPPING:
            cls = CONFIG_MAPPING[rbln_config_class_name]
        else:
            raise ValueError(f"Configuration for {rbln_config_class_name} not found.")
    return cls


def load_config(path: str) -> Tuple[Type["RBLNModelConfig"], Dict[str, Any]]:
    path = Path(path)
    if path.is_dir():
        path = path / "rbln_config.json"

    with open(path, "r") as jsonf:
        config_file = json.load(jsonf)

    if "_meta" in config_file:
        is_legacy_rbln_config = True

        if is_legacy_rbln_config:
            raise RuntimeError(
                f"`{path}` is an old version. Please recompile the model to get the latest config file."
            )

    cls_name = config_file["cls_name"]
    cls = get_rbln_config_class(cls_name)
    return cls, config_file


class RBLNAutoConfig:
    """
    Resolver and factory for RBLN model configurations.

    This class selects the concrete `RBLNModelConfig` subclass, validates the
    provided data, and returns a frozen configuration object that serves as the
    single source of truth during export and load. It does not define the schema
    or control model behavior.
    """

    def __new__(cls, **kwargs):
        cls_name = kwargs.get("cls_name")
        if cls_name is None:
            raise ValueError("`cls_name` is required.")
        cls = get_rbln_config_class(cls_name)
        return cls(**kwargs)

    @staticmethod
    def load_from_dict(config_dict: Dict[str, Any]) -> "RBLNModelConfig":
        """
        Build a `RBLNModelConfig` from a plain dictionary.

        The dictionary must contain `cls_name`, which identifies the concrete
        configuration class to instantiate. All other keys are forwarded to the
        target class initializer. This method does not mutate `config_dict`.

        Args:
            config_dict: Mapping typically created by `json.load` or `yaml.safe_load`.
                For example, the parsed contents of `rbln_config.json`.

        Returns:
            RBLNModelConfig: A configuration instance. The specific subclass is
            selected by `config_dict["cls_name"]`.

        Raises:
            ValueError: If `cls_name` is missing.
            Exception: Any error raised by the target config class during init.

        Examples:
            >>> data = {
            ...     "cls_name": "RBLNLlamaForCausalLMConfig",
            ...     "create_runtimes": False,
            ...     "tensor_parallel_size": 4
            ... }
            >>> cfg = RBLNAutoConfig.load_from_dict(data)
        """
        cls_name = config_dict.get("cls_name")
        if cls_name is None:
            raise ValueError("`cls_name` is required.")
        cls = get_rbln_config_class(cls_name)
        return cls(**config_dict)

    @staticmethod
    def register(config: Type["RBLNModelConfig"], exist_ok=False):
        """
        Register a new configuration for this class.

        Args:
            config (RBLNModelConfig): The config to register.
            exist_ok (bool): Whether to allow registering an already registered model.
        """
        if not issubclass(config, RBLNModelConfig):
            raise ValueError("`config` must be a subclass of RBLNModelConfig.")

        native_cls = getattr(importlib.import_module("optimum.rbln"), config.__name__, None)
        if config.__name__ in CONFIG_MAPPING or native_cls is not None:
            if not exist_ok:
                raise ValueError(f"Configuration for {config.__name__} already registered.")

        CONFIG_MAPPING[config.__name__] = config

    @staticmethod
    def load(
        path: str,
        passed_rbln_config: Optional["RBLNModelConfig"] = None,
        kwargs: Optional[Dict[str, Any]] = None,
        return_unused_kwargs: bool = False,
    ) -> Union["RBLNModelConfig", Tuple["RBLNModelConfig", Dict[str, Any]]]:
        """
        Load RBLNModelConfig from a path.
        Class name is automatically inferred from the `rbln_config.json` file.

        Args:
            path (str): Path to the RBLNModelConfig.
            passed_rbln_config (Optional["RBLNModelConfig"]): RBLNModelConfig to pass its runtime options.

        Returns:
            RBLNModelConfig: The loaded RBLNModelConfig.
        """
        if kwargs is None:
            kwargs = {}
        cls, config_file = load_config(path)

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
            submodule_config.update(rbln_submodule_kwargs.pop(submodule, {}))
            config_file[submodule] = RBLNAutoConfig.load_from_dict(submodule_config)

        if passed_rbln_config is not None:
            config_file.update(passed_rbln_config._runtime_options)
            # TODO(jongho): Reject if the passed_rbln_config has different attributes from the config_file

        config_file.update(rbln_runtime_kwargs)

        rbln_config = cls(**config_file)

        if len(rbln_kwargs) > 0:
            for key, value in rbln_kwargs.items():
                if getattr(rbln_config, key) != value:
                    raise ValueError(
                        f"Cannot set the following arguments: {list(rbln_kwargs.keys())} "
                        f"Since the value is already set to {getattr(rbln_config, key)}"
                    )

        if return_unused_kwargs:
            return cls(**config_file), kwargs
        else:
            return cls(**config_file)


class RBLNModelConfig(RBLNSerializableConfigProtocol):
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

        # Method 4: Combining a config object with override parameters
        # (rbln_ prefixed parameters take precedence over rbln_config values)
        model = RBLNResNetForImageClassification.from_pretrained(
            "model_id",
            export=True,
            rbln_config=config,
            rbln_image_size=320,  # This overrides the value in config
            rbln_device=1         # This sets a new value
        )
        ```


        Save and load configuration:
        ```python
        # Save to disk
        config.save("/path/to/model")

        # Using AutoConfig
        loaded_config = RBLNAutoConfig.load("/path/to/model")
        ```


        Converting between configuration formats:
        ```python
        # Converting a dictionary to a config instance
        config_dict = {
            "image_size": 224,
            "batch_size": 8,
            "create_runtimes": True
        }
        config = RBLNResNetForImageClassificationConfig(**config_dict)
        ```

        Configuration for language models:
        ```python
        from optimum.rbln import RBLNLlamaForCausalLMConfig, RBLNCompileConfig

        # Configure a LLaMA for RBLN
        config = RBLNLlamaForCausalLMConfig(
            max_seq_len=4096,
            device=[0, 1, 2, 3],
            tensor_parallel_size=4  # For multi-NPU parallel inference
        )
        ```

        Working with models that have submodules:
        ```python
        from optimum.rbln import RBLNLlavaNextForConditionalGeneration

        # Configuring a model with submodules
        # LlavaNext has a vision_tower and a language_model submodule
        model = RBLNLlavaNextForConditionalGeneration.from_pretrained(
            "llava-hf/llava-v1.6-mistral-7b-hf",
            export=True,
            rbln_config={
                # Main model's (projector, which is not a submodule) configuration
                "create_runtimes": True,
                "device": 0,

                # Submodule configurations as nested dictionaries
                "vision_tower": {
                    "image_size": 336,
                },
                "language_model": {
                    "tensor_parallel_size": 4,  # Distribute across 4 NPUs
                    "max_seq_len": 8192,
                    "use_inputs_embeds": True,
                    "batch_size": 1,
                },
            },
        )
        ```

        Advanced multi-device deployment with tensor parallelism:
        ```python
        from optimum.rbln import RBLNLlamaForCausalLMConfig

        # Setup a complex multi-device configuration for large language models
        llm_config = RBLNLlamaForCausalLMConfig(
            # Split model across 8 NPUs
            tensor_parallel_size=8,

            # Runtime options
            device=[8, 9, 10, 11, 12, 13, 14, 15],
            create_runtimes=True,
            activate_profiler=True,  # Enable profiling for performance analysis

            # Model-specific parameters for the LLM
            max_seq_len=131072,
            batch_size=4,
            attn_impl="flash_attn",
        )
        ```

        Compilation without runtime creation (create_runtimes=False):
        ```python
        from optimum.rbln import RBLNLlamaForCausalLM, RBLNLlamaForCausalLMConfig

        # Compile a model on a machine without NPU or for later use
        config = RBLNLlamaForCausalLMConfig(
            create_runtimes=False,  # Compile only, don't create runtime
            npu="RBLN-CA25",  # Specify target NPU for compilation
            max_seq_len=4096,
            tensor_parallel_size=4,
            batch_size=1
        )

        # Export the model - will compile but not create runtimes
        model = RBLNLlamaForCausalLM.from_pretrained(
            "meta-llama/Llama-2-7b-hf",
            export=True,
            rbln_config=config
        )

        # Save the compiled model for later use on NPU
        model.save_pretrained("./compiled_llama_model")

        # Later, on a machine with the target NPU
        inference_model = RBLNLlamaForCausalLM.from_pretrained(
            "./compiled_llama_model",
            rbln_create_runtimes=True,  # Now create runtimes (Optional)
        )
        ```

        Two-stage workflow with separate compilation and runtime:
        ```python
        from optimum.rbln import RBLNResNetForImageClassification

        # Stage 1: Model engineer compiles model (can be on any machine)
        def compile_model():
            model = RBLNResNetForImageClassification.from_pretrained(
                "microsoft/resnet-50",
                export=True,
                rbln_create_runtimes=False,
                rbln_npu="RBLN-CA25",
                rbln_image_size=224
            )
            model.save_pretrained("./compiled_model")
            print("Model compiled and saved, ready for deployment")

        # Stage 2: Deployment engineer loads model on NPU
        def deploy_model():
            model = RBLNResNetForImageClassification.from_pretrained(
                "./compiled_model",
                rbln_create_runtimes=True,
            )
            print("Model loaded and ready for inference")
            return model
        ```
    """

    non_save_attributes = [
        "_frozen",
        "_runtime_options",
        "npu",
        "dtype",
        "tensor_parallel_size",
        "create_runtimes",
        "device",
        "device_map",
        "activate_profiler",
        "timeout",
    ]
    submodules: List[str] = []
    subclass_non_save_attributes = []
    _allow_no_compile_cfgs = False

    def initialize_submodule_config(
        self,
        submodule_config: Optional[Union[Dict[str, Any], "RBLNModelConfig"]] = None,
        force_kwargs: bool = False,
        **kwargs: Any,
    ) -> "RBLNModelConfig":
        if submodule_config is None:
            submodule_config = {}

        if isinstance(submodule_config, RBLNModelConfig):
            return submodule_config

        if isinstance(submodule_config, dict):
            from_predecessor = self._runtime_options.copy()
            from_predecessor.update(
                {
                    "npu": self.npu,
                    "tensor_parallel_size": self.tensor_parallel_size,
                    "optimum_rbln_version": self.optimum_rbln_version,
                }
            )
            from_predecessor.update(kwargs)

            init_kwargs = from_predecessor
            init_kwargs.update(submodule_config)

            if force_kwargs:
                for key, value in kwargs.items():
                    if key in init_kwargs:
                        if init_kwargs[key] != value:
                            raise ValueError(
                                f"Parameter conflict for '{key}': submodule_config has {init_kwargs[key]}, "
                                f"but kwargs has {value}. Using kwargs value: {value}"
                            )
                        init_kwargs[key] = value

            if "cls_name" in init_kwargs:
                config_cls = get_rbln_config_class(init_kwargs["cls_name"])
            else:
                return init_kwargs

            submodule_config = config_cls(**init_kwargs)

        if not isinstance(submodule_config, RBLNModelConfig):
            raise TypeError(f"Invalid submodule config type: {type(submodule_config)}")

        return submodule_config

    def filter_parameters(self, config_cls: Type["RBLNModelConfig"], parameters: Dict[str, Any]) -> Dict[str, Any]:
        import importlib

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

    def __setattr__(self, key, value):
        if (
            key != "_attributes_map"
            and key not in self.non_save_attributes
            and key not in self.subclass_non_save_attributes
        ):
            self._attributes_map[key] = value

        if hasattr(self, "_frozen") and self._frozen:
            if not hasattr(self, key) or getattr(self, key) != value:
                raise RuntimeError(
                    f"`{self.__class__.__name__}` is frozen. Cannot update or set attribute after freezing."
                )

        # If the submodule is a dict, Instantiate the submodule config class
        if key in self.submodules and isinstance(value, dict) and (cls_name := value.get("cls_name")):
            rbln_config_cls = getattr(importlib.import_module("optimum.rbln"), cls_name)
            value = rbln_config_cls(**value)

        # Forbid setting keyword-only arguments
        # keyword-only arguments should be translated to other attributes, not set directly
        _keyword_only_args = set()
        init_signature = inspect.signature(self.__class__.__init__)
        for param_name, param in init_signature.parameters.items():
            if param.kind == inspect.Parameter.KEYWORD_ONLY:
                _keyword_only_args.add(param_name)

        if key in _keyword_only_args:
            raise AttributeError(
                f"Cannot set attribute '{key}'. This is an internal error. Please report it to the developers."
            )

        super().__setattr__(key, value)

    @deprecate_kwarg(
        old_name="_torch_dtype",
        new_name="dtype",
        version="0.12.0",
        deprecated_type=torch.dtype,
        value_replacer=RBLNCompileConfig.normalize_dtype,
        raise_if_greater_or_equal_version=False,
    )
    def __init__(
        self,
        cls_name: Optional[str] = None,
        create_runtimes: Optional[bool] = None,
        device: Optional[Union[int, List[int]]] = None,
        device_map: Optional[Dict[str, Union[int, List[int]]]] = None,
        activate_profiler: Optional[bool] = None,
        npu: Optional[str] = None,
        tensor_parallel_size: Optional[int] = None,
        timeout: Optional[int] = None,
        optimum_rbln_version: Optional[str] = None,
        dtype: Optional[Union[str, torch.dtype]] = None,
        _compile_cfgs: Optional[List[RBLNCompileConfig]] = None,
        *,
        optimize_host_memory: Optional[bool] = None,
        **kwargs: Any,
    ):
        """
        Initialize a RBLN model configuration with runtime options and compile configurations.

        Args:
            cls_name (Optional[str]): The class name of the configuration. Defaults to the current class name.
            create_runtimes (Optional[bool]): Whether to create RBLN runtimes. Defaults to True.
            device (Optional[Union[int, List[int]]]): The device(s) to load the model onto. Can be a single device ID or a list.
            device_map (Optional[Dict[str, Union[int, List[int]]]]): Mapping from compiled model names to device IDs.
            activate_profiler (Optional[bool]): Whether to activate the profiler for performance analysis.
            npu (Optional[str]): The NPU device name to use for compilation.
            tensor_parallel_size (Optional[int]): Size for tensor parallelism to distribute the model across devices.
            timeout (Optional[int]): The timeout for the runtime in seconds. If it isn't provided, it will be set to 60 by default.
            optimum_rbln_version (Optional[str]): The optimum-rbln version used for this configuration.
            dtype (Optional[Union[str, torch.dtype]]): The data type to use for the model.
            _compile_cfgs (List[RBLNCompileConfig]): List of compilation configurations for the model.
            kwargs: Additional keyword arguments.

        Raises:
            ValueError: If unexpected keyword arguments are provided.


        """
        self._attributes_map = {}
        self._frozen = False

        self.cls_name = cls_name
        if self.cls_name is None:
            self.cls_name = self.__class__.__name__

        self._runtime_options = {}
        self._runtime_options["create_runtimes"] = create_runtimes
        self._runtime_options["device"] = device
        self._runtime_options["device_map"] = device_map
        self._runtime_options["activate_profiler"] = activate_profiler
        self._runtime_options["timeout"] = timeout

        if optimize_host_memory is not None:
            logger.warning("`optimize_host_memory` is deprecated and will be removed in future versions.")

        # Automatically pass npu, tensor_parallel_size to compile_cfgs
        self.npu = npu
        self.tensor_parallel_size = tensor_parallel_size

        if dtype is not None and isinstance(dtype, torch.dtype):
            dtype = RBLNCompileConfig.normalize_dtype(dtype)
        self._dtype = dtype or "float32"
        self.optimum_rbln_version = optimum_rbln_version
        if self.optimum_rbln_version is None:
            self.optimum_rbln_version = __version__

        compile_cfgs = _compile_cfgs if _compile_cfgs is not None else []
        self._compile_cfgs: List[RBLNCompileConfig] = compile_cfgs

        if not isinstance(self._compile_cfgs, list):
            raise ValueError("`compile_cfgs` must be a list of `RBLNCompileConfig`.")
        if len(self._compile_cfgs) > 0 and not isinstance(self._compile_cfgs[0], RBLNCompileConfig):
            self.set_compile_cfgs([RBLNCompileConfig(**cfg) for cfg in self._compile_cfgs])

        if len(kwargs) > 0:
            if optimum_rbln_version is not None:  # loaded from file
                if Version(__version__) < Version(optimum_rbln_version):
                    diff = "newer"
                elif Version(__version__) > Version(optimum_rbln_version):
                    diff = "older"
                else:
                    diff = None
                if diff is not None:
                    raise ValueError(
                        f"Unexpected arguments: {kwargs.keys()}\n"
                        f"Maybe you are trying to load a model compiled with {diff} version of optimum-rbln. "
                        "It is recommended to use the same version to compile and load the model.\n"
                        f"Current version: {__version__}, Loaded version: {optimum_rbln_version}"
                    )

            raise ValueError(f"Unexpected arguments: {kwargs.keys()}")

    @property
    def torch_dtype(self):
        logger.warning_once("`torch_dtype` is deprecated. Use `dtype` instead.")
        return self.dtype

    @torch_dtype.setter
    def torch_dtype(self, torch_dtype: Union[str, torch.dtype]):
        logger.warning_once("`torch_dtype` is deprecated. Use `dtype` instead.")
        self.dtype = torch_dtype

    @property
    def dtype(self):
        return getattr(torch, self._dtype)

    @dtype.setter
    def dtype(self, dtype: Union[str, torch.dtype]):
        if isinstance(dtype, torch.dtype):
            dtype = RBLNCompileConfig.normalize_dtype(dtype)

        self._dtype = dtype

    @property
    def rbln_model_cls_name(self) -> str:
        return self.__class__.__name__[:-6]

    @property
    def rbln_model_cls(self) -> Type:
        rbln_model_cls = getattr(importlib.import_module("optimum.rbln"), self.rbln_model_cls_name, None)
        if rbln_model_cls is None:
            raise ValueError(
                f"RBLN model class {self.rbln_model_cls_name} not found. This is an internal error. "
                "Please report it to the developers."
            )
        return rbln_model_cls

    def _prepare_for_serialization(self) -> Dict[str, Any]:
        # Prepare the attributes map for serialization by converting nested RBLNModelConfig
        # objects to their serializable form.
        serializable_map = {}
        for key, value in self._attributes_map.items():
            if isinstance(value, RBLNSerializableConfigProtocol):
                # Convert nested RBLNModelConfig to its serializable form
                serializable_map[key] = value._prepare_for_serialization()
            elif key == "_dtype":
                serializable_map["dtype"] = value
            elif key == "_compile_cfgs":
                serializable_map[key] = [cfg.asdict() for cfg in value]
            else:
                serializable_map[key] = value

        return serializable_map

    def __repr__(self):
        repr_dict = self._prepare_for_serialization()
        return json.dumps(repr_dict, indent=2)

    @property
    def compile_cfgs(self):
        return self._compile_cfgs

    @compile_cfgs.setter
    def compile_cfgs(self, compile_cfgs: List[RBLNCompileConfig]):
        raise RuntimeError("`compile_cfgs` cannot be set directly. Please use `set_compile_cfgs` instead.")

    def set_compile_cfgs(self, compile_cfgs: List[RBLNCompileConfig]):
        if not isinstance(compile_cfgs, list):
            raise ValueError("`compile_cfgs` must be a list of `RBLNCompileConfig`.")
        if len(compile_cfgs) == 0:
            raise ValueError("`compile_cfgs` must contain at least one `RBLNCompileConfig`.")
        if not isinstance(compile_cfgs[0], RBLNCompileConfig):
            raise ValueError("`compile_cfgs` must contain only `RBLNCompileConfig`.")

        self._compile_cfgs = compile_cfgs
        for compile_cfg in self._compile_cfgs:
            compile_cfg.npu = self.npu
            compile_cfg.tensor_parallel_size = self.tensor_parallel_size

        target_npu = self.npu or next((cfg.npu for cfg in self._compile_cfgs if cfg.npu is not None), None)
        warn_deprecated_npu(target_npu)

    def freeze(self):
        if self._frozen:
            raise RuntimeError(f"`{self.__class__.__name__}` is already frozen.")

        if (
            not isinstance(self._compile_cfgs, list)
            or len(self._compile_cfgs) == 0
            or not all(isinstance(cfg, RBLNCompileConfig) for cfg in self._compile_cfgs)
        ):
            if not self._allow_no_compile_cfgs:
                raise RuntimeError("`compile_cfgs` must contain at least one `RBLNCompileConfig` before freezing.")

        for submodule_name in self.submodules:
            submodule_config = getattr(self, submodule_name, None)
            if not isinstance(submodule_config, RBLNModelConfig):
                raise ValueError(f"`{submodule_name}` must be an instance of `RBLNModelConfig` before freezing.")

            if not submodule_config.is_frozen():
                raise ValueError(f"`{submodule_name}` config must be frozen before freezing super config.")

        self._frozen = True

    def is_frozen(self):
        return self._frozen

    def save(self, path: str):
        if not self._frozen:
            raise RuntimeError("`RBLNModelConfig` is not frozen. Please call `set_compile_cfgs` first.")

        # save as json file without runtime attributes
        path = Path(path)
        if path.is_dir():
            path = path / "rbln_config.json"

        with open(path, "w") as jsonf:
            serializable_data = self._prepare_for_serialization()
            json.dump(serializable_data, jsonf, indent=2)

    @classmethod
    def load(cls, path: str, **kwargs: Any) -> "RBLNModelConfig":
        """
        Load a RBLNModelConfig from a path.

        Args:
            path (str): Path to the RBLNModelConfig file or directory containing the config file.
            kwargs: Additional keyword arguments to override configuration values.
                      Keys starting with 'rbln_' will have the prefix removed and be used
                      to update the configuration.

        Returns:
            RBLNModelConfig: The loaded configuration instance.

        Note:
            This method loads the configuration from the specified path and applies any
            provided overrides. If the loaded configuration class doesn't match the expected
            class, a warning will be logged.
        """
        cls_reserved, config_file = load_config(path)

        if cls_reserved != cls:
            logger.warning(f"Expected {cls.__name__}, but got {cls_reserved.__name__}.")

        rbln_keys = [key for key in kwargs.keys() if key.startswith("rbln_")]
        rbln_kwargs = {key[5:]: kwargs.pop(key) for key in rbln_keys}
        config_file.update(rbln_kwargs)

        return cls(**config_file)

    @classmethod
    def initialize_from_kwargs(
        cls: Type["RBLNModelConfig"],
        rbln_config: Optional[Union[Dict[str, Any], "RBLNModelConfig"]] = None,
        **kwargs: Any,
    ) -> Tuple["RBLNModelConfig", Dict[str, Any]]:
        # Initialize RBLNModelConfig from kwargs.
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

    def get_default_values_for_original_cls(self, func_name: str, keys: List[str]) -> Dict[str, Any]:
        # Get default values for original class attributes from RBLNModelConfig.
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

    @property
    def create_runtimes(self):
        context = ContextRblnConfig.get_current_context()["create_runtimes"]
        if context is not None:
            return context
        elif self._runtime_options["create_runtimes"] is None:
            return True
        return self._runtime_options["create_runtimes"]

    @create_runtimes.setter
    def create_runtimes(self, create_runtimes: bool):
        self._runtime_options["create_runtimes"] = create_runtimes

    @property
    def device(self):
        context = ContextRblnConfig.get_current_context()["device"]
        if context is not None:
            return context
        return self._runtime_options["device"]

    @device.setter
    def device(self, device: Union[int, List[int]]):
        self._runtime_options["device"] = device

    @property
    def device_map(self):
        context = ContextRblnConfig.get_current_context()["device_map"]
        if context:
            return context
        elif self._runtime_options["device_map"] is None:
            rbln_device_map = {}
            device_val = self.device
            for cfg in self.compile_cfgs:
                rbln_device_map[cfg.compiled_model_name] = device_val
            return rbln_device_map
        return self._runtime_options["device_map"]

    @device_map.setter
    def device_map(self, device_map: Dict[str, Union[int, List[int]]]):
        self._runtime_options["device_map"] = device_map

    @property
    def activate_profiler(self):
        context = ContextRblnConfig.get_current_context()["activate_profiler"]
        if context is not None:
            return context
        return self._runtime_options["activate_profiler"]

    @activate_profiler.setter
    def activate_profiler(self, activate_profiler: bool):
        self._runtime_options["activate_profiler"] = activate_profiler

    @property
    def timeout(self):
        context = ContextRblnConfig.get_current_context()["timeout"]
        if context is not None:
            return context
        return self._runtime_options["timeout"]

    @timeout.setter
    def timeout(self, timeout: int):
        self._runtime_options["timeout"] = timeout
