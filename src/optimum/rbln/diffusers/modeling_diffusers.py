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

import copy
import importlib
from os import PathLike
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Type, Union

import torch

from ..configuration_utils import ContextRblnConfig, RBLNModelConfig, get_rbln_config_class
from ..modeling import RBLNModel
from ..utils.decorator_utils import remove_compile_time_kwargs
from ..utils.logging import get_logger
from ..utils.model_utils import get_rbln_model_cls


logger = get_logger(__name__)

if TYPE_CHECKING:
    from diffusers.pipelines.controlnet.multicontrolnet import MultiControlNetModel


class RBLNDiffusionMixinConfig(RBLNModelConfig):
    """
    Configuration class for RBLN diffusion pipelines.
    """

    pass


class RBLNDiffusionMixin:
    """
    RBLNDiffusionMixin provides essential functionalities for compiling Stable Diffusion pipeline components to run on RBLN NPUs.
    This mixin class serves as a base for implementing RBLN-compatible Stable Diffusion pipelines. It contains shared logic for
    handling the core components of Stable Diffusion.

    To use this mixin:

    1. Create a new pipeline class that inherits from both this mixin and the original StableDiffusionPipeline.
    2. Define the required _submodules and _optional_submodules class variable listing the components to be compiled.

    Example:
        ```python
        class RBLNStableDiffusionPipeline(RBLNDiffusionMixin, StableDiffusionPipeline):
            _submodules = ["text_encoder", "unet", "vae"]
        ```

    Class Variables:
        - `_submodules`: List of submodule names that should be compiled (typically ["text_encoder", "unet", "vae"])
        - `_optional_submodules`: List of submodule names compiled without inheriting RBLNModel (typically ["safety_checker"])

    Methods:
        from_pretrained: Creates and optionally compiles a model from a pretrained checkpoint

    Notes:
        - When `export=True`, all compatible submodules will be compiled for NPU inference
        - The compilation config can be customized per submodule by including submodule names
          as keys in rbln_config
    """

    _connected_classes = {}
    _submodules = []
    _optional_submodules = []
    _prefix = {}

    @staticmethod
    def _maybe_apply_and_fuse_lora(
        model: torch.nn.Module,
        lora_ids: Optional[Union[str, List[str]]] = None,
        lora_weights_names: Optional[Union[str, List[str]]] = None,
        lora_scales: Optional[Union[float, List[float]]] = None,
    ) -> torch.nn.Module:
        lora_ids = [lora_ids] if isinstance(lora_ids, str) else lora_ids
        lora_weights_names = [lora_weights_names] if isinstance(lora_weights_names, str) else lora_weights_names
        lora_scales = [lora_scales] if isinstance(lora_scales, float) else lora_scales

        # adapt lora weight into pipeline before compilation
        if lora_ids and lora_weights_names:
            if len(lora_ids) == 1:
                if len(lora_ids) != len(lora_weights_names):
                    raise ValueError(
                        f"You must define the same number of lora ids ({len(lora_ids)} and lora weights ({len(lora_weights_names)}))"
                    )
                else:
                    model.load_lora_weights(lora_ids[0], weight_name=lora_weights_names[0])
                    model.fuse_lora(lora_scale=lora_scales[0] if lora_scales else 1.0)
            elif len(lora_ids) > 1:
                if not len(lora_ids) == len(lora_weights_names):
                    raise ValueError(
                        f"If you fuse {len(lora_ids)} lora models, but you must define the same number for lora weights and adapters."
                    )

                adapter_names = [f"adapter_{i}" for i in range(len(lora_ids))]

                for lora_id, lora_weight, adapter_name in zip(lora_ids, lora_weights_names, adapter_names):
                    model.load_lora_weights(lora_id, weight_name=lora_weight, adapter_name=adapter_name)

                if lora_scales:
                    model.set_adapters(adapter_names, adapter_weights=lora_scales)

                model.fuse_lora()
        return model

    @classmethod
    def get_rbln_config_class(cls) -> Type[RBLNModelConfig]:
        # Lazily loads and caches the corresponding RBLN model config class.
        if "_rbln_config_class" not in cls.__dict__ or cls._rbln_config_class is None:
            rbln_config_class_name = cls.__name__ + "Config"
            cls._rbln_config_class = get_rbln_config_class(rbln_config_class_name)
        return cls._rbln_config_class

    @classmethod
    def get_hf_class(cls):
        if "_hf_class" not in cls.__dict__ or cls._hf_class is None:
            hf_cls_name = cls.__name__[4:]
            library = importlib.import_module("diffusers")
            cls._hf_class = getattr(library, hf_cls_name, None)
        return cls._hf_class

    @classmethod
    def from_pretrained(
        cls,
        model_id: str,
        *,
        export: bool = None,
        model_save_dir: Optional[PathLike] = None,
        rbln_config: Optional[Dict[str, Any]] = None,
        lora_ids: Optional[Union[str, List[str]]] = None,
        lora_weights_names: Optional[Union[str, List[str]]] = None,
        lora_scales: Optional[Union[float, List[float]]] = None,
        **kwargs: Any,
    ) -> "RBLNDiffusionMixin":
        """
        Load a pretrained diffusion pipeline from a model checkpoint, with optional compilation for RBLN NPUs.

        This method has two distinct operating modes:
            - When `export=True`: Takes a PyTorch-based diffusion model, compiles it for RBLN NPUs, and loads the compiled model
            - When `export=False`: Loads an already compiled RBLN model from `model_id` without recompilation

        It supports various diffusion pipelines including Stable Diffusion, Kandinsky, ControlNet, and other diffusers-based models.

        Args:
            model_id (`str`):
                The model ID or path to the pretrained model to load. Can be either:

                - A model ID from the HuggingFace Hub
                - A local path to a saved model directory
            export:
                If True, takes a PyTorch model from `model_id` and compiles it for RBLN NPU execution.
                If False, loads an already compiled RBLN model from `model_id` without recompilation.
            model_save_dir:
                Directory to save the compiled model artifacts. Only used when `export=True`.
                If not provided and `export=True`, a temporary directory is used.
            rbln_config:
                Configuration options for RBLN compilation. Can include settings for specific submodules
                such as `text_encoder`, `unet`, and `vae`. Configuration can be tailored to the specific
                pipeline being compiled.
            lora_ids:
                LoRA adapter ID(s) to load and apply before compilation. LoRA weights are fused
                into the model weights during compilation. Only used when `export=True`.
            lora_weights_names:
                Names of specific LoRA weight files to load, corresponding to lora_ids. Only used when `export=True`.
            lora_scales:
                Scaling factor(s) to apply to the LoRA adapter(s). Only used when `export=True`.
            kwargs:
                Additional arguments to pass to the underlying diffusion pipeline constructor or the
                RBLN compilation process. These may include parameters specific to individual submodules
                or the particular diffusion pipeline being used.

        Returns:
            A compiled or loaded diffusion pipeline that can be used for inference on RBLN NPU.
                The returned object is an instance of the class that called this method, inheriting from RBLNDiffusionMixin.
        """
        rbln_config, kwargs = cls.get_rbln_config_class().initialize_from_kwargs(rbln_config, **kwargs)

        if export is None:
            export = any(
                not RBLNModel._is_compiled(
                    model_id,
                    token=kwargs.get("token"),
                    revision=kwargs.get("revision"),
                    force_download=kwargs.get("force_download", False),
                    cache_dir=kwargs.get("cache_dir"),
                    subfolder=submodule_name,
                    local_files_only=kwargs.get("local_files_only", False),
                )
                for submodule_name in cls._submodules
            )

        if export:
            # keep submodules if user passed any of them.
            passed_submodules = {
                name: kwargs.pop(name)
                for name in cls._submodules + cls._optional_submodules
                if isinstance(kwargs.get(name), RBLNModel)
            }

        else:
            # raise error if any of submodules are torch module.
            model_index_config = cls.load_config(pretrained_model_name_or_path=model_id)
            for submodule_name in cls._submodules + cls._optional_submodules:
                passed_submodule = kwargs.get(submodule_name, None)

                if passed_submodule is None:
                    module_name, class_name = model_index_config[submodule_name]
                    if module_name != "optimum.rbln":
                        raise ValueError(
                            f"Invalid module_name '{module_name}' found in model_index.json for "
                            f"submodule '{submodule_name}'. "
                            "Expected 'optimum.rbln'. Please check the model_index.json configuration."
                            "If you want to compile, set `export=True`."
                        )

                    submodule_cls = get_rbln_model_cls(class_name)
                    submodule_config = getattr(rbln_config, submodule_name)
                    submodule = submodule_cls.from_pretrained(
                        model_id, export=False, subfolder=submodule_name, rbln_config=submodule_config
                    )

                else:
                    if passed_submodule.__class__.__name__.startswith("RBLN"):
                        submodule = passed_submodule

                    elif isinstance(passed_submodule, torch.nn.Module):
                        raise AssertionError(
                            f"{submodule_name} is not compiled torch module. If you want to compile, set `export=True`."
                        )

                kwargs[submodule_name] = submodule

        with ContextRblnConfig(
            device=rbln_config.device,
            device_map=rbln_config.device_map,
            create_runtimes=rbln_config.create_runtimes,
            activate_profiler=rbln_config.activate_profiler,
            timeout=rbln_config.timeout,
        ):
            model = super().from_pretrained(pretrained_model_name_or_path=model_id, **kwargs)

        if not export:
            return model

        model = cls._maybe_apply_and_fuse_lora(
            model,
            lora_ids=lora_ids,
            lora_weights_names=lora_weights_names,
            lora_scales=lora_scales,
        )

        if cls._load_connected_pipes:
            compiled_submodules = cls._compile_pipelines(model, passed_submodules, model_save_dir, rbln_config)
        else:
            compiled_submodules = cls._compile_submodules(model, passed_submodules, model_save_dir, rbln_config)
        return cls._construct_pipe(model, compiled_submodules, model_save_dir, rbln_config)

    @classmethod
    def _compile_pipelines(
        cls,
        model: torch.nn.Module,
        passed_submodules: Dict[str, RBLNModel],
        model_save_dir: Optional[PathLike],
        rbln_config: "RBLNDiffusionMixinConfig",
    ) -> Dict[str, RBLNModel]:
        compiled_submodules = {}
        for connected_pipe_name, connected_pipe_cls in cls._connected_classes.items():
            connected_pipe_submodules = {}
            prefix = cls._prefix.get(connected_pipe_name, "")
            for submodule_name in connected_pipe_cls._submodules:
                connected_pipe_submodules[submodule_name] = passed_submodules.get(prefix + submodule_name, None)
            connected_pipe = getattr(model, connected_pipe_name)
            connected_pipe_compiled_submodules = connected_pipe_cls._compile_submodules(
                connected_pipe,
                connected_pipe_submodules,
                model_save_dir,
                getattr(rbln_config, connected_pipe_name),
                prefix,
            )
            for submodule_name, compiled_submodule in connected_pipe_compiled_submodules.items():
                compiled_submodules[prefix + submodule_name] = compiled_submodule
        return compiled_submodules

    @classmethod
    def _compile_submodules(
        cls,
        model: torch.nn.Module,
        passed_submodules: Dict[str, RBLNModel],
        model_save_dir: Optional[PathLike],
        rbln_config: RBLNDiffusionMixinConfig,
        prefix: Optional[str] = "",
    ) -> Dict[str, RBLNModel]:
        compiled_submodules = {}

        for submodule_name in cls._submodules:
            submodule = passed_submodules.get(submodule_name) or getattr(model, submodule_name, None)

            if getattr(rbln_config, submodule_name, None) is None:
                raise ValueError(f"RBLN config for submodule {submodule_name} is not provided.")

            submodule_rbln_cls: Type[RBLNModel] = getattr(rbln_config, submodule_name).rbln_model_cls
            rbln_config = submodule_rbln_cls.update_rbln_config_using_pipe(model, rbln_config, submodule_name)

            if submodule is None:
                raise ValueError(f"submodule ({submodule_name}) cannot be accessed since it is not provided.")
            elif isinstance(submodule, RBLNModel):
                pass
            elif submodule_name == "controlnet" and hasattr(submodule, "nets"):
                submodule = cls._compile_multicontrolnet(
                    controlnets=submodule,
                    model_save_dir=model_save_dir,
                    controlnet_rbln_config=getattr(rbln_config, submodule_name),
                    prefix=prefix,
                )
            elif isinstance(submodule, torch.nn.Module):
                subfolder = prefix + submodule_name
                submodule = submodule_rbln_cls.from_model(
                    model=submodule,
                    subfolder=subfolder,
                    model_save_dir=model_save_dir,
                    rbln_config=getattr(rbln_config, submodule_name),
                )
            else:
                raise ValueError(f"Unknown class of submodule({submodule_name}) : {submodule.__class__.__name__} ")

            compiled_submodules[submodule_name] = submodule
        return compiled_submodules

    @classmethod
    def _compile_multicontrolnet(
        cls,
        controlnets: "MultiControlNetModel",
        model_save_dir: Optional[PathLike],
        controlnet_rbln_config: RBLNModelConfig,
        prefix: Optional[str] = "",
    ):
        # Compile multiple ControlNet models for a MultiControlNet setup
        from .models.controlnet import RBLNControlNetModel
        from .pipelines.controlnet import RBLNMultiControlNetModel

        compiled_controlnets = []
        for i, controlnet in enumerate(controlnets.nets):
            _controlnet_rbln_config = copy.deepcopy(controlnet_rbln_config)
            compiled_controlnets.append(
                RBLNControlNetModel.from_model(
                    model=controlnet,
                    subfolder=f"{prefix}controlnet" if i == 0 else f"{prefix}controlnet_{i}",
                    model_save_dir=model_save_dir,
                    rbln_config=_controlnet_rbln_config,
                )
            )
        return RBLNMultiControlNetModel(compiled_controlnets)

    @classmethod
    def _construct_pipe(cls, model, submodules, model_save_dir, rbln_config):
        # Construct finalize pipe setup with compiled submodules and configurations
        if model_save_dir is not None:
            # To skip saving original pytorch modules
            for submodule_name in cls._submodules:
                delattr(model, submodule_name)

            if cls._load_connected_pipes:
                for connected_pipe_name, connected_pipe_cls in cls._connected_classes.items():
                    for submodule_name in connected_pipe_cls._submodules:
                        delattr(getattr(model, connected_pipe_name), submodule_name)

            # Direct calling of `save_pretrained` causes config.unet = (None, None).
            # So config must be saved again, later.
            model.save_pretrained(model_save_dir)
            # FIXME: Here, model touches its submodules such as model.unet,
            # Causing warning messeages.

        update_dict = {}
        for submodule_name in cls._submodules + cls._optional_submodules:
            # replace submodule
            if submodule_name in submodules:
                setattr(model, submodule_name, submodules[submodule_name])
                update_dict[submodule_name] = ("optimum.rbln", submodules[submodule_name].__class__.__name__)
            else:
                # It assumes that the modules in _optional_components is compiled
                # and already registered as an attribute of the model.
                update_dict[submodule_name] = ("optimum.rbln", getattr(model, submodule_name).__class__.__name__)

        if cls._load_connected_pipes:
            for connected_pipe_name, connected_pipe_cls in cls._connected_classes.items():
                prefix = cls._prefix.get(connected_pipe_name, "")
                for submodule_name in connected_pipe_cls._submodules:
                    setattr(getattr(model, connected_pipe_name), submodule_name, submodules[prefix + submodule_name])

        # Update config to be able to load from model directory.
        #
        # e.g)
        # update_dict = {
        #     "vae": ("optimum.rbln", "RBLNAutoencoderKL"),
        #     "text_encoder": ("optimum.rbln", "RBLNCLIPTextModel"),
        #     "unet": ("optimum.rbln", "RBLNUNet2DConditionModel"),
        # }
        model.register_to_config(**update_dict)

        if model_save_dir:
            # overwrite to replace incorrect config
            model.save_config(model_save_dir)

        # Keep compiled_model objs to further analysis. -> TODO: remove soon...
        model.compiled_models = []
        for name in cls._submodules:
            submodule = getattr(model, name)
            model.compiled_models.extend(submodule.compiled_models)

        return model

    def get_compiled_image_size(self):
        if hasattr(self, "vae") and hasattr(self.vae, "image_size"):
            compiled_image_size = self.vae.image_size
        else:
            compiled_image_size = None
        return compiled_image_size

    def handle_additional_kwargs(self, **kwargs):
        # Function to handle additional compile-time parameters during inference.

        # If the additional variable is determined by another module, this method should be overrided.

        # Example:
        #     ```python
        #     if hasattr(self, "movq"):
        #         compiled_image_size = self.movq.image_size
        #         kwargs["height"] = compiled_image_size[0]
        #         kwargs["width"] = compiled_image_size[1]

        #     compiled_num_frames = self.unet.rbln_config.num_frames
        #     if compiled_num_frames is not None:
        #         kwargs["num_frames"] = compiled_num_frames
        #     return kwargs
        #     ```
        return kwargs

    @remove_compile_time_kwargs
    def __call__(self, *args, **kwargs):
        kwargs = self.handle_additional_kwargs(**kwargs)
        return super().__call__(*args, **kwargs)
