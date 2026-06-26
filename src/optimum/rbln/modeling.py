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

from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union, get_args, get_origin, get_type_hints

import rebel
import torch
from huggingface_hub.constants import HUGGINGFACE_HUB_CACHE
from transformers import PretrainedConfig
from transformers.modeling_outputs import BaseModelOutput

from .configuration_utils import DEFAULT_COMPILED_MODEL_NAME, RBLNModelConfig
from .modeling_base import RBLNBaseModel
from .utils.logging import get_logger


if TYPE_CHECKING:
    from transformers import PreTrainedModel


logger = get_logger(__name__)


class RBLNModel(RBLNBaseModel):
    @classmethod
    def update_kwargs(cls, kwargs):
        # Update user-given kwargs to get proper pytorch model.

        return kwargs

    @classmethod
    def save_torch_artifacts(
        cls,
        model: "PreTrainedModel",
        save_dir_path: Path,
        subfolder: str,
        rbln_config: RBLNModelConfig,
    ):
        # If you are unavoidably running on a CPU rather than an RBLN device,
        # store the torch tensor, weight, etc. in this function.
        pass

    @classmethod
    def _wrap_model_if_needed(cls, model: torch.nn.Module, rbln_config: RBLNModelConfig) -> torch.nn.Module:
        # Wrap the model if needed.
        return model

    @classmethod
    def get_compiled_model(cls, model: "PreTrainedModel", rbln_config: RBLNModelConfig):
        if rbln_config._allow_no_compile_cfgs:
            return {}

        model = cls._wrap_model_if_needed(model, rbln_config)
        rbln_compile_config = rbln_config.compile_cfgs[0]
        compiled_model = cls.compile(
            model,
            rbln_compile_config=rbln_compile_config,
            create_runtimes=rbln_config.create_runtimes,
            device=rbln_config.device,
        )
        return compiled_model

    @classmethod
    def _update_rbln_config(
        cls,
        preprocessors: Optional[Any],
        model: Optional["PreTrainedModel"] = None,
        model_config: Optional["PretrainedConfig"] = None,
        rbln_config: Optional[RBLNModelConfig] = None,
    ) -> RBLNModelConfig:
        # Default implementation: return config as-is
        # Subclasses should override to set compile_cfgs if needed
        return rbln_config

    @classmethod
    def _reconstruct_model_if_needed(cls, model: "PreTrainedModel"):
        return model

    @classmethod
    def from_model(
        cls,
        model: "PreTrainedModel",
        config: Optional[PretrainedConfig] = None,
        rbln_config: Optional[Union[RBLNModelConfig, Dict]] = None,
        model_save_dir: Optional[Union[str, Path, TemporaryDirectory]] = None,
        subfolder: str = "",
        **kwargs: Any,
    ) -> "RBLNModel":
        """
        Converts and compiles a pre-trained HuggingFace library model into a RBLN model.
        This method performs the actual model conversion and compilation process.

        Args:
            model (PreTrainedModel): The PyTorch model to be compiled.
                The object must be an instance of the HuggingFace transformers PreTrainedModel class.
            config (Optional[PretrainedConfig]): The configuration object associated with the model.
            rbln_config (Optional[Union[RBLNModelConfig, Dict]]): Configuration for RBLN model compilation and runtime.
                This can be provided as a dictionary or an instance of the model's configuration class (e.g., `RBLNLlamaForCausalLMConfig` for Llama models).
                For detailed configuration options, see the specific model's configuration class documentation.
            kwargs: Additional keyword arguments. Arguments with the prefix `rbln_` are passed to rbln_config, while the remaining arguments are passed to the HuggingFace library.

        The method performs the following steps:

        1. Compiles the PyTorch model into an optimized RBLN graph
        2. Configures the model for the specified NPU device
        3. Creates the necessary runtime objects if requested
        4. Saves the compiled model and configurations

        Returns:
            (RBLNModel): A RBLN model instance ready for inference on RBLN NPU devices.
        """

        model = cls._reconstruct_model_if_needed(model)
        preprocessors = kwargs.pop("preprocessors", [])
        rbln_config, kwargs = cls.prepare_rbln_config(rbln_config=rbln_config, **kwargs)

        # Directory to save compile artifacts(.rbln) and original configs
        if model_save_dir is None:
            save_dir = TemporaryDirectory()
            save_dir_path = Path(save_dir.name)
        else:
            save_dir = model_save_dir
            if isinstance(save_dir, TemporaryDirectory):
                save_dir_path = Path(model_save_dir.name)
            else:
                save_dir_path = Path(model_save_dir)
                save_dir_path.mkdir(exist_ok=True)

        # Save configs
        if config is None:
            config = model.config

        if hasattr(model, "can_generate") and model.can_generate():
            import json

            generation_config = model.generation_config
            generation_config_path = save_dir_path / subfolder / "generation_config.json"

            generation_config.save_pretrained(generation_config_path.parent)
            local_config = json.loads(generation_config_path.read_text(encoding="utf-8"))
            local_config["transformers_version"] = generation_config.transformers_version
            generation_config_path.write_text(json.dumps(local_config, indent=2) + "\n", encoding="utf-8")

        if not isinstance(config, PretrainedConfig):  # diffusers config
            config = PretrainedConfig(**config)

        # Save preprocessor
        for preprocessor in preprocessors:
            preprocessor.save_pretrained(save_dir_path / subfolder)

        # Load submodules
        if len(cls._rbln_submodules) > 0:
            rbln_submodules = cls._load_submodules(
                model=model,
                model_save_dir=save_dir,
                rbln_config=rbln_config,
                preprocessors=preprocessors,
                **kwargs,
            )
        else:
            rbln_submodules = []

        # Get compilation arguments (e.g. input_info)
        rbln_config: RBLNModelConfig = cls.update_rbln_config(
            preprocessors=preprocessors, model=model, model_config=config, rbln_config=rbln_config
        )

        # torchscript should be True for jit to work
        torchscript_backup = config.torchscript
        config.torchscript = True

        compiled_model: Union[rebel.RBLNCompiledModel, Dict[str, rebel.RBLNCompiledModel]] = cls.get_compiled_model(
            model, rbln_config=rbln_config
        )

        # Save compiled models (.rbln)
        (save_dir_path / subfolder).mkdir(exist_ok=True)
        if not isinstance(compiled_model, dict):
            compiled_models = {DEFAULT_COMPILED_MODEL_NAME: compiled_model}
        else:
            compiled_models = compiled_model
        for compiled_model_name, cm in compiled_models.items():
            cm.save(save_dir_path / subfolder / f"{compiled_model_name}.rbln")
        rbln_config.save(save_dir_path / subfolder)

        config.torchscript = torchscript_backup
        config.save_pretrained(save_dir_path / subfolder)

        # Save torch artifacts (e.g. embedding matrix if needed.)
        cls.save_torch_artifacts(model, save_dir_path=save_dir_path, subfolder=subfolder, rbln_config=rbln_config)

        # Instantiate
        return cls._from_pretrained(
            model_id=save_dir_path,
            config=config,
            model_save_dir=save_dir,
            subfolder=subfolder,
            rbln_config=rbln_config,
            rbln_compiled_models=compiled_models,
            rbln_submodules=rbln_submodules,
            **kwargs,
        )

    @classmethod
    def get_pytorch_model(
        cls,
        model_id: str,
        use_auth_token: Optional[Union[bool, str]] = None,
        revision: Optional[str] = None,
        force_download: bool = False,
        cache_dir: Optional[str] = HUGGINGFACE_HUB_CACHE,
        subfolder: str = "",
        local_files_only: bool = False,
        trust_remote_code: bool = False,
        # Some rbln-config should be applied before loading torch module (i.e. quantized llm)
        rbln_config: Optional[RBLNModelConfig] = None,
        **kwargs,
    ) -> "PreTrainedModel":
        kwargs = cls.update_kwargs(kwargs)

        return cls.get_hf_class().from_pretrained(
            model_id,
            subfolder=subfolder,
            revision=revision,
            cache_dir=cache_dir,
            use_auth_token=use_auth_token,
            local_files_only=local_files_only,
            force_download=force_download,
            trust_remote_code=trust_remote_code,
            **kwargs,
        )

    @classmethod
    def _create_runtimes(
        cls,
        compiled_models: List[rebel.RBLNCompiledModel],
        rbln_config: RBLNModelConfig,
    ) -> List[rebel.Runtime]:
        if len(rbln_config.compile_cfgs) == 0:
            return []

        if DEFAULT_COMPILED_MODEL_NAME not in rbln_config.device_map:
            cls._raise_missing_compiled_file_error([DEFAULT_COMPILED_MODEL_NAME])

        return [
            rebel.Runtime(
                compiled_model,
                tensor_type="pt",
                device=rbln_config.device_map[DEFAULT_COMPILED_MODEL_NAME],
                activate_profiler=rbln_config.activate_profiler,
                timeout=rbln_config.timeout,
            )
            for compiled_model in compiled_models
        ]

    def forward(self, *args: Any, return_dict: Optional[bool] = None, **kwargs: Any) -> Any:
        """
        Defines the forward pass of `RBLNModel`. The interface mirrors HuggingFace conventions so it can act as a drop-in
        replacement in many cases.

        This method executes the compiled RBLN model on RBLN NPU devices while remaining fully compatible with Hugging Face
        Transformers and Diffusers APIs. In practice, `RBLNModel` can replace models built on `torch.nn.Module` — including
        `transformers.PreTrainedModel` implementations and Diffusers components based on `diffusers.ModelMixin` — enabling
        seamless integration into existing workflows.

        Args:
            args: Variable length argument list containing model inputs. The format matches the original
                HuggingFace model's forward method signature (e.g., input_ids, attention_mask for
                transformers models, or sample, timestep for diffusers models).
            return_dict:
                Whether to return outputs as a dictionary-like object or as a tuple. When `None`:
                - For transformers models: Uses `self.config.use_return_dict` (typically `True`)
                - For diffusers models: Defaults to `True`
            kwargs: Arbitrary keyword arguments containing additional model inputs and parameters,
                matching the original HuggingFace model's interface.

        Returns:
            Model outputs in the same format as the original HuggingFace model.

            If `return_dict=True`, Returns a dictionary-like object (e.g., BaseModelOutput,
                CausalLMOutput) with named fields such as `logits`, `hidden_states`, etc.
            If `return_dict=False`, Returns a tuple containing the raw model outputs.

        Note:
            - This method maintains the exact same interface as the original HuggingFace model's forward method
            - The compiled model runs on RBLN NPU hardware for accelerated inference
            - All HuggingFace model features (generation, attention patterns, etc.) are preserved
            - Can be used directly in HuggingFace pipelines, transformers.Trainer, and other workflows
        """
        if self.hf_library_name == "transformers":
            return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        else:
            return_dict = True if return_dict is None else return_dict

        # Get output from the model
        output = self.model[0](*args, **kwargs)

        # Format output according to task requirements
        return self._prepare_output(output, return_dict)

    @classmethod
    def get_hf_output_class(cls):
        # Dynamically gets the output class from the corresponding HuggingFace model class.
        if "_output_class" in cls.__dict__ and cls._output_class is not None:
            return cls._output_class

        hf_class = cls.get_hf_class()
        if hf_class is None:
            raise ValueError(f"No HuggingFace model class found for {cls.__name__}")

        hints = get_type_hints(hf_class.forward) if hasattr(hf_class, "forward") else {}
        ret = hints.get("return")

        if ret is not None:
            candidates = get_args(ret) if get_origin(ret) is Union else (ret,)

            for t in candidates:
                if t is type(None):  # Skip NoneType in Union
                    continue
                mod = getattr(t, "__module__", "")
                if "transformers" in mod or "diffusers" in mod:
                    cls._output_class = t
                    return t

        # Fallback to BaseModelOutput
        cls._output_class = BaseModelOutput
        return BaseModelOutput

    def _prepare_output(self, output, return_dict):
        # Prepare model output based on return_dict flag.
        # This method can be overridden by subclasses to provide task-specific output handling.
        tuple_output = (output,) if not isinstance(output, (tuple, list)) else tuple(output)
        if not return_dict:
            return tuple_output
        else:
            output_class = self.get_hf_output_class()
            if hasattr(output_class, "loss"):
                tuple_output = (None,) + tuple_output

            # Truncate if we have too many outputs, otherwise use as is
            if hasattr(output_class, "__annotations__"):
                num_fields = len(output_class.__annotations__)
                if len(tuple_output) > num_fields:
                    tuple_output = tuple_output[:num_fields]
                    logger.warning(
                        f"Truncating output to {num_fields} fields for {output_class.__name__}. "
                        f"Expected {num_fields} fields, but got {len(tuple_output)} fields."
                        "This is unexpected. Please report this issue to the developers."
                    )

            return output_class(*tuple_output)
