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
import os
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Type, Union

import rebel
import torch
from transformers import AutoConfig, AutoModel, GenerationConfig, PretrainedConfig
from transformers.utils.hub import PushToHubMixin

from .configuration_utils import RBLNAutoConfig, RBLNCompileConfig, RBLNModelConfig, get_rbln_config_class
from .utils.hub import pull_compiled_model_from_hub, validate_files
from .utils.logging import get_logger
from .utils.runtime_utils import UnavailableRuntime, tp_and_devices_are_ok
from .utils.save_utils import maybe_load_preprocessors
from .utils.submodule import SubModulesMixin


if TYPE_CHECKING:
    from transformers import AutoFeatureExtractor, AutoProcessor, AutoTokenizer, PreTrainedModel

logger = get_logger(__name__)


class PreTrainedModel:  # noqa: F811
    pass


class RBLNBaseModelConfig(RBLNModelConfig):
    pass


class RBLNBaseModel(SubModulesMixin, PushToHubMixin, PreTrainedModel):
    model_type = "rbln_model"
    auto_model_class = AutoModel
    config_class = AutoConfig
    config_name = "config.json"
    hf_library_name = "transformers"
    _supports_non_fp32 = False

    def __init__(
        self,
        models: List[rebel.Runtime],
        config: "PretrainedConfig",
        rbln_config: RBLNModelConfig,
        model_save_dir: Optional[Union[str, Path, TemporaryDirectory]] = None,
        subfolder: str = "",
        rbln_compiled_models: Optional[rebel.RBLNCompiledModel] = None,
        rbln_submodules: Optional[List["RBLNBaseModel"]] = None,
        **kwargs,
    ):
        self.model = models
        self.config = config
        self.rbln_config = rbln_config
        if not rbln_config.is_frozen():
            raise RuntimeError("`rbln_config` must be frozen. Please call `rbln_config.freeze()` first.")
        self.compiled_models = rbln_compiled_models

        # Registers the RBLN classes into the transformers AutoModel classes to avoid warnings when creating
        # a pipeline https://github.com/huggingface/transformers/blob/3d3204c025b6b5de013e07dd364208e28b4d9589/src/transformers/pipelines/base.py#L940
        AutoConfig.register(self.model_type, AutoConfig)
        if hasattr(self.auto_model_class, "register"):
            self.auto_model_class.register(AutoConfig, self.__class__)

        # copied from tranformers PreTrainedModel __init__
        if self.can_generate():
            gen_config_dir = model_save_dir.name if isinstance(model_save_dir, TemporaryDirectory) else model_save_dir
            self.generation_config = GenerationConfig.from_pretrained(gen_config_dir, trust_remote_code=True)
        else:
            self.generation_config = None

        if self.generation_config is not None:
            self.generation_config.use_cache = True

        self.device = torch.device("cpu")
        self.training = False
        self.dtype = rbln_config.dtype

        # FIXME :: model_save_dir is not used after initialized. (This can be used when save/load)
        # This attribute is needed to keep one reference on the temporary directory, since garbage collecting it
        # would end-up removing the directory containing the underlying RBLN model.
        self._model_save_dir_tempdirectory_instance = None
        if isinstance(model_save_dir, TemporaryDirectory):
            self._model_save_dir_tempdirectory_instance = model_save_dir
            self.model_save_dir = Path(model_save_dir.name)
        elif isinstance(model_save_dir, str):
            self.model_save_dir = Path(model_save_dir)
        else:
            self.model_save_dir = model_save_dir
        self.subfolder = subfolder

        if rbln_submodules is None:
            rbln_submodules = []
        self.rbln_submodules = rbln_submodules
        self.__post_init__(**kwargs)

    @classmethod
    def _load_compiled_model_dir(
        cls,
        model_id: Union[str, Path],
        token: Optional[Union[bool, str]] = None,
        revision: Optional[str] = None,
        force_download: bool = False,
        cache_dir: Optional[str] = None,
        subfolder: str = "",
        local_files_only: bool = False,
    ) -> str:
        # Load the directory containing the compiled model files.
        model_path = Path(model_id)

        if model_path.is_dir():
            model_path = model_path / subfolder
            rbln_files = list(model_path.glob("*.rbln"))
            rbln_config_filenames = list(model_path.glob("rbln_config.json"))
            validate_files(rbln_files, rbln_config_filenames, f"directory {model_path}")
        else:
            model_path = pull_compiled_model_from_hub(
                model_id=model_id,
                subfolder=subfolder,
                token=token,
                revision=revision,
                cache_dir=cache_dir,
                force_download=force_download,
                local_files_only=local_files_only,
            )

        return str(model_path)

    @classmethod
    def _load_compiled_models(cls, model_path: str, expected_compiled_model_names: List[str]):
        compiled_models = Path(model_path).glob("*.rbln")
        expected_compiled_models = [
            Path(model_path) / f"{compiled_model_name}.rbln" for compiled_model_name in expected_compiled_model_names
        ]
        unexpected_compiled_models = [cm for cm in compiled_models if cm not in expected_compiled_models]
        if unexpected_compiled_models:
            # TODO(jongho): fix after May release. raise error if unexpected compiled models are found
            logger.warning(
                f"Unexpected compiled models found: {[cm.name for cm in unexpected_compiled_models]}. "
                f"Please check the model path: {model_path}"
            )

        rbln_compiled_models = {}
        for compiled_model in expected_compiled_models:
            if not compiled_model.exists():
                raise FileNotFoundError(
                    f"Expected RBLN compiled model '{compiled_model.name}' not found at '{model_path}'. "
                    "Please ensure all models specified in `rbln_config` are present."
                )
            rbln_compiled_models[compiled_model.stem] = rebel.RBLNCompiledModel(compiled_model)
        return rbln_compiled_models

    @classmethod
    def _from_pretrained(
        cls,
        model_id: Union[str, Path],
        config: Optional["PretrainedConfig"] = None,
        token: Optional[Union[bool, str]] = None,
        revision: Optional[str] = None,
        force_download: bool = False,
        cache_dir: Optional[str] = None,
        subfolder: str = "",
        local_files_only: bool = False,
        trust_remote_code: bool = False,
        model_save_dir: Optional[Union[str, Path, TemporaryDirectory]] = None,
        # passed from compile function
        rbln_config: Optional[RBLNModelConfig] = None,
        rbln_compiled_models: Optional[Dict[str, rebel.RBLNCompiledModel]] = None,
        rbln_submodules: Optional[List["RBLNBaseModel"]] = None,
        **kwargs,
    ) -> "RBLNBaseModel":
        if rbln_compiled_models is None:
            model_path_subfolder = cls._load_compiled_model_dir(
                model_id=model_id,
                token=token,
                revision=revision,
                force_download=force_download,
                cache_dir=cache_dir,
                subfolder=subfolder,
                local_files_only=local_files_only,
            )

            if isinstance(rbln_config, dict):
                rbln_config_as_kwargs = {f"rbln_{key}": value for key, value in rbln_config.items()}
                kwargs.update(rbln_config_as_kwargs)
                rbln_config = None
            elif isinstance(rbln_config, RBLNModelConfig) and rbln_config.rbln_model_cls_name != cls.__name__:
                raise ValueError(
                    f"Cannot use the passed rbln_config. Its model class name ({rbln_config.rbln_model_cls_name}) "
                    f"does not match the expected model class name ({cls.__name__})."
                )

            rbln_config, kwargs = RBLNAutoConfig.load(
                model_path_subfolder, passed_rbln_config=rbln_config, kwargs=kwargs, return_unused_kwargs=True
            )

            if rbln_config.rbln_model_cls_name != cls.__name__:
                raise NameError(
                    f"Cannot load the model. The model was originally compiled using "
                    f"{rbln_config.rbln_model_cls_name}, but you are trying to load it with {cls.__name__}."
                    "Please use the same model class that was used during compilation."
                )

            if len(cls._rbln_submodules) > 0:
                if rbln_submodules is None:
                    rbln_submodules = cls._load_submodules(model_save_dir=model_id, rbln_config=rbln_config, **kwargs)
            elif rbln_submodules is None:
                rbln_submodules = []

            rbln_config.freeze()

            if config is None:
                if cls.hf_library_name == "transformers":
                    config = AutoConfig.from_pretrained(
                        model_path_subfolder,
                        cache_dir=cache_dir,
                        force_download=force_download,
                        revision=revision,
                        token=token,
                        trust_remote_code=trust_remote_code,
                    )
                elif cls.hf_library_name == "diffusers":
                    # import here to prevent diffusers dependency
                    # TODO(jongho): Remove diffusers dependency if use transformers only.
                    from diffusers.configuration_utils import ConfigMixin

                    class DummyConfigMixin(ConfigMixin):
                        # Just to load config, We need to specify `config_name`
                        config_name = "config.json"

                    config = DummyConfigMixin.load_config(
                        model_id,
                        cache_dir=cache_dir,
                        force_download=force_download,
                        local_files_only=local_files_only,
                        revision=revision,
                        token=token,
                        subfolder=subfolder,
                    )
                    config = PretrainedConfig(**config)

            compiled_model_names = [cfg.compiled_model_name for cfg in rbln_config.compile_cfgs]
            rbln_compiled_models = cls._load_compiled_models(model_path_subfolder, compiled_model_names)

            if subfolder != "":
                model_save_dir = Path(model_path_subfolder).absolute().parent
            else:
                model_save_dir = Path(model_path_subfolder).absolute()

        return cls._from_compiled_models(
            rbln_compiled_models=rbln_compiled_models,
            rbln_config=rbln_config,
            config=config,
            model_save_dir=model_save_dir,
            subfolder=subfolder,
            rbln_submodules=rbln_submodules,
            **kwargs,
        )

    @classmethod
    def _from_compiled_models(
        cls,
        rbln_compiled_models: Dict[str, rebel.RBLNCompiledModel],
        rbln_config: RBLNModelConfig,
        config: "PretrainedConfig",
        model_save_dir: Union[Path, str],
        subfolder: Union[Path, str],
        rbln_submodules: Optional[List["RBLNBaseModel"]] = None,
        **kwargs,
    ):
        if rbln_submodules is None:
            rbln_submodules = []

        if isinstance(model_save_dir, str):
            model_save_dir = Path(model_save_dir)

        # FIXME:: Should we convert it?
        compiled_model_names = [cfg.compiled_model_name for cfg in rbln_config.compile_cfgs]
        rbln_compiled_models = [rbln_compiled_models[cm_name] for cm_name in compiled_model_names]

        # create runtimes only if `rbln_create_runtimes` is enabled
        try:
            models = (
                cls._create_runtimes(rbln_compiled_models, rbln_config)
                if rbln_config.create_runtimes
                else UnavailableRuntime()
            )

        except rebel.core.exception.RBLNRuntimeError as e:
            error_msg = (
                f"\nFailed to create RBLN runtime: {str(e)}\n\n"
                f"If you only need to compile the model without loading it to NPU, you can use:\n"
                f"  from_pretrained(..., rbln_create_runtimes=False) or\n"
                f"  from_pretrained(..., rbln_config={{..., 'create_runtimes': False}})\n\n"
                f"To check your NPU status, run the 'rbln-stat' command in your terminal.\n"
                f"Make sure your NPU is properly installed and operational."
            )
            raise rebel.core.exception.RBLNRuntimeError(error_msg) from e

        return cls(
            models,
            config,
            rbln_config,
            model_save_dir=model_save_dir,
            subfolder=subfolder,
            rbln_compiled_models=rbln_compiled_models,
            rbln_submodules=rbln_submodules,
            **kwargs,
        )

    @classmethod
    def _export(cls, model_id: Union[str, Path], **kwargs) -> "RBLNBaseModel":
        subfolder = kwargs.get("subfolder", "")
        model_save_dir = kwargs.pop("model_save_dir", None)

        rbln_config, kwargs = cls.prepare_rbln_config(**kwargs)

        model: "PreTrainedModel" = cls.get_pytorch_model(model_id=model_id, rbln_config=rbln_config, **kwargs)
        preprocessors = maybe_load_preprocessors(model_id, subfolder=subfolder)
        return cls.from_model(
            model, preprocessors=preprocessors, model_save_dir=model_save_dir, rbln_config=rbln_config, **kwargs
        )

    @classmethod
    def prepare_rbln_config(
        cls, rbln_config: Optional[Union[Dict[str, Any], RBLNModelConfig]] = None, **kwargs
    ) -> Tuple[RBLNModelConfig, Dict[str, Any]]:
        # Extract rbln-config from kwargs and convert it to RBLNModelConfig.

        config_cls = cls.get_rbln_config_class()
        rbln_config, kwargs = config_cls.initialize_from_kwargs(rbln_config, **kwargs)
        return rbln_config, kwargs

    @classmethod
    def _is_compiled(
        cls,
        model_id: Union[str, Path],
        token: Optional[Union[bool, str]] = None,
        revision: Optional[str] = None,
        force_download: bool = False,
        cache_dir: Optional[str] = None,
        subfolder: str = "",
        local_files_only: bool = False,
    ) -> bool:
        # Check if the model is already compiled.
        try:
            cls._load_compiled_model_dir(
                model_id=model_id,
                token=token,
                revision=revision,
                force_download=force_download,
                cache_dir=cache_dir,
                subfolder=subfolder,
                local_files_only=local_files_only,
            )
            return True
        except (FileNotFoundError, KeyError):
            return False

    @classmethod
    def from_pretrained(
        cls: Type["RBLNBaseModel"],
        model_id: Union[str, Path],
        export: Optional[bool] = None,
        rbln_config: Optional[Union[Dict, RBLNModelConfig]] = None,
        **kwargs: Any,
    ) -> "RBLNBaseModel":
        """
        The `from_pretrained()` function is utilized in its standard form as in the HuggingFace transformers library.
        User can use this function to load a pre-trained model from the HuggingFace library and convert it to a RBLN model to be run on RBLN NPUs.

        Args:
            model_id (Union[str, Path]): The model id of the pre-trained model to be loaded.
                It can be downloaded from the HuggingFace model hub or a local path, or a model id of a compiled model using the RBLN Compiler.
            export (Optional[bool]): A boolean flag to indicate whether the model should be compiled.
                If None, it will be determined based on the existence of the compiled model files in the model_id.
            rbln_config (Optional[Union[Dict, RBLNModelConfig]]): Configuration for RBLN model compilation and runtime.
                This can be provided as a dictionary or an instance of the model's configuration class (e.g., `RBLNLlamaForCausalLMConfig` for Llama models).
                For detailed configuration options, see the specific model's configuration class documentation.
            kwargs: Additional keyword arguments. Arguments with the prefix `rbln_` are passed to rbln_config, while the remaining arguments are passed to the HuggingFace library.

        Returns:
            (RBLNModel): A RBLN model instance ready for inference on RBLN NPU devices.
        """

        if isinstance(model_id, Path):
            model_id = model_id.as_posix()

        if export is None:
            export = not cls._is_compiled(
                model_id=model_id,
                token=kwargs.get("token"),
                revision=kwargs.get("revision"),
                force_download=kwargs.get("force_download", False),
                cache_dir=kwargs.get("cache_dir"),
                subfolder=kwargs.get("subfolder", ""),
                local_files_only=kwargs.get("local_files_only", False),
            )

        from_pretrained_method = cls._export if export else cls._from_pretrained
        return from_pretrained_method(model_id=model_id, **kwargs, rbln_config=rbln_config)

    @classmethod
    def compile(
        cls,
        model,
        rbln_compile_config: RBLNCompileConfig,
        create_runtimes: bool,
        device: Union[int, List[int]],
        **kwargs,
    ):
        if create_runtimes:
            runtime_cannot_be_created = tp_and_devices_are_ok(
                tensor_parallel_size=rbln_compile_config.tensor_parallel_size,
                device=device,
                npu=rbln_compile_config.npu,
            )
            if runtime_cannot_be_created:
                raise ValueError(runtime_cannot_be_created)

        compiled_model = rebel.compile_from_torch(
            model,
            input_info=rbln_compile_config.input_info,
            npu=rbln_compile_config.npu,
            tensor_parallel_size=rbln_compile_config.tensor_parallel_size,
            **kwargs,
        )
        return compiled_model

    @classmethod
    def update_rbln_config(
        cls,
        preprocessors: Optional[Union["AutoFeatureExtractor", "AutoProcessor", "AutoTokenizer"]],
        model: "PreTrainedModel",
        model_config: "PretrainedConfig",
        rbln_config: RBLNModelConfig,
    ) -> RBLNModelConfig:
        rbln_config.dtype = model.dtype
        if not cls._supports_non_fp32 and rbln_config.dtype != torch.float32:
            raise NotImplementedError(
                f"Currently, {cls.__name__} does not support non-fp32 dtype. Please use float32 dtype."
            )
        rbln_config = cls._update_rbln_config(
            preprocessors=preprocessors, model=model, model_config=model_config, rbln_config=rbln_config
        )
        rbln_config.freeze()
        if rbln_config.rbln_model_cls_name != cls.__name__:
            raise NameError(
                f"Cannot get the rbln config. {cls.__name__} is not the same as {rbln_config.rbln_model_cls_name}. "
                "This is an internal error. Please report it to the developers."
            )
        return rbln_config

    @classmethod
    def get_hf_class(cls):
        # Lazily loads and caches the corresponding HuggingFace model class.
        # Removes 'RBLN' prefix from the class name to get the original class name
        # (e.g., RBLNLlamaForCausalLM -> LlamaForCausalLM) and imports it from
        # the transformers/diffusers module.

        # Returns:
        #     type: The original HuggingFace model class
        if "_hf_class" not in cls.__dict__ or cls._hf_class is None:
            hf_cls_name = cls.__name__[4:]
            library = importlib.import_module(cls.hf_library_name)
            cls._hf_class = getattr(library, hf_cls_name, None)
        return cls._hf_class

    @classmethod
    def get_rbln_config_class(cls) -> Type[RBLNModelConfig]:
        # Lazily loads and caches the corresponding RBLN model config class.
        if "_rbln_config_class" not in cls.__dict__ or cls._rbln_config_class is None:
            rbln_config_class_name = cls.__name__ + "Config"
            cls._rbln_config_class = get_rbln_config_class(rbln_config_class_name)
        return cls._rbln_config_class

    def can_generate(self):
        return False

    def to(self, *args, **kwargs):
        return self

    def parameters(self):
        # A dummy parameter generator for compatibility.

        # This method mimics the interface of torch.nn.Module.parameters()
        # specifically for code that uses `next(model.parameters())` to infer
        # the device or dtype. It yields a single dummy tensor on CPU with model dtype.

        # Warning:
        #     This does NOT yield the actual model parameters used by the RBLN runtime.
        #     Code relying on iterating through all model parameters will not work as expected.
        yield torch.tensor([1.0], dtype=self.dtype, device=torch.device("cpu"))

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def __repr__(self):
        has_submodules = len(self.rbln_submodules) > 0
        repr_str: str = f"<{self.__class__.__name__}>\n"
        repr_str += f"- Total {len(self.model)} Runtimes"
        repr_str += f" and {len(self.rbln_submodules)} Submodules\n" if has_submodules else "\n"
        repr_str += "[Runtimes]\n"
        repr_str += "\n".join([repr(model) for model in self.model])
        repr_str += "\n"

        if has_submodules > 0:
            for i, submodule in enumerate(self.rbln_submodules):
                repr_str += f"[Submodules {i} : {self._rbln_submodules[i]['name']}]\n"
                repr_str += repr(submodule) + "\n"

        return repr_str

    def __post_init__(self, **kwargs):
        pass

    def save_pretrained(
        self,
        save_directory: Union[str, Path],
        push_to_hub: bool = False,
        **kwargs,
    ):
        """
        Saves a model and its configuration file to a directory, so that it can be re-loaded using the
        [`~optimum.rbln.modeling_base.RBLNBaseModel.from_pretrained`] class method.

        Args:
            save_directory (Union[str, Path]):
                Directory where to save the model file.
            push_to_hub (bool):
                Whether or not to push your model to the HuggingFace model hub after saving it.

        """
        if os.path.isfile(save_directory):
            logger.error(f"Provided path ({save_directory}) should be a directory, not a file")
            return

        # Normalize paths to handle relative paths and symlinks
        real_save_dir = Path(self.model_save_dir).resolve() / self.subfolder
        save_directory_path = Path(save_directory).resolve()

        if not os.path.exists(real_save_dir) or not os.path.isdir(real_save_dir):
            raise FileNotFoundError(
                f"Unable to save the model. The model directory '{real_save_dir}' does not exist or is not accessible. "
                f"Cannot save to the specified destination '{save_directory}'. "
                f"Please ensure the model directory exists and you have the necessary permissions to access it."
            )

        if isinstance(self.config, PretrainedConfig):
            self.config.save_pretrained(real_save_dir)

        if save_directory_path == real_save_dir:
            raise FileExistsError(
                f"Cannot save model to '{save_directory}'. This directory already exists and contains the model files."
            )

        # Create a temporary directory with normalized path
        tmp_dir = str(save_directory_path) + ".tmp"
        try:
            # Remove temporary directory if it exists from a previous failed attempt
            if os.path.exists(tmp_dir):
                shutil.rmtree(tmp_dir)

            # First copy everything to a temporary directory
            shutil.copytree(real_save_dir, tmp_dir)

            # If everything succeeded, move files to target directory
            if os.path.exists(save_directory_path):
                # Merge files from tmp_dir into existing directory
                def _merge_dir(src_root: str, dst_root: str):
                    for name in os.listdir(src_root):
                        src_item = os.path.join(src_root, name)
                        dst_item = os.path.join(dst_root, name)

                        if os.path.islink(src_item) or os.path.isfile(src_item):
                            os.makedirs(os.path.dirname(dst_item), exist_ok=True)
                            if os.path.isdir(dst_item) and not os.path.islink(dst_item):
                                shutil.rmtree(dst_item)
                            os.replace(src_item, dst_item)
                        elif os.path.isdir(src_item):
                            if os.path.islink(dst_item) or os.path.isfile(dst_item):
                                os.remove(dst_item)
                            os.makedirs(dst_item, exist_ok=True)
                            _merge_dir(src_item, dst_item)
                        else:
                            # Fallback for special file types
                            os.replace(src_item, dst_item)

                _merge_dir(tmp_dir, str(save_directory_path))

                # Remove the temporary directory tree after merge
                shutil.rmtree(tmp_dir)
            else:
                # If target doesn't exist, just rename tmp_dir to target
                os.rename(tmp_dir, save_directory_path)

        except Exception as e:
            # Clean up the temporary directory if anything fails
            if os.path.exists(tmp_dir):
                shutil.rmtree(tmp_dir)
            raise e  # Re-raise the exception after cleanup

        if push_to_hub:
            repo_id = kwargs.pop("repo_id", None)
            if repo_id is None:
                raise ValueError("`repo_id` must be provided to push the model to the HuggingFace model hub.")
            return super().push_to_hub(repo_id=repo_id, **kwargs)

    @staticmethod
    def _raise_missing_compiled_file_error(missing_files: List[str]):
        # Raises a KeyError with a message indicating missing compiled model files.

        if len(missing_files) == 1:
            message = f"The rbln model folder is missing the required '{missing_files[0]}.rbln' file. "
        else:
            files_str = ", ".join([f"'{f}.rbln'" for f in missing_files])
            message = (
                "The rbln model folder is missing required files. "
                f"Ensure that {files_str} files are present in the folder. "
            )
        message += (
            "These files are necessary for loading the rbln model. "
            "If these files are missing, please recompile the model using the latest optimum-rbln "
            "and ensure the compilation completes successfully."
        )
        raise KeyError(message)
