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
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Type, Union

from transformers import PretrainedConfig

from ..configuration_utils import RBLNModelConfig, get_rbln_config_class
from ..utils.model_utils import get_rbln_model_cls


if TYPE_CHECKING:
    from transformers import AutoFeatureExtractor, AutoProcessor, AutoTokenizer, PreTrainedModel

    from ..modeling import RBLNModel


class SubModulesMixin:
    """
    _rbln_submodules = [
        {"name": "vision_tower"},
        {"name": "language_model"},
    ]
    """

    _rbln_submodules: List[Dict[str, Any]] = []

    def __init__(self, *, rbln_submodules: Optional[List["RBLNModel"]] = None, **kwargs) -> None:
        if rbln_submodules is None:
            rbln_submodules = []
        for submodule_meta, submodule in zip(self._rbln_submodules, rbln_submodules):
            setattr(self, submodule_meta["name"], submodule)

    @classmethod
    def _get_submodule_config_class(
        cls, cls_name: str, submodule_rbln_config: Dict[str, Any]
    ) -> Type[RBLNModelConfig]:
        if isinstance(submodule_rbln_config, dict) and "cls_name" in submodule_rbln_config:
            config_cls_name = submodule_rbln_config["cls_name"]
            return get_rbln_config_class(config_cls_name)
        return get_rbln_config_class(f"RBLN{cls_name}Config")

    @classmethod
    def _update_submodule_config(
        cls,
        model: "PreTrainedModel",
        rbln_config: RBLNModelConfig,
        preprocessors: Optional[Union["AutoFeatureExtractor", "AutoProcessor", "AutoTokenizer"]],
    ):
        return rbln_config

    @classmethod
    def _update_submodule_rbln_config(
        cls,
        submodule_name: str,
        submodule_cls: Type["RBLNModel"],
        model: "PreTrainedModel",
        submodule_config: PretrainedConfig,
        submodule_rbln_config: RBLNModelConfig,
        preprocessors: Optional[Union["AutoFeatureExtractor", "AutoProcessor", "AutoTokenizer"]],
    ):
        return submodule_rbln_config

    @classmethod
    def _export_submodules_from_model(
        cls, model: "PreTrainedModel", model_save_dir: str, rbln_config: RBLNModelConfig, **kwargs
    ) -> List["RBLNModel"]:
        rbln_submodules = []
        submodule_prefix = getattr(cls, "_rbln_submodule_prefix", None)
        submodule_postfix = getattr(cls, "_rbln_submodule_postfix", None)
        preprocessors = kwargs.pop("preprocessors", [])

        for submodule in cls._rbln_submodules:
            submodule_name = submodule["name"]
            if submodule_prefix is not None:
                torch_submodule: PreTrainedModel = getattr(model, submodule_prefix)
                torch_submodule = getattr(torch_submodule, submodule_name)
            elif submodule_postfix is not None:
                torch_submodule: PreTrainedModel = getattr(model, submodule_name)
                torch_submodule = getattr(torch_submodule, submodule_postfix)
            else:
                torch_submodule: PreTrainedModel = getattr(model, submodule_name)

            cls_name = torch_submodule.__class__.__name__
            submodule_cls: Type["RBLNModel"] = get_rbln_model_cls(f"RBLN{cls_name}")
            submodule_rbln_config = getattr(rbln_config, submodule_name) or {}
            submodule_config_cls = cls._get_submodule_config_class(cls_name, submodule_rbln_config)

            if isinstance(submodule_rbln_config, dict):
                filtered_kwargs = rbln_config.filter_parameters(submodule_config_cls, submodule_rbln_config)
                filtered_kwargs["cls_name"] = submodule_config_cls.__name__
                submodule_rbln_config = submodule_config_cls(**filtered_kwargs)
            elif not isinstance(submodule_rbln_config, submodule_config_cls):
                config_dict = {k: v for k, v in submodule_rbln_config.__dict__.items() if not k.startswith("_")}
                filtered_kwargs = rbln_config.filter_parameters(submodule_config_cls, config_dict)
                filtered_kwargs["cls_name"] = submodule_config_cls.__name__
                submodule_rbln_config = submodule_config_cls(**filtered_kwargs)

            submodule_rbln_config = cls._update_submodule_rbln_config(
                submodule_name=submodule_name,
                submodule_cls=submodule_cls,
                model=model,
                submodule_config=torch_submodule.config,
                submodule_rbln_config=submodule_rbln_config,
                preprocessors=preprocessors,
            )
            setattr(rbln_config, submodule_name, submodule_rbln_config)
            submodule_rbln_config = submodule_cls._update_submodule_config(model, submodule_rbln_config, preprocessors)

            rbln_submodule = submodule_cls.from_model(
                model=torch_submodule,
                config=torch_submodule.config,
                subfolder=submodule_name,
                model_save_dir=model_save_dir,
                rbln_config=submodule_rbln_config,
                **kwargs,
            )

            rbln_submodules.append(rbln_submodule)

        return rbln_submodules

    @classmethod
    def _load_submodules_from_compiled_models(cls, model_save_dir: str, rbln_config: RBLNModelConfig, **kwargs):
        rbln_submodules = []

        for submodule in cls._rbln_submodules:
            submodule_name = submodule["name"]

            # Get cls name for call the constructor of the rbln class
            submodule_rbln_config = getattr(rbln_config, submodule_name)

            # RBLNModelConfig -> RBLNModel
            submodule_cls = get_rbln_model_cls(submodule_rbln_config.rbln_model_cls_name)

            json_file_path = Path(model_save_dir) / submodule_name / "config.json"
            config = PretrainedConfig.from_json_file(json_file_path)

            rbln_submodule = submodule_cls._from_pretrained(
                model_id=model_save_dir,
                config=config,
                subfolder=submodule_name,
                rbln_config=submodule_rbln_config,
                **kwargs,
            )

            # update submodule's rbln_config since it is updated in the from_pretrained method
            setattr(rbln_config, submodule_name, rbln_submodule.rbln_config)
            rbln_submodules.append(rbln_submodule)

        return rbln_submodules

    @classmethod
    def _load_submodules(cls, model_save_dir, rbln_config: RBLNModelConfig, model=None, **kwargs):
        # Two ways :
        # 1. Compile from pytorch object
        # 2. Load from compiled file
        if model is not None:
            return cls._export_submodules_from_model(
                model=model, model_save_dir=model_save_dir, rbln_config=rbln_config, **kwargs
            )

        else:
            return cls._load_submodules_from_compiled_models(
                model_save_dir=model_save_dir, rbln_config=rbln_config, **kwargs
            )
