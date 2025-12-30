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

import os
import pathlib
from functools import partial
from typing import Any, Dict, Optional, Tuple, Union
from unittest.mock import patch

import rebel
import torch
from diffusers.utils import is_cosmos_guardrail_available
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer, SiglipProcessor

from .... import RBLNAutoModelForCausalLM, RBLNSiglipVisionModel
from ....utils.runtime_utils import RBLNPytorchRuntime, UnavailableRuntime
from .configuration_cosmos_guardrail import RBLNCosmosSafetyCheckerConfig


if is_cosmos_guardrail_available():
    from cosmos_guardrail import CosmosSafetyChecker
    from cosmos_guardrail.cosmos_guardrail import (
        COSMOS_GUARDRAIL_CHECKPOINT,
        Blocklist,
        GuardrailRunner,
        LlamaGuard3,
        ModelConfig,
        RetinaFaceFilter,
        SafetyClassifier,
        SigLIPEncoder,
        VideoContentSafetyFilter,
        VideoSafetyModel,
    )
    from retinaface.data import cfg_re50

    COSMOS_AVAILABLE = True
else:
    COSMOS_AVAILABLE = False

    class FailToImportCosmosGuardrail(torch.nn.Module): ...

    class CosmosSafetyChecker(FailToImportCosmosGuardrail): ...

    COSMOS_GUARDRAIL_CHECKPOINT = None

    class LlamaGuard3(FailToImportCosmosGuardrail): ...

    class Blocklist(FailToImportCosmosGuardrail): ...

    class GuardrailRunner(FailToImportCosmosGuardrail): ...

    class ModelConfig(FailToImportCosmosGuardrail): ...

    class RetinaFaceFilter(FailToImportCosmosGuardrail): ...

    class SafetyClassifier(FailToImportCosmosGuardrail): ...

    class SigLIPEncoder(FailToImportCosmosGuardrail): ...

    class VideoContentSafetyFilter(FailToImportCosmosGuardrail): ...

    class VideoSafetyModel(FailToImportCosmosGuardrail): ...

    cfg_re50 = None


def is_compiled_dir(dir: str) -> bool:
    # walk directory and check if there is any *.rbln files in that dir.
    if not os.path.exists(dir):
        return False

    for _, _, files in os.walk(dir):
        for file in files:
            if file.endswith(".rbln"):
                return True
    return False


def get_image_features(
    self,
    pixel_values: torch.Tensor,
    return_dict: bool = True,
    output_attentions: bool = False,
    output_hidden_states: bool = False,
    interpolate_pos_encoding: bool = False,
):
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    return self(
        pixel_values,
        return_dict=return_dict,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        interpolate_pos_encoding=interpolate_pos_encoding,
    )[1]


class RBLNSigLIPEncoder(SigLIPEncoder):
    def __init__(
        self,
        model_name: str = "google/siglip-so400m-patch14-384",
        checkpoint_id: str = COSMOS_GUARDRAIL_CHECKPOINT,
        rbln_config: Optional[RBLNCosmosSafetyCheckerConfig] = None,
    ):
        torch.nn.Module.__init__(self)
        if is_compiled_dir(checkpoint_id):
            self.checkpoint_dir = (
                pathlib.Path(checkpoint_id) / "video_content_safety_filter" / "siglip_encoder"
            ).as_posix()
            self.processor = SiglipProcessor.from_pretrained(self.checkpoint_dir)

            # We don't use RBLNSiglipModel, but we need to override get_image_features to return pooler_output
            self.model = RBLNSiglipVisionModel.from_pretrained(
                self.checkpoint_dir, rbln_config=rbln_config.siglip_encoder
            )
        else:
            super().__init__(model_name, checkpoint_id)
            model = self.model
            del self.model
            self.model = RBLNSiglipVisionModel.from_model(model, rbln_config=rbln_config.siglip_encoder)
        self.rbln_config = rbln_config

        # Override get_image_features to return pooler_output
        self.model.get_image_features = lambda *args, **kwargs: get_image_features(self.model, *args, **kwargs)

    def save_pretrained(self, checkpoint_id: str):
        cache_dir = (pathlib.Path(checkpoint_id) / "video_content_safety_filter" / "siglip_encoder").as_posix()
        self.model.save_pretrained(cache_dir)
        self.processor.save_pretrained(cache_dir)


class RBLNRetinaFaceFilter(RetinaFaceFilter):
    def __init__(
        self,
        checkpoint_id: str = COSMOS_GUARDRAIL_CHECKPOINT,
        batch_size: int = 1,
        confidence_threshold: float = 0.7,
        rbln_config: Optional[RBLNCosmosSafetyCheckerConfig] = None,
    ):
        torch.nn.Module.__init__(self)
        if is_compiled_dir(checkpoint_id):
            self.compiled_model = rebel.RBLNCompiledModel(
                pathlib.Path(checkpoint_id) / "face_blur_filter" / "retinaface.rbln"
            )
            self.cfg = cfg_re50
            self.batch_size = batch_size
            self.confidence_threshold = confidence_threshold
            self.cfg["pretrain"] = False
        else:
            with patch("torch.load", partial(torch.load, weights_only=True, map_location=torch.device("cpu"))):
                super().__init__(checkpoint_id)
            net = self.net
            del self.net
            self.compiled_model = rebel.compile_from_torch(
                net,
                input_info=[
                    (
                        "frames",
                        [
                            self.batch_size,
                            3,
                            rbln_config.face_blur_filter.image_size[0],
                            rbln_config.face_blur_filter.image_size[1],
                        ],
                        "float32",
                    )
                ],
                npu=rbln_config.face_blur_filter.npu,
            )

        self.rbln_config = rbln_config

        try:
            runtime = (
                rebel.Runtime(
                    self.compiled_model,
                    tensor_type="pt",
                    device=self.rbln_config.face_blur_filter.device,
                    activate_profiler=rbln_config.face_blur_filter.activate_profiler,
                )
                if self.rbln_config.face_blur_filter.create_runtimes
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

        self.net = RBLNPytorchRuntime(runtime)

    def save_pretrained(self, checkpoint_id: str):
        cache_path = pathlib.Path(checkpoint_id) / "face_blur_filter"
        cache_path.mkdir(parents=True, exist_ok=True)
        self.compiled_model.save(cache_path / "retinaface.rbln")


class RBLNVideoSafetyModel(VideoSafetyModel):
    def __init__(
        self,
        config: ModelConfig,
        checkpoint_id: str = COSMOS_GUARDRAIL_CHECKPOINT,
        rbln_config: Optional["RBLNCosmosSafetyCheckerConfig"] = None,
    ):
        torch.nn.Module.__init__(self)
        self.config = config
        self.num_classes = config.num_classes
        self.rbln_config = rbln_config

        if is_compiled_dir(checkpoint_id):
            self.compiled_model = rebel.RBLNCompiledModel(
                pathlib.Path(checkpoint_id) / "video_content_safety_filter" / "safety_filter.rbln"
            )
        else:
            # Load model from checkpoint
            network = SafetyClassifier(
                input_size=self.rbln_config.video_safety_model.input_size, num_classes=self.num_classes
            )
            network.eval()

            checkpoint_dir = snapshot_download(checkpoint_id)
            checkpoint_dir = (pathlib.Path(checkpoint_dir) / "video_content_safety_filter").as_posix()

            safety_filter_local_path = os.path.join(checkpoint_dir, "safety_filter.pt")
            checkpoint = torch.load(safety_filter_local_path, weights_only=True)
            network.load_state_dict({k.replace("network.", ""): v for k, v in checkpoint["model"].items()})

            self.compiled_model = rebel.compile_from_torch(
                network,
                input_info=[
                    (
                        "data",
                        [
                            self.rbln_config.video_safety_model.batch_size,
                            self.rbln_config.video_safety_model.input_size,
                        ],
                        "float32",
                    )
                ],
                npu=self.rbln_config.video_safety_model.npu,
            )

        try:
            runtime = (
                rebel.Runtime(
                    self.compiled_model,
                    tensor_type="pt",
                    device=self.rbln_config.video_safety_model.device,
                    activate_profiler=rbln_config.video_safety_model.activate_profiler,
                )
                if self.rbln_config.video_safety_model.create_runtimes
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

        self.network = RBLNPytorchRuntime(runtime)

    def save_pretrained(self, checkpoint_id: str):
        cache_path = pathlib.Path(checkpoint_id) / "video_content_safety_filter"
        cache_path.mkdir(parents=True, exist_ok=True)
        self.compiled_model.save(cache_path / "safety_filter.rbln")

    def parameters(self):
        yield torch.tensor([1.0], dtype=torch.float32, device=torch.device("cpu"))


class RBLNVideoContentSafetyFilter(VideoContentSafetyFilter):
    def __init__(
        self,
        checkpoint_id: str = COSMOS_GUARDRAIL_CHECKPOINT,
        rbln_config: Optional["RBLNCosmosSafetyCheckerConfig"] = None,
    ):
        torch.nn.Module.__init__(self)
        self.rbln_config = rbln_config
        self.encoder = RBLNSigLIPEncoder(checkpoint_id=checkpoint_id, rbln_config=rbln_config)

        model_config = ModelConfig(input_size=1152, num_classes=7)
        self.model = RBLNVideoSafetyModel(model_config, checkpoint_id=checkpoint_id, rbln_config=rbln_config)

    def save_pretrained(self, checkpoint_id: str):
        self.model.save_pretrained(checkpoint_id)
        self.encoder.save_pretrained(checkpoint_id)


class RBLNLlamaGuard3(LlamaGuard3):
    def __init__(
        self,
        checkpoint_id: str = COSMOS_GUARDRAIL_CHECKPOINT,
        base_model_id: str = "meta-llama/Llama-Guard-3-8B",
        rbln_config: Optional[RBLNCosmosSafetyCheckerConfig] = None,
    ) -> None:
        if is_compiled_dir(checkpoint_id):
            torch.nn.Module.__init__(self)
            cache_dir = pathlib.Path(checkpoint_id) / "llamaguard3"
            self.tokenizer = AutoTokenizer.from_pretrained(cache_dir)
            self.model = RBLNAutoModelForCausalLM.from_pretrained(cache_dir, rbln_config=rbln_config.llamaguard3)

        else:
            super().__init__(checkpoint_id, base_model_id)
            model = self.model
            del self.model
            self.model = RBLNAutoModelForCausalLM.from_model(model, rbln_config=rbln_config.llamaguard3)

        self.rbln_config = rbln_config
        self.dtype = torch.bfloat16
        self.device = torch.device("cpu")

    def save_pretrained(self, checkpoint_id: str):
        cache_dir = pathlib.Path(checkpoint_id) / "llamaguard3"
        self.model.save_pretrained(cache_dir)
        self.tokenizer.save_pretrained(cache_dir)


class RBLNCosmosSafetyChecker(CosmosSafetyChecker):
    """
    RBLN-accelerated implementation of Cosmos Safety Checker.
    """

    def __init__(
        self,
        checkpoint_id: str = COSMOS_GUARDRAIL_CHECKPOINT,
        llamaguard_model_id: str = "meta-llama/Llama-Guard-3-8B",
        rbln_config: Optional[RBLNCosmosSafetyCheckerConfig] = None,
    ) -> None:
        torch.nn.Module.__init__(self)
        if not COSMOS_AVAILABLE:
            raise ImportError(
                "`cosmos_guardrail` is not installed. Please install it to use the safety checker for Cosmos: `pip install cosmos_guardrail`."
            )

        if rbln_config is None:
            rbln_config = RBLNCosmosSafetyCheckerConfig()
        elif isinstance(rbln_config, dict):
            rbln_config = RBLNCosmosSafetyCheckerConfig(**rbln_config)

        self.text_guardrail = GuardrailRunner(
            safety_models=[
                Blocklist(COSMOS_GUARDRAIL_CHECKPOINT),  # Changed since it cannot be saved
                RBLNLlamaGuard3(
                    checkpoint_id=checkpoint_id,
                    base_model_id=llamaguard_model_id,
                    rbln_config=rbln_config,
                ),
            ]
        )

        self.video_guardrail = GuardrailRunner(
            safety_models=[RBLNVideoContentSafetyFilter(checkpoint_id=checkpoint_id, rbln_config=rbln_config)],
            postprocessors=[RBLNRetinaFaceFilter(checkpoint_id=checkpoint_id, rbln_config=rbln_config)],
        )

        self.rbln_config = rbln_config

    def save_pretrained(self, save_dir: str):
        for text_safety_models in self.text_guardrail.safety_models:
            if isinstance(text_safety_models, RBLNLlamaGuard3):
                text_safety_models.save_pretrained(save_dir)

        for video_safety_models in self.video_guardrail.safety_models:
            if isinstance(video_safety_models, RBLNVideoContentSafetyFilter):
                video_safety_models.save_pretrained(save_dir)

        for postprocessors in self.video_guardrail.postprocessors:
            if isinstance(postprocessors, RBLNRetinaFaceFilter):
                postprocessors.save_pretrained(save_dir)

        self.rbln_config._frozen = True  # Ad-hoc to save config
        self.rbln_config.save(save_dir)

    @classmethod
    def from_pretrained(
        cls,
        checkpoint_id: str,
        rbln_config: Optional[RBLNCosmosSafetyCheckerConfig] = None,
        subfolder: Optional[str] = None,
        export: Optional[bool] = True,
        **kwargs,
    ):
        rbln_config, kwargs = cls.prepare_rbln_config(rbln_config=rbln_config, **kwargs)

        if len(kwargs) > 0:
            raise ValueError(f"Unexpected arguments: {kwargs.keys()}")

        if subfolder is not None:
            checkpoint_id = os.path.join(checkpoint_id, subfolder)

        return cls(checkpoint_id=checkpoint_id, rbln_config=rbln_config)

    @classmethod
    def prepare_rbln_config(
        cls, rbln_config: Optional[Union[Dict[str, Any], RBLNCosmosSafetyCheckerConfig]] = None, **kwargs
    ) -> Tuple[RBLNCosmosSafetyCheckerConfig, Dict[str, Any]]:
        # Extract rbln-config from kwargs and convert it to RBLNCosmosSafetyCheckerConfig.
        rbln_config, kwargs = RBLNCosmosSafetyCheckerConfig.initialize_from_kwargs(rbln_config, **kwargs)
        return rbln_config, kwargs
