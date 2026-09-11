# Copyright 2026 Rebellions Inc. All rights reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from typing import Any

from diffusers import Cosmos2_5_TransferPipeline
from diffusers.schedulers import UniPCMultistepScheduler
from transformers import AutoTokenizer

from ....transformers.models.qwen2_5_vl import RBLNQwen2_5_VLForConditionalGeneration
from ....utils.logging import get_logger
from ...configurations.pipelines.configuration_cosmos import RBLNCosmos2_5_TransferPipelineConfig
from ...modeling_diffusers import RBLNDiffusionMixin
from ...models.autoencoders.autoencoder_kl_wan import RBLNAutoencoderKLWan
from ...models.controlnets.controlnet_cosmos import RBLNCosmosControlNetModel
from ...models.transformers.transformer_cosmos import RBLNCosmosTransformer3DModel
from .cosmos_guardrail import RBLNCosmosSafetyChecker


logger = get_logger(__name__)


class RBLNCosmos2_5_TransferPipeline(RBLNDiffusionMixin, Cosmos2_5_TransferPipeline):
    """
    RBLN-accelerated implementation of the Cosmos-Transfer2.5 pipeline.

    Transfer2.5 generates videos that follow a control signal (edge/depth/seg/blur maps): a
    4-block ControlNet computes per-step residuals that are injected into the Predict2.5 base
    transformer. Long videos are generated auto-regressively in `num_frames_per_chunk` windows,
    so the compiled shapes are per chunk and the chunk loop runs on the host.

    The official Hub layout keeps the ControlNet on its own revision, so a typical export is:

    ```python
    controlnet = CosmosControlNetModel.from_pretrained(
        "nvidia/Cosmos-Transfer2.5-2B", revision="diffusers/controlnet/general/edge"
    )
    pipe = RBLNCosmos2_5_TransferPipeline.from_pretrained(
        "nvidia/Cosmos-Transfer2.5-2B", revision="diffusers/general",
        controlnet=controlnet, export=True, ...
    )
    ```
    """

    original_class = Cosmos2_5_TransferPipeline
    _submodules = ["text_encoder", "transformer", "vae", "controlnet"]
    _optional_submodules = ["safety_checker"]

    def __init__(
        self,
        text_encoder: RBLNQwen2_5_VLForConditionalGeneration,
        tokenizer: AutoTokenizer,
        transformer: RBLNCosmosTransformer3DModel,
        vae: RBLNAutoencoderKLWan,
        scheduler: UniPCMultistepScheduler,
        controlnet: RBLNCosmosControlNetModel,
        safety_checker: RBLNCosmosSafetyChecker = None,
    ):
        if safety_checker is None:
            safety_checker = RBLNCosmosSafetyChecker()

        super().__init__(
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            transformer=transformer,
            vae=vae,
            scheduler=scheduler,
            controlnet=controlnet,
            safety_checker=safety_checker,
        )

    def handle_additional_kwargs(self, **kwargs):
        if (
            "max_sequence_length" in kwargs
            and kwargs["max_sequence_length"] != self.transformer.rbln_config.max_seq_len
        ):
            logger.warning(
                f"The transformer in this pipeline is compiled with 'max_seq_len={self.transformer.rbln_config.max_seq_len}'. 'max_sequence_length' set by the user will be ignored"
            )
            kwargs.pop("max_sequence_length")

        # The chunk window is a compiled shape; the total output length (num_frames) stays free
        # because transfer generates long videos chunk by chunk on the host.
        compiled_num_frames = self.transformer.rbln_config.num_frames
        if compiled_num_frames is not None:
            if (
                kwargs.get("num_frames_per_chunk") is not None
                and kwargs["num_frames_per_chunk"] != compiled_num_frames
            ):
                logger.warning(
                    f"The transformer in this pipeline is compiled with 'num_frames={compiled_num_frames}' per chunk. "
                    "'num_frames_per_chunk' set by the user will be ignored"
                )
            kwargs["num_frames_per_chunk"] = compiled_num_frames

        for key in ("height", "width"):
            compiled_value = getattr(self.transformer.rbln_config, key, None)
            if compiled_value is None:
                continue
            if kwargs.get(key) is not None and kwargs[key] != compiled_value:
                raise ValueError(
                    f"The transformer in this pipeline is compiled with '{key}={compiled_value}', "
                    f"but '{key}={kwargs[key]}' was requested. Recompile the pipeline with the "
                    f"desired value, or drop '{key}' to use the compiled one."
                )
            kwargs[key] = compiled_value
        return kwargs

    @classmethod
    def from_pretrained(
        cls,
        model_id: str,
        *,
        export: bool = False,
        safety_checker: RBLNCosmosSafetyChecker | None = None,
        rbln_config: dict[str, Any] | RBLNCosmos2_5_TransferPipelineConfig | None = None,
        **kwargs: dict[str, Any],
    ):
        rbln_config, kwargs = cls.get_rbln_config_class().initialize_from_kwargs(rbln_config, **kwargs)
        if safety_checker is None and export:
            safety_checker = RBLNCosmosSafetyChecker(rbln_config=rbln_config.safety_checker)

        return super().from_pretrained(
            model_id, export=export, safety_checker=safety_checker, rbln_config=rbln_config, **kwargs
        )
