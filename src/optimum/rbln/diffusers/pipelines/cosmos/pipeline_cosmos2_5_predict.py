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

from diffusers import Cosmos2_5_PredictBasePipeline
from diffusers.schedulers import UniPCMultistepScheduler
from transformers import AutoTokenizer

from ....transformers.models.qwen2_5_vl import RBLNQwen2_5_VLForConditionalGeneration
from ....utils.logging import get_logger
from ...configurations.pipelines.configuration_cosmos import RBLNCosmos2_5_PredictBasePipelineConfig
from ...modeling_diffusers import RBLNDiffusionMixin
from ...models.autoencoders.autoencoder_kl_wan import RBLNAutoencoderKLWan
from ...models.transformers.transformer_cosmos import RBLNCosmosTransformer3DModel
from .cosmos_guardrail import RBLNCosmosSafetyChecker


logger = get_logger(__name__)


class RBLNCosmos2_5_PredictBasePipeline(RBLNDiffusionMixin, Cosmos2_5_PredictBasePipeline):
    """
    RBLN-accelerated implementation of Cosmos-Predict2.5 pipeline.

    This pipeline compiles Cosmos-Predict2.5 models to run efficiently on RBLN NPUs, enabling high-performance
    inference for generating videos that follow physical laws with enhanced visual quality.

    One pipeline serves the three conditioning modes: Text2World (no visual input), Image2World
    (`image=...`) and Video2World (`video=...`).
    """

    original_class = Cosmos2_5_PredictBasePipeline
    _submodules = ["text_encoder", "transformer", "vae"]
    _optional_submodules = ["safety_checker"]

    def __init__(
        self,
        text_encoder: RBLNQwen2_5_VLForConditionalGeneration,
        tokenizer: AutoTokenizer,
        transformer: RBLNCosmosTransformer3DModel,
        vae: RBLNAutoencoderKLWan,
        scheduler: UniPCMultistepScheduler,
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
            safety_checker=safety_checker,
        )

    def handle_additional_kwargs(self, **kwargs):
        # Fill in a compiled shape only when the caller left it out, so the default of the HF
        # `__call__` does not stand in for what this pipeline was compiled with. A caller who
        # asks for a different shape keeps their value and gets the runtime's own error.
        compiled_max_seq_len = self.transformer.rbln_config.max_seq_len
        if compiled_max_seq_len is not None and kwargs.get("max_sequence_length") is None:
            kwargs["max_sequence_length"] = compiled_max_seq_len
        compiled_num_frames = self.transformer.rbln_config.num_frames
        if compiled_num_frames is not None and kwargs.get("num_frames") is None:
            kwargs["num_frames"] = compiled_num_frames
        return kwargs

    @classmethod
    def from_pretrained(
        cls,
        model_id: str,
        *,
        export: bool = False,
        safety_checker: RBLNCosmosSafetyChecker | None = None,
        rbln_config: dict[str, Any] | RBLNCosmos2_5_PredictBasePipelineConfig | None = None,
        **kwargs: dict[str, Any],
    ):
        rbln_config, kwargs = cls.get_rbln_config_class().initialize_from_kwargs(rbln_config, **kwargs)
        if safety_checker is None and export:
            safety_checker = RBLNCosmosSafetyChecker(rbln_config=rbln_config.safety_checker)

        return super().from_pretrained(
            model_id, export=export, safety_checker=safety_checker, rbln_config=rbln_config, **kwargs
        )


__all__ = [
    "RBLNCosmos2_5_PredictBasePipeline",
]
