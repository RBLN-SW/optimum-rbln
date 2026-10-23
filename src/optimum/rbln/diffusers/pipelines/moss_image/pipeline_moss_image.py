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

from ...configurations import RBLNMOSSImagePipelineConfig
from ...modeling_diffusers import RBLNDiffusionMixin
from . import pipeline_moss_image_upstream
from .pipeline_moss_image_upstream import MOSSImagePipeline


# The library name under which the checkpoint's `model_index.json` lists its transformer and scheduler.
_REMOTE_CODE_MODULE = "pipeline_moss_image"


class RBLNMOSSImagePipeline(RBLNDiffusionMixin, MOSSImagePipeline):
    """
    RBLN-accelerated implementation of the MOSS-Image text-to-image pipeline.

    The pipeline compiles the Qwen3.5 text encoder, the transformer and the VAE decoder for one image size and one
    prompt length. The transformer, the scheduler and the pipeline itself ship with the checkpoint as remote code;
    this package carries a copy of that code, so `trust_remote_code` is not needed.

    Examples:
        ```python
        import torch
        from optimum.rbln import RBLNMOSSImagePipeline

        pipe = RBLNMOSSImagePipeline.from_pretrained(
            "OpenMOSS-Team/MOSS-Image-Base",
            export=True,
            torch_dtype={"default": torch.bfloat16, "vae": torch.float32},
            rbln_config={
                "height": 1024,
                "width": 1024,
                "text_encoder": {"num_devices": 4, "device": [0, 1, 2, 3]},
                "transformer": {"num_devices": 4, "device": [4, 5, 6, 7]},
                "vae": {"device": 8},
            },
        )
        image = pipe("A cat", generator=torch.Generator("cpu").manual_seed(42)).images[0]
        ```
    """

    original_class = MOSSImagePipeline
    _hf_class = MOSSImagePipeline
    _rbln_config_class = RBLNMOSSImagePipelineConfig
    _submodules = ["text_encoder", "transformer", "vae"]

    @classmethod
    def load_config(cls, *args: Any, **kwargs: Any) -> Any:
        # The checkpoint lists its transformer and scheduler under the remote-code module `pipeline_moss_image`.
        # Pointing them at the copy in this package makes diffusers load them from here.
        config = super().load_config(*args, **kwargs)
        config_dict = config[0] if isinstance(config, tuple) else config
        for name, value in config_dict.items():
            if isinstance(value, list) and value[0] == _REMOTE_CODE_MODULE:
                config_dict[name] = [pipeline_moss_image_upstream.__name__, value[1]]
        return config

    def handle_additional_kwargs(self, **kwargs: Any) -> dict[str, Any]:
        # The graph reads a fixed number of text tokens; a longer prompt is truncated to it, as `max_sequence_length`
        # already truncates upstream.
        compiled_max_sequence_length = self.transformer.rbln_config.max_sequence_length
        max_sequence_length = kwargs.get("max_sequence_length")
        if max_sequence_length is None or max_sequence_length > compiled_max_sequence_length:
            kwargs["max_sequence_length"] = compiled_max_sequence_length
        return kwargs


__all__ = [
    "RBLNMOSSImagePipeline",
]
