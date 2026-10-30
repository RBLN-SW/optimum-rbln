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

from ...modeling_diffusers import RBLNDiffusionMixin
from .pipeline_moss_image_upstream import MOSSImagePipeline


class RBLNMOSSImagePipeline(RBLNDiffusionMixin, MOSSImagePipeline):
    """
    RBLN-accelerated implementation of MOSS-Image pipeline for text-to-image generation.
    """

    original_class = MOSSImagePipeline
    _hf_class = MOSSImagePipeline
    _submodules = ["text_encoder", "transformer", "vae"]

    @classmethod
    def load_config(cls, *args: Any, **kwargs: Any) -> Any:
        config = super().load_config(*args, **kwargs)
        config_dict = config[0] if isinstance(config, tuple) else config
        for name, value in config_dict.items():
            # Load the checkpoint's remote-code module `pipeline_moss_image` from the copy in this package.
            if isinstance(value, list) and value[0] == "pipeline_moss_image":
                config_dict[name] = [MOSSImagePipeline.__module__, value[1]]
        return config

    def handle_additional_kwargs(self, **kwargs):
        # If there is no max_sequence_length of kwargs,
        # it is filled based on the compiled value.
        compiled_max_sequence_length = self.transformer.rbln_config.max_sequence_length
        if compiled_max_sequence_length is not None and kwargs.get("max_sequence_length") is None:
            kwargs["max_sequence_length"] = compiled_max_sequence_length
        return kwargs


__all__ = [
    "RBLNMOSSImagePipeline",
]
