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

from ...configuration_generic import RBLNImageModelConfig, RBLNModelForImageClassificationConfig


class Dinov2ConfigMixin:
    def __init__(self, encoder_chunk_size: int | None = None, **kwargs):
        """Optionally split encoder layers into full-batch jobs sharing a device activation.

        None retains the single compiled model. A positive integer caps the number of
        encoder layers in each job; it does not change the compiled batch size.
        Job duration and memory requirements must be validated on the target NPU.
        """
        if encoder_chunk_size is not None and (type(encoder_chunk_size) is not int or encoder_chunk_size <= 0):
            raise ValueError("encoder_chunk_size must be a positive integer or None")
        super().__init__(**kwargs)
        self.encoder_chunk_size = encoder_chunk_size


class RBLNDinov2ModelConfig(Dinov2ConfigMixin, RBLNImageModelConfig):
    pass


class RBLNDinov2ForImageClassificationConfig(Dinov2ConfigMixin, RBLNModelForImageClassificationConfig):
    pass


__all__ = ["RBLNDinov2ForImageClassificationConfig", "RBLNDinov2ModelConfig"]
