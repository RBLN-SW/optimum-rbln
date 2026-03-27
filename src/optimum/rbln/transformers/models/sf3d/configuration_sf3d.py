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

from typing import Any, Optional

from ....configuration_utils import RBLNModelConfig


class RBLNSF3DForImageTo3DConfig(RBLNModelConfig):
    """
    Configuration class for RBLNSF3DForImageTo3D.

    This configuration class stores the configuration parameters specific to
    RBLN-optimized SF3D (Stable Fast 3D) models for single-image 3D reconstruction tasks.

    Args:
        batch_size: Batch size for image encoding (default 1).
        image_size: Input image size (default 512).
        query_chunk_size: Chunk size for NPU triplane query + decoder.
            The isosurface grid and UV texels are processed in fixed-size
            chunks on the NPU. Larger values give fewer NPU calls but use
            more memory. Set to ``None`` to run triplane query and decoder
            entirely on CPU (default 131072).
    """

    def __init__(
        self,
        batch_size: Optional[int] = None,
        image_size: int = 512,
        query_chunk_size: Optional[int] = 131072,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.batch_size = batch_size or 1
        self.image_size = image_size
        self.query_chunk_size = query_chunk_size
