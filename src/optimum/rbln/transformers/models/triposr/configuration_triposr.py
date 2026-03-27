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


class RBLNTripoSRForImageTo3DConfig(RBLNModelConfig):
    """
    Configuration class for RBLNTripoSRForImageTo3D.

    This configuration class stores the configuration parameters specific to
    RBLN-optimized TripoSR models for single-image 3D reconstruction tasks.
    """

    def __init__(
        self,
        batch_size: Optional[int] = None,
        image_size: int = 512,
        query_chunk_size: int = 16384,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.batch_size = batch_size or 1
        self.image_size = image_size
        self.query_chunk_size = query_chunk_size
