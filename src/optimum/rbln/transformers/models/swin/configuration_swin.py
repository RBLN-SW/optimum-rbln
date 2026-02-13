# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

from pydantic import Field

from ...configuration_generic import RBLNModelForImageClassificationConfig


class RBLNSwinBackboneConfig(RBLNModelForImageClassificationConfig):
    """Configuration for RBLN Swin backbone."""

    output_hidden_states: bool | None = Field(default=None, description="Whether to output hidden states.")
    output_attentions: bool | None = Field(default=None, description="Whether to output attentions.")
