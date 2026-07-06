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

from ...configuration_generic import RBLNModelForImageClassificationConfig


class RBLNSwinBackboneConfig(RBLNModelForImageClassificationConfig):
    def __init__(
        self,
        image_size: int | tuple[int, int] | None = None,
        batch_size: int | None = None,
        output_hidden_states: bool | None = None,
        output_attentions: bool | None = None,
        **kwargs: Any,
    ):
        """
        Args:
            batch_size (int | None): The batch size for text processing. Defaults to 1.
            kwargs: Additional arguments passed to the parent RBLNModelConfig.

        Raises:
            ValueError: If batch_size is not a positive integer.
        """
        super().__init__(batch_size=batch_size, image_size=image_size, **kwargs)
        self.output_hidden_states = output_hidden_states
        self.output_attentions = output_attentions
