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

from ..configuration_utils import RBLNModelConfig


class RBLNChronos2ModelConfig(RBLNModelConfig):
    """Static batch, context and forecast capacities for Chronos-2 inference."""

    def __init__(
        self,
        batch_size: int = 1,
        context_length: int | None = None,
        prediction_length: int | None = None,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        for name, value in [
            ("batch_size", batch_size),
            ("context_length", context_length),
            ("prediction_length", prediction_length),
        ]:
            if value is not None and (type(value) is not int or value <= 0):
                raise ValueError(f"{name} must be a positive integer, got {value}")
        self.batch_size = batch_size
        self.context_length = context_length
        self.prediction_length = prediction_length


__all__ = ["RBLNChronos2ModelConfig"]
