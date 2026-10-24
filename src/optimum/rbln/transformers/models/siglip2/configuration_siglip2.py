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

from ....configuration_utils import RBLNModelConfig


class RBLNSiglip2ModelConfig(RBLNModelConfig):
    def __init__(self, batch_size=None, text_batch_size=None, max_seq_len=None, max_patches=None, **kwargs):
        """Configure independent image/text batches and static token capacities.

        Args:
            batch_size: Image batch size, defaulting to one.
            text_batch_size: Text batch size, defaulting to the image batch size.
            max_seq_len: Text length, defaulting to the checkpoint positional limit.
            max_patches: Image patch budget, defaulting to the native patch budget.
            kwargs: Additional compiler and runtime configuration.
        """
        super().__init__(**kwargs)
        self.batch_size = 1 if batch_size is None else batch_size
        self.text_batch_size = self.batch_size if text_batch_size is None else text_batch_size
        self.max_seq_len = max_seq_len
        self.max_patches = max_patches
        for name in ("batch_size", "text_batch_size", "max_seq_len", "max_patches"):
            value = getattr(self, name)
            if value is not None and (type(value) is not int or value < 1):
                raise ValueError(f"{name} must be a positive integer")


__all__ = ["RBLNSiglip2ModelConfig"]
