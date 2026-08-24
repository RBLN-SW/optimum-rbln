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


from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Union

from torch.nn import Module

from ...modeling import RBLNModel


if TYPE_CHECKING:
    import rebel


class LoopProcessor(Module, ABC):
    def __init__(self, model: Union[RBLNModel, "rebel.Runtime"]):
        super().__init__()
        self.model = model

    def __repr__(self) -> str:
        return repr(self.model)

    def _is_batch_implemented(self) -> bool:
        return self._forward_batch.__func__ is not LoopProcessor._forward_batch

    def forward(self, *args, force_loop: bool = False, **kwargs) -> Any:
        if not force_loop and self._is_batch_implemented():
            return self._forward_batch(*args, **kwargs)
        else:
            return self._forward_loop(*args, **kwargs)

    def _forward_loop(self, *args, **kwargs) -> Any:
        batch_size = self._get_batch_size(*args, **kwargs)

        if not isinstance(batch_size, int) or batch_size == 0:
            return self._process_outputs([])

        common_inputs = self._prepare_inputs_before_loop(*args, **kwargs)

        outputs = []
        for i in range(batch_size):
            item_args, item_kwargs = self._prepare_inputs_for_iteration(i, common_inputs, *args, **kwargs)
            item_output = self.model(*item_args, **item_kwargs)
            outputs.append(item_output)

        return self._process_outputs(outputs, **kwargs)

    def _forward_batch(self, *args, **kwargs) -> Any:
        raise NotImplementedError("The batch processing logic (_forward_batch) is not implemented in this class.")

    @abstractmethod
    def _get_batch_size(self, *args, **kwargs) -> int:
        pass

    @abstractmethod
    def _prepare_inputs_for_iteration(
        self, index: int, common_inputs: dict[str, Any], *args, **kwargs
    ) -> tuple[list[Any], dict[str, Any]]:
        pass

    def _prepare_inputs_before_loop(self, *args, **kwargs) -> dict[str, Any]:
        pass

    @abstractmethod
    def _process_outputs(self, outputs: list[Any], **kwargs) -> Any:
        pass
