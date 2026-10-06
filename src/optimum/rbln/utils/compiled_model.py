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

"""Models compiled into rbln functions, saved as function files with the values of their weights."""

import shutil
from collections.abc import Collection, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

import rbln


if TYPE_CHECKING:
    from ..configuration_utils import RBLNCompileConfig


class RBLNWeights:
    """The weights the compiled models of one torch module read.

    While the process has the module they come from its parameters and buffers; once saved, or
    when loaded, the value file `<name>.rblnv` that holds them for every compiled model of the
    module, each weight in the layout its functions read it in.
    """

    def __init__(
        self, name: str, state_dict: Mapping[str, torch.Tensor] | None = None, path: Path | None = None
    ) -> None:
        self.name = name
        self._state_dict = state_dict
        self._path = path
        self._values: rbln.Values | None = None
        self.members: list["RBLNCompiledModel"] = []

    @classmethod
    def of(cls, model: torch.nn.Module, name: str) -> "RBLNWeights":
        state = {
            **dict(model.named_buffers(remove_duplicate=False)),
            **dict(model.named_parameters(remove_duplicate=False)),
        }
        return cls(name, state_dict=state)

    @classmethod
    def load(cls, path: Path) -> "RBLNWeights":
        return cls(path.name.removesuffix(".rblnv"), path=path)

    @property
    def filename(self) -> str:
        return f"{self.name}.rblnv"

    def save(self, directory: Path) -> None:
        """Writes the value file into `directory`, once for all its compiled models; the weights are
        read from it from then on, so the module they came from can go."""
        path = directory / self.filename
        if self._path is not None and self._path.resolve() == path.resolve():
            return
        if self._state_dict is not None:
            functions = [f for member in self.members for f in member.functions]
            rbln.save_values(path, functions, self._state_dict)
        else:
            shutil.copyfile(self._path, path)
        self._state_dict, self._path, self._values = None, path, None

    def materialize(self, functions: list[rbln.Function], devices: Any) -> list[dict[str, Any]]:
        """For each of `functions`, the tensors of its weights on `devices`; functions that hold a
        weight alike share one tensor of it."""
        if self._state_dict is not None:
            return rbln.materialize(functions, self._state_dict, devices=devices)
        if self._values is None:
            self._values = rbln.load_values(self._path)
        return rbln.materialize(functions, values=self._values, devices=devices)


class RBLNCompiledModel:
    """The functions a model is compiled into, one for each bucket of input shapes it was compiled
    for, and the weights they read.

    Saved at `<name>.rbln`, it is the function file of the first bucket there and of the others
    beside it at `<name>.rbln.<bucket>`, with the value file of its weights.
    """

    def __init__(self, functions: list[rbln.Function], weights: RBLNWeights) -> None:
        self.functions = functions
        self.weights = weights
        weights.members.append(self)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        for bucket, function in enumerate(self.functions):
            function.save(bucket_path(path, bucket))
        self.weights.save(path.parent)

    @classmethod
    def load(cls, path: str | Path, buckets: int, weights: RBLNWeights) -> "RBLNCompiledModel":
        path = Path(path)
        return cls([rbln.load(bucket_path(path, bucket)) for bucket in range(buckets)], weights)

    def __repr__(self) -> str:
        return f"RBLNCompiledModel({self.functions!r})"


def bucket_path(path: Path, bucket: int) -> Path:
    return path.with_suffix(f"{path.suffix}.{bucket}") if bucket else path


def input_types(
    input_info: list[tuple[str, list[int], str]], dynamic: Collection[str] = ()
) -> dict[str, rbln.TensorType]:
    """`input_info` as the types of the inputs, by name, in the order the model takes them. The
    outermost axis of an input `dynamic` names is the number of blocks of a paged cache, left open
    from the extent `input_info` gives it on."""
    types = {}
    for name, shape, dtype in input_info:
        if name in dynamic:
            shape = [rbln.Dynamic("num_blocks", min=shape[0]), *shape[1:]]
        types[name] = rbln.TensorType(shape, dtype)
    return types


def compile_model(
    model: torch.nn.Module,
    compile_config: "RBLNCompileConfig",
    weights: RBLNWeights | None = None,
    asked: Mapping[str, rbln.TensorType] | None = None,
    dynamic: Collection[str] = (),
) -> RBLNCompiledModel:
    """Compiles `model` for each bucket of inputs `compile_config` gives.

    `weights` gathers the compiled models of one module, which save their weights in one value
    file; by default the compiled model has a value file of its own. `asked` gives, by name, the
    type an input must have, such as the `type` of an arg another function shares with this one,
    and `dynamic` the inputs whose outermost axis is the number of blocks of a paged cache.
    """
    if weights is None:
        weights = RBLNWeights.of(model, compile_config.compiled_model_name)
    infos = compile_config.input_info if compile_config.is_multiple_input_info else [compile_config.input_info]
    functions = []
    for info in infos:
        types = input_types(info, dynamic)
        if asked:
            types.update({name: t for name, t in asked.items() if name in types})
        functions.append(rbln.compile(model, types, npu=compile_config.npu, devices=compile_config.num_devices or 1))
    compile_config.values_file = weights.filename
    return RBLNCompiledModel(functions, weights)


def _grid(function: rbln.Function) -> list[list[int]]:
    return [[0] * len(chiplets) for chiplets in function.program_nbytes]


def _add(grid: list[list[int]], arg: rbln.Arg, blocks: int | None = None) -> None:
    for shard in arg.shards:
        nbytes = shard["min_nbytes"]
        if blocks is not None and shard.get("step_nbytes"):
            nbytes += (blocks - arg.logical.shape[0]) * shard["step_nbytes"]
        grid[shard["node"]][shard["chiplet"]] += nbytes


def device_usage(
    compiled_models: Sequence[RBLNCompiledModel], shared: Collection[str] = ()
) -> dict[str, list[list[int]]]:
    """By kind, the bytes of device memory `compiled_models` take on each chiplet of each node when
    loaded together, apart from the tensors `shared` names, which they share with one another:
    "Weight", each weight once for the compiled models of one module that hold it alike;
    "Kernel", the program and constants of each function; "Intermediate", the scratch of each; and
    "IO", the device copies of the inputs and outputs each takes from the host."""
    first = compiled_models[0].functions[0]
    usage = {kind: _grid(first) for kind in ("Weight", "Kernel", "Intermediate", "IO")}
    seen = set()
    for compiled_model in compiled_models:
        for function in compiled_model.functions:
            for node, chiplets in enumerate(function.program_nbytes):
                for chiplet, nbytes in enumerate(chiplets):
                    usage["Kernel"][node][chiplet] += nbytes
            for a in function.args:
                if not a.shards or a.name in shared or a.host_bindable:
                    continue
                if a.sources and a.access != "write":
                    key = (id(compiled_model.weights), a.name, a.type_id, tuple(s.get("pool") for s in a.shards))
                    if key not in seen:
                        seen.add(key)
                        _add(usage["Weight"], a)
                else:
                    _add(usage["Intermediate" if a.name == "scratch" else "IO"], a)
    return usage


def tensor_sizes(
    compiled_model: RBLNCompiledModel, names: Collection[str], blocks: int | None = None
) -> dict[str, list[list[int]]]:
    """The bytes a tensor of each arg `names` gives takes on each chiplet of each node, with
    `blocks` along a dynamic outermost axis, or the least the arg takes."""
    function = compiled_model.functions[0]
    sizes = {}
    for name in names:
        grid = _grid(function)
        arg = function.arg(name)
        _add(grid, arg, blocks if arg.logical.dynamic_axes else None)
        sizes[name] = grid
    return sizes
