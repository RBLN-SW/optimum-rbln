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

import math
import re
import threading
from collections.abc import Mapping, Sequence
from functools import cache
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np
import torch
from rebel import v2


if TYPE_CHECKING:
    from .compiled_model import RBLNCompiledModel


def device_count() -> int:
    """The NPUs the process may use: those `RBLN_DEVICES` names, or every NPU of the host."""
    try:
        return v2.device_count()
    except RuntimeError:
        return 0


def npu_is_available(device: int = 0) -> bool:
    return 0 <= device < device_count()


@cache
def get_npu_name(device: int = 0) -> str | None:
    """The kind of NPU device `device` of the process is, or None when it has no such device."""
    return v2.Device(device).npu if npu_is_available(device) else None


def _resolve_npu(npu: str | None = None) -> str:
    if npu is None:
        npu = get_npu_name(0)
        if npu is None:
            raise RuntimeError("No NPU is available to get available DRAM size.")
    return npu


def _dram_spec(npu: str) -> tuple[int, int]:
    if npu.startswith("RBLN-CR"):
        return 144 * 2**30, 1 * 2**30
    elif npu.startswith("RBLN-CA"):
        return 16 * 2**30, 288 * 2**20
    raise ValueError(f"Unknown npu name: {npu}")


def get_available_dram_per_chiplet(num_chiplets: int, npu: str | None = None) -> int:
    """
    Get the available DRAM per chiplet. Device DRAM is physically partitioned across
    chiplets, so an allocation pinned to a chiplet must fit within this amount, not the
    node total.

    Args:
        num_chiplets : int
            Number of chiplets the device DRAM is split across.
        npu : str | None, default=None
            The NPU to get the available DRAM size. Resolved from the local device if None.

    Returns:
        int
            The available DRAM per chiplet in bytes.
    """
    npu = _resolve_npu(npu)
    dram_nbytes, sys_per_chiplet = _dram_spec(npu)
    return dram_nbytes // num_chiplets - sys_per_chiplet


_BYTE_UNITS = {"B": 1, "KB": 2**10, "MB": 2**20, "GB": 2**30, "TB": 2**40}


def parse_byte_size(value: int | str) -> int:
    """Parse a byte size given as an int (bytes) or a string like "10GB" / "512MB".

    Units are case-insensitive and binary (KB=2**10 ... TB=2**40). Returns a positive int of bytes.
    """
    if isinstance(value, bool):
        raise ValueError(f"Invalid byte size: {value!r}")
    if isinstance(value, int):
        nbytes = value
    elif isinstance(value, str):
        match = re.fullmatch(r"\s*(\d+)\s*([KMGT]?B)?\s*", value, re.IGNORECASE)
        if not match:
            raise ValueError(
                f"Invalid byte size {value!r}. Expected an integer optionally suffixed with "
                "B, KB, MB, GB, or TB (e.g. '10GB')."
            )
        unit = match.group(2)
        nbytes = int(match.group(1)) * (_BYTE_UNITS[unit.upper()] if unit else 1)
    else:
        raise ValueError(f"Invalid byte size type: {type(value).__name__}")
    if nbytes <= 0:
        raise ValueError(f"Byte size must be positive, got {nbytes}.")
    return nbytes


def resolve_npu_or_none(npu: str | None = None) -> str | None:
    """The target NPU: the name pinned on the config, else the attached device's, else None.

    Unlike `_resolve_npu` this does not raise — callers that only pick defaults or bounds must
    keep working on a host with no NPU attached.
    """
    if npu is not None:
        return npu
    return get_npu_name(0)


def npu_is_cr13_or_later(npu: str | None = None) -> bool:
    """Whether the NPU is RBLN-CR13 or later — every CR except CR03 (rebel-compiler's `_is_evt1`)."""
    npu = resolve_npu_or_none(npu)
    if not npu:
        return False
    normalized = normalize_npu(npu)
    return normalized.startswith("RBLN-CR") and normalized != "RBLN-CR0"


def normalize_npu(npu: str) -> str:
    """Normalize the NPU string by removing the form factor."""
    match = re.match(r"(RBLN-CA|RBLN-CR)(\d+)", npu)
    if match:
        prefix, num = match.groups()
        if len(num) == 1:
            # Convert "RBLN-CAx" → "RBLN-CA0"
            # (e.g., "RBLN-CA2" -> "RBLN-CA0")
            npu = f"{prefix}0"
        elif len(num) == 2:
            # Strip form factor (e.g., "RBLN-CA15" → "RBLN-CA1")
            npu = f"{prefix}{num[:-1]}"
    return npu


def tp_and_devices_are_ok(
    num_devices: int | None = None,
    device: int | list[int] | None = None,
    npu: str | None = None,
) -> str | None:
    if num_devices is None:
        num_devices = 1

    if device is None:
        device = list(range(num_devices))
    elif isinstance(device, int):
        device = [device]
    elif isinstance(device, list):
        if any(not isinstance(d, int) for d in device):
            return "Device must be a(n) (list of) integer(s)."
        if len(device) != num_devices:
            return f"The number of devices ({len(device)}) does not match `num_devices` ({num_devices})."
    else:
        return f"Invalid device: {device}"

    for device_id in device:
        if device_id < 0:  # if any device is dummy device, skip it
            return None
        if get_npu_name(device_id) is None:
            return (
                f"Device {device_id} is not a valid NPU device. Please check your NPU status with 'rbln-smi' command."
            )

    if device_count() < num_devices:
        return f"`num_devices` ({num_devices}) is greater than the number of available devices {device_count()}."

    if npu is not None:
        for device_id in device:
            npu_name = get_npu_name(device_id)
            if normalize_npu(npu_name) != normalize_npu(npu):
                return f"Device {device_id} ({npu_name}) is not on the same NPU as {npu}."

    return None


def open_devices(device: int | Sequence[int] | None, count: int, npu: str) -> list[v2.Device]:
    """The devices a function compiled for `count` devices of kind `npu` runs on when given
    `device`: devices of the process by number, the first `count` when None, or for a negative
    number a dummy device of the kind, which takes no NPU memory and runs nothing."""
    ids = list(range(count)) if device is None else [device] if isinstance(device, int) else list(device)
    if any(i < 0 for i in ids):
        if count != 1:
            raise RuntimeError(f"a dummy device runs a model compiled for one device, not {count}")
        return [v2.Device.open_dummy(npu)]
    if len(ids) != count:
        raise RuntimeError(f"The model is compiled for {count} devices, not {ids}.")
    return [v2.Device(ids[0])] if count == 1 else list(v2.Device.group(ids))


def _is_scratch(arg: v2.Arg) -> bool:
    return arg.name == "scratch" and not arg.sources


def _torch_of(value: Any) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value
    array = np.asarray(value)
    if array.dtype.name == "bfloat16":
        return torch.from_numpy(array.view(np.int16)).view(torch.bfloat16)
    return torch.from_numpy(array)


def _decodable_into(arg: v2.Arg, target: torch.Tensor) -> torch.Tensor | None:
    """`target` viewed as the logical value of result `arg`, to decode straight into, or None when
    it cannot take that value as it is."""
    logical = arg.logical
    if logical.dynamic_axes or target.device.type != "cpu" or target.requires_grad:
        return None
    if not target.is_contiguous() or str(target.dtype).removeprefix("torch.") != logical.dtype:
        return None
    if target.numel() != math.prod(logical.shape):
        return None
    return target.view(list(logical.shape))


class RBLNRuntime:
    """A compiled model loaded on NPUs, which a call runs on torch tensors.

    Each bucket of the compiled model runs with an executor of its own, and a call runs the one
    whose input shapes are those of the inputs given. Every executor binds the weights and the
    tensors given as `tensors` that its function takes, such as a KV cache other compiled models
    share; the inputs left, in the order the model takes them, are what a call gives. A result that
    is one of these inputs comes back as the value given for it, and one that is a tensor the model
    updates in place as an empty tensor.
    """

    def __init__(
        self,
        compiled_model: "RBLNCompiledModel",
        devices: list[v2.Device],
        weights: list[dict[str, Any]],
        tensors: Mapping[str, Any] | None = None,
    ) -> None:
        self.compiled_model = compiled_model
        self.devices = devices
        names = {a.name for a in compiled_model.functions[0].args}
        self.shared = {name: t for name, t in (tensors or {}).items() if name in names}
        self.executors: list[v2.Executor] = []
        for function, held in zip(compiled_model.functions, weights, strict=True):
            bound = {**held, **self.shared}
            for a in function.args:
                if _is_scratch(a) and a.name not in bound:
                    bound[a.name] = v2.empty_like(a, devices)
            self.executors.append(v2.Executor(function, devices, tensors=bound))
        first = compiled_model.functions[0]
        bound = set(self.executors[0].tensors)
        self.input_names = [
            a.name for a in first.args if a.shards and a.used and a.access == "read" and a.name not in bound
        ]
        self._results = [[function.arg(name) for name in function.results] for function in compiled_model.functions]
        self._buckets = {
            tuple(tuple(function.arg(name).logical.shape) for name in self.input_names): index
            for index, function in enumerate(compiled_model.functions)
        }

    def __call__(self, *args: Any, out: Any = None, **kwargs: Any) -> Any:
        return self.forward(*args, out=out, **kwargs)

    def forward(self, *args: Any, out: Any = None, **kwargs: Any) -> Any:
        inputs = list(args) + [kwargs[name] for name in self.input_names[len(args) :] if name in kwargs]
        index = self._bucket(inputs)
        executor, results = self.executors[index], self._results[index]
        if out is None:
            outputs = [
                torch.empty(0) if isinstance(result, (v2.Tensor, v2.HostTensor)) else _torch_of(result)
                for result in executor(*inputs)
            ]
            return outputs[0] if len(outputs) == 1 else outputs
        targets = [out] if isinstance(out, torch.Tensor) else list(out)
        targets += [None] * (len(results) - len(targets))
        into = [
            _decodable_into(a, t) if t is not None and a.access == "write" else None
            for a, t in zip(results, targets, strict=True)
        ]
        outputs = []
        for target, given, result in zip(targets, into, executor(*inputs, out=into), strict=True):
            if isinstance(result, (v2.Tensor, v2.HostTensor)):
                outputs.append(torch.empty(0))
            elif given is not None:
                outputs.append(target if target is not None else given)
            elif target is not None:
                outputs.append(target.copy_(_torch_of(result).reshape(target.shape)))
            else:
                outputs.append(_torch_of(result))
        return outputs[0] if len(outputs) == 1 else outputs

    def _bucket(self, inputs: list[Any]) -> int:
        if len(self._buckets) == 1:
            return 0
        shapes = tuple(tuple(np.shape(x)) for x in inputs)
        index = self._buckets.get(shapes)
        if index is None:
            raise TypeError(
                f"No bucket takes inputs of shapes {list(shapes)}; the buckets take {list(self._buckets)}."
            )
        return index

    def copy_kv_cache(self, src_block: int, dst_block: int) -> None:
        """Copies block `src_block` of every shared tensor into block `dst_block` on its devices,
        the blocks of a paged KV cache being its outermost axis. A block is copied whole: a request
        reads no position of it past those it writes after the ones it shares."""
        function = self.compiled_model.functions[0]
        for name, tensor in self.shared.items():
            arg = function.arg(name)
            v2.copy(arg.view(tensor, dst_block, dst_block + 1), arg.view(tensor, src_block, src_block + 1))

    def __repr__(self) -> str:
        return f"RBLNRuntime({self.compiled_model!r}, devices={self.devices!r})"


def zeroed_tensors(
    function: v2.Function, names: Sequence[str], devices: list[v2.Device], **axes: int
) -> dict[str, v2.Tensor]:
    """A zeroed tensor of each arg of `function` that `names` gives, on `devices`, with the dynamic
    axes `axes` names at their values, for the compiled models that share it to bind."""
    tensors = {}
    for name in names:
        arg = function.arg(name)
        dynamic = {d.name for d in arg.logical.dynamic_axes}
        tensors[name] = v2.empty_like(arg, devices, **{k: v for k, v in axes.items() if k in dynamic})
        for shard in tensors[name].shards:
            shard.device.fill(shard, 0, shard.nbytes, 0)
    return tensors


def create_runtimes(
    compiled_models: Sequence["RBLNCompiledModel"],
    devices: Sequence[int | Sequence[int] | None],
    tensors: Mapping[str, Any] | None = None,
) -> list[RBLNRuntime]:
    """Loads `compiled_models` on their devices, each compiled model on the devices at its place in
    `devices`. Compiled models of one module on the same devices share the tensors of the weights
    they hold alike, and every runtime binds `tensors` where its functions take them."""
    opened = []
    for compiled_model, device in zip(compiled_models, devices, strict=True):
        first = compiled_model.functions[0]
        opened.append(open_devices(device, first.num_devices, first.npu))
    held: dict[int, list[dict[str, Any]]] = {}
    groups: dict[tuple[int, tuple[int, ...]], list[int]] = {}
    for i, (compiled_model, ds) in enumerate(zip(compiled_models, opened, strict=True)):
        groups.setdefault((id(compiled_model.weights), tuple((d.id, d.npu) for d in ds)), []).append(i)
    for members in groups.values():
        functions = [f for i in members for f in compiled_models[i].functions]
        weights = compiled_models[members[0]].weights.materialize(functions, opened[members[0]])
        for i in members:
            held[i] = weights[: len(compiled_models[i].functions)]
            weights = weights[len(compiled_models[i].functions) :]
    return [
        RBLNRuntime(compiled_model, ds, held[i], tensors)
        for i, (compiled_model, ds) in enumerate(zip(compiled_models, opened, strict=True))
    ]


class RBLNPytorchRuntime:
    mandatory_members: ClassVar[list[str]] = []

    def __init__(self, runtime: "RBLNRuntime", **kwargs) -> None:
        self.runtime = runtime
        for key, value in kwargs.items():
            setattr(self, key, value)
        for mandatory_member in self.mandatory_members:
            if mandatory_member not in kwargs:
                raise AttributeError(f"`{mandatory_member}` should be assigned to {self.__class__.__name__} objects.")

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.forward(*args, **kwds)

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        # filtering useless args or kwarg such as None.
        tensor_args = [arg for arg in args if isinstance(arg, torch.Tensor)]
        tensor_kwargs = {
            key: value for key, value in kwargs.items() if isinstance(value, torch.Tensor) or key == "out"
        }
        return self.runtime(*tensor_args, **tensor_kwargs)

    def __repr__(self) -> str:
        return repr(self.runtime)

    def parameters(self):
        yield torch.tensor([1.0], dtype=torch.float32, device=torch.device("cpu"))


class UnavailableRuntime:
    """
    A placeholder class used when model runtimes are not created.

    This class is returned by RBLNBaseModel._from_compiled_models when rbln_config.create_runtimes=False.
    It provides proper error messages when users attempt to use a model that was loaded without
    runtime creation.

    Usage:
        1. When compiling models on machines without NPU hardware
        2. When preparing models for later deployment
        3. When only model compilation is needed, not inference

    To use a model with runtimes, either:
        - Load the model with from_pretrained(..., rbln_create_runtimes=True)
        - Or set rbln_config={"create_runtimes": True} during loading
    """

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Raises a RuntimeError when the model is called without runtimes."""
        raise self.forward(*args, **kwargs)

    def __len__(self) -> int:
        """Returns 0 since no runtimes are available."""
        return 0

    def __getitem__(self, idx: int) -> Any:
        """Returns self for any index, allowing iteration to work with appropriate errors."""
        return self

    def __iter__(self):
        """Returns an iterator with self as the only item."""
        return iter([self])

    def forward(self, *args: list["torch.Tensor"], **kwargs: "torch.Tensor"):
        """Raises a detailed RuntimeError explaining why inference cannot be performed."""
        raise RuntimeError(
            "Cannot perform inference: RBLN runtime is not available.\n\n"
            "This model was loaded with create_runtimes=False. To use this model for inference:\n"
            "1. Load the model with runtime creation enabled:\n"
            "   model = RBLNModel.from_pretrained(..., rbln_create_runtimes=True)\n"
            "2. Ensure your NPU hardware is properly configured (check with 'rbln-smi' command)\n"
            "3. If you're on a machine without NPU hardware, you need to transfer the model files\n"
            "   to a compatible system with NPU support."
        )

    def __repr__(self) -> str:
        """Returns a detailed string representation of the UnavailableRuntime."""
        return "<UnavailableRuntime: Model loaded without runtime creation (create_runtimes=False)>"


class ContextRblnConfig:
    _local = threading.local()

    def __init__(
        self,
        device=None,
        device_map=None,
        create_runtimes=None,
        activate_profiler=None,
        timeout=None,
    ):
        self.device = device
        self.device_map = device_map
        self.create_runtimes = create_runtimes
        self.activate_profiler = activate_profiler
        self.timeout = timeout
        self._previous_context = None

    def __enter__(self):
        self._previous_context = {
            "device": getattr(self._local, "device", None),
            "device_map": getattr(self._local, "device_map", None),
            "create_runtimes": getattr(self._local, "create_runtimes", None),
            "activate_profiler": getattr(self._local, "activate_profiler", None),
            "timeout": getattr(self._local, "timeout", None),
        }

        if self.device is not None:
            self._local.device = self.device
        if self.device_map is not None:
            self._local.device_map = self.device_map
        if self.create_runtimes is not None:
            self._local.create_runtimes = self.create_runtimes
        if self.activate_profiler is not None:
            self._local.activate_profiler = self.activate_profiler
        if self.timeout is not None:
            self._local.timeout = self.timeout
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._previous_context is not None:
            self._local.device = self._previous_context["device"]
            self._local.device_map = self._previous_context["device_map"]
            self._local.create_runtimes = self._previous_context["create_runtimes"]
            self._local.activate_profiler = self._previous_context["activate_profiler"]
            self._local.timeout = self._previous_context["timeout"]

    @classmethod
    def get_current_context(cls):
        return {
            "device": getattr(cls._local, "device", None),
            "device_map": getattr(cls._local, "device_map", None),
            "create_runtimes": getattr(cls._local, "create_runtimes", None),
            "activate_profiler": getattr(cls._local, "activate_profiler", None),
            "timeout": getattr(cls._local, "timeout", None),
        }
