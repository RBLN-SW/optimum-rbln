import contextlib
import json
import math
import os
from pathlib import Path
from unittest.mock import patch

import rebel
import torch
from rebel.compile_context import CompileContext
from torch.overrides import TorchFunctionMode
from torch.utils._pytree import tree_flatten, tree_map

from optimum.rbln.modeling import RBLNModel
from optimum.rbln.modeling_base import RBLNBaseModel


REAL_COMPILE_ENV = "OPTIMUM_RBLN_REAL_COMPILE"

_SCALAR_READS = {
    torch.Tensor.item,
    torch.Tensor.__bool__,
    torch.Tensor.__int__,
    torch.Tensor.__index__,
    torch.Tensor.__float__,
    torch.Tensor.tolist,
}

_VALUE_NUMEL = 2**24

# Depth of `_export` calls; checkpoints only load on meta inside one.
_exporting = 0


class FakeCompileError(RuntimeError):
    pass


def is_fake_compile() -> bool:
    return os.environ.get(REAL_COMPILE_ENV) != "1"


def _fake_npu() -> str:
    return os.environ.get("OPTIMUM_RBLN_FAKE_NPU") or os.environ.get("RBLN_FORCE_NPU_NAME") or "RBLN-CA25"


def _fake_device_count() -> int:
    return int(os.environ.get("OPTIMUM_RBLN_FAKE_DEVICE_COUNT", "8"))


def _torch_dtype(dtype: str | torch.dtype) -> torch.dtype:
    return dtype if isinstance(dtype, torch.dtype) else getattr(torch, dtype)


def _dtype_name(dtype: str | torch.dtype) -> str:
    return str(dtype).removeprefix("torch.")


@contextlib.contextmanager
def _on_meta(module: torch.nn.Module):
    # Swap every host tensor the module holds for a meta one, and put them back afterwards. Yields the values of the
    # swapped buffers and tensor attributes; weights are not kept, as a fake checkpoint has none.
    # Meta tensors the forward attaches (e.g. cached tables) are dropped afterwards, so they do not outlive the trace.
    swapped, stores, values = [], [], {}
    for m in module.modules():
        for store in (m._parameters, m._buffers, vars(m)):
            stores.append((store, set(store)))
            for key, value in list(store.items()):
                if isinstance(value, torch.Tensor) and not value.is_meta:
                    swapped.append((store, key, value))
                    store[key] = value.detach().to("meta")
                    if not isinstance(value, torch.nn.Parameter):
                        values[id(store[key])] = (store[key], value.detach())
    try:
        yield values
    finally:
        for store, keys in stores:
            for key in set(store) - keys:
                if isinstance(store[key], torch.Tensor) and store[key].is_meta:
                    del store[key]
        for store, key, value in reversed(swapped):
            store[key] = value


def _to_cpu(value):
    if isinstance(value, torch.device) and value.type == "meta" or isinstance(value, str) and value == "meta":
        return torch.device("cpu")
    return value


def _written(func, args, kwargs):
    # Tensors an in-place op writes into: `x.add_()`, `x += y`, `x[i] = y`, `out=`.
    name = getattr(func, "__name__", "")
    inplace = name.endswith("_") and not name.startswith("__") or name.startswith("__i") or name == "__setitem__"
    return [t for t in (args[0] if inplace and args else None, kwargs.get("out")) if isinstance(t, torch.Tensor)]


class _MetaProbe(TorchFunctionMode):
    # Records the tensors that reach an op (attribute reads like `.shape` do not count) and answers scalar reads with
    # real values. Every op whose meta inputs all have a value is also run on those values on the CPU, so a value
    # computed from the inputs, buffers and tensor attributes is known; one computed from a weight is not.
    def __init__(self, values):
        super().__init__()
        self.used = set()
        self.values = values  # id(meta tensor) -> (meta tensor, cpu tensor); the meta tensor keeps its id alive

    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        if func in _SCALAR_READS and isinstance(args[0], torch.Tensor) and args[0].is_meta:
            if id(args[0]) not in self.values:
                raise FakeCompileError(
                    f"`{func.__name__}` reads a value computed from the weights, which are not loaded"
                )
            return func(self.values[id(args[0])][1])
        tensors = [t for t in tree_flatten((args, kwargs))[0] if isinstance(t, torch.Tensor)]
        if getattr(func, "__name__", None) != "__get__":
            self.used.update(map(id, tensors))
        out = func(*args, **kwargs)
        outs = [t for t in tree_flatten(out)[0] if isinstance(t, torch.Tensor)]
        written = _written(func, args, kwargs)
        reals = None
        if (
            all(id(t) in self.values for t in tensors if t.is_meta)
            and (written or any(t.is_meta for t in outs))
            and sum(t.numel() for t in outs if t.is_meta) <= _VALUE_NUMEL
        ):

            def value(x):
                return self.values[id(x)][1] if isinstance(x, torch.Tensor) and x.is_meta else _to_cpu(x)

            # Values are best effort: one that cannot be computed is left unknown, and only reading it fails.
            try:
                with torch.device("cpu"):
                    real = func(*tree_map(value, args), **tree_map(value, kwargs))
                reals = [t for t in tree_flatten(real)[0] if isinstance(t, torch.Tensor)]
            except Exception:
                reals = None
        if reals is None or len(reals) != len(outs):
            for t in outs + written:
                self.values.pop(id(t), None)
        else:  # an in-place op has written into the values of `written` already
            self.values.update((id(m), (m, v)) for m, v in zip(outs, reals, strict=True) if m.is_meta)
        return out


def _infer_outputs(mod, input_info, example_inputs):
    if example_inputs is None:
        example_inputs = [
            torch.zeros(shape, dtype=_torch_dtype(dtype), device="cpu" if math.prod(shape) <= _VALUE_NUMEL else "meta")
            for _, shape, dtype in input_info
        ]
    inputs = [t.to("meta") for t in example_inputs]
    try:
        # The real compile traces under torch.export; wrappers take their export-only branches there.
        with torch.no_grad(), _on_meta(mod) as values:
            # The inputs hold the zeros the real trace runs on too; the KV caches it gets as meta have no values.
            values.update(
                (id(meta), (meta, example))
                for meta, example in zip(inputs, example_inputs, strict=True)
                if not example.is_meta
            )
            probe = _MetaProbe(values)
            with torch.device("meta"), probe, patch.object(torch.compiler, "is_exporting", lambda: True):
                outputs = mod(*inputs)
    except Exception as e:
        raise FakeCompileError(
            f"Fake compile could not run {type(mod).__name__} on meta tensors: {e}\n"
            "Make it run on meta, or mark the test class @requires_compile to compile it for real."
        ) from e
    leaves, _ = tree_flatten(outputs)
    outputs = [[list(t.shape), _dtype_name(t.dtype)] for t in leaves if isinstance(t, torch.Tensor)]
    unused = [name for (name, _, _), t in zip(input_info, inputs, strict=True) if id(t) not in probe.used]
    return outputs, unused


class FakeCompiledModel:
    def __init__(
        self,
        path: str | os.PathLike | None = None,
        *,
        signatures: list[dict] | None = None,
        num_devices: int = 1,
        npu: str | None = None,
    ):
        if path is not None:
            try:
                spec = json.loads(Path(path).read_text())
                signatures, num_devices = spec["fake_rbln_signatures"], spec["num_devices"]
                npu = spec.get("npu")
            except (UnicodeDecodeError, json.JSONDecodeError, KeyError) as e:
                raise RuntimeError(f"{path} is a real compiled model; fake compile can only load its own.") from e
        self.signatures = signatures
        self.num_devices = num_devices
        self._meta = {"npu": npu or _fake_npu()}

    def save(self, path: str | os.PathLike):
        Path(path).write_text(
            json.dumps(
                {
                    "fake_rbln_signatures": self.signatures,
                    "num_devices": self.num_devices,
                    "npu": self._meta["npu"],
                }
            )
        )

    # Memory queries behind kvcache_num_blocks estimation: nothing allocated, so every block fits.
    def get_alloc_per_node_by_key(self):
        return {}

    def get_alloc_per_chiplet_by_key(self):
        return {}

    def exp_get_dram_tensor_sizes(self):
        return {}

    def exp_multiply_buffer_size(self, *args, **kwargs):
        pass

    def exp_rescale_buffer_size(self, *args, **kwargs):
        pass


def fake_compile_from_torch(mod, input_info=None, example_inputs=None, compile_context=None, **kwargs):
    # Multiple input_info compile one graph per input set; the runtime dispatches on the inputs given.
    input_infos = [input_info] if isinstance(input_info[0][0], str) else input_info
    # Static inputs live on the device and are not passed at run time.
    static = sorted(getattr(compile_context, "_fake_static_names", ()))
    signatures = []
    for info in input_infos:
        info = [[name, list(shape), _dtype_name(dtype)] for name, shape, dtype in info]
        outputs, unused = _infer_outputs(mod, info, example_inputs if len(input_infos) == 1 else None)
        signatures.append({"inputs": info, "static": static, "unused": unused, "outputs": outputs})
    num_devices = kwargs.get("num_devices") or kwargs.get("tensor_parallel_size") or 1
    return FakeCompiledModel(signatures=signatures, num_devices=num_devices, npu=kwargs.get("npu"))


class FakeRuntime:
    def __init__(self, compiled_model: FakeCompiledModel, device: int | list[int] | None = None, **kwargs):
        if device is None:
            devices = list(range(compiled_model.num_devices))
        else:
            devices = [device] if isinstance(device, int) else list(device)
        if all(d >= 0 for d in devices) and len(devices) != compiled_model.num_devices:  # negative ids are dummies
            raise RuntimeError(f"Compiled for {compiled_model.num_devices} device(s), got device={device}.")
        if any(d >= _fake_device_count() for d in devices):
            raise RuntimeError(f"device={device} is out of the {_fake_device_count()} fake devices.")
        self.signatures = compiled_model.signatures
        self.kwargs = kwargs
        names = [name for name, _, _ in self._runtime_inputs(self.signatures[0])]
        self._index_to_input_name = dict(enumerate(names))

    @staticmethod
    def _runtime_inputs(signature):
        skipped = set(signature["static"]) | set(signature["unused"])
        return [spec for spec in signature["inputs"] if spec[0] not in skipped]

    def get_executor_count(self):
        return len(self.signatures)

    def _mismatch(self, signature, args, kwargs):
        specs = self._runtime_inputs(signature)
        positional, named = specs[: len(args)], {spec[0]: spec for spec in specs[len(args) :]}
        if len(args) + len(kwargs) != len(specs) or not set(kwargs) <= set(named):
            return f"expected inputs {[s[0] for s in specs]}, got {len(args)} positional and {sorted(kwargs)}"
        pairs = list(zip(positional, args, strict=True)) + [(named[k], v) for k, v in kwargs.items()]
        for (name, shape, dtype), tensor in pairs:
            if list(tensor.shape) != shape or _dtype_name(tensor.dtype) != dtype:
                return f"`{name}`: expected {shape} {dtype}, got {list(tensor.shape)} {_dtype_name(tensor.dtype)}"
        return None

    def __call__(self, *args, out=None, **kwargs):
        mismatches = []
        for signature in self.signatures:
            mismatch = self._mismatch(signature, args, kwargs)
            if mismatch is None:
                return self._outputs(signature, out)
            mismatches.append(mismatch)
        raise AssertionError(f"Runtime inputs do not match the compiled input_info: {'; '.join(mismatches)}")

    def _outputs(self, signature, out):
        specs = signature["outputs"]
        if out is not None:
            buffers = [out] if isinstance(out, torch.Tensor) else list(out)
            if [list(b.shape) for b in buffers] != [s[0] for s in specs]:
                raise AssertionError(
                    f"`out` buffers {[list(b.shape) for b in buffers]} do not match outputs {[s[0] for s in specs]}"
                )
            for buffer in buffers:
                buffer.zero_()
            return out
        outputs = [torch.zeros(shape, dtype=_torch_dtype(dtype)) for shape, dtype in specs]
        return outputs[0] if len(outputs) == 1 else outputs

    def __repr__(self):
        return f"<FakeRuntime inputs={list(self._index_to_input_name.values())}>"


def _mark_static_address(original):
    def mark_static_address(self, tensor, name=None):
        if not hasattr(self, "_fake_static_names"):
            self._fake_static_names = set()
        self._fake_static_names.add(name)
        return original(self, tensor, name)

    return mark_static_address


def _counting_export(original):
    @classmethod
    def _export(cls, *args, **kwargs):
        global _exporting
        _exporting += 1
        try:
            return original.__func__(cls, *args, **kwargs)
        finally:
            _exporting -= 1

    return _export


def _meta_get_pytorch_model(original):
    # Only inside `_export`: tests calling get_pytorch_model directly check the loaded weights.
    @classmethod
    def get_pytorch_model(cls, *args, **kwargs):
        if _exporting and cls.hf_library_name == "transformers":
            kwargs["device_map"] = "meta"
        return original.__func__(cls, *args, **kwargs)

    return get_pytorch_model


def _zeros_for_meta(obj):
    if isinstance(obj, torch.Tensor):
        return torch.zeros_like(obj, device="cpu") if obj.is_meta else obj
    if isinstance(obj, dict):
        new = type(obj)((key, _zeros_for_meta(value)) for key, value in obj.items())
        if hasattr(obj, "_metadata"):  # state_dict versions, read back by load_state_dict
            new._metadata = obj._metadata
        return new
    if isinstance(obj, (list, tuple)):
        return type(obj)(_zeros_for_meta(value) for value in obj)
    return obj


def _save_meta_as_zeros(original):
    # Host-side weights (torch_artifacts.pth, query_tokens.pth) are saved from the meta checkpoint; give them
    # zeros so the loaded model can run them. Other saves are left alone.
    def save(obj, f, *args, **kwargs):
        if str(f).endswith(".pth"):
            obj = _zeros_for_meta(obj)
        return original(obj, f, *args, **kwargs)

    return save


@contextlib.contextmanager
def fake_rbln():
    """Patch compiler/runtime APIs only when fake compilation is enabled."""
    if not is_fake_compile():
        yield
        return

    with contextlib.ExitStack() as stack:
        for target, name, value in [
            (rebel, "compile_from_torch", fake_compile_from_torch),
            (rebel, "RBLNCompiledModel", FakeCompiledModel),
            (rebel, "Runtime", FakeRuntime),
            (rebel, "npu_is_available", lambda *args, **kwargs: True),
            (
                rebel,
                "get_npu_name",
                lambda device_id=0, *args, **kwargs: _fake_npu() if device_id < _fake_device_count() else None,
            ),
            (rebel, "device_count", lambda *args, **kwargs: _fake_device_count()),
            (CompileContext, "mark_static_address", _mark_static_address(CompileContext.mark_static_address)),
            (RBLNBaseModel, "_export", _counting_export(RBLNBaseModel.__dict__["_export"])),
            (RBLNModel, "get_pytorch_model", _meta_get_pytorch_model(RBLNModel.__dict__["get_pytorch_model"])),
            (torch, "save", _save_meta_as_zeros(torch.save)),
        ]:
            stack.enter_context(patch.object(target, name, value))
        yield
