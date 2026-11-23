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

from __future__ import annotations

import json
import math
import re
from bisect import bisect_left, insort
from collections import defaultdict
from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING

import torch


if TYPE_CHECKING:
    from ..configuration_utils import RBLNCompileConfig
    from .models.decoderonly.configuration_decoderonly import RBLNDecoderOnlyModelConfig


def has_device_cache_inputs(compiled_model, config: RBLNDecoderOnlyModelConfig) -> bool:
    """Older artifacts own their KV buffers and omit them from the executor's input signature."""
    names = {meta.name for meta in config.cache_metas}
    executor = compiled_model.select_executor()
    profiles = (executor.get_input_profile(i) for i in range(executor.num_inputs))
    return any(profile.name in names and profile.device == "rbln" for profile in profiles)


def cache_shape_override(config: RBLNDecoderOnlyModelConfig, compile_config: RBLNCompileConfig):
    from rebel.input_info import SymbolInfo

    dynamic_names = {meta.name for meta in config.cache_metas if meta.can_resize}
    symbols = {}
    shapes = []
    for idx, (name, shape, _) in enumerate(compile_config.input_info):
        dims = [("const", dim) for dim in shape]
        if name in dynamic_names:
            dims[0] = ("symbol", "num_blocks")
            if not symbols:
                symbols["num_blocks"] = SymbolInfo("num_blocks", idx, 0, 1, config.num_full_blocks, shape[0])
        shapes.append(dims)
    return (symbols, shapes) if symbols else None


def runtime_cache_device(runtimes) -> torch.device:
    from torch_rbln._internal.rsd_utils import get_physical_device_ids

    groups = {
        tuple(rt._runtime_handle.get_context().get_first_user_device_ids(rt._compiled_model.get_num_devices()))
        for rt in runtimes
    }
    if len(groups) != 1:
        raise ValueError(f"All runtimes sharing a KV cache must use the same device group, got {groups}.")
    devices = list(groups.pop())
    for index in range(torch.rbln.device_count()):
        if get_physical_device_ids(index) == devices:
            return torch.device("rbln", index)
    raise ValueError(
        f"No torch-rbln logical device maps to {devices}. Configure RBLN_DEVICE_MAP before importing torch."
    )


class DeviceKVCache:
    def __init__(self, config: RBLNDecoderOnlyModelConfig, device: torch.device):
        self.device = device
        self.specs = {
            meta.name: (
                tuple([config.kvcache_num_blocks, *meta.shape[1:]] if meta.can_resize else meta.shape),
                getattr(torch, meta.dtype),
            )
            for meta in config.cache_metas
        }
        self._tensors: dict[str, torch.Tensor] | None = None

    @property
    def tensors(self) -> dict[str, torch.Tensor]:
        if self._tensors is None:
            self._tensors = {
                name: torch.zeros(shape, dtype=dtype, device=self.device)
                for name, (shape, dtype) in self.specs.items()
            }
        return self._tensors

    def bind(self, tensors: Mapping[str, torch.Tensor]) -> None:
        if tensors.keys() != self.specs.keys():
            raise ValueError(f"Expected cache tensors {list(self.specs)}, got {list(tensors)}.")
        for name, (shape, dtype) in self.specs.items():
            t = tensors[name]
            if not isinstance(t, torch.Tensor) or t.device != self.device or t.shape != shape or t.dtype != dtype:
                raise ValueError(f"{name} requires a tensor with shape {shape} and dtype {dtype} on {self.device}.")
            if not t.is_contiguous() or t.storage_offset() or t.nbytes != t.untyped_storage().nbytes():
                raise ValueError(f"{name} requires a contiguous whole allocation.")
        if len({t.data_ptr() for t in tensors.values()}) != len(tensors):
            raise ValueError("Different cache inputs must have distinct allocations.")
        self._tensors = dict(tensors)


class DecoderCacheRuntime:
    """Supply the shared cache inputs when a decoder wrapper invokes its runtime."""

    def __init__(self, runtime, cache: DeviceKVCache):
        self.runtime = runtime
        self.cache = cache
        names = [runtime._index_to_input_name[i] for i in sorted(runtime._index_to_input_name)]
        self.cache_names = [name for name in names if name in cache.specs]
        self._index_to_input_name = dict(enumerate(name for name in names if name not in cache.specs))

    def __getattr__(self, name):
        return getattr(self.runtime, name)

    def run(self, *args, **kwargs):
        if len(args) > len(self._index_to_input_name):
            raise TypeError("Too many positional inputs for the decoder runtime.")
        inputs = {self._index_to_input_name[i]: arg for i, arg in enumerate(args)}
        if kwargs.keys() & (inputs.keys() | set(self.cache_names)):
            raise TypeError("A runtime input was supplied more than once.")
        inputs.update(kwargs)
        tensors = self.cache.tensors
        return self.runtime.run(**inputs, **{name: tensors[name] for name in self.cache_names})

    __call__ = run


def _extent(expr, blocks):
    if isinstance(expr, int):
        return expr
    op, *args = expr
    if op == "symbol":
        return blocks
    if op == "prod":
        return math.prod(_extent(arg, blocks) for arg in args)
    if op == "sum":
        return sum(_extent(arg, blocks) for arg in args)
    raise ValueError(f"Unknown KV placement expression {expr!r}.")


def _placement_itemsize(dtype: str) -> int:
    if dtype.lower() == "bool":
        return 1
    bits = re.search(r"(\d+)$", dtype)
    if bits is None or int(bits[1]) % 8:
        raise ValueError(f"Unsupported physical KV cache dtype {dtype!r}.")
    return int(bits[1]) // 8


def _shards_per_block(profile) -> list[tuple[int, int, int]]:
    placement = profile.physical_placement
    if placement is None or profile.variable_shape_dims != {0}:
        raise ValueError(f"Cache {profile.name} has no dynamic block placement.")
    itemsize = _placement_itemsize(placement.dtype)
    requests = []
    for shard in placement.shards:
        dims = [dim.to_py() for dim in shard.slice_shape]
        zero, one, two = (math.prod(_extent(dim, blocks) for dim in dims) for blocks in (0, 1, 2))
        if zero or two != 2 * one:
            raise ValueError(f"Cache {profile.name} has a nonlinear block placement.")
        requests.append((shard.node_id, shard.chiplet_id, one * itemsize))
    return requests


def cache_shards(compiled_models, config):
    """Return (node, chiplet, fixed bytes or bytes per block) for each distinct cache input."""
    metas = {meta.name: meta for meta in config.cache_metas}
    layouts = {}
    shards = {}
    for model in compiled_models.values():
        executor = model.select_executor()
        for idx in range(executor.num_inputs):
            profile = executor.get_input_profile(idx)
            name = profile.name
            if name not in metas:
                continue
            placement = json.loads(executor.get_input_device_config(idx).json)["on_device"]
            layout = placement["hash"]
            if name in layouts:
                if layouts[name] != layout:
                    raise ValueError(f"Compiled runtimes require different layouts for shared cache {name}.")
                continue
            layouts[name] = layout
            if metas[name].can_resize:
                shards[name] = _shards_per_block(profile)
            else:
                shards[name] = [(a["node_id"], a["chiplet_id"], a["size"]) for a in placement["allocs"]]
    if missing := metas.keys() - shards.keys():
        raise ValueError(f"No device input placement for caches {sorted(missing)}.")
    return shards


def allocator_reserved(requests: Iterable[int]) -> int:
    """Replay the runtime caching allocator's segment sizes and best-fit splits."""
    mib = 1 << 20
    pools = {True: [], False: []}
    reserved = 0
    for request in requests:
        if request == 0:
            continue
        size = (request + 4095) // 4096 * 4096
        small = size <= mib
        pool = pools[small]
        idx = bisect_left(pool, size)
        if idx < len(pool):
            segment = pool.pop(idx)
        else:
            if small:
                segment = 2 * mib
            elif size <= 10 * mib:
                segment = 20 * mib
            else:
                segment = (size + 2 * mib - 1) // (2 * mib) * 2 * mib
            reserved += segment
        left = segment - size
        if (small and left >= 4096) or (not small and left > mib):
            insort(pool, left)
    return reserved


def cache_requests(shards, config, blocks):
    requests = defaultdict(list)
    for meta in config.cache_metas:
        for node, chiplet, size in shards[meta.name]:
            requests[node, chiplet].append(size * blocks if meta.can_resize else size)
    return requests


def fit_cache_blocks(shards, config, room: dict[tuple[int, int], int]) -> int:
    fixed = cache_requests(shards, config, 0)
    base = {unit: sum(reqs) for unit, reqs in fixed.items()}
    dynamic = any(meta.can_resize for meta in config.cache_metas)
    lower = config.num_min_blocks if dynamic else config.kvcache_num_blocks
    upper = config.num_full_blocks if dynamic else config.kvcache_num_blocks
    if missing := base.keys() - room.keys():
        raise ValueError(f"Missing device memory information for cache chiplets {sorted(missing)}.")
    if any(available < 0 for available in room.values()):
        raise ValueError(f"Insufficient memory for the loaded model before allocating {lower} KV blocks.")
    for unit, requests in cache_requests(shards, config, 1).items():
        slope = sum(requests) - base[unit]
        if slope:
            upper = min(upper, (room[unit] - base[unit]) // slope)
    # Crossing the allocator's 10 MiB threshold can reduce the reserved size.
    for blocks in range(upper, lower - 1, -1):
        requests = cache_requests(shards, config, blocks)
        if all(allocator_reserved(values) <= room[unit] for unit, values in requests.items()):
            return blocks
    raise ValueError(f"Insufficient memory for the required {lower} KV blocks.")
