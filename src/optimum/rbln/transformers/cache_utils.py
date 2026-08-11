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

"""Metadata classes describing the on-device caches of a decoder-only model.

Analogous to Transformers' ``cache_utils`` (which keeps ``DynamicCache`` / sliding-window /
linear-attention layers as distinct types), this module holds a small polymorphic hierarchy:

    CacheMeta                     # base: name, layer_index, shape, dtype; serialization
    ├─ KVCacheMeta                # paged KV: [num_blocks, num_heads, block_size, head_dim]
    │   ├─ FullAttentionKVCacheMeta      # resizable when is_auto (grown after compile)
    │   └─ SlidingWindowKVCacheMeta      # fixed-size window
    └─ LinearAttentionCacheMeta   # conv/recurrent state; raw model-computed shape, never resized

Each subclass owns its ``can_resize`` / ``compile_shape`` and a class-level ``layer_type`` tag, so
the previous single-dataclass-with-a-``layer_type``-string design (and its magic-string ``can_resize``)
is gone. Instances are built through the ``make()`` dispatcher / the subclass ``from_config()``
factories, which compute the derived ``shape`` and construct the meta.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar

from ..configuration_utils import RBLNSerializableConfigProtocol


if TYPE_CHECKING:
    from .models.decoderonly.configuration_decoderonly import RBLNDecoderOnlyModelForCausalLMConfig


@dataclass
class CacheMeta(RBLNSerializableConfigProtocol):
    """
    Base metadata describing a decoder cache tensor for one transformer layer.

    Concrete subclasses distinguish the cache algorithm (full/sliding-window paged KV, linear
    attention state). Each subclass owns its ``can_resize`` / ``compile_shape`` semantics and a
    class-level ``layer_type`` tag; that avoids the previous single-dataclass-with-a-``layer_type``
    -string design.

    Instances are created through the ``make()`` dispatcher / the subclass ``from_config()`` factories,
    which compute the derived ``shape`` and construct the meta; the dataclass constructor is just the
    low-level mechanism those factories call.

    Attributes:
        name (str): Logical name of the cache tensor.
        layer_index (int): Index of the transformer layer this cache belongs to.
        shape (list[int]): Final tensor shape stored by the factory.
        dtype (str): Data type of the cache buffer ("float16", "float32", ...).
    """

    name: str
    layer_index: int
    shape: list[int]
    dtype: str

    # Class-level cache-algorithm tag; overridden per subclass and emitted into the serialized config.
    layer_type: ClassVar[str] = "cache"

    def _prepare_for_serialization(self) -> dict[str, Any]:
        # Emit the historical schema so serialized rbln_config.json stays consistent across cache types
        # and with older artifacts. ``is_auto`` only exists on the full-attention subclass.
        return {
            "name": self.name,
            "layer_index": self.layer_index,
            "shape": self.shape,
            "layer_type": self.layer_type,
            "is_auto": getattr(self, "is_auto", False),
            "dtype": self.dtype,
        }

    @property
    def can_resize(self) -> bool:
        return False

    @property
    def compile_shape(self) -> list[int]:
        return self.shape


@dataclass
class KVCacheMeta(CacheMeta):
    """Paged KV cache: ``shape == [num_blocks, num_heads, block_size, head_dim]``."""

    @property
    def num_blocks(self) -> int:
        return self.shape[0]

    @property
    def block_size(self) -> int:
        return self.shape[2]

    @staticmethod
    def _validate_num_blocks(num_blocks: int) -> None:
        if num_blocks <= 0:
            raise ValueError("`num_blocks` must be greater than 0 when using KV cache.")


@dataclass
class FullAttentionKVCacheMeta(KVCacheMeta):
    """Full-attention paged KV cache. The only resizable cache: when ``is_auto`` it compiles with a
    single block and is grown to the estimated block count after compilation."""

    is_auto: bool
    layer_type: ClassVar[str] = "full_attention"

    @property
    def can_resize(self) -> bool:
        return self.is_auto

    @property
    def compile_shape(self) -> list[int]:
        return [1, self.shape[1], self.shape[2], self.shape[3]] if self.is_auto else self.shape

    @classmethod
    def from_config(
        cls,
        name: str,
        layer_index: int,
        num_key_value_heads: int,
        head_dim: int,
        dtype: str,
        rbln_config: "RBLNDecoderOnlyModelForCausalLMConfig",
    ) -> "FullAttentionKVCacheMeta":
        block_size = rbln_config.kvcache_block_size
        if rbln_config.is_auto_num_blocks:
            num_blocks, is_auto = rbln_config.num_full_blocks, True
        else:
            num_blocks, is_auto = rbln_config.kvcache_num_blocks, False
        cls._validate_num_blocks(num_blocks)
        return cls(
            name=name,
            layer_index=layer_index,
            shape=[num_blocks, num_key_value_heads, block_size, head_dim],
            dtype=dtype,
            is_auto=is_auto,
        )


@dataclass
class SlidingWindowKVCacheMeta(KVCacheMeta):
    """Sliding-window paged KV cache. Fixed size (one block per batch item), never resized."""

    layer_type: ClassVar[str] = "sliding_attention"

    @classmethod
    def from_config(
        cls,
        name: str,
        layer_index: int,
        num_key_value_heads: int,
        head_dim: int,
        dtype: str,
        rbln_config: "RBLNDecoderOnlyModelForCausalLMConfig",
    ) -> "SlidingWindowKVCacheMeta":
        block_size = rbln_config.sliding_window
        num_blocks = rbln_config.batch_size
        cls._validate_num_blocks(num_blocks)
        return cls(
            name=name,
            layer_index=layer_index,
            shape=[num_blocks, num_key_value_heads, block_size, head_dim],
            dtype=dtype,
        )


@dataclass
class LinearAttentionCacheMeta(CacheMeta):
    """Linear-attention (e.g. GatedDeltaNet) state: a fixed-size conv/recurrent tensor whose raw shape
    is computed by the model. Not a paged KV cache — no block/head layout, never resized."""

    layer_type: ClassVar[str] = "linear_attention"

    @classmethod
    def from_config(cls, name: str, layer_index: int, *, shape: list[int], dtype: str) -> "LinearAttentionCacheMeta":
        return cls(name=name, layer_index=layer_index, shape=shape, dtype=dtype)
