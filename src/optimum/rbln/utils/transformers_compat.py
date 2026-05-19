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

"""Compatibility shim for transformers 4.x and 5.x.

optimum-rbln supports both transformers 4.57.x and 5.x. The handful of
public/semi-public symbols that actually moved between the two are
re-exported (or wrapped) from here so that the rest of the codebase can
import them from a single place.

Items confirmed unchanged in v5.8.1 and therefore NOT shimmed here:
    - transformers.modeling_attn_mask_utils._prepare_4d_{,causal_}attention_mask
    - transformers.utils._LazyModule
    - transformers.utils.hub.PushToHubMixin
    - transformers.modeling_utils.{PreTrainedModel, get_state_dict_dtype}
    - transformers.dynamic_module_utils.get_class_from_dynamic_module
    - transformers.models.whisper.generation_whisper.WhisperGenerationMixin
    - transformers.integrations.mxfp4.Mxfp4GptOssExperts
    - transformers.cache_utils.{Cache, DynamicCache, StaticCache, SlidingWindowCache}
"""

from typing import Any, Optional

from .import_utils import is_transformers_version


_IS_TRANSFORMERS_V5 = is_transformers_version(">=", "5.0")


# --- no_init_weights -------------------------------------------------------
# v4.x: transformers.modeling_utils.no_init_weights
# v5.x: transformers.initialization.no_init_weights (modeling_utils export removed)
try:
    from transformers.initialization import no_init_weights  # type: ignore[attr-defined]
except ImportError:
    from transformers.modeling_utils import no_init_weights  # type: ignore[no-redef]


# --- Auto classes ----------------------------------------------------------
# v4.x: AutoModelForVision2Seq exists; AutoModelForImageTextToText is an alias.
# v5.x: AutoModelForVision2Seq is removed; AutoModelForImageTextToText is canonical.
from transformers import AutoModelForImageTextToText  # noqa: E402


AutoModelForVision2Seq = AutoModelForImageTextToText

# v4.x exposes both MODEL_FOR_VISION_2_SEQ_MAPPING and the newer
# MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING. v5.x dropped the former. Provide the old
# name as an alias of the new one in v5 so that downstream auto-classes can
# remain ABI-stable.
from transformers.models.auto.modeling_auto import (  # noqa: E402
    MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING,
    MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES,
)


try:
    from transformers.models.auto.modeling_auto import (
        MODEL_FOR_VISION_2_SEQ_MAPPING,
        MODEL_FOR_VISION_2_SEQ_MAPPING_NAMES,
    )
except ImportError:
    MODEL_FOR_VISION_2_SEQ_MAPPING = MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING
    MODEL_FOR_VISION_2_SEQ_MAPPING_NAMES = MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES


# --- RoPE config helpers ---------------------------------------------------
# v4.x: rope params are flat attributes on the config (config.rope_theta).
# v5.x: rope params live inside the dict config.rope_parameters.


def get_rope_param(config: Any, name: str, default: Any = None) -> Any:
    """Read a RoPE parameter (e.g. 'rope_theta') across transformers versions."""
    if _IS_TRANSFORMERS_V5:
        rp = getattr(config, "rope_parameters", None)
        if isinstance(rp, dict) and name in rp:
            return rp[name]
    return getattr(config, name, default)


def set_rope_param(config: Any, name: str, value: Any) -> None:
    """Write a RoPE parameter (e.g. 'rope_theta') across transformers versions."""
    if _IS_TRANSFORMERS_V5 and hasattr(config, "rope_parameters"):
        if config.rope_parameters is None:
            config.rope_parameters = {}
        config.rope_parameters[name] = value
    else:
        setattr(config, name, value)


# --- VLM submodule access --------------------------------------------------
# transformers PR #42156 moved VLM sub-modules (vision_tower,
# multi_modal_projector, language_model, ...) off the conditional-generation
# head and onto the underlying *Model class in v5: e.g.
#     v4: paligemma_for_cond_gen.multi_modal_projector
#     v5: paligemma_for_cond_gen.model.multi_modal_projector


def get_vlm_submodule(model, name):
    """Return a VLM sub-module that may live on the head or under `.model`."""
    sub = getattr(model, name, None)
    if sub is not None:
        return sub
    inner = getattr(model, "model", None)
    if inner is not None:
        sub = getattr(inner, name, None)
        if sub is not None:
            return sub
    raise AttributeError(f"{type(model).__name__} has no submodule {name!r}")


# --- image processor size --------------------------------------------------
# v4.x: image processors expose `size` as a plain dict ({"height": ..., "width": ...}).
# v5.x: `size` is a SizeDict dataclass with attribute access and no `.keys()`.

_SIZE_DICT_FIELDS = ("height", "width", "longest_edge", "shortest_edge", "max_height", "max_width")


def processor_size_keys(size):
    """Return the set of populated size keys for either v4 dict or v5 SizeDict."""
    if hasattr(size, "keys"):
        return set(size.keys())
    return {k for k in _SIZE_DICT_FIELDS if getattr(size, k, None) is not None}


def processor_size_get(size, key, default=None):
    """Read a size entry from either v4 dict or v5 SizeDict."""
    if hasattr(size, "get"):
        return size.get(key, default)
    return getattr(size, key, default)


# --- token / use_auth_token ------------------------------------------------
# v5.x dropped use_auth_token from PreTrainedModel.from_pretrained;
# only `token` is accepted. Public-facing entrypoints in optimum-rbln keep
# accepting both for backward compatibility and route everything through
# this helper.


def normalize_token_kwarg(
    use_auth_token: Optional[Any] = None,
    token: Optional[Any] = None,
) -> Optional[Any]:
    """Collapse use_auth_token / token into the v5-compatible single 'token'.

    Emits a FutureWarning when use_auth_token is supplied. Returns the resolved
    token value, which callers should forward as `token=` to transformers
    APIs (never as `use_auth_token=`).
    """
    if use_auth_token is not None:
        import warnings

        warnings.warn(
            "`use_auth_token` is deprecated and will be removed in a future release. Please use `token` instead.",
            FutureWarning,
            stacklevel=3,
        )
        if token is None:
            token = use_auth_token
    return token


# --- unsupported (model x transformers v5) combinations --------------------
# Populated as we discover models that don't yet work on v5. Each entry is
# a free-form key (typically the RBLN class name or model architecture
# string) that callers pass to `assert_supported_on_current_transformers`.

_UNSUPPORTED_ON_V5: set = {
    # v5 renamed MixtralDecoderLayer.block_sparse_moe; the RBLN wrapper still
    # reaches for that attribute by name during compile. Pin transformers<5 to
    # use this model until the wrapper is ported.
    "RBLNMixtralForCausalLM",
    # v5 Qwen3MoeSparseMoeBlock dropped the `num_experts` attribute the RBLN
    # wrapper depends on. Pin transformers<5 to use this model.
    "RBLNQwen3MoeForCausalLM",
    # v5 Gemma3TextConfig nests rope_parameters per layer_type
    # (sliding/full) and dropped rope_theta / rope_local_base_freq flat
    # attrs. The RBLN wrapper still assumes the v4 layout. Pin transformers<5.
    "RBLNGemma3ForCausalLM",
}


def assert_supported_on_current_transformers(model_kind: str) -> None:
    """Raise RuntimeError if `model_kind` is known-broken on the running transformers."""
    if _IS_TRANSFORMERS_V5 and model_kind in _UNSUPPORTED_ON_V5:
        raise RuntimeError(
            f"{model_kind} is not yet supported on transformers>=5.0. "
            f"Please install transformers<5 until support is added."
        )


__all__ = [
    "AutoModelForVision2Seq",
    "MODEL_FOR_VISION_2_SEQ_MAPPING",
    "MODEL_FOR_VISION_2_SEQ_MAPPING_NAMES",
    "assert_supported_on_current_transformers",
    "get_rope_param",
    "get_vlm_submodule",
    "no_init_weights",
    "normalize_token_kwarg",
    "processor_size_get",
    "processor_size_keys",
    "set_rope_param",
]
