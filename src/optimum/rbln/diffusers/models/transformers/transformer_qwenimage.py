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

import types
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import torch
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.transformers.transformer_qwenimage import (
    QwenImageTransformer2DModel,
    QwenImageTransformerBlock,
)
from transformers import PretrainedConfig

from transformers import PretrainedConfig

from transformers import PretrainedConfig

from ....configuration_utils import RBLNCompileConfig, RBLNModelConfig
from ....modeling import RBLNModel
from ....utils.logging import get_logger
from ...configurations import RBLNQwenImageTransformer2DModelConfig


if TYPE_CHECKING:
    from transformers import AutoFeatureExtractor, AutoProcessor, AutoTokenizer, PreTrainedModel

    from ...modeling_diffusers import RBLNDiffusionMixin, RBLNDiffusionMixinConfig

logger = get_logger(__name__)


# ═══════════════════════════════════════════════════════════════════════
# Compile-time patches
#
# RBLN compiler does not support several ops used by the upstream
# QwenImageTransformer2DModel.  The three helpers below are applied
# inside get_compiled_model (try/finally) so that the compile graph
# is clean while the original module-level functions are restored
# immediately afterwards.
# ═══════════════════════════════════════════════════════════════════════


def _apply_rotary_emb_real(x, freqs_cis, use_real=True, use_real_unbind_dim=-1):
    """Real-number replacement for ``apply_rotary_emb_qwen``.

    Mathematically equivalent to the original complex multiplication::

        (a + jb)(c + jd) = (ac - bd) + j(ad + bc)

    Uses the **interleaved** convention (same as ``use_real_unbind_dim=-1``
    in diffusers): adjacent elements ``(x[2k], x[2k+1])`` form a pair
    rotated by frequency ``k``.

    ``freqs_cis`` is a ``(cos, sin)`` tuple with shape ``[S, D]`` where
    values are repeat-interleaved: ``[c0, c0, c1, c1, …]``.
    """
    cos, sin = freqs_cis                           # each [S, D]
    cos = cos[None, :, None, :].to(x.device)       # [1, S, 1, D]
    sin = sin[None, :, None, :].to(x.device)       # [1, S, 1, D]
    x_real, x_imag = x.reshape(*x.shape[:-1], -1, 2).unbind(-1)  # each [B, S, H, D/2]
    x_rotated = torch.stack([-x_imag, x_real], dim=-1).flatten(-2)  # [B, S, H, D]
    return (x.float() * cos + x_rotated.float() * sin).to(x.dtype)


def _modulate_no_where(self, x, mod_params, index=None):
    """Arithmetic-lerp replacement for ``QwenImageTransformerBlock._modulate``.

    Replaces ``torch.where(index == 0, a, b)`` with
    ``a * (1 - idx) + b * idx`` to remove the boolean branch.
    """
    shift, scale, gate = mod_params.chunk(3, dim=-1)

    if index is not None:
        n = shift.size(0) // 2
        idx = index.unsqueeze(-1).to(x.dtype)
        inv = 1 - idx
        shift = shift[:n].unsqueeze(1) * inv + shift[n:].unsqueeze(1) * idx
        scale = scale[:n].unsqueeze(1) * inv + scale[n:].unsqueeze(1) * idx
        gate = gate[:n].unsqueeze(1) * inv + gate[n:].unsqueeze(1) * idx
    else:
        shift = shift.unsqueeze(1)
        scale = scale.unsqueeze(1)
        gate = gate.unsqueeze(1)

    return x * (1 + scale) + shift, gate


def _patch_rope_to_real(transformer, img_shapes, txt_seq_lens, dtype):
    """Pre-compute ``QwenEmbedRope`` complex frequencies as real buffers.

    After this call the ``pos_embed.forward`` of *transformer* returns
    ``((vid_cos, vid_sin), (txt_cos, txt_sin))`` — pure real tensors —
    so that no complex numbers enter the compile graph.
    """
    embed = transformer.pos_embed

    with torch.no_grad():
        vid_freqs, txt_freqs = embed(img_shapes, txt_seq_lens=txt_seq_lens)

    def _to_cos_sin(freqs):
        c, s = freqs.real, freqs.imag   # each [S, D/2]
        # repeat-interleave: [c0,c0,c1,c1,...] to match interleaved pairing
        c = torch.stack([c, c], dim=-1).reshape(c.shape[0], -1)
        s = torch.stack([s, s], dim=-1).reshape(s.shape[0], -1)
        return (
            c.to(dtype).contiguous(),
            s.to(dtype).contiguous(),
        )

    vid_cos, vid_sin = _to_cos_sin(vid_freqs)
    txt_cos, txt_sin = _to_cos_sin(txt_freqs)

    embed.register_buffer("_vid_cos", vid_cos)
    embed.register_buffer("_vid_sin", vid_sin)
    embed.register_buffer("_txt_cos", txt_cos)
    embed.register_buffer("_txt_sin", txt_sin)
    embed.pos_freqs = None
    embed.neg_freqs = None

    def _forward_real(self, *args, **kwargs):
        return (self._vid_cos, self._vid_sin), (self._txt_cos, self._txt_sin)

    embed.forward = types.MethodType(_forward_real, embed)


# ═══════════════════════════════════════════════════════════════════════
# Wrapper
# ═══════════════════════════════════════════════════════════════════════


class QwenImageTransformer2DModelWrapper(torch.nn.Module):
    """Thin wrapper that fixes compile-time constants and maps the
    four tensor inputs to the upstream ``QwenImageTransformer2DModel``
    interface.

    ``img_shapes`` / ``txt_seq_lens`` are Python lists stored as
    attributes (compile-time constants).

    ``encoder_hidden_states_mask`` (float32, 1=valid / 0=pad) is
    converted to an additive attention bias and injected via
    ``attention_kwargs`` so that the model's own
    ``compute_text_seq_len_from_mask`` is bypassed (it relies on
    bool ops that the RBLN compiler cannot handle).
    """

    _MASK_NEG_INF: float = -10000.0

    def __init__(
        self,
        model: QwenImageTransformer2DModel,
        img_shapes: list,
        txt_seq_lens: list,
    ) -> None:
        super().__init__()
        self.model = model
        self.img_shapes = img_shapes
        self.txt_seq_lens = txt_seq_lens

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor,
        timestep: torch.FloatTensor,
        encoder_hidden_states_mask: torch.FloatTensor,
    ) -> torch.Tensor:
        batch_size = hidden_states.shape[0]
        image_seq_len = hidden_states.shape[1]

        text_attn_bias = (1.0 - encoder_hidden_states_mask) * self._MASK_NEG_INF
        image_attn_bias = torch.zeros(
            batch_size, image_seq_len,
            device=hidden_states.device, dtype=hidden_states.dtype,
        )
        joint_attention_mask = torch.cat([text_attn_bias, image_attn_bias], dim=1)

        return self.model(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            encoder_hidden_states_mask=None,
            timestep=timestep,
            img_shapes=self.img_shapes,
            txt_seq_lens=self.txt_seq_lens,
            guidance=None,
            attention_kwargs={"attention_mask": joint_attention_mask},
            return_dict=False,
        )


# ═══════════════════════════════════════════════════════════════════════
# RBLN model
# ═══════════════════════════════════════════════════════════════════════


class RBLNQwenImageTransformer2DModel(RBLNModel):
    """RBLN implementation of ``QwenImageTransformer2DModel``.

    Three compile-time patches are applied inside ``get_compiled_model``
    and restored immediately after compilation:

    1. **Real RoPE** – replaces ``apply_rotary_emb_qwen`` (complex ops).
    2. **Arithmetic _modulate** – replaces ``torch.where`` branch.
    3. **Pre-computed RoPE buffers** – eliminates runtime ``torch.polar``.
    """

    hf_library_name = "diffusers"
    auto_model_class = QwenImageTransformer2DModel
    _output_class = Transformer2DModelOutput
    _supports_non_fp32=True

    def __post_init__(self, **kwargs):
        super().__post_init__(**kwargs)

    @contextmanager
    def cache_context(self, name: str):
        """No-op context manager; RBLN compiled models don't use diffusers caching."""
        yield

    # ── helpers ────────────────────────────────────────────────────────

    @classmethod
    def _get_compile_time_constants(cls, rbln_config):
        """Return ``(img_shapes, txt_seq_lens)`` derived from config."""
        sample_h, sample_w = rbln_config._sample_size
        patch_size = 2
        packed_h = sample_h // patch_size
        packed_w = sample_w // patch_size
        single = (1, packed_h, packed_w)
        img_shapes = [[single] * rbln_config.num_img_groups]
        txt_seq_lens = [rbln_config.prompt_embed_length] * rbln_config.batch_size
        return img_shapes, txt_seq_lens

    # ── wrap / compile ────────────────────────────────────────────────

    @classmethod
    def _wrap_model_if_needed(cls, model: torch.nn.Module, rbln_config: RBLNModelConfig) -> torch.nn.Module:
        img_shapes, txt_seq_lens = cls._get_compile_time_constants(rbln_config)
        return QwenImageTransformer2DModelWrapper(model, img_shapes, txt_seq_lens).eval()

    @classmethod
    def get_compiled_model(cls, model, rbln_config: RBLNQwenImageTransformer2DModelConfig):
        import diffusers.models.transformers.transformer_qwenimage as _tq

        original_rotary = _tq.apply_rotary_emb_qwen
        original_modulate = QwenImageTransformerBlock._modulate

        try:
            # Patch 1 – real-number RoPE
            _tq.apply_rotary_emb_qwen = _apply_rotary_emb_real
            # Patch 2 – arithmetic _modulate
            QwenImageTransformerBlock._modulate = _modulate_no_where
            # Patch 3 – pre-computed RoPE buffers
            img_shapes, txt_seq_lens = cls._get_compile_time_constants(rbln_config)
            _patch_rope_to_real(model, img_shapes, txt_seq_lens, dtype=torch.float32)

            wrapped = cls._wrap_model_if_needed(model, rbln_config)
            compiled_model = cls.compile(
                wrapped,
                rbln_compile_config=rbln_config.compile_cfgs[0],
                create_runtimes=rbln_config.create_runtimes,
                device=rbln_config.device,
            )
        finally:
            _tq.apply_rotary_emb_qwen = original_rotary
            QwenImageTransformerBlock._modulate = original_modulate

        return compiled_model

    # ── config ────────────────────────────────────────────────────────

    @classmethod
    def update_rbln_config_using_pipe(
        cls, pipe: "RBLNDiffusionMixin", rbln_config: "RBLNDiffusionMixinConfig", submodule_name: str
    ) -> "RBLNDiffusionMixinConfig":
        if rbln_config.transformer._sample_size is None:
            if rbln_config.image_size is not None:
                vae_sf = pipe.vae_scale_factor
                rbln_config.transformer._sample_size = (
                    rbln_config.image_size[0] // vae_sf,
                    rbln_config.image_size[1] // vae_sf,
                )
            else:
                rbln_config.transformer._sample_size = pipe.default_sample_size
        return rbln_config

    @classmethod
    def _update_rbln_config(
        cls,
        preprocessors: Union["AutoFeatureExtractor", "AutoProcessor", "AutoTokenizer"],
        model: "PreTrainedModel",
        model_config: "PretrainedConfig",
        rbln_config: RBLNQwenImageTransformer2DModelConfig,
    ) -> RBLNQwenImageTransformer2DModelConfig:
        def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
            if isinstance(cfg, dict):
                return cfg.get(key, default)
            return getattr(cfg, key, default)

        if rbln_config._sample_size is None:
            rbln_config._sample_size = model_config.sample_size
        if isinstance(rbln_config._sample_size, int):
            rbln_config._sample_size = (rbln_config._sample_size, rbln_config._sample_size)

        sample_h, sample_w = rbln_config._sample_size
        packed_h, packed_w = sample_h // 2, sample_w // 2
        total_seq_len = packed_h * packed_w * rbln_config.num_img_groups

        input_info = [
            (
                "hidden_states",
                [rbln_config.batch_size, total_seq_len, model_config.in_channels],
                rbln_config.dtype,
            ),
            (
                "encoder_hidden_states",
                [rbln_config.batch_size, rbln_config.prompt_embed_length, model_config.joint_attention_dim],
                rbln_config.dtype,
            ),
            (
                "timestep",
                [rbln_config.batch_size],
                rbln_config.dtype,
            ),
            (
                "encoder_hidden_states_mask",
                [rbln_config.batch_size, rbln_config.prompt_embed_length],
                rbln_config.dtype,
            ),
        ]

        rbln_config.set_compile_cfgs([RBLNCompileConfig(input_info=input_info)])
        return rbln_config

    @property
    def compiled_batch_size(self):
        return self.rbln_config.compile_cfgs[0].input_info[0][1][0]

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor = None,
        encoder_hidden_states_mask: torch.FloatTensor = None,
        timestep: torch.LongTensor = None,
        img_shapes: list = None,
        guidance: torch.Tensor = None,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        return_dict: bool = True,
        **kwargs,
    ) -> Union[Transformer2DModelOutput, Tuple]:
        compiled_seq_len = self.rbln_config.prompt_embed_length
        actual_seq_len = encoder_hidden_states.shape[1]

        if encoder_hidden_states_mask is not None:
            mask = encoder_hidden_states_mask.float()
            if actual_seq_len < compiled_seq_len:
                mask = torch.nn.functional.pad(mask, (0, compiled_seq_len - actual_seq_len), value=0.0)
        else:
            mask = torch.ones(
                encoder_hidden_states.shape[0], compiled_seq_len,
                device=encoder_hidden_states.device, dtype=encoder_hidden_states.dtype,
            )
            if actual_seq_len < compiled_seq_len:
                mask[:, actual_seq_len:] = 0.0

        if actual_seq_len < compiled_seq_len:
            encoder_hidden_states = torch.nn.functional.pad(
                encoder_hidden_states, (0, 0, 0, compiled_seq_len - actual_seq_len), value=0.0,
            )

        return super().forward(
            hidden_states,
            encoder_hidden_states,
            timestep,
            mask,
            return_dict=return_dict,
        )
