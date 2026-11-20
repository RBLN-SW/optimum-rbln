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
from collections import OrderedDict
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any, Union

import torch
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.transformers.transformer_qwenimage import (
    QwenImageTransformer2DModel,
    QwenImageTransformerBlock,
    QwenTimestepProjEmbeddings,
)
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
    cos, sin = freqs_cis  # each [S, D]
    cos = cos[None, :, None, :].to(x.device)  # [1, S, 1, D]
    sin = sin[None, :, None, :].to(x.device)  # [1, S, 1, D]
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
        c, s = freqs.real, freqs.imag  # each [S, D/2]
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
# Modulation hoisting
#
# Every block owns ``img_mod`` / ``txt_mod`` = SiLU + Linear(dim, 6*dim), whose
# only input is the timestep embedding: 12.7 GiB of weights (60 blocks) that
# never look at a token.  Compiled into the graph they are kept on a single
# memory node of the chip (rebel 0.11.2, RBLN-CR13: the transformer takes 30.7
# of that node's 35 GiB), so no other runtime fits beside the transformer.
# Computing them on the host and feeding the results as one small input frees
# that node.  Measured on the real model (first denoising step, fp32 reference):
#   hoist txt_mod only   6.4 GiB off node 0, same speed, error unchanged (8.8 -> 9.0% at 8 blocks)
#   hoist img_mod too    balanced nodes and a 17% faster forward, but the compiler
#                        repartitions the image stream and the error grows with depth
#                        (11% vs 4% after 60 blocks)  -> opt-in via hoist_modulation="all"
# ═══════════════════════════════════════════════════════════════════════

MODULATION_WEIGHTS_FILE = "modulation.pth"


class _ModulationHolder:
    """Slot the wrapper fills with the ``mod_params`` input on every forward."""

    def __init__(self) -> None:
        self.params: torch.Tensor | None = None


class _ModulationFromInput(torch.nn.Module):
    """Stand-in for a block's ``img_mod`` / ``txt_mod``: returns this block's precomputed rows."""

    def __init__(self, holder: _ModulationHolder, layer_idx: int, row_start: int, row_end: int) -> None:
        super().__init__()
        self._holder = holder
        self.layer_idx, self.row_start, self.row_end = layer_idx, row_start, row_end

    def forward(self, temb: torch.Tensor) -> torch.Tensor:
        return self._holder.params[self.layer_idx, self.row_start : self.row_end]


class QwenImageModulationNet(torch.nn.Module):
    """Host-side copy of the hoisted layers: timestep -> modulation parameters of every block.

    Output ``[num_layers, rows, 6*dim]``. With ``mode="all"`` the first ``img_rows`` rows are ``img_mod(temb)``
    (``2*B`` rows when ``zero_cond_t`` doubles the timestep, else ``B``) followed by ``B`` rows of ``txt_mod``;
    with ``mode="txt"`` only the ``B`` ``txt_mod`` rows are produced.
    """

    def __init__(self, num_layers: int, inner_dim: int, zero_cond_t: bool, mode: str = "txt") -> None:
        super().__init__()
        if mode not in ("txt", "all"):
            raise ValueError(f"mode must be 'txt' or 'all', got {mode!r}")
        self.mode, self.zero_cond_t = mode, zero_cond_t
        self.time_text_embed = QwenTimestepProjEmbeddings(embedding_dim=inner_dim)
        mod = lambda: torch.nn.Sequential(torch.nn.SiLU(), torch.nn.Linear(inner_dim, 6 * inner_dim, bias=True))  # noqa: E731
        self.img_mod = torch.nn.ModuleList(mod() for _ in range(num_layers)) if mode == "all" else None
        self.txt_mod = torch.nn.ModuleList(mod() for _ in range(num_layers))

    @classmethod
    def from_transformer(cls, model: QwenImageTransformer2DModel, mode: str) -> "QwenImageModulationNet":
        cfg = model.config
        net = cls(
            cfg.num_layers,
            cfg.num_attention_heads * cfg.attention_head_dim,
            bool(getattr(cfg, "zero_cond_t", False)),
            mode,
        )
        net.time_text_embed.load_state_dict(model.time_text_embed.state_dict())
        for i, block in enumerate(model.transformer_blocks):
            if net.img_mod is not None:
                net.img_mod[i].load_state_dict(block.img_mod.state_dict())
            net.txt_mod[i].load_state_dict(block.txt_mod.state_dict())
        return net.to(next(model.parameters()).dtype).eval()

    @torch.no_grad()
    def forward(self, timestep: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        # Mirrors QwenImageTransformer2DModel.forward: cast, double for zero_cond_t, embed, modulate. Computed in
        # this module's own dtype (fp32 on the host, see __post_init__) and rounded once, to `dtype`, at the end.
        compute_dtype = self.txt_mod[0][1].weight.dtype
        timestep = timestep.to(compute_dtype)
        if self.zero_cond_t:
            timestep = torch.cat([timestep, timestep * 0], dim=0)
        temb = self.time_text_embed(timestep, timestep)  # second arg only supplies the dtype
        txt_temb = torch.chunk(temb, 2, dim=0)[0] if self.zero_cond_t else temb
        if self.mode == "all":
            rows = [
                torch.cat([im(temb), tx(txt_temb)], dim=0) for im, tx in zip(self.img_mod, self.txt_mod, strict=True)
            ]
        else:
            rows = [tx(txt_temb) for tx in self.txt_mod]
        return torch.stack(rows, dim=0).to(dtype)


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
        hoist_modulation: str = "none",
        batch_size: int = 1,
    ) -> None:
        super().__init__()
        self.model = model
        self.img_shapes = img_shapes
        self.txt_seq_lens = txt_seq_lens
        self.hoist_modulation = hoist_modulation
        self._holder = _ModulationHolder()
        self._replaced: list[tuple[torch.nn.Module, torch.nn.Module, torch.nn.Module]] = []
        if hoist_modulation in ("txt", "all"):
            img_rows = (
                (2 * batch_size if getattr(model.config, "zero_cond_t", False) else batch_size)
                if hoist_modulation == "all"
                else 0
            )
            for i, block in enumerate(model.transformer_blocks):
                self._replaced.append((block, block.img_mod, block.txt_mod))
                if hoist_modulation == "all":
                    block.img_mod = _ModulationFromInput(self._holder, i, 0, img_rows)
                block.txt_mod = _ModulationFromInput(self._holder, i, img_rows, img_rows + batch_size)

    def restore(self) -> None:
        """Put the original ``img_mod`` / ``txt_mod`` modules back (the hoisted weights are saved from them)."""
        for block, img_mod, txt_mod in self._replaced:
            block.img_mod, block.txt_mod = img_mod, txt_mod
        self._replaced.clear()

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor,
        timestep: torch.FloatTensor,
        encoder_hidden_states_mask: torch.FloatTensor,
        mod_params: torch.FloatTensor | None = None,
    ) -> torch.Tensor:
        if self.hoist_modulation in ("txt", "all"):
            self._holder.params = mod_params
        batch_size = hidden_states.shape[0]
        image_seq_len = hidden_states.shape[1]

        text_attn_bias = (1.0 - encoder_hidden_states_mask) * self._MASK_NEG_INF
        image_attn_bias = torch.zeros(
            batch_size,
            image_seq_len,
            device=hidden_states.device,
            dtype=hidden_states.dtype,
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

    With ``hoist_modulation`` (default ``"txt"``) the blocks' ``txt_mod`` (and with
    ``"all"`` also ``img_mod``) are replaced by an input during compilation; their
    weights are saved to ``modulation.pth`` and evaluated on the host per timestep.
    """

    hf_library_name = "diffusers"
    auto_model_class = QwenImageTransformer2DModel
    _output_class = Transformer2DModelOutput
    _supports_non_fp32 = True
    _comp_dtype = "bfloat"

    def __post_init__(self, **kwargs):
        super().__post_init__(**kwargs)
        self._modulation_net: QwenImageModulationNet | None = None
        self._modulation_cache: "OrderedDict[tuple, torch.Tensor]" = OrderedDict()
        # The artifact is authoritative: a graph compiled with hoisting takes a 5th input, `mod_params`.
        # (The config attribute may be unset on artifacts compiled before this option existed.)
        mod_input = next(
            (entry for entry in self.rbln_config.compile_cfgs[0].input_info if entry[0] == "mod_params"), None
        )
        if mod_input is not None:
            cfg = self.config
            mode = self.rbln_config.hoist_modulation
            if mode not in ("txt", "all"):  # config predates the mode string: infer from the input's row count
                mode = "txt" if mod_input[1][1] == self.rbln_config.batch_size else "all"
            net = QwenImageModulationNet(
                cfg.num_layers,
                cfg.num_attention_heads * cfg.attention_head_dim,
                bool(getattr(cfg, "zero_cond_t", False)),
                mode,
            )
            candidates = [
                Path(self.model_save_dir) / self.subfolder / MODULATION_WEIGHTS_FILE,
                Path(self.model_save_dir) / MODULATION_WEIGHTS_FILE,
            ]
            path = next((c for c in candidates if c.is_file()), None)
            if path is None:
                raise FileNotFoundError(
                    f"{MODULATION_WEIGHTS_FILE} not found under {self.model_save_dir}: this transformer was compiled "
                    "with hoist_modulation=True and needs the hoisted weights next to compiled_model.rbln."
                )
            net.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
            # fp32 on the host (12.7 GiB of RAM for txt, 25 GiB for all, on the 60-block model).
            self._modulation_net = net.float().eval().requires_grad_(False)

    def _runtime_dtype(self) -> torch.dtype:
        dtype = self.rbln_config.dtype
        return getattr(torch, dtype) if isinstance(dtype, str) else dtype

    def _modulation_params(self, timestep: torch.Tensor) -> torch.Tensor:
        """Modulation parameters for this timestep, cached: a schedule reuses the same timesteps every request."""
        key = tuple(timestep.detach().float().flatten().tolist())
        cached = self._modulation_cache.get(key)
        if cached is not None:
            self._modulation_cache.move_to_end(key)
            return cached
        params = self._modulation_net(timestep.detach().cpu(), self._runtime_dtype())
        self._modulation_cache[key] = params
        if len(self._modulation_cache) > 256:
            self._modulation_cache.popitem(last=False)
        return params

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
        return QwenImageTransformer2DModelWrapper(
            model,
            img_shapes,
            txt_seq_lens,
            hoist_modulation=rbln_config.hoist_modulation or "none",
            batch_size=rbln_config.batch_size,
        ).eval()

    @classmethod
    def get_compiled_model(cls, model, rbln_config: RBLNQwenImageTransformer2DModelConfig):
        import diffusers.models.transformers.transformer_qwenimage as _tq

        original_rotary = _tq.apply_rotary_emb_qwen
        original_modulate = QwenImageTransformerBlock._modulate
        wrapped = None

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
            if wrapped is not None:
                wrapped.restore()  # save_torch_artifacts below needs the real img_mod / txt_mod

        return compiled_model

    @classmethod
    def save_torch_artifacts(
        cls,
        model: QwenImageTransformer2DModel,
        save_dir_path: Path,
        subfolder: str,
        rbln_config: RBLNQwenImageTransformer2DModelConfig,
    ) -> None:
        if rbln_config.hoist_modulation in ("txt", "all"):
            net = QwenImageModulationNet.from_transformer(model, rbln_config.hoist_modulation)
            torch.save(net.state_dict(), Path(save_dir_path) / subfolder / MODULATION_WEIGHTS_FILE)

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

        if rbln_config.hoist_modulation is None:
            rbln_config.hoist_modulation = "txt"

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
        if rbln_config.hoist_modulation in ("txt", "all"):
            inner_dim = _cfg_get(model_config, "num_attention_heads") * _cfg_get(model_config, "attention_head_dim")
            img_rows = 0
            if rbln_config.hoist_modulation == "all":
                img_rows = (
                    2 * rbln_config.batch_size
                    if _cfg_get(model_config, "zero_cond_t", False)
                    else rbln_config.batch_size
                )
            input_info.append(
                (
                    "mod_params",
                    [_cfg_get(model_config, "num_layers"), img_rows + rbln_config.batch_size, 6 * inner_dim],
                    rbln_config.dtype,
                )
            )

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
        attention_kwargs: dict[str, Any] | None = None,
        return_dict: bool = True,
        **kwargs,
    ) -> Transformer2DModelOutput | tuple:
        compiled_seq_len = self.rbln_config.prompt_embed_length
        actual_seq_len = encoder_hidden_states.shape[1]

        if encoder_hidden_states_mask is not None:
            # Match the dtype this graph was compiled at -- the runtime rejects any
            # other. `encoder_hidden_states` carries it, and the `else` branch below
            # already builds its mask that way.
            mask = encoder_hidden_states_mask.to(encoder_hidden_states.dtype)
            if actual_seq_len < compiled_seq_len:
                mask = torch.nn.functional.pad(mask, (0, compiled_seq_len - actual_seq_len), value=0.0)
        else:
            mask = torch.ones(
                encoder_hidden_states.shape[0],
                compiled_seq_len,
                device=encoder_hidden_states.device,
                dtype=encoder_hidden_states.dtype,
            )
            if actual_seq_len < compiled_seq_len:
                mask[:, actual_seq_len:] = 0.0

        if actual_seq_len < compiled_seq_len:
            encoder_hidden_states = torch.nn.functional.pad(
                encoder_hidden_states,
                (0, 0, 0, compiled_seq_len - actual_seq_len),
                value=0.0,
            )

        inputs = [hidden_states, encoder_hidden_states, timestep, mask]
        if self._modulation_net is not None:
            inputs.append(self._modulation_params(timestep))
        return super().forward(*inputs, return_dict=return_dict)


__all__ = [
    "RBLNQwenImageTransformer2DModel",
]
