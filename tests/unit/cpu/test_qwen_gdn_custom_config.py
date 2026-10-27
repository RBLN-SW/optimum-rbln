import pytest
import torch
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5GatedDeltaNet as NativeGdn

from optimum.rbln.transformers.models.qwen3_5.configuration_qwen3_5 import (
    RBLNQwen3_5TextModelConfig,
)
from optimum.rbln.transformers.models.qwen3_5.modeling_qwen3_5 import (
    _qwen3_5_linear_state_shapes,
    validate_gdn_custom_kernel,
)
from optimum.rbln.transformers.models.qwen3_5.qwen3_5_architecture import Qwen3_5GatedDeltaNet

from ._qwen3_5_gdn_reference import NativeGatedDeltaNetReference


# The custom core only exists on a rebel-compiler that registers the GDN ops.
gdn_ops = pytest.importorskip("rebel.ops.torch_custom_ops.gdn")
block_state, unblock_state = gdn_ops.block_state, gdn_ops.unblock_state


def _gdn_models(ratio, prefill_size, gate_bias, batch_size=1):
    torch.manual_seed(42)
    hf_config = Qwen3_5TextConfig(
        hidden_size=64,
        linear_num_key_heads=2,
        linear_num_value_heads=2 * ratio,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
    )
    native = NativeGdn(hf_config, layer_idx=0)
    if gate_bias:
        native.in_proj_a.bias = torch.nn.Parameter(torch.randn(2 * ratio) * 0.1)
        native.in_proj_b.bias = torch.nn.Parameter(torch.randn(2 * ratio) * 0.1)
    rbln_config = RBLNQwen3_5TextModelConfig(max_seq_len=4096, batch_size=batch_size, prefill_chunk_size=prefill_size)
    models = [core(native, rbln_config, layer_idx=0) for core in (NativeGatedDeltaNetReference, Qwen3_5GatedDeltaNet)]
    return hf_config, models


def _grouped(flat, ratio):
    """Flat (B, K-1, conv_dim) cache -> (B, K-1, Hk*(2+r), 128), the ring's slot-major layout."""
    batch = flat.shape[0]
    parts = flat.split((256, 256, 256 * ratio), dim=2)
    grouped = torch.cat([x.reshape(batch, 3, 2, -1, 128).permute(0, 2, 3, 1, 4) for x in parts], dim=2)
    return grouped.reshape(batch, -1, 3, 128).transpose(1, 2).contiguous()


def _flat(grouped, ratio, position):
    """The ring (B, K-1, Hk*(2+r), 128) back to the flat window the reference keeps.

    The ring puts position p in slot p % W, so its slots are the window rolled by where the
    sequence stands. `position` is the sequence length after the call, and the window's oldest row
    is position - (K - 1), which sits in slot (position - 3) % 3 == position % 3.
    """
    batch = grouped.shape[0]
    grouped = torch.roll(grouped, shifts=-(position % 3), dims=1)
    grouped = grouped.transpose(1, 2).reshape(batch, 2, 2 + ratio, 3, 128)
    return torch.cat(
        [x.permute(0, 3, 1, 2, 4).reshape(batch, 3, -1) for x in grouped.split((1, 1, ratio), dim=2)], dim=2
    )


def _custom_states(native_states):
    """The native (conv, recurrent) caches in the custom core's layouts."""
    conv, recurrent = native_states
    ratio = conv.shape[-1] // 256 - 2
    batch = recurrent.shape[0]
    return _grouped(conv, ratio), block_state(recurrent.reshape(batch, 2, ratio, 128, 128))


def _conv_tables(position, valid, reset, tokens, batch=1):
    """The ring's index tables for one call, as `RBLNQwen3_5RuntimeModel._conv_tables` builds them.

    Row b reads cache block b, as decode lays the cache out, and every row of a call stands at the
    same position here.
    """
    from rebel.ops.torch_custom_ops.causal_conv1d_ring import ring_tables

    full = lambda value: torch.full((batch,), value, dtype=torch.int32)  # noqa: E731
    return ring_tables(
        torch.arange(batch, dtype=torch.int32),
        full(position),  # absolute position of this call's first token
        full(valid),
        full(0 if reset else 1),
        tokens,
        3,
        4,
    )


def _step(models, states, hidden, kwargs):
    """Runs both cores and carries their states.

    The reference hands its new caches back. The custom core writes both of its own -- the conv cache
    through `causal_conv1d_*`, the recurrent one through `gdn_*` -- and returns the hidden states alone,
    so the pair in `states` is the same pair throughout and is compared in place.
    """
    reference, custom = models
    native = reference(hidden, *states[0], **kwargs[0])
    states[0] = (native[1], native[2] if len(native) == 3 else states[0][1])
    conv, recurrent = states[1]
    return [native[0], custom(hidden, conv, recurrent, conv_cache=conv, **kwargs[1])]


def _compare(outputs, states, ratio, position):
    # The custom core matches the native path; its caches hold the native caches' values.
    (native, custom), (native_states, custom_states) = outputs, states
    torch.testing.assert_close(custom, native, rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(_flat(custom_states[0], ratio, position), native_states[0], rtol=0, atol=0)
    recurrent = unblock_state(custom_states[1]).reshape_as(native_states[1])
    torch.testing.assert_close(recurrent, native_states[1], rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("ratio", [1, 2, 3, 5])
@pytest.mark.parametrize("prefill_size", [128, 384, 512])
@pytest.mark.parametrize("gate_bias", [False, True])
def test_gdn_custom_model_prefill_decode_and_reset(ratio, prefill_size, gate_bias):
    hf_config, models = _gdn_models(ratio, prefill_size, gate_bias)
    conv_dim, units = 512 + 256 * ratio, 2 * ratio
    initial = (torch.randn(1, 3, conv_dim), torch.randn(1, units * 128, 128))
    states = [initial, _custom_states(initial)]
    assert _qwen3_5_linear_state_shapes(hf_config, 1) == (
        (1, 3, 2 * (2 + ratio), 128),
        (1, 2, ratio, *gdn_ops.STATE_BLOCKS),
    )
    # The ring is addressed by absolute position, so the calls below carry one: a reset starts a
    # fresh sequence at 0, and every call advances it by the tokens it actually holds.
    position = 0
    with torch.inference_mode():
        for phase, valid, reset in (
            ("prefill", prefill_size, True),
            ("prefill", prefill_size - 37, False),
            ("prefill", 1, False),
            ("prefill", 2, False),
            ("decode", 1, False),
            ("decode", 1, False),
            ("prefill", 100, True),
            ("prefill", 1, True),
        ):
            seq = prefill_size if phase == "prefill" else 1
            position = 0 if reset else position
            hidden = torch.randn(1, seq, 64)
            kwargs = {}
            if phase == "prefill":
                mask = torch.zeros(1, seq, 1)
                mask[:, :valid] = 1
                kwargs = {
                    "valid_mask": mask,
                    "query_position": torch.tensor(valid - 1),
                }
            per_model, norm_shapes = [], []
            for model in models:
                model.phase = phase
                model_kwargs = dict(kwargs)
                if phase == "prefill":
                    # one 0/1 value per cache row, and prefill carries one sequence
                    model_kwargs["state_mask"] = torch.full((1,), 0.0 if reset else 1.0)
                per_model.append(model_kwargs)
            # The reference derives the ring's indices itself; the custom core takes them as the
            # tables the runtime builds (see `RBLNQwen3_5RuntimeModel._conv_tables`).
            per_model[1].pop("query_position", None)
            per_model[1]["conv_tables"] = _conv_tables(position, valid, reset, seq)

            def record(module, args, shapes=norm_shapes):
                shapes.append(tuple(tuple(value.shape) for value in args))

            # Both models wrap the native layer's one norm.
            hook = models[0].norm.register_forward_pre_hook(record)
            outputs = _step(models, states, hidden, per_model)
            hook.remove()
            position += valid
            custom_shape = (1, 2, ratio, seq, 128)
            assert norm_shapes == [((seq * units, 128),) * 2, (custom_shape,) * 2]
            _compare(outputs, states, ratio, position)


@pytest.mark.parametrize("ratio", [1, 3])
def test_gdn_custom_model_batched_decode(ratio):
    batch = 3
    _, models = _gdn_models(ratio, 512, gate_bias=True, batch_size=batch)
    initial = (torch.randn(batch, 3, 512 + 256 * ratio), torch.randn(batch, 2 * ratio * 128, 128) * 0.1)
    states = [initial, _custom_states(initial)]
    # `_grouped` seeds slot 0 with the oldest row, which is where a sequence of 3 tokens leaves it,
    # so the first decode is position 3.
    position = 3
    with torch.inference_mode():
        for _ in range(2):
            for model in models:
                model.phase = "decode"
            tables = _conv_tables(position, 1, reset=False, tokens=1, batch=batch)
            outputs = _step(models, states, torch.randn(batch, 1, 64), [{}, {"conv_tables": tables}])
            position += 1
            _compare(outputs, states, ratio, position)


def _text_config(**overrides):
    dims = {"linear_num_value_heads": 6, "linear_key_head_dim": 128, "linear_value_head_dim": 128, **overrides}
    return Qwen3_5TextConfig(hidden_size=64, linear_num_key_heads=2, **dims)


@pytest.mark.parametrize(
    "overrides, gdn_chunk_size, npu, message",
    [
        ({"linear_key_head_dim": 64}, 128, None, "linear_key_head_dim"),
        ({"linear_value_head_dim": 64}, 128, None, "linear_value_head_dim"),
        ({}, 64, None, "gdn_chunk_size"),
    ],
)
def test_gdn_custom_kernel_rejects_what_the_ops_do_not_serve(overrides, gdn_chunk_size, npu, message):
    # The core always runs as the ops, so a configuration they cannot take fails at config time with a
    # message that names the field, not at compile time as a shape mismatch.
    config = RBLNQwen3_5TextModelConfig(
        max_seq_len=4096, batch_size=1, prefill_chunk_size=512, gdn_chunk_size=gdn_chunk_size, npu=npu
    )
    with pytest.raises(ValueError, match=message):
        validate_gdn_custom_kernel(_text_config(**overrides), config)


@pytest.mark.parametrize("npu", ["RBLN-CA25", "RBLN-CA22", "RBLN-CR13", None])
def test_gdn_custom_kernel_takes_every_target(npu):
    # No target is refused here, an unknown one (no device, no `npu`) included: the compile decides.
    config = RBLNQwen3_5TextModelConfig(
        max_seq_len=4096, batch_size=1, prefill_chunk_size=512, gdn_chunk_size=128, npu=npu
    )
    validate_gdn_custom_kernel(_text_config(), config)


def test_gdn_custom_kernel_validates_before_the_defaults_are_filled():
    # _update_rbln_config validates before super() fills the defaults, so every field it reads must
    # already be set. `prefill_chunk_size` is not: set_default_values owns it, and a config that
    # leaves it out carries None until then.
    config = RBLNQwen3_5TextModelConfig(max_seq_len=4096, batch_size=1)
    assert config.prefill_chunk_size is None

    validate_gdn_custom_kernel(_text_config(), config)
