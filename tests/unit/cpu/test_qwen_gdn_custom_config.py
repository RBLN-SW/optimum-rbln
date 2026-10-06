import json

import pytest
import torch
from rebel.ops.torch_custom_ops.gdn import block_state, unblock_state
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5GatedDeltaNet as NativeGdn

from optimum.rbln.transformers.models.qwen3_5.configuration_qwen3_5 import (
    RBLNQwen3_5ForCausalLMConfig,
    RBLNQwen3_5ForConditionalGenerationConfig,
    RBLNQwen3_5ModelConfig,
    RBLNQwen3_5TextModelConfig,
)
from optimum.rbln.transformers.models.qwen3_5.modeling_qwen3_5 import (
    _qwen3_5_linear_mask_shapes,
    _qwen3_5_linear_state_shapes,
    _qwen3_5_resolve_gdn_custom_kernel,
)
from optimum.rbln.transformers.models.qwen3_5.qwen3_5_architecture import Qwen3_5GatedDeltaNet


@pytest.mark.parametrize(
    "config_class",
    [
        RBLNQwen3_5ForCausalLMConfig,
        RBLNQwen3_5ForConditionalGenerationConfig,
        RBLNQwen3_5ModelConfig,
        RBLNQwen3_5TextModelConfig,
    ],
)
def test_gdn_custom_kernel_is_recorded_and_serialized(config_class, tmp_path):
    kwargs = {"max_seq_len": 4096, "batch_size": 1, "prefill_chunk_size": 512}
    if config_class in (RBLNQwen3_5ForConditionalGenerationConfig, RBLNQwen3_5ModelConfig):
        kwargs.update(
            use_inputs_embeds=True,
            visual={"cls_name": "RBLNQwen3_5VisionModelConfig", "max_seq_len": 1024},
        )
    assert config_class(**kwargs).gdn_custom_kernel is None
    config = config_class(gdn_custom_kernel=True, **kwargs)
    config.save(tmp_path)
    assert config_class.from_pretrained(tmp_path).gdn_custom_kernel is True
    # An export from before the custom core loads with the native core.
    path = tmp_path / "rbln_config.json"
    saved = json.loads(path.read_text())
    del saved["gdn_custom_kernel"]
    path.write_text(json.dumps(saved))
    assert config_class.from_pretrained(tmp_path).gdn_custom_kernel is None


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
    models = [
        Qwen3_5GatedDeltaNet(
            native,
            RBLNQwen3_5TextModelConfig(
                max_seq_len=4096,
                batch_size=batch_size,
                prefill_chunk_size=prefill_size,
                gdn_custom_kernel=custom,
            ),
            layer_idx=0,
        )
        for custom in (False, True)
    ]
    return hf_config, models


def _grouped(flat, ratio):
    """Flat (B, K-1, conv_dim) cache -> (B, Hk*(2+r)*(K-1), 128) grouped by Q/K head."""
    batch = flat.shape[0]
    parts = flat.split((256, 256, 256 * ratio), dim=2)
    grouped = torch.cat([x.reshape(batch, 3, 2, -1, 128).permute(0, 2, 3, 1, 4) for x in parts], dim=2)
    return grouped.reshape(batch, -1, 128)


def _flat(grouped, ratio):
    batch = grouped.shape[0]
    grouped = grouped.reshape(batch, 2, 2 + ratio, 3, 128)
    return torch.cat(
        [x.permute(0, 3, 1, 2, 4).reshape(batch, 3, -1) for x in grouped.split((1, 1, ratio), dim=2)], dim=2
    )


def _custom_states(native_states):
    """The native (conv, recurrent) caches in the custom core's layouts."""
    conv, recurrent = native_states
    ratio = conv.shape[-1] // 256 - 2
    batch = recurrent.shape[0]
    return _grouped(conv, ratio), block_state(recurrent.reshape(batch, 2, ratio, 128, 128))


def _step(models, states, hidden, kwargs):
    """Runs both models and carries their states: the native one returns its new recurrent state, the custom one
    updates its recurrent cache in place and returns None for it."""
    outputs = []
    for index, model in enumerate(models):
        output = model(hidden, *states[index], **kwargs[index])
        assert (output[2] is None) is model.gdn_custom_kernel
        states[index] = (output[1], states[index][1] if output[2] is None else output[2])
        outputs.append(output[0])
    return outputs


def _compare(outputs, states, ratio):
    # The custom core matches the native path; its caches hold the native caches' values.
    (native, custom), (native_states, custom_states) = outputs, states
    torch.testing.assert_close(custom, native, rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(_flat(custom_states[0], ratio), native_states[0], rtol=0, atol=0)
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
    assert _qwen3_5_linear_state_shapes(hf_config, 1) == ((1, 3, conv_dim), (1, units * 128, 128))
    assert _qwen3_5_linear_state_shapes(hf_config, 1, True) == (
        (1, 2 * (2 + ratio) * 3, 128),
        (1, 2, ratio, 2, 128, 64),
    )
    assert _qwen3_5_linear_mask_shapes(hf_config, 1, True) == ((1, 2 * (2 + ratio) * 3, 128), (1,))
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
                    conv_mask, recurrent_mask = _qwen3_5_linear_mask_shapes(hf_config, 1, model.gdn_custom_kernel)
                    model_kwargs["conv_state_mask"] = torch.full(conv_mask, 0.0 if reset else 1.0)
                    model_kwargs["recurrent_state_mask"] = torch.full(recurrent_mask, 0.0 if reset else 1.0)
                per_model.append(model_kwargs)

            def record(module, args, shapes=norm_shapes):
                shapes.append(tuple(tuple(value.shape) for value in args))

            # Both models wrap the native layer's one norm.
            hook = models[0].norm.register_forward_pre_hook(record)
            outputs = _step(models, states, hidden, per_model)
            hook.remove()
            custom_shape = (1, 2, ratio, seq, 128)
            assert norm_shapes == [((seq * units, 128),) * 2, (custom_shape,) * 2]
            _compare(outputs, states, ratio)


@pytest.mark.parametrize("ratio", [1, 3])
def test_gdn_custom_model_batched_decode(ratio):
    batch = 3
    _, models = _gdn_models(ratio, 512, gate_bias=True, batch_size=batch)
    initial = (torch.randn(batch, 3, 512 + 256 * ratio), torch.randn(batch, 2 * ratio * 128, 128) * 0.1)
    states = [initial, _custom_states(initial)]
    with torch.inference_mode():
        for _ in range(2):
            for model in models:
                model.phase = "decode"
            outputs = _step(models, states, torch.randn(batch, 1, 64), [{}, {}])
            _compare(outputs, states, ratio)


@pytest.mark.parametrize(
    "overrides, prefill_size, gdn_chunk_size, expected",
    [
        ({}, 512, 128, True),
        ({"linear_num_value_heads": 10}, 384, 128, True),
        ({"linear_key_head_dim": 64}, 512, 128, False),
        ({}, 64, 64, False),
    ],
)
def test_gdn_custom_kernel_serves_supported_configs(overrides, prefill_size, gdn_chunk_size, expected):
    dims = {"linear_num_value_heads": 6, "linear_key_head_dim": 128, "linear_value_head_dim": 128, **overrides}
    text_config = Qwen3_5TextConfig(hidden_size=64, linear_num_key_heads=2, **dims)
    native = NativeGdn(text_config, layer_idx=0)

    def rbln_config(**options):
        return RBLNQwen3_5TextModelConfig(
            max_seq_len=4096,
            batch_size=1,
            prefill_chunk_size=prefill_size,
            gdn_chunk_size=gdn_chunk_size,
            **options,
        )

    assert Qwen3_5GatedDeltaNet(native, rbln_config(), layer_idx=0).gdn_custom_kernel is expected
    # An export records the choice; a recorded choice wins.
    config = rbln_config()
    _qwen3_5_resolve_gdn_custom_kernel(text_config, config)
    assert config.gdn_custom_kernel is expected
    assert Qwen3_5GatedDeltaNet(native, rbln_config(gdn_custom_kernel=False), layer_idx=0).gdn_custom_kernel is False


@pytest.mark.parametrize("npu, expected", [("RBLN-CA25", True), ("RBLN-CA22", True), ("RBLN-CR13", False)])
def test_gdn_custom_kernel_runs_on_atom_only(npu, expected):
    text_config = Qwen3_5TextConfig(
        hidden_size=64,
        linear_num_key_heads=2,
        linear_num_value_heads=6,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
    )
    config = RBLNQwen3_5TextModelConfig(
        max_seq_len=4096, batch_size=1, prefill_chunk_size=512, gdn_chunk_size=128, npu=npu
    )
    _qwen3_5_resolve_gdn_custom_kernel(text_config, config)
    assert config.gdn_custom_kernel is expected
