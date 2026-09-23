import json

import pytest
import torch
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5GatedDeltaNet as NativeGdn

from optimum.rbln.transformers.models.qwen3_5.configuration_qwen3_5 import (
    RBLNQwen3_5ForCausalLMConfig,
    RBLNQwen3_5ForConditionalGenerationConfig,
    RBLNQwen3_5ModelConfig,
    RBLNQwen3_5TextModelConfig,
)
from optimum.rbln.transformers.models.qwen3_5.modeling_qwen3_5 import _qwen3_5_linear_state_shapes
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
@pytest.mark.parametrize("grouped_conv_state", [False, True])
def test_gdn_custom_kernel_is_opt_in_and_serialized(config_class, grouped_conv_state, tmp_path):
    kwargs = {"max_seq_len": 4096, "batch_size": 1, "prefill_chunk_size": 512}
    if config_class in (RBLNQwen3_5ForConditionalGenerationConfig, RBLNQwen3_5ModelConfig):
        kwargs.update(
            use_inputs_embeds=True,
            visual={"cls_name": "RBLNQwen3_5VisionModelConfig", "max_seq_len": 1024},
        )
    assert config_class(**kwargs).gdn_custom_kernel is False
    assert config_class(**kwargs).gdn_grouped_conv_state is False
    with pytest.raises(ValueError, match="requires gdn_custom_kernel"):
        config_class(gdn_grouped_conv_state=True, **kwargs)
    config = config_class(gdn_custom_kernel=True, gdn_grouped_conv_state=grouped_conv_state, **kwargs)
    config.save(tmp_path)
    assert config_class.from_pretrained(tmp_path).gdn_custom_kernel is True
    assert config_class.from_pretrained(tmp_path).gdn_grouped_conv_state is grouped_conv_state
    path = tmp_path / "rbln_config.json"
    saved = json.loads(path.read_text())
    del saved["gdn_grouped_conv_state"]
    path.write_text(json.dumps(saved))
    assert config_class.from_pretrained(tmp_path).gdn_grouped_conv_state is False


@pytest.mark.parametrize("gate_bias", [False, True])
@pytest.mark.parametrize("mask_heads", [1, 6])
def test_gdn_custom_model_prefill_decode_and_reset(gate_bias, mask_heads):
    torch.manual_seed(42)
    hf_config = Qwen3_5TextConfig(
        hidden_size=64,
        linear_num_key_heads=2,
        linear_num_value_heads=6,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
    )
    native = NativeGdn(hf_config, layer_idx=0)
    if gate_bias:
        native.in_proj_a.bias = torch.nn.Parameter(torch.randn(6) * 0.1)
        native.in_proj_b.bias = torch.nn.Parameter(torch.randn(6) * 0.1)
    models = [
        Qwen3_5GatedDeltaNet(
            native,
            RBLNQwen3_5TextModelConfig(
                max_seq_len=4096,
                batch_size=1,
                prefill_chunk_size=512,
                gdn_custom_kernel=custom,
                gdn_grouped_conv_state=grouped,
            ),
            layer_idx=0,
        )
        for custom, grouped in ((False, False), (True, False), (True, True))
    ]
    initial = (torch.randn(1, 3, 1280), torch.randn(1, 768, 128))
    grouped_initial = torch.cat(
        [x.reshape(1, 3, 2, -1, 128).permute(0, 2, 3, 1, 4) for x in initial[0].split((256, 256, 768), dim=2)],
        dim=2,
    ).reshape(1, 30, 128)
    states = [initial, initial, (grouped_initial, initial[1])]
    assert _qwen3_5_linear_state_shapes(hf_config, 1) == ((1, 3, 1280), (1, 768, 128))
    assert _qwen3_5_linear_state_shapes(hf_config, 1, True) == ((1, 30, 128), (1, 768, 128))
    with torch.inference_mode():
        for phase, valid, reset in (
            ("prefill", 512, True),
            ("prefill", 294, False),
            ("prefill", 1, False),
            ("prefill", 2, False),
            ("decode", 1, False),
            ("decode", 1, False),
            ("prefill", 128, True),
            ("prefill", 1, True),
            ("prefill", 2, True),
        ):
            seq = 512 if phase == "prefill" else 1
            hidden = torch.randn(1, seq, 64)
            kwargs = {}
            if phase == "prefill":
                mask = torch.zeros(1, seq, mask_heads)
                mask[:, :valid] = 1
                kwargs = {
                    "valid_mask": mask,
                    "query_position": torch.tensor(valid - 1),
                    "conv_state_mask": torch.full_like(initial[0], 0 if reset else 1),
                    "recurrent_state_mask": torch.full_like(initial[1], 0 if reset else 1),
                }
            outputs = []
            for index, model in enumerate(models):
                model.phase = phase
                model_kwargs = dict(kwargs)
                if phase == "prefill":
                    model_kwargs["conv_state_mask"] = torch.full_like(states[index][0], 0 if reset else 1)
                norm_shapes = []
                with model.norm.register_forward_pre_hook(
                    lambda module, args, shapes=norm_shapes: shapes.append(tuple(tuple(value.shape) for value in args))
                ):
                    outputs.append(model(hidden, *states[index], **model_kwargs))
                shape = (1, 2, 3, seq, 128) if model.gdn_custom_kernel else (seq * 6, 128)
                assert norm_shapes == [(shape, shape)]
                states[index] = outputs[-1][1:]
            grouped_cache = outputs[2][1].reshape(1, 2, 5, 3, 128)
            legacy_cache = torch.cat(
                [x.permute(0, 3, 1, 2, 4).reshape(1, 3, -1) for x in grouped_cache.split((1, 1, 3), dim=2)],
                dim=2,
            )
            for expected, actual in zip(outputs[1], (outputs[2][0], legacy_cache, outputs[2][2]), strict=True):
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            for expected, actual in zip(outputs[0], outputs[1], strict=True):
                torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)
            torch.testing.assert_close(outputs[1][1], outputs[0][1], rtol=0, atol=0)
