"""CPU math/identity-gate tests; this never compiles or executes an NPU graph."""

import pytest
import torch
from chronos.chronos2.config import Chronos2CoreConfig
from chronos.chronos2.model import Chronos2Model

from scripts.diagnostics.chronos2_encoder0_probe import (
    Encoder0BoundaryProbe,
    require_original_final_identity,
    tensor_sha256,
)
from scripts.diagnostics.chronos2_encoder_segments import EncoderJob


def test_augmented_graph_keeps_original_trained_module_ownership_and_final_math():
    torch.manual_seed(42)
    model = (
        Chronos2Model(
            Chronos2CoreConfig(
                d_model=32,
                d_kv=8,
                d_ff=64,
                num_layers=3,
                num_heads=4,
                dropout_rate=0,
                chronos_config={
                    "context_length": 64,
                    "input_patch_size": 16,
                    "input_patch_stride": 16,
                    "output_patch_size": 16,
                    "max_output_patches": 4,
                    "quantiles": [0.1, 0.5, 0.9],
                    "use_reg_token": True,
                    "use_arcsinh": True,
                    "time_encoding_scale": 64,
                },
            )
        )
        .float()
        .eval()
    )
    layers = model.encoder.block[:3]
    original, probe = EncoderJob(layers).eval(), Encoder0BoundaryProbe(layers).eval()
    assert {id(p) for p in original.parameters()} == {id(p) for p in probe.parameters()}
    hidden = torch.randn(4, 8, 32)
    attention = torch.ones(4, 8)
    attention[3, :4] = 0
    groups = torch.tensor([0, 0, 0, 3])
    time = model.encoder._expand_and_invert_time_attention_mask(attention, torch.float32)
    group = model.encoder._construct_and_invert_group_time_mask(groups, attention, torch.float32)
    native_boundaries = []
    handles = [
        layer.register_forward_hook(lambda _module, _inputs, output: native_boundaries.append(output[0]))
        for layer in layers[0].layer[:2]
    ]
    handles.append(
        layers[0].layer[2].register_forward_hook(lambda _module, _inputs, output: native_boundaries.append(output))
    )
    handles.extend(
        layer.register_forward_hook(lambda _module, _inputs, output: native_boundaries.append(output.hidden_states))
        for layer in layers[1:]
    )
    with torch.no_grad():
        expected = original(hidden, time, group)
        for handle in handles:
            handle.remove()
        actual = probe(hidden, time, group)
    assert len(actual) == 5
    for boundary, native in zip(actual, native_boundaries, strict=True):
        torch.testing.assert_close(boundary, native, rtol=0, atol=0)
    torch.testing.assert_close(actual[-1], expected, rtol=0, atol=0)
    assert tensor_sha256(actual[-1]) == tensor_sha256(expected)
    assert require_original_final_identity(actual, tensor_sha256(expected), hidden.shape)[
        "valid_for_original_encoder0_attribution"
    ]


def test_changed_final_bytes_invalidates_probe_before_boundary_attribution():
    outputs = [torch.zeros(4, 8, 32) for _ in range(5)]
    original_sha = tensor_sha256(outputs[-1])
    outputs[-1][0, 0, 0] = 1
    with pytest.raises(ValueError, match="INVALID PROBE"):
        require_original_final_identity(outputs, original_sha, (4, 8, 32))
