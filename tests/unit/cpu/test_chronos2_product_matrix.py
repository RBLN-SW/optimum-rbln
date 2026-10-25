"""Bind full-batch compiled interfaces and trained tensors to the native model."""

import json
from pathlib import Path

import pytest

from scripts.diagnostics import chronos2_official_matrix as matrix
from scripts.diagnostics.chronos2_official_matrix import check_placement, expected_interfaces, native_parameter


@pytest.mark.parametrize(
    ("stage", "key", "expected"),
    [
        ("compiled_model", "model.encoder.block.9.weight", "encoder.block.9.weight"),
        ("embedding", "shared.weight", "shared.weight"),
        ("encoder_6", "layers.2.layer.0.weight", "encoder.block.8.layer.0.weight"),
        ("forecast", "head.residual_layer.weight", "output_patch_embedding.residual_layer.weight"),
        ("forecast", "layer_norm.weight", "encoder.final_layer_norm.weight"),
    ],
)
def test_native_trained_tensor_mapping(stage, key, expected):
    assert native_parameter(stage, key) == expected


def test_unknown_weight_owner_is_rejected():
    with pytest.raises(ValueError, match="Unmapped"):
        native_parameter("forecast", "unknown.weight")


def test_full_joint_batch_and_forecast_interfaces():
    inputs, outputs = expected_interfaces("encoder_9", 512)
    assert inputs[0][1] == [512, 577, 768]
    assert inputs[2][1] == [577, 1, 512, 512]
    assert outputs == [[[512, 577, 768], "float32"]]
    _, forecast = expected_interfaces("forecast", 512)
    assert forecast == [[[512, 21, 1024], "float32"], [[512, 21, 1024], "float32"]]


def test_existing_multifunction_graph_profile_is_preserved():
    profiles = json.loads(Path(matrix.__file__).with_name("chronos2_sdk343_placement.json").read_text())["profiles"]
    stats = profiles["RBLN-CA25:1"]["stats"]
    assert stats[0]["device_func_count"] == 13
    check_placement(stats, "RBLN-CA25", 1, "compiled_model")
    with pytest.raises(ValueError, match="placement differs"):
        check_placement(
            [{"compile_id": 0, "device_func_count": 1, "host_op_reason": {}}], "RBLN-CA25", 1, "compiled_model"
        )


def test_new_encoder_stage_does_not_accept_host_fallback():
    with pytest.raises(ValueError, match="placement differs"):
        check_placement(
            [{"compile_id": 0, "device_func_count": 2, "host_op_reason": {}}], "RBLN-CA25", 512, "encoder_0"
        )
