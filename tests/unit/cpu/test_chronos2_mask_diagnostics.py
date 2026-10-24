"""Mask semantics and pre-assert capture controls; no NPU or full mask allocation."""

from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch
from chronos.chronos2.model import Chronos2Encoder

from scripts.diagnostics.chronos2_mask_diagnostics import MaskDiagnostics, mask_comparison, tensor_digest
from scripts.diagnostics.chronos2_sequential_runtime import SequentialRuntimeChain


def masks():
    context = torch.tensor([[0, 1, 1], [0, 0, 1], [0, 0, 0], [0, 0, 0]], dtype=torch.bool)
    groups = torch.tensor([0, 0, 2, 3])
    attention = torch.cat((context.float(), torch.ones(4, 2)), 1)
    time = Chronos2Encoder._expand_and_invert_time_attention_mask(attention, torch.float32)
    group = Chronos2Encoder._construct_and_invert_group_time_mask(groups, attention, torch.float32)
    return context, groups, time, group


def test_physical_sentinel_difference_is_recorded_without_hiding_mask_membership():
    context, groups, time, group = masks()
    actual_time = torch.where(time == 0, 0.0, -8581545984.0)
    actual_group = torch.where(group == 0, 0.0, -8581545984.0)
    report, slices = mask_comparison(context, groups, actual_time, actual_group, active_batch=2, chunk=2)
    assert report["group_zero_membership_mismatches"] == 0
    assert report["time_zero_membership_mismatches"] == 0
    assert report["blocked_not_negative"] == 0
    assert report["actual_group"]["sha256"] != report["native_cpu_group_sha256"]
    torch.testing.assert_close(slices["native_cpu_time"], time, rtol=0, atol=0)


def test_stride_preservation_and_query_key_axis_error_are_distinguished():
    context, groups, time, group = masks()
    same_values_different_strides = group.transpose(2, 3).contiguous().transpose(2, 3)
    assert tensor_digest(same_values_different_strides) == tensor_digest(group)
    correct, _ = mask_comparison(context, groups, time, same_values_different_strides, 2, chunk=2)
    assert correct["group_zero_membership_mismatches"] == 0
    wrong, _ = mask_comparison(context, groups, time, group.transpose(2, 3), 2, chunk=2)
    assert wrong["active_query_membership_mismatches"] > 0


def test_one_cross_group_permission_error_is_detected():
    context, groups, time, group = masks()
    actual = group.clone()
    actual[-1, 0, 0, 3] = 0
    report, _ = mask_comparison(context, groups, time, actual, 2, chunk=2)
    assert report["group_zero_membership_mismatches"] == 1
    assert report["active_query_membership_mismatches"] == 1


def test_forecasts_are_persisted_before_original_parity_assertion(tmp_path):
    events = []
    observer = MaskDiagnostics(tmp_path, events.append)
    inputs = {"context": torch.ones(3, 7), "group_ids": torch.full((3,), 7)}
    expected = SimpleNamespace(quantile_preds=torch.ones(3, 2, 4))
    actual = SimpleNamespace(quantile_preds=torch.zeros(3, 2, 4))
    observer.reference(None, inputs, expected)
    with observer.product_request(inputs):
        observer.product_output(actual)
    with pytest.raises(AssertionError):
        assert torch.equal(actual.quantile_preds, expected.quantile_preds)
    saved = torch.load(tmp_path / "request-0001/forecast-before-original-assert.pt", weights_only=True)
    torch.testing.assert_close(saved["actual_quantile_preds"], actual.quantile_preds)
    torch.testing.assert_close(saved["original_cpu_expected_quantile_preds"], expected.quantile_preds)
    assert events[0]["phase"] == "mask_diagnostics_before_original_assert"


def test_stage_observer_runs_outside_runtime_lifetime(tmp_path):
    active, seen = [], []

    @contextmanager
    def loader(path, device):
        active.append(path.stem)

        def runtime(*inputs):
            if path.stem == "embedding":
                return inputs[0] + 1, torch.ones(1), torch.ones(1)
            return (inputs[0] + 2, inputs[0] + 3)

        try:
            yield runtime
        finally:
            active.clear()

    def observe(phase, call, stage, inputs, output):
        assert not active
        seen.append((phase, stage["name"]))

    stages = [{"name": n, "file": n + ".rbln", "alloc_per_node": [1]} for n in ("embedding", "forecast")]
    chain = SequentialRuntimeChain(tmp_path, stages, 0, lambda value: None, loader=loader, observer=observe)
    output = chain(torch.zeros(1))
    torch.testing.assert_close(output[0], torch.tensor([3.0]))
    assert seen == [(phase, name) for name in ("embedding", "forecast") for phase in ("before", "after")]


def test_observer_preserves_real_segmented_cpu_forecast_and_captures_masks(tmp_path):
    import copy
    import json

    from chronos.chronos2.config import Chronos2CoreConfig
    from chronos.chronos2.model import Chronos2Model

    from optimum.rbln import RBLNChronos2Model, RBLNChronos2ModelConfig
    from scripts.diagnostics.chronos2_encoder_segments import make_jobs

    torch.manual_seed(42)
    native = (
        Chronos2Model(
            Chronos2CoreConfig(
                d_model=32,
                d_kv=8,
                d_ff=64,
                num_layers=1,
                num_heads=4,
                dropout_rate=0,
                chronos_config={
                    "context_length": 64,
                    "input_patch_size": 16,
                    "input_patch_stride": 16,
                    "output_patch_size": 16,
                    "max_output_patches": 3,
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
    config = RBLNChronos2ModelConfig(batch_size=4, context_length=64, prediction_length=48)
    RBLNChronos2Model._update_rbln_config(None, native, native.config, config)
    config.freeze()
    jobs = make_jobs(native, config.compile_cfgs[0], 1)
    modules = {cfg.compiled_model_name: job for job, cfg in jobs}
    stages = [{"name": name, "file": name + ".rbln", "alloc_per_node": [1]} for name in modules]

    @contextmanager
    def loader(path, device):
        yield modules[path.stem]

    observer = MaskDiagnostics(tmp_path / "capture", lambda event: None)
    chain = SequentialRuntimeChain(tmp_path, stages, 0, lambda event: None, loader=loader, observer=observer.stage)
    product = RBLNChronos2Model([chain], copy.deepcopy(native.config), config)
    context = torch.stack([torch.sin(torch.arange(31).float() / 10) + i for i in range(3)])
    context[:, ::5] = torch.nan
    future = torch.full((3, 48), torch.nan)
    future[0] = torch.cos(torch.arange(48) / 9)
    inputs = {
        "context": context,
        "group_ids": torch.full((3,), 7),
        "future_covariates": future,
        "num_output_patches": 3,
    }
    with torch.no_grad():
        expected = native(**inputs)
        observer.reference(native, inputs, expected)
        with observer.product_request(inputs):
            actual = product(**inputs)
            observer.product_output(actual)
        chain.observer = None
        unobserved = product(**inputs)
    torch.testing.assert_close(actual.quantile_preds, unobserved.quantile_preds, rtol=0, atol=0)
    torch.testing.assert_close(actual.quantile_preds, expected.quantile_preds)
    report = json.loads((tmp_path / "capture/request-0001/embedding-mask-comparison.json").read_text())
    assert report["group_zero_membership_mismatches"] == 0
    assert report["time_zero_membership_mismatches"] == 0
    assert (tmp_path / "capture/request-0001/normalized-inputs.pt").exists()
    assert (tmp_path / "capture/request-0001/forecast-before-original-assert.pt").exists()
