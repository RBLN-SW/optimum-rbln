"""CPU proof for job boundaries; this never compiles or uses an NPU."""

import copy

import pytest
import torch
from chronos.chronos2.config import Chronos2CoreConfig
from chronos.chronos2.model import Chronos2Model

from optimum.rbln import RBLNChronos2Model, RBLNChronos2ModelConfig
from optimum.rbln.chronos.chronos2_architecture import Chronos2ModelWrapper
from scripts.diagnostics.chronos2_encoder_segments import RuntimeChain, make_jobs


@pytest.fixture
def native():
    torch.manual_seed(42)
    return (
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


def config(native, batch):
    result = RBLNChronos2ModelConfig(batch_size=batch, prediction_length=48)
    RBLNChronos2Model._update_rbln_config(None, native, native.config, result)
    result.freeze()
    return result


@pytest.mark.parametrize("chunk", [1, 2, 3])
@pytest.mark.parametrize(
    "scenario", ["maximum", "grouped-covariates", "partial-grouped-covariates", "independent-missing", "all-missing"]
)
def test_all_native_scenarios_preserve_full_forecast(native, chunk, scenario):
    cfg = config(native, 4)
    jobs = make_jobs(native, cfg.compile_cfgs[0], chunk)
    segmented = RBLNChronos2Model([RuntimeChain([job for job, _ in jobs])], copy.deepcopy(native.config), cfg)
    original = RBLNChronos2Model([Chronos2ModelWrapper(native)], copy.deepcopy(native.config), cfg)
    batch = 3 if scenario == "partial-grouped-covariates" else 4
    length = 64 if scenario == "maximum" else 31
    context = torch.stack([torch.sin(torch.arange(length).float() / 10) + i for i in range(batch)])
    groups = torch.arange(batch)
    inputs = {"context": context, "group_ids": groups, "num_output_patches": 3}
    if "grouped-covariates" in scenario:
        groups[:] = 7
        context[:, ::5] = torch.nan
        future = torch.full((batch, 48), torch.nan)
        future[0] = torch.cos(torch.arange(48) / 9)
        inputs["future_covariates"] = future
    elif scenario == "independent-missing":
        context[:, ::3] = torch.nan
    elif scenario == "all-missing":
        context[:] = torch.nan
    with torch.no_grad():
        expected = native(**inputs).quantile_preds
        actual = segmented(**inputs).quantile_preds
        torch.testing.assert_close(actual, original(**inputs).quantile_preds, atol=0, rtol=0)
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(segmented(**inputs).quantile_preds, actual, atol=0, rtol=0)


def test_batch512_contract_never_slices_cross_series_attention(native):
    cfg = config(native, 512)
    jobs = make_jobs(native, cfg.compile_cfgs[0], 2)
    assert [cfg.compiled_model_name for _, cfg in jobs] == ["embedding", "encoder_0", "encoder_2", "forecast"]
    for _job, cfg in jobs[1:-1]:
        assert cfg.input_info == [
            ("hidden", [512, 8, 32], "float32"),
            ("attention_mask", [512, 1, 1, 8], "float32"),
            ("group_time_mask", [8, 1, 512, 512], "float32"),
        ]
    assert [len(job.layers) for job, _ in jobs[1:-1]] == [2, 1]


@pytest.mark.parametrize("chunk", [0, -1, True, 1.5, 4])
def test_invalid_encoder_boundaries_are_rejected(native, chunk):
    with pytest.raises(ValueError, match="chunk_size"):
        make_jobs(native, config(native, 4).compile_cfgs[0], chunk)
