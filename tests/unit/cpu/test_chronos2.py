import copy

import pytest
import torch
from chronos.chronos2.config import Chronos2CoreConfig
from chronos.chronos2.model import Chronos2Model

from optimum.rbln import RBLNChronos2Model, RBLNChronos2ModelConfig
from optimum.rbln.chronos.chronos2_architecture import Chronos2ModelWrapper


@pytest.fixture
def native():
    torch.manual_seed(42)
    config = Chronos2CoreConfig(
        d_model=32,
        d_kv=8,
        d_ff=64,
        num_layers=2,
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
    return Chronos2Model(config).float().eval()


def optimized(native, batch=4, prediction_length=48):
    config = RBLNChronos2ModelConfig(batch_size=batch, prediction_length=prediction_length)
    RBLNChronos2Model._update_rbln_config(None, native, native.config, config)
    config.freeze()
    return RBLNChronos2Model([Chronos2ModelWrapper(native)], copy.deepcopy(native.config), config)


def test_native_max_defaults(native):
    config = RBLNChronos2ModelConfig()
    RBLNChronos2Model._update_rbln_config(None, native, native.config, config)
    assert config.context_length == 64
    assert config.prediction_length == 64
    assert config.compile_cfgs[0].input_info == [
        ("patched_context", [1, 4, 48], "float32"),
        ("context_mask", [1, 4], "bool"),
        ("patched_future", [1, 4, 48], "float32"),
        ("group_ids", [1], "int64"),
    ]


@pytest.mark.parametrize("group_ids", [torch.tensor([3, 3, -2]), torch.tensor([1, 2, 3])])
@pytest.mark.parametrize("length", [31, 64])
def test_full_forecast_with_groups_missing_data_and_covariates(native, group_ids, length):
    model = optimized(native)
    context = torch.randn(3, length)
    context[0, ::3] = torch.nan
    context[2] = torch.nan
    covariates = torch.full((3, 36), torch.nan)
    covariates[1] = torch.linspace(-1, 1, 36)
    with torch.no_grad():
        expected = native(
            context=context, group_ids=group_ids, future_covariates=covariates, num_output_patches=3
        ).quantile_preds
        actual = model(
            context=context, group_ids=group_ids, future_covariates=covariates, num_output_patches=3
        ).quantile_preds
    torch.testing.assert_close(actual, expected)
    assert actual.isfinite().all()
    assert not list(model.preprocessor.patch.parameters())
    assert not list(model.preprocessor.instance_norm.parameters())


def test_rejects_different_future_token_count(native):
    with pytest.raises(ValueError, match="num_output_patches"):
        optimized(native)(context=torch.ones(1, 64), num_output_patches=2)


@pytest.mark.parametrize("kwargs", [{"batch_size": 0}, {"context_length": 0}, {"prediction_length": 0}])
def test_invalid_configuration(kwargs):
    with pytest.raises(ValueError, match="positive integer"):
        RBLNChronos2ModelConfig(**kwargs)


def test_native_forecasting_pipeline(native):
    from chronos import Chronos2Pipeline

    context = torch.sin(torch.arange(31, dtype=torch.float32)).reshape(1, 1, -1)
    expected = Chronos2Pipeline(native).predict(
        context, prediction_length=36, context_length=64, batch_size=4, unrolled_quantiles=[0.1, 0.5, 0.9]
    )
    actual = Chronos2Pipeline(optimized(native)).predict(
        context, prediction_length=36, context_length=64, batch_size=4, unrolled_quantiles=[0.1, 0.5, 0.9]
    )
    assert len(actual) == len(expected)
    for result, reference in zip(actual, expected, strict=True):
        torch.testing.assert_close(result, reference)


def test_explicit_masks_and_extreme_group_ids(native):
    model = optimized(native)
    context = torch.randn(2, 64)
    context_mask = torch.ones_like(context)
    context_mask[:, ::3] = 0
    future = torch.randn(2, 37)
    future_mask = torch.ones_like(future)
    future_mask[:, 1::2] = 0
    future[future_mask == 0] = torch.nan
    groups = torch.tensor([torch.iinfo(torch.int64).max, torch.iinfo(torch.int64).min])
    inputs = {
        "context": context,
        "context_mask": context_mask,
        "group_ids": groups,
        "future_covariates": future,
        "future_covariates_mask": future_mask,
        "num_output_patches": 3,
    }
    with torch.no_grad():
        expected = native(**inputs).quantile_preds
        actual = model(**inputs).quantile_preds
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(model(**inputs, return_dict=False)[0], actual)


def test_preserves_native_covariate_validation(native):
    with pytest.raises(ValueError, match="NaN"):
        optimized(native)(
            context=torch.ones(1, 64),
            future_covariates=torch.full((1, 16), torch.nan),
            future_covariates_mask=torch.ones(1, 16),
        )


def test_configuration_roundtrip_and_patch_rounding(tmp_path):
    config = RBLNChronos2ModelConfig(batch_size=4, context_length=64, prediction_length=36)
    config.save(tmp_path)
    loaded = RBLNChronos2ModelConfig.from_pretrained(tmp_path)
    assert loaded.batch_size == 4
    assert loaded.context_length == 64
    assert loaded.prediction_length == 36


@pytest.mark.parametrize("split", [False, True])
def test_preserves_fp32_output_sum_before_inverse_arcsinh(native, split):
    model = optimized(native, batch=1)
    shape = (1, 3, 48)
    branches = (torch.full(shape, 16.0, dtype=torch.bfloat16), torch.full(shape, 0.0625, dtype=torch.bfloat16))
    model.model = [lambda *inputs: branches if split else branches[0].float() + branches[1].float()]
    result = model(context=torch.full((1, 64), torch.nan)).quantile_preds
    assert result.dtype == torch.float32
    torch.testing.assert_close(result, torch.sinh(torch.full(shape, 16.0625)), rtol=1e-6, atol=0)


def test_preserves_output_head_layer_norm(native):
    from chronos.chronos2.layers import Chronos2LayerNorm

    head = native.output_patch_embedding
    head.use_layer_norm = True
    head.layer_norm = Chronos2LayerNorm(native.num_quantiles * native.chronos_config.output_patch_size)
    inputs = {"context": torch.randn(1, 64), "num_output_patches": 3}
    with torch.no_grad():
        expected = native(**inputs).quantile_preds
        actual = optimized(native)(**inputs).quantile_preds
    torch.testing.assert_close(actual, expected)
