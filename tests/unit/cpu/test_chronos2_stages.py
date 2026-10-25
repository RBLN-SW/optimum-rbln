"""Preserve full Chronos geometry while bounding CA25 runtime residency."""

import copy

import pytest
import torch
from chronos.chronos2.config import Chronos2CoreConfig
from chronos.chronos2.model import Chronos2Model

from optimum.rbln import RBLNChronos2Model, RBLNChronos2ModelConfig


@pytest.fixture
def native():
    torch.manual_seed(42)
    config = Chronos2CoreConfig(
        d_model=32,
        d_kv=8,
        d_ff=64,
        num_layers=4,
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


def test_ca25_large_batch_compiles_all_layers_without_slicing_groups(native, monkeypatch):
    seen = []

    def compile(cls, model, rbln_compile_config, create_runtimes, device):
        seen.append((model, rbln_compile_config, create_runtimes))
        return object()

    monkeypatch.setattr(RBLNChronos2Model, "compile", classmethod(compile))
    config = RBLNChronos2ModelConfig(batch_size=512, npu="RBLN-CA25", num_devices=1)
    RBLNChronos2Model._update_rbln_config(None, native, native.config, config)
    models = RBLNChronos2Model.get_compiled_model(native, config)
    assert isinstance(models, dict)
    assert list(models) == ["embedding", "encoder_0", "encoder_3", "forecast"]
    assert [cfg.compiled_model_name for cfg in config.compile_cfgs] == list(models)
    original = {p.data_ptr() for p in native.parameters()}
    partitioned = {p.data_ptr() for model, _, _ in seen for p in model.parameters()}
    assert partitioned == original
    assert all(not create for _, _, create in seen)
    assert [len(model.layers) for model, _, _ in seen[1:-1]] == [3, 1]
    for _, cfg, _ in seen[1:-1]:
        assert cfg.input_info == [
            ("hidden", [512, 9, 32], "float32"),
            ("attention_mask", [512, 1, 1, 9], "float32"),
            ("group_time_mask", [9, 1, 512, 512], "float32"),
        ]


@pytest.mark.parametrize(("npu", "batch"), [("RBLN-CA25", 256), ("RBLN-CR13", 512)])
def test_existing_single_graph_routes_remain_selected(native, monkeypatch, npu, batch):
    seen = []

    def compile(cls, model, rbln_compile_config, create_runtimes, device):
        seen.append(rbln_compile_config.compiled_model_name)
        return object()

    monkeypatch.setattr(RBLNChronos2Model, "compile", classmethod(compile))
    config = RBLNChronos2ModelConfig(batch_size=batch, npu=npu, num_devices=1)
    RBLNChronos2Model._update_rbln_config(None, native, native.config, config)
    assert not isinstance(RBLNChronos2Model.get_compiled_model(native, config), dict)
    assert seen == ["compiled_model"]


@pytest.mark.parametrize(
    "scenario", ["maximum", "grouped-covariates", "partial-grouped-covariates", "independent-missing", "all-missing"]
)
def test_stage_runtime_full_forecast_and_bounded_lifetime(native, monkeypatch, tmp_path, scenario):
    from optimum.rbln.chronos.chronos2_stages import Chronos2SequentialRuntime

    active = []
    order = []

    class Runtime:
        def __init__(self, stage, name):
            assert not active
            active.append(name)
            self.stage, self.name = stage, name

        def __call__(self, *inputs):
            order.append(self.name)
            return self.stage(*inputs)

        def __del__(self):
            active.remove(self.name)

    class Compiled:
        def __init__(self, stage, name):
            self.stage, self.name = stage, name

        def create_runtime(self, **kwargs):
            return Runtime(self.stage, self.name)

    def compile(cls, model, rbln_compile_config, create_runtimes, device):
        return Compiled(model, rbln_compile_config.compiled_model_name)

    monkeypatch.setattr(RBLNChronos2Model, "compile", classmethod(compile))
    config = RBLNChronos2ModelConfig(batch_size=512, npu="RBLN-CA25", num_devices=1)
    RBLNChronos2Model._update_rbln_config(None, native, native.config, config)
    compiled = RBLNChronos2Model.get_compiled_model(native, config)
    config.freeze()
    config.save(tmp_path / "rbln_config.json")
    loaded = RBLNChronos2ModelConfig.from_pretrained(tmp_path)
    loaded.freeze()
    assert [(c.compiled_model_name, c.input_info) for c in loaded.compile_cfgs] == [
        (c.compiled_model_name, c.input_info) for c in config.compile_cfgs
    ]
    chain = RBLNChronos2Model._create_runtimes(list(compiled.values()), loaded)[0]
    assert isinstance(chain, Chronos2SequentialRuntime)
    model = RBLNChronos2Model([chain], copy.deepcopy(native.config), loaded)
    batch = 3 if scenario == "partial-grouped-covariates" else 512
    context = torch.sin(torch.arange(64).float() / 10).repeat(batch, 1)
    groups = torch.arange(batch)
    inputs = {"context": context, "group_ids": groups, "num_output_patches": 4}
    if "grouped-covariates" in scenario:
        groups[:] = 7
        context[:, ::5] = torch.nan
        future = torch.full((batch, 64), torch.nan)
        future[0] = torch.cos(torch.arange(64).float() / 9)
        inputs["future_covariates"] = future
    elif scenario == "independent-missing":
        context[:, ::3] = torch.nan
    elif scenario == "all-missing":
        context[:] = torch.nan
    with torch.no_grad():
        expected = native(**inputs).quantile_preds
        actual = model(**inputs).quantile_preds
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(model(**inputs).quantile_preds, actual, atol=0, rtol=0)
    assert not active
    assert order == list(compiled) * 2
