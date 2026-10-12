from types import SimpleNamespace

import pytest

from optimum.rbln.transformers.modeling_attention_utils import RBLNDecoderOnlyFlashAttentionMixin


class FakeCompiledModel:
    # alloc[key][node_id][chiplet_id]
    def __init__(self, alloc, dram=None):
        self.alloc = alloc
        self.dram = dram or {}

    def get_alloc_per_chiplet_by_key(self):
        return self.alloc

    def exp_get_dram_tensor_sizes(self):
        return self.dram


def fake_model(alloc, device):
    return SimpleNamespace(compiled_models=[FakeCompiledModel(alloc)], rbln_config=SimpleNamespace(device=device))


def collect(lm_device, colocated, monkeypatch):
    monkeypatch.setattr(
        "optimum.rbln.transformers.modeling_attention_utils.get_available_dram_per_chiplet", lambda n, npu: 1000
    )
    lm = {"prefill": FakeCompiledModel({"Kernel": [[100], [100]]}, dram={"kv": [[10], [10]]})}
    rbln_config = SimpleNamespace(device=lm_device, memory_budget=None, npu=None)
    alloc, _, _, chiplets = RBLNDecoderOnlyFlashAttentionMixin._collect_chiplet_kvcache_inputs(
        lm, rbln_config, colocated
    )
    return dict(alloc), chiplets


def test_no_colocated_models_is_unchanged(monkeypatch):
    alloc, chiplets = collect([0, 1], None, monkeypatch)
    assert alloc == {(0, 0): 100, (1, 0): 100}
    assert chiplets == {(0, 0), (1, 0)}


def test_disjoint_devices_reserve_nothing(monkeypatch):
    vision = fake_model({"Kernel": [[700], [700]]}, device=[2, 3])
    alloc, _ = collect([0, 1], [vision], monkeypatch)
    assert alloc == {(0, 0): 100, (1, 0): 100}


def test_overlapping_devices_reserve_on_matching_local_node(monkeypatch):
    # vision node 0 -> device 1 (LM node 1), vision node 1 -> device 2 (not ours)
    vision = fake_model({"Kernel": [[700], [700]], "Weight": [[50], [50]]}, device=[1, 2])
    alloc, _ = collect([0, 1], [vision], monkeypatch)
    assert alloc == {(0, 0): 100, (1, 0): 100 + 750}


def test_unset_devices_are_treated_as_overlapping(monkeypatch):
    vision = fake_model({"Kernel": [[700], [700]]}, device=None)
    alloc, _ = collect(None, [vision], monkeypatch)
    assert alloc == {(0, 0): 800, (1, 0): 800}


def test_int_device_and_dram_tensor_of_colocated_are_counted(monkeypatch):
    vision = fake_model({"Kernel": [[700]], "DramTensor": [[30]]}, device=1)
    alloc, _ = collect([0, 1], [vision], monkeypatch)
    assert alloc == {(0, 0): 100, (1, 0): 830}


@pytest.mark.parametrize("compiled_models", [None, []])
def test_model_without_compiled_models_is_skipped(monkeypatch, compiled_models):
    empty = SimpleNamespace(compiled_models=compiled_models, rbln_config=SimpleNamespace(device=None))
    alloc, _ = collect(None, [empty], monkeypatch)
    assert alloc == {(0, 0): 100, (1, 0): 100}


def test_reservation_lowers_estimated_blocks(monkeypatch):
    monkeypatch.setattr(
        "optimum.rbln.transformers.modeling_attention_utils.get_available_dram_per_chiplet", lambda n, npu: 2**21 * 64
    )
    lm = {"prefill": FakeCompiledModel({"Kernel": [[0]]}, dram={"kv": [[2**21]]})}
    rbln_config = SimpleNamespace(
        device=None,
        memory_budget=None,
        npu=None,
        phases=["prefill", "decode"],
        num_full_blocks=64,
        cache_metas=[SimpleNamespace(name="kv", can_resize=True)],
    )
    assert RBLNDecoderOnlyFlashAttentionMixin.estimate_num_kvcache_blocks(lm, rbln_config) == 64
    vision = fake_model({"Kernel": [[2**21 * 16]]}, device=None)
    assert (
        RBLNDecoderOnlyFlashAttentionMixin.estimate_num_kvcache_blocks(lm, rbln_config, colocated_models=[vision])
        == 48
    )
