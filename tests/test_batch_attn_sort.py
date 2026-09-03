import pytest
import torch
from transformers.generation.utils import GenerationMixin

from optimum.rbln.transformers.models.decoderonly.configuration_decoderonly import (
    RBLNDecoderOnlyModelForCausalLMConfig,
)
from optimum.rbln.transformers.models.decoderonly.generation_decoderonly import (
    RBLNDecoderOnlyGenerationMixin,
    _expand_batch_perm_idx,
    _permute_flat_segments,
)
from optimum.rbln.transformers.models.decoderonly.modeling_decoderonly import RBLNDecoderOnlyModelForCausalLM
from optimum.rbln.utils.runtime_utils import npu_is_cr13_or_later


LENGTHS = torch.tensor([3, 7, 5, 7])
SORT_IDX = torch.argsort(LENGTHS, descending=True)
UNSORT_IDX = torch.argsort(SORT_IDX)


def test_expand_batch_perm_idx():
    perm = torch.tensor([2, 0, 1])
    assert _expand_batch_perm_idx(perm, 3).tolist() == [2, 0, 1]
    assert _expand_batch_perm_idx(perm, 6).tolist() == [4, 5, 0, 1, 2, 3]


def test_permute_flat_segments():
    # samples own [2, 1, 3] rows; permutation [2, 0, 1] moves whole segments
    x = torch.arange(6)
    out = _permute_flat_segments(x, [2, 1, 3], torch.tensor([2, 0, 1]))
    assert out.tolist() == [3, 4, 5, 0, 1, 2]

    # roundtrip: permuting back with the inverse restores the original
    inv = torch.argsort(torch.tensor([2, 0, 1]))
    back = _permute_flat_segments(out, [3, 2, 1], inv)
    assert torch.equal(back, x)


def test_unsort_tensor_roundtrip():
    x = torch.arange(8).reshape(4, 2)
    assert torch.equal(
        RBLNDecoderOnlyGenerationMixin._unsort_generation_outputs(x.index_select(0, SORT_IDX), UNSORT_IDX), x
    )

    # num_return_sequences=2: rows grouped per sample via repeat_interleave
    x2 = x.repeat_interleave(2, dim=0)
    sorted_x2 = x.index_select(0, SORT_IDX).repeat_interleave(2, dim=0)
    assert torch.equal(RBLNDecoderOnlyGenerationMixin._unsort_generation_outputs(sorted_x2, UNSORT_IDX), x2)


def test_unsort_model_output_fields():
    x = torch.arange(8).reshape(4, 2)
    sorted_x = x.index_select(0, SORT_IDX)

    class Output:
        sequences = sorted_x
        scores = (sorted_x.float(), sorted_x.float() * 2)
        logits = None
        attentions = ((sorted_x.float().unsqueeze(1),),)
        hidden_states = None

    out = RBLNDecoderOnlyGenerationMixin._unsort_generation_outputs(Output(), UNSORT_IDX)
    assert torch.equal(out.sequences, x)
    assert torch.equal(out.scores[1], x.float() * 2)
    assert torch.equal(out.attentions[0][0], x.float().unsqueeze(1))
    assert out.logits is None


class _FakeCausalLM(RBLNDecoderOnlyGenerationMixin):
    def __init__(self, use_batch_attn_opt):
        self.rbln_config = type("Cfg", (), {"use_batch_attn_opt": use_batch_attn_opt})()


def test_generate_fast_path_sorts_and_unsorts(monkeypatch):
    captured = {}

    def fake_generate(self, input_ids, **kwargs):
        captured["input_ids"] = input_ids
        captured["kwargs"] = kwargs
        return input_ids

    monkeypatch.setattr(GenerationMixin, "generate", fake_generate)

    ids = torch.tensor([[9, 1, 2, 3], [9, 9, 9, 5], [9, 9, 6, 7]])
    mask = torch.tensor([[0, 1, 1, 1], [0, 0, 0, 1], [0, 0, 1, 1]])
    result = _FakeCausalLM(use_batch_attn_opt=True).generate(ids, attention_mask=mask)

    expected_sort = torch.tensor([0, 2, 1])  # lengths 3, 1, 2 -> descending
    assert captured["kwargs"]["inputs_sorted"] is True
    assert torch.equal(captured["input_ids"], ids.index_select(0, expected_sort))
    assert torch.equal(captured["kwargs"]["attention_mask"], mask.index_select(0, expected_sort))
    assert torch.equal(result, ids)


def test_generate_fast_path_disabled(monkeypatch):
    captured = {}

    def fake_generate(self, input_ids, **kwargs):
        captured["kwargs"] = kwargs
        return input_ids

    monkeypatch.setattr(GenerationMixin, "generate", fake_generate)

    ids = torch.tensor([[1, 2], [3, 4]])
    result = _FakeCausalLM(use_batch_attn_opt=False).generate(ids)
    assert "inputs_sorted" not in captured["kwargs"]
    assert torch.equal(result, ids)


def test_generate_config_without_field(monkeypatch):
    # Multimodal top-level configs are plain RBLNModelConfig without use_batch_attn_opt
    captured = {}

    def fake_generate(self, input_ids, **kwargs):
        captured["kwargs"] = kwargs
        return input_ids

    monkeypatch.setattr(GenerationMixin, "generate", fake_generate)

    model = _FakeCausalLM(use_batch_attn_opt=False)
    model.rbln_config = type("Cfg", (), {})()
    ids = torch.tensor([[1, 2], [3, 4]])
    result = model.generate(ids)
    assert "inputs_sorted" not in captured["kwargs"]
    assert torch.equal(result, ids)


class _FakeForwardModel:
    rbln_config = type("Cfg", (), {"use_batch_attn_opt": True})()
    _sort = RBLNDecoderOnlyModelForCausalLM._maybe_sort_inputs_for_batch_attn_opt


def _prefill_inputs():
    mask = torch.tensor([[0, 1, 1, 1], [0, 0, 0, 1], [0, 0, 1, 1]])
    return {
        "input_ids": torch.arange(12).reshape(3, 4),
        "inputs_embeds": None,
        "cache_position": None,
        "attention_mask": mask,
        "generate_idx": mask.sum(-1, keepdim=True).int(),
        "padded_cache_lengths": torch.zeros(3, 1, dtype=torch.int32),
        "position_ids": None,
        "token_type_ids": None,
        "lora_int_ids": None,
    }


def test_forward_sort_prefill_then_decode():
    model = _FakeForwardModel()
    assert model._sort(_prefill_inputs()) is not None
    assert model._rbln_sort_idx.tolist() == [0, 2, 1]

    decode = dict(_prefill_inputs(), cache_position=torch.ones(3, 1, dtype=torch.int32))
    mask_before = decode["attention_mask"]
    assert model._sort(decode) is not None
    # attention_mask is unused at decode and must not be copied per step
    assert decode["attention_mask"] is mask_before


def test_forward_sort_decode_hardening():
    model = _FakeForwardModel()
    decode = dict(_prefill_inputs(), cache_position=torch.ones(3, 1, dtype=torch.int32))

    model._rbln_sort_idx = None
    model._rbln_unsort_idx = None
    with pytest.raises(RuntimeError, match="established at prefill"):
        model._sort(dict(decode))

    model._sort(_prefill_inputs())
    bad = {k: (v[:2] if isinstance(v, torch.Tensor) else v) for k, v in decode.items()}
    with pytest.raises(RuntimeError, match="does not match"):
        model._sort(bad)


def test_forward_sort_inputs_sorted_skips():
    model = _FakeForwardModel()
    model._sort(_prefill_inputs())
    assert model._sort(_prefill_inputs(), inputs_sorted=True) is None
    assert model._rbln_sort_idx is None


def test_use_batch_attn_opt_serialized():
    cfg = RBLNDecoderOnlyModelForCausalLMConfig(max_seq_len=1024, _use_batch_attn_opt=True, npu="RBLN-CR31")
    assert cfg.use_batch_attn_opt is True
    serialized = cfg._prepare_for_serialization()
    assert serialized.get("_use_batch_attn_opt") is True
    assert "npu" not in serialized
    # round-trip: the serialized form reconstructs through __init__
    assert RBLNDecoderOnlyModelForCausalLMConfig(max_seq_len=1024, _use_batch_attn_opt=True).use_batch_attn_opt is True


def test_use_batch_attn_opt_not_user_settable():
    with pytest.raises(ValueError, match="[Uu]nexpected"):
        RBLNDecoderOnlyModelForCausalLMConfig(max_seq_len=1024, use_batch_attn_opt=True)

    cfg = RBLNDecoderOnlyModelForCausalLMConfig(max_seq_len=1024)
    assert cfg.use_batch_attn_opt is None
    with pytest.raises(AttributeError):
        cfg.use_batch_attn_opt = True


def test_npu_is_cr13_or_later():
    assert npu_is_cr13_or_later("RBLN-CR13") is True
    assert npu_is_cr13_or_later("RBLN-CR31") is True
    assert npu_is_cr13_or_later("RBLN-CR03") is False
    assert npu_is_cr13_or_later("RBLN-CA25") is False
