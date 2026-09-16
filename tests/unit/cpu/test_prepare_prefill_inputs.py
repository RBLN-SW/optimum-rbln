from types import SimpleNamespace

import pytest
import torch

from optimum.rbln.transformers.models.decoderonly.decoderonly_runtime_utils import RBLNRuntimeModel
from optimum.rbln.transformers.models.gemma3.gemma3_runtime_utils import RBLNGemma3RuntimeModel
from optimum.rbln.transformers.models.gemma4.gemma4_runtime_utils import RBLNGemma4RuntimeModel
from optimum.rbln.transformers.models.qwen3_vl.qwen3_vl_runtime_utils import RBLNQwen3VLRuntimeModel
from optimum.rbln.transformers.models.qwen3_vl_moe.qwen3_vl_moe_runtime_utils import RBLNQwen3VLMoeRuntimeModel


CHUNK = 8
MAX_SEQ = 64
HIDDEN = 4
HEAD_DIM = 2

# every runtime class that reaches `_prepare_prefill_inputs`: the base text-only one, the two
# chunked-multimodal mixin users, and the qwen3_vl pair that carries the deepstack tensors
ALL_RUNTIMES = [
    RBLNRuntimeModel,
    RBLNGemma3RuntimeModel,
    RBLNGemma4RuntimeModel,
    RBLNQwen3VLRuntimeModel,
    RBLNQwen3VLMoeRuntimeModel,
]
MIXIN_RUNTIMES = [RBLNGemma3RuntimeModel, RBLNGemma4RuntimeModel]
QWEN3VL_RUNTIMES = [RBLNQwen3VLRuntimeModel, RBLNQwen3VLMoeRuntimeModel]

FIELDS = (
    "inputs",
    "cache_position",
    "chunked_attention_mask",
    "position_ids",
    "position_embed",
    "padded_cache_lengths",
    "query_length",
    "token_type_ids",
    "visual_pos_mask",
    "deepstack_embeds",
)


def _runtime(cls, **overrides):
    # `_prepare_prefill_inputs` only reads `self.rbln_config`, so skip __init__ (it would need a
    # compiled runtime) and hand the method a config stub.
    model = object.__new__(cls)
    config = {
        "prefill_chunk_size": CHUNK,
        "max_seq_len": MAX_SEQ,
        "dtype": torch.float32,
        "use_attention_mask": True,
        "use_position_ids": True,
        "use_inputs_embeds": True,
        "can_generate": True,
        "use_image_prefill": False,
        "image_prefill_chunk_size": 2 * CHUNK,
    }
    config.update(overrides)
    model.rbln_config = SimpleNamespace(**config)
    return model


def _prepare(model, inputs, **kwargs):
    # the base classes return 8 fields, the qwen3_vl pair appends the two deepstack ones
    return dict(zip(FIELDS, model._prepare_prefill_inputs(inputs, **kwargs), strict=False))


def _embeds(seq_len):
    return torch.arange(seq_len * HIDDEN, dtype=torch.float32).reshape(1, seq_len, HIDDEN)


def _position_embed(seq_len):
    return torch.arange(2 * seq_len * HEAD_DIM, dtype=torch.float32).reshape(2, 1, 1, seq_len, HEAD_DIM)


def _left_pad_mask(seq_len, pad):
    return torch.cat([torch.zeros(pad, dtype=torch.int64), torch.ones(seq_len - pad, dtype=torch.int64)])


def _right_pad_mask(seq_len, keep):
    return torch.cat([torch.ones(keep, dtype=torch.int64), torch.zeros(seq_len - keep, dtype=torch.int64)])


@pytest.mark.parametrize("cls", ALL_RUNTIMES)
def test_left_padding_is_stripped_from_every_input(cls):
    seq_len, pad = 20, 5
    query_length = seq_len - pad
    inputs, position_embed = _embeds(seq_len), _position_embed(seq_len)
    token_type_ids = torch.zeros(1, seq_len, dtype=torch.int64)
    token_type_ids[0, pad + 2 : pad + 6] = 1

    out = _prepare(
        _runtime(cls),
        inputs,
        attention_mask=_left_pad_mask(seq_len, pad),
        position_embed=position_embed,
        token_type_ids=token_type_ids,
    )

    assert out["query_length"] == query_length
    assert out["inputs"].shape[1] == CHUNK * 2
    assert torch.equal(out["inputs"][:, :query_length], inputs[:, pad:])
    assert torch.equal(out["position_embed"][:, :, :, :query_length, :], position_embed[:, :, :, pad:, :])
    assert torch.equal(out["token_type_ids"][:, :query_length], token_type_ids[:, pad:])
    assert torch.equal(out["cache_position"][:, :query_length], torch.arange(query_length).unsqueeze(0))
    assert out["padded_cache_lengths"] == 0


@pytest.mark.parametrize("cls", ALL_RUNTIMES)
def test_right_padding_is_stripped_when_not_generating(cls):
    seq_len, keep = 20, 13
    inputs = _embeds(seq_len)

    out = _prepare(
        _runtime(cls, can_generate=False),
        inputs,
        attention_mask=_right_pad_mask(seq_len, keep),
    )

    assert out["query_length"] == keep
    assert torch.equal(out["inputs"][:, :keep], inputs[:, :keep])


@pytest.mark.parametrize("cls", ALL_RUNTIMES)
def test_unmasked_inputs_are_padded_to_a_chunk_multiple(cls):
    seq_len = 13
    inputs = _embeds(seq_len)

    out = _prepare(_runtime(cls), inputs)

    assert out["query_length"] == seq_len
    assert out["inputs"].shape[1] == CHUNK * 2
    assert torch.equal(out["inputs"][:, :seq_len], inputs)
    assert (out["inputs"][:, seq_len:] == 0).all()
    # the pad region of cache_position is zero-filled, not a continued arange
    assert (out["cache_position"][:, seq_len:] == 0).all()


@pytest.mark.parametrize("cls", ALL_RUNTIMES)
def test_rejects_masks_the_chunked_prefill_cannot_honor(cls):
    seq_len = 16
    inputs = _embeds(seq_len)
    model = _runtime(cls)

    with pytest.raises(ValueError, match="1D tensor"):
        _prepare(model, inputs, attention_mask=torch.ones(1, seq_len, dtype=torch.int64))

    interleaved = torch.tensor([1, 0] * (seq_len // 2), dtype=torch.int64)
    with pytest.raises(ValueError, match="group all 1s together"):
        _prepare(model, inputs, attention_mask=interleaved)

    with pytest.raises(ValueError, match="at least one real token"):
        _prepare(model, inputs, attention_mask=torch.zeros(seq_len, dtype=torch.int64))

    with pytest.raises(ValueError, match="left padded for generation"):
        _prepare(model, inputs, attention_mask=_right_pad_mask(seq_len, 10))


@pytest.mark.parametrize("cls", ALL_RUNTIMES)
def test_rejects_inputs_longer_than_max_seq_len(cls):
    with pytest.raises(ValueError, match="exceeds the maximum allowed sequence length"):
        _prepare(_runtime(cls), _embeds(MAX_SEQ + 1))


def test_base_attention_mask_buffer_shapes():
    inputs = _embeds(13)

    with_position_ids = _prepare(_runtime(RBLNRuntimeModel), inputs)["chunked_attention_mask"]
    assert with_position_ids.shape == (1, MAX_SEQ)

    without_position_ids = _prepare(_runtime(RBLNRuntimeModel, use_position_ids=False), inputs)[
        "chunked_attention_mask"
    ]
    assert without_position_ids.shape == (1, 1, CHUNK, MAX_SEQ)

    assert _prepare(_runtime(RBLNRuntimeModel, use_attention_mask=False), inputs)["chunked_attention_mask"] is None


def test_base_position_ids_default_to_cache_position():
    out = _prepare(_runtime(RBLNRuntimeModel), _embeds(13))
    assert torch.equal(out["position_ids"], out["cache_position"])


def test_base_pads_input_ids_on_the_token_axis():
    seq_len = 13
    input_ids = torch.arange(seq_len, dtype=torch.int64).unsqueeze(0)

    out = _prepare(_runtime(RBLNRuntimeModel, use_inputs_embeds=False), input_ids)

    assert out["inputs"].shape == (1, CHUNK * 2)
    assert torch.equal(out["inputs"][:, :seq_len], input_ids)


@pytest.mark.parametrize("cls", MIXIN_RUNTIMES)
def test_mixin_flattens_the_attention_mask_buffer(cls):
    out = _prepare(_runtime(cls, use_position_ids=False), _embeds(13))
    assert out["chunked_attention_mask"].shape == (1, MAX_SEQ)


@pytest.mark.parametrize("cls", MIXIN_RUNTIMES)
def test_mixin_pads_caller_supplied_position_ids_to_cache_position(cls):
    seq_len = 13
    position_ids = torch.arange(seq_len, dtype=torch.int32).unsqueeze(0)

    out = _prepare(_runtime(cls), _embeds(seq_len), position_ids=position_ids)

    assert out["position_ids"].shape == out["cache_position"].shape
    assert torch.equal(out["position_ids"][:, :seq_len], position_ids)


@pytest.mark.parametrize("cls", MIXIN_RUNTIMES)
@pytest.mark.parametrize("bucket, expected_extra", [(2 * CHUNK, 2 * CHUNK), ([CHUNK, 4 * CHUNK], 4 * CHUNK)])
def test_image_prefill_reserves_the_largest_chunk_bucket(cls, bucket, expected_extra):
    seq_len = 13
    aligned = CHUNK * 2
    token_type_ids = torch.zeros(1, seq_len, dtype=torch.int64)

    out = _prepare(
        _runtime(cls, use_image_prefill=True, image_prefill_chunk_size=bucket),
        _embeds(seq_len),
        token_type_ids=token_type_ids,
    )

    assert out["inputs"].shape[1] == aligned + expected_extra
    assert out["cache_position"].shape[1] == aligned + expected_extra
    assert out["position_ids"].shape[1] == aligned + expected_extra
    assert (out["token_type_ids"][:, aligned:] == -1).all()
    assert out["query_length"] == seq_len


@pytest.mark.parametrize("cls", MIXIN_RUNTIMES)
def test_no_image_prefill_keeps_the_chunk_aligned_width(cls):
    out = _prepare(_runtime(cls, use_image_prefill=False), _embeds(13))
    assert out["inputs"].shape[1] == CHUNK * 2


def _deepstack(seq_len, span, num_layers=3):
    visual_pos_mask = torch.zeros(1, seq_len, dtype=torch.bool)
    visual_pos_mask[0, span[0] : span[1]] = True
    deepstack_embeds = torch.zeros(num_layers, seq_len, HIDDEN, dtype=torch.float32)
    deepstack_embeds[:, span[0] : span[1]] = torch.arange(
        num_layers * (span[1] - span[0]) * HIDDEN, dtype=torch.float32
    ).reshape(num_layers, span[1] - span[0], HIDDEN)
    return visual_pos_mask, deepstack_embeds


@pytest.mark.parametrize("cls", QWEN3VL_RUNTIMES)
def test_deepstack_follows_the_left_padding_that_inputs_lose(cls):
    # `visual_pos_mask` / `deepstack_embeds` are built on the padded batch coordinates, so a
    # sequence that is not the longest in its batch must have them compacted with the same mask
    # as `inputs` - otherwise the visual features land `left_pad` tokens off.
    seq_len, pad = 20, 5
    query_length = seq_len - pad
    visual_pos_mask, deepstack_embeds = _deepstack(seq_len, (pad + 3, pad + 8))

    out = _prepare(
        _runtime(cls),
        _embeds(seq_len),
        attention_mask=_left_pad_mask(seq_len, pad),
        visual_pos_mask=visual_pos_mask,
        deepstack_embeds=deepstack_embeds,
    )

    assert out["visual_pos_mask"].shape == (1, CHUNK * 2)
    assert out["deepstack_embeds"].shape == (3, CHUNK * 2, HIDDEN)
    assert torch.nonzero(out["visual_pos_mask"][0]).flatten().tolist() == [3, 4, 5, 6, 7]
    assert torch.equal(out["visual_pos_mask"][:, :query_length], visual_pos_mask[:, pad:])
    assert torch.equal(out["deepstack_embeds"][:, :query_length], deepstack_embeds[:, pad:])
    assert not out["visual_pos_mask"][:, query_length:].any()
    assert (out["deepstack_embeds"][:, query_length:] == 0).all()


@pytest.mark.parametrize("cls", QWEN3VL_RUNTIMES)
def test_deepstack_follows_right_padding_on_the_embedding_path(cls):
    seq_len, keep = 20, 13
    visual_pos_mask, deepstack_embeds = _deepstack(seq_len, (3, 8))

    out = _prepare(
        _runtime(cls, can_generate=False),
        _embeds(seq_len),
        attention_mask=_right_pad_mask(seq_len, keep),
        visual_pos_mask=visual_pos_mask,
        deepstack_embeds=deepstack_embeds,
    )

    assert out["visual_pos_mask"].shape == (1, CHUNK * 2)
    assert torch.equal(out["visual_pos_mask"][:, :keep], visual_pos_mask[:, :keep])
    assert torch.equal(out["deepstack_embeds"][:, :keep], deepstack_embeds[:, :keep])


@pytest.mark.parametrize("cls", QWEN3VL_RUNTIMES)
def test_deepstack_without_padding_is_only_chunk_aligned(cls):
    seq_len = 13
    visual_pos_mask, deepstack_embeds = _deepstack(seq_len, (3, 8))

    out = _prepare(
        _runtime(cls),
        _embeds(seq_len),
        visual_pos_mask=visual_pos_mask,
        deepstack_embeds=deepstack_embeds,
    )

    assert torch.equal(out["visual_pos_mask"][:, :seq_len], visual_pos_mask)
    assert torch.equal(out["deepstack_embeds"][:, :seq_len], deepstack_embeds)


@pytest.mark.parametrize("cls", QWEN3VL_RUNTIMES)
def test_text_only_prefill_passes_deepstack_through_as_none(cls):
    out = _prepare(_runtime(cls), _embeds(13), attention_mask=_left_pad_mask(13, 4))
    assert out["visual_pos_mask"] is None
    assert out["deepstack_embeds"] is None
