from types import SimpleNamespace

import pytest
import torch

from optimum.rbln.transformers.models.decoderonly.generation_decoderonly import RBLNDecoderOnlyGenerationMixin
from optimum.rbln.transformers.models.exaone4_5.modeling_exaone4_5 import RBLNExaone4_5_ForConditionalGeneration
from optimum.rbln.transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import RBLNQwen2_5_VLForConditionalGeneration
from optimum.rbln.transformers.models.qwen2_vl.modeling_qwen2_vl import RBLNQwen2VLForConditionalGeneration
from optimum.rbln.transformers.models.qwen3_vl.modeling_qwen3_vl import RBLNQwen3VLForConditionalGeneration
from optimum.rbln.transformers.utils.generation_multimodal import RBLNVisionBatchSortMixin, _permute_flat_segments


VS, IMG, VID = 90, 91, 92


def test_permute_flat_segments():
    # samples own [2, 1, 3] rows; permutation [2, 0, 1] moves whole segments
    x = torch.arange(6)
    out = _permute_flat_segments(x, [2, 1, 3], torch.tensor([2, 0, 1]))
    assert out.tolist() == [3, 4, 5, 0, 1, 2]

    # roundtrip: permuting back with the inverse restores the original
    inv = torch.argsort(torch.tensor([2, 0, 1]))
    back = _permute_flat_segments(out, [3, 2, 1], inv)
    assert torch.equal(back, x)


def _make_model(cls, requires_batch_sort=True, **config_attrs):
    model = object.__new__(cls)
    model.rbln_config = SimpleNamespace(requires_batch_sort=requires_batch_sort)
    model.config = SimpleNamespace(vision_start_token_id=VS, image_token_id=IMG, video_token_id=VID, **config_attrs)
    return model


def test_qwen2_vl_sort_vision_inputs():
    model = _make_model(RBLNQwen2VLForConditionalGeneration)
    # lengths [4, 6, 5]; images per sample [2, 0, 1]; one video on sample 2
    ids = torch.tensor(
        [
            [0, 0, VS, IMG, VS, IMG],
            [1, 2, 3, 4, 5, 6],
            [0, VS, IMG, VS, VID, 8],
        ]
    )
    mask = torch.tensor([[0, 0, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1], [0, 1, 1, 1, 1, 1]])
    grids = torch.tensor([[1, 2, 2], [1, 2, 4], [1, 2, 6]])  # 4 + 8 patches (s0), 12 patches (s2)
    pixels = torch.arange(24 * 3).reshape(24, 3)
    video_grids = torch.tensor([[2, 2, 2]])
    video_pixels = torch.arange(8 * 3).reshape(8, 3)

    kwargs = {
        "attention_mask": mask,
        "pixel_values": pixels,
        "image_grid_thw": grids,
        "pixel_values_videos": video_pixels,
        "video_grid_thw": video_grids,
    }
    sorted_ids, unsort_idx = model._sort_generation_inputs(ids, kwargs)

    sort_idx = torch.tensor([1, 2, 0])  # lengths 6, 5, 4 descending
    assert torch.equal(sorted_ids, ids.index_select(0, sort_idx))
    assert torch.equal(kwargs["attention_mask"], mask.index_select(0, sort_idx))
    assert kwargs["inputs_sorted"] is True
    # image order becomes: (none from s1), s2's, then s0's two
    assert torch.equal(kwargs["image_grid_thw"], grids[[2, 0, 1]])
    assert torch.equal(kwargs["pixel_values"], torch.cat([pixels[12:24], pixels[0:12]]))
    # single video: segments unchanged
    assert torch.equal(kwargs["video_grid_thw"], video_grids)
    assert torch.equal(kwargs["pixel_values_videos"], video_pixels)
    # roundtrip restores original order
    assert torch.equal(RBLNDecoderOnlyGenerationMixin._unsort_generation_outputs(sorted_ids, unsort_idx), ids)


def test_qwen2_5_vl_videos_and_second_per_grid_ts():
    model = _make_model(RBLNQwen2_5_VLForConditionalGeneration)
    # lengths [4, 6, 5]; videos per sample [1, 0, 2]
    ids = torch.tensor(
        [
            [0, 0, VS, VID, 7, 7],
            [1, 2, 3, 4, 5, 6],
            [0, VS, VID, VS, VID, 9],
        ]
    )
    mask = torch.tensor([[0, 0, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1], [0, 1, 1, 1, 1, 1]])
    video_grids = torch.tensor([[1, 2, 2], [1, 2, 4], [2, 2, 2]])  # 4 (s0), 8 + 8 (s2)
    video_pixels = torch.arange(20 * 2).reshape(20, 2)
    ts = torch.tensor([0.1, 0.2, 0.3])

    kwargs = {
        "attention_mask": mask,
        "pixel_values_videos": video_pixels,
        "video_grid_thw": video_grids,
        "second_per_grid_ts": ts,
    }
    sorted_ids, unsort_idx = model._sort_generation_inputs(ids, kwargs)

    sort_idx = torch.tensor([1, 2, 0])
    assert torch.equal(sorted_ids, ids.index_select(0, sort_idx))
    assert torch.equal(kwargs["video_grid_thw"], video_grids[[1, 2, 0]])
    assert torch.equal(kwargs["pixel_values_videos"], torch.cat([video_pixels[4:20], video_pixels[0:4]]))
    assert torch.equal(kwargs["second_per_grid_ts"], torch.tensor([0.2, 0.3, 0.1]))
    assert torch.equal(RBLNDecoderOnlyGenerationMixin._unsort_generation_outputs(sorted_ids, unsort_idx), ids)


def test_qwen2_5_vl_second_per_grid_ts_list():
    model = _make_model(RBLNQwen2_5_VLForConditionalGeneration)
    ids = torch.tensor([[0, 0, VS, VID, 7, 7], [1, 2, 3, 4, 5, 6], [0, VS, VID, VS, VID, 9]])
    mask = torch.tensor([[0, 0, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1], [0, 1, 1, 1, 1, 1]])
    kwargs = {
        "attention_mask": mask,
        "video_grid_thw": torch.tensor([[1, 2, 2], [1, 2, 4], [2, 2, 2]]),
        "second_per_grid_ts": [0.1, 0.2, 0.3],
    }
    model._sort_generation_inputs(ids, kwargs)
    assert kwargs["second_per_grid_ts"] == [0.2, 0.3, 0.1]


def test_qwen3_vl_chunked_video_grid_rows():
    model = _make_model(RBLNQwen3VLForConditionalGeneration, vision_config=SimpleNamespace(spatial_merge_size=2))
    # sample 0: one video split into 2 temporal chunks (2 markers, one grid row with t=2)
    # sample 1: one video with 1 chunk
    ids = torch.tensor([[0, VS, VID, VS, VID, 7], [1, 2, 3, VS, VID, 6]])
    mask = torch.tensor([[0, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1]])
    video_grids = torch.tensor([[2, 2, 2], [1, 2, 2]])  # 8 patches (s0), 4 patches (s1)
    video_pixels = torch.arange(12 * 2).reshape(12, 2)

    kwargs = {"attention_mask": mask, "pixel_values_videos": video_pixels, "video_grid_thw": video_grids}
    sorted_ids, _ = model._sort_generation_inputs(ids, kwargs)

    assert torch.equal(sorted_ids, ids.index_select(0, torch.tensor([1, 0])))
    assert torch.equal(kwargs["video_grid_thw"], video_grids[[1, 0]])
    assert torch.equal(kwargs["pixel_values_videos"], torch.cat([video_pixels[8:12], video_pixels[0:8]]))


def test_qwen3_vl_precomputed_embeds():
    model = _make_model(RBLNQwen3VLForConditionalGeneration, vision_config=SimpleNamespace(spatial_merge_size=2))
    # images per sample [1, 1]; merged tokens = patches // 4: 2 (s0), 4 (s1)
    ids = torch.tensor([[0, VS, IMG, 5, 5, 5], [VS, IMG, 3, 4, 5, 6]])
    mask = torch.tensor([[0, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1]])
    grids = torch.tensor([[1, 2, 4], [1, 4, 4]])
    embeds = torch.arange(6 * 2).reshape(6, 2).float()
    deepstack = [embeds * 10, embeds * 100]

    kwargs = {
        "attention_mask": mask,
        "image_grid_thw": grids,
        "image_embeds": embeds,
        "deepstack_image_embeds": deepstack,
    }
    model._sort_generation_inputs(ids, kwargs)

    assert torch.equal(kwargs["image_grid_thw"], grids[[1, 0]])
    assert torch.equal(kwargs["image_embeds"], torch.cat([embeds[2:6], embeds[0:2]]))
    for scale, layer in zip((10, 100), kwargs["deepstack_image_embeds"], strict=True):
        assert torch.equal(layer, torch.cat([embeds[2:6], embeds[0:2]]) * scale)


def test_qwen3_vl_moe_inherits_mixin():
    from optimum.rbln.transformers.models.qwen3_vl_moe.modeling_qwen3_vl_moe import (
        RBLNQwen3VLMoeForConditionalGeneration,
    )

    assert issubclass(RBLNQwen3VLMoeForConditionalGeneration, RBLNVisionBatchSortMixin)
    assert RBLNQwen3VLMoeForConditionalGeneration._video_grid_rows_are_chunks is True


def test_qwen3_5_inherits_mixin():
    from optimum.rbln.transformers.models.qwen3_5.modeling_qwen3_5 import RBLNQwen3_5ForConditionalGeneration

    assert issubclass(RBLNQwen3_5ForConditionalGeneration, RBLNVisionBatchSortMixin)
    assert RBLNQwen3_5ForConditionalGeneration._video_grid_rows_are_chunks is True


def test_exaone4_5_token_count_rows():
    model = _make_model(RBLNExaone4_5_ForConditionalGeneration, vision_config=SimpleNamespace(spatial_merge_size=2))
    # no vision_start marker; merged tokens per grid: 1, 2 (sample 0) and 4 (sample 2)
    ids = torch.tensor(
        [
            [0, 0, 0, IMG, IMG, IMG, 5],
            [1, 2, 3, 4, 5, 6, 7],
            [0, IMG, IMG, IMG, IMG, 9, 9],
        ]
    )
    mask = torch.tensor([[0, 0, 0, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1], [0, 1, 1, 1, 1, 1, 1]])
    grids = torch.tensor([[1, 2, 2], [1, 2, 4], [1, 4, 4]])  # 4 + 8 patches (s0), 16 patches (s2)
    pixels = torch.arange(28 * 3).reshape(28, 3)

    kwargs = {"attention_mask": mask, "pixel_values": pixels, "image_grid_thw": grids}
    sorted_ids, unsort_idx = model._sort_generation_inputs(ids, kwargs)

    sort_idx = torch.tensor([1, 2, 0])  # lengths 7, 6, 4 descending
    assert torch.equal(sorted_ids, ids.index_select(0, sort_idx))
    assert kwargs["inputs_sorted"] is True
    assert torch.equal(kwargs["image_grid_thw"], grids[[2, 0, 1]])
    assert torch.equal(kwargs["pixel_values"], torch.cat([pixels[12:28], pixels[0:12]]))
    assert torch.equal(RBLNDecoderOnlyGenerationMixin._unsort_generation_outputs(sorted_ids, unsort_idx), ids)


def test_exaone4_5_misaligned_grids_raise():
    model = _make_model(RBLNExaone4_5_ForConditionalGeneration, vision_config=SimpleNamespace(spatial_merge_size=2))
    # sample 0 has 2 image tokens but the first grid yields 4 merged tokens
    ids = torch.tensor([[IMG, IMG, 3, 4], [1, 2, 3, 4]])
    kwargs = {"pixel_values": torch.zeros(16, 3), "image_grid_thw": torch.tensor([[1, 4, 4]])}
    with pytest.raises(RuntimeError, match="Cannot map image inputs"):
        model._sort_generation_inputs(ids, kwargs)


def test_gate_off_noop():
    model = _make_model(RBLNQwen2VLForConditionalGeneration, requires_batch_sort=False)
    ids = torch.tensor([[0, VS, IMG, 3], [1, 2, 3, 4]])
    grids = torch.tensor([[1, 2, 2]])
    pixels = torch.zeros(4, 3)
    kwargs = {"pixel_values": pixels, "image_grid_thw": grids}
    out_ids, unsort_idx = model._sort_generation_inputs(ids, kwargs)
    assert out_ids is ids
    assert unsort_idx is None
    assert kwargs["pixel_values"] is pixels
    assert kwargs["image_grid_thw"] is grids
    assert "inputs_sorted" not in kwargs


def test_require_sorted_batch_inputs_guard():
    model = _make_model(RBLNQwen2VLForConditionalGeneration)
    batch = torch.zeros(2, 4, dtype=torch.long)
    with pytest.raises(RuntimeError, match="sorted by sequence length"):
        model._require_sorted_batch_inputs(batch, False)
    model._require_sorted_batch_inputs(batch, True)
    model._require_sorted_batch_inputs(batch[:1], False)
    model._require_sorted_batch_inputs(None, False)

    off = _make_model(RBLNQwen2VLForConditionalGeneration, requires_batch_sort=False)
    off._require_sorted_batch_inputs(batch, False)


def test_sort_requires_input_ids_for_vision():
    model = _make_model(RBLNQwen2VLForConditionalGeneration)
    kwargs = {
        "inputs_embeds": torch.zeros(2, 4, 8),
        "pixel_values": torch.zeros(4, 3),
        "image_grid_thw": torch.tensor([[1, 2, 2]]),
    }
    with pytest.raises(RuntimeError, match="requires input_ids"):
        model._sort_generation_inputs(None, kwargs)
