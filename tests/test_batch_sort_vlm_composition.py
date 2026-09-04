import pytest
import torch

from optimum.rbln.transformers.models.gemma3.modeling_gemma3 import RBLNGemma3ForConditionalGeneration
from optimum.rbln.transformers.models.gemma4.modeling_gemma4 import RBLNGemma4ForConditionalGeneration
from optimum.rbln.transformers.models.idefics3.modeling_idefics3 import RBLNIdefics3ForConditionalGeneration
from optimum.rbln.transformers.models.llava.modeling_llava import RBLNLlavaForConditionalGeneration
from optimum.rbln.transformers.models.llava_next.modeling_llava_next import RBLNLlavaNextForConditionalGeneration
from optimum.rbln.transformers.models.paligemma.modeling_paligemma import RBLNPaliGemmaForConditionalGeneration


IMG = 99
VID = 98


def _fake_model(cls, config_fields, lm_attr="language_model", use_batch_attn_opt=True):
    model = cls.__new__(cls)
    lm_config = type("Cfg", (), {"use_batch_attn_opt": use_batch_attn_opt})()
    setattr(model, lm_attr, type("LM", (), {"rbln_config": lm_config})())
    model.config = type("Config", (), config_fields)()
    return model


def _images_by_sample(image_tensor, images_per_sample, order):
    # expected image-first layout after permuting samples by `order`
    segments = torch.split(image_tensor, images_per_sample, dim=0)
    return torch.cat([segments[i] for i in order], dim=0)


# lengths [4, 6, 5] -> descending sample order [1, 2, 0]
MASK = torch.tensor([[1, 1, 1, 1, 0, 0], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 0]])
SORT = torch.tensor([1, 2, 0])


def test_gemma3_sorts_flat_pixel_values():
    model = _fake_model(RBLNGemma3ForConditionalGeneration, {"image_token_index": IMG, "mm_tokens_per_image": 2})
    # images per sample: [1, 2, 0], each image = one run of 2 placeholder tokens
    input_ids = torch.tensor([[1, IMG, IMG, 2, 0, 0], [IMG, IMG, 3, IMG, IMG, 4], [5, 6, 7, 8, 9, 0]])
    pixel_values = torch.arange(3, dtype=torch.float).view(3, 1, 1, 1)
    kwargs = {"attention_mask": MASK.clone(), "pixel_values": pixel_values}

    sorted_ids, unsort_idx = model._sort_generation_inputs(input_ids, kwargs)

    assert torch.equal(sorted_ids, input_ids.index_select(0, SORT))
    assert torch.equal(kwargs["attention_mask"], MASK.index_select(0, SORT))
    assert torch.equal(kwargs["pixel_values"], _images_by_sample(pixel_values, [1, 2, 0], SORT.tolist()))
    assert kwargs["inputs_sorted"] is True
    assert torch.equal(model._unsort_generation_outputs(sorted_ids, unsort_idx), input_ids)


def test_gemma3_gate_off_is_noop():
    model = _fake_model(
        RBLNGemma3ForConditionalGeneration,
        {"image_token_index": IMG, "mm_tokens_per_image": 2},
        use_batch_attn_opt=None,
    )
    input_ids = torch.tensor([[1, 2], [3, 4]])
    kwargs = {"attention_mask": torch.tensor([[1, 1], [1, 0]]), "pixel_values": torch.zeros(2, 1, 1, 1)}
    sorted_ids, unsort_idx = model._sort_generation_inputs(input_ids, kwargs)
    assert sorted_ids is input_ids
    assert unsort_idx is None
    assert "inputs_sorted" not in kwargs


def test_gemma3_batch_one_is_noop():
    model = _fake_model(RBLNGemma3ForConditionalGeneration, {"image_token_index": IMG, "mm_tokens_per_image": 2})
    input_ids = torch.tensor([[1, IMG, IMG]])
    kwargs = {"pixel_values": torch.zeros(1, 1, 1, 1)}
    sorted_ids, unsort_idx = model._sort_generation_inputs(input_ids, kwargs)
    assert sorted_ids is input_ids
    assert unsort_idx is None
    assert "inputs_sorted" not in kwargs


def test_gemma4_sorts_images_and_videos():
    model = _fake_model(RBLNGemma4ForConditionalGeneration, {"image_token_id": IMG, "video_token_id": VID})
    # images per sample [1, 0, 1]; one video (2 frames = 2 runs) on sample 1
    input_ids = torch.tensor([[1, IMG, IMG, 2, 0, 0], [VID, 3, VID, 4, 5, 6], [7, IMG, IMG, IMG, 8, 0]])
    pixel_values = torch.arange(2, dtype=torch.float).view(2, 1, 1)
    image_position_ids = torch.arange(2).view(2, 1)
    pixel_values_videos = torch.arange(2, dtype=torch.float).view(1, 2, 1)
    video_position_ids = torch.arange(2).view(1, 2, 1)
    mm_token_type_ids = torch.arange(18).view(3, 6)
    kwargs = {
        "attention_mask": MASK.clone(),
        "pixel_values": pixel_values,
        "image_position_ids": image_position_ids,
        "pixel_values_videos": pixel_values_videos,
        "video_position_ids": video_position_ids,
        "mm_token_type_ids": mm_token_type_ids,
    }

    sorted_ids, unsort_idx = model._sort_generation_inputs(input_ids, kwargs)

    assert torch.equal(sorted_ids, input_ids.index_select(0, SORT))
    assert torch.equal(kwargs["mm_token_type_ids"], mm_token_type_ids.index_select(0, SORT))
    assert torch.equal(kwargs["pixel_values"], _images_by_sample(pixel_values, [1, 0, 1], SORT.tolist()))
    assert torch.equal(kwargs["image_position_ids"], _images_by_sample(image_position_ids, [1, 0, 1], SORT.tolist()))
    # single video belongs to sample 1, which moves to row 0: layout unchanged
    assert torch.equal(kwargs["pixel_values_videos"], pixel_values_videos)
    assert torch.equal(kwargs["video_position_ids"], video_position_ids)
    assert kwargs["inputs_sorted"] is True
    assert torch.equal(model._unsort_generation_outputs(sorted_ids, unsort_idx), input_ids)


def test_llava_run_counting_and_image_sizes():
    model = _fake_model(RBLNLlavaForConditionalGeneration, {"image_token_index": IMG, "image_seq_length": 2})
    # images per sample [1, 2, 0] as separated runs
    input_ids = torch.tensor([[1, IMG, IMG, 2, 0, 0], [IMG, IMG, 3, IMG, IMG, 4], [5, 6, 7, 8, 9, 0]])
    pixel_values = torch.arange(3, dtype=torch.float).view(3, 1, 1, 1)
    image_sizes = torch.tensor([[8, 8], [16, 16], [32, 32]])
    kwargs = {"attention_mask": MASK.clone(), "pixel_values": pixel_values, "image_sizes": image_sizes}

    sorted_ids, unsort_idx = model._sort_generation_inputs(input_ids, kwargs)

    assert torch.equal(sorted_ids, input_ids.index_select(0, SORT))
    assert torch.equal(kwargs["pixel_values"], _images_by_sample(pixel_values, [1, 2, 0], SORT.tolist()))
    assert torch.equal(kwargs["image_sizes"], _images_by_sample(image_sizes, [1, 2, 0], SORT.tolist()))
    assert kwargs["inputs_sorted"] is True
    assert torch.equal(model._unsort_generation_outputs(sorted_ids, unsort_idx), input_ids)


def test_llava_adjacent_images_fall_back_to_token_count():
    model = _fake_model(RBLNLlavaForConditionalGeneration, {"image_token_index": IMG, "image_seq_length": 2})
    # sample 1 has two adjacent images (one merged run of 4 tokens)
    input_ids = torch.tensor([[1, IMG, IMG, 2, 0, 0], [IMG, IMG, IMG, IMG, 3, 4], [5, 6, 7, 8, 9, 0]])
    pixel_values = torch.arange(3, dtype=torch.float).view(3, 1, 1, 1)
    kwargs = {"attention_mask": MASK.clone(), "pixel_values": pixel_values}

    sorted_ids, _ = model._sort_generation_inputs(input_ids, kwargs)
    assert torch.equal(sorted_ids, input_ids.index_select(0, SORT))
    assert torch.equal(kwargs["pixel_values"], _images_by_sample(pixel_values, [1, 2, 0], SORT.tolist()))


def test_llava_unmappable_images_raise():
    model = _fake_model(RBLNLlavaForConditionalGeneration, {"image_token_index": IMG, "image_seq_length": None})
    # 3 images but only 2 runs and no per-image token count to fall back on
    input_ids = torch.tensor([[1, IMG, IMG, 2, 0, 0], [IMG, IMG, IMG, IMG, 3, 4]])
    kwargs = {
        "attention_mask": torch.tensor([[1, 1, 1, 1, 0, 0], [1, 1, 1, 1, 1, 1]]),
        "pixel_values": torch.zeros(3, 1, 1, 1),
    }
    with pytest.raises(RuntimeError, match="Cannot map image inputs"):
        model._sort_generation_inputs(input_ids, kwargs)


def test_llava_pixtral_images_via_image_sizes():
    # pixtral: [IMG_BREAK] splits an image into one run per patch row, so per-image
    # [IMG] counts come from image_sizes ((H/stride) * (W/stride))
    IMG_BREAK = 97
    vision_config = type("VisionCfg", (), {"patch_size": 2})()
    model = _fake_model(
        RBLNLlavaForConditionalGeneration,
        {"image_token_index": IMG, "image_seq_length": None, "vision_config": vision_config},
    )
    # image sizes (4,4)->4 tokens, (2,4)->2, (2,2)->1; images per sample [1, 2, 0]
    input_ids = torch.tensor(
        [
            [IMG, IMG, IMG_BREAK, IMG, IMG, 1],
            [IMG, IMG, 2, IMG, 3, 4],
            [5, 6, 7, 8, 9, 0],
        ]
    )
    pixel_values = torch.arange(3, dtype=torch.float).view(3, 1, 1, 1)
    image_sizes = torch.tensor([[4, 4], [2, 4], [2, 2]])
    kwargs = {"attention_mask": MASK.clone(), "pixel_values": pixel_values, "image_sizes": image_sizes}

    sorted_ids, unsort_idx = model._sort_generation_inputs(input_ids, kwargs)

    assert torch.equal(sorted_ids, input_ids.index_select(0, SORT))
    assert torch.equal(kwargs["pixel_values"], _images_by_sample(pixel_values, [1, 2, 0], SORT.tolist()))
    assert torch.equal(kwargs["image_sizes"], _images_by_sample(image_sizes, [1, 2, 0], SORT.tolist()))
    assert torch.equal(model._unsort_generation_outputs(sorted_ids, unsort_idx), input_ids)


def test_llava_pixtral_mismatched_totals_raise():
    vision_config = type("VisionCfg", (), {"patch_size": 2})()
    model = _fake_model(
        RBLNLlavaForConditionalGeneration,
        {"image_token_index": IMG, "image_seq_length": None, "vision_config": vision_config},
    )
    # row 0 carries 3 [IMG] tokens but no image combination (4, 2, 1) sums to 3 greedily
    input_ids = torch.tensor([[IMG, IMG, IMG, 1, 2, 3], [4, 5, 6, 7, 8, 9], [10, 11, 12, 13, 14, 0]])
    kwargs = {
        "attention_mask": MASK.clone(),
        "pixel_values": torch.zeros(3, 1, 1, 1),
        "image_sizes": torch.tensor([[4, 4], [2, 4], [2, 2]]),
    }
    with pytest.raises(RuntimeError, match="Cannot map image inputs"):
        model._sort_generation_inputs(input_ids, kwargs)


def test_llava_next_sorts_patched_pixel_values():
    model = _fake_model(RBLNLlavaNextForConditionalGeneration, {"image_token_index": IMG})
    # variable-length runs per image; images per sample [1, 2, 0]
    input_ids = torch.tensor([[1, IMG, IMG, IMG, 0, 0], [IMG, 2, IMG, IMG, 3, 4], [5, 6, 7, 8, 9, 0]])
    pixel_values = torch.arange(6, dtype=torch.float).view(3, 2, 1, 1, 1)
    image_sizes = torch.tensor([[8, 8], [16, 16], [32, 32]])
    kwargs = {"attention_mask": MASK.clone(), "pixel_values": pixel_values, "image_sizes": image_sizes}

    sorted_ids, unsort_idx = model._sort_generation_inputs(input_ids, kwargs)

    assert torch.equal(sorted_ids, input_ids.index_select(0, SORT))
    assert torch.equal(kwargs["pixel_values"], _images_by_sample(pixel_values, [1, 2, 0], SORT.tolist()))
    assert torch.equal(kwargs["image_sizes"], _images_by_sample(image_sizes, [1, 2, 0], SORT.tolist()))
    assert kwargs["inputs_sorted"] is True
    assert torch.equal(model._unsort_generation_outputs(sorted_ids, unsort_idx), input_ids)


def test_idefics3_sorts_batch_first_pixel_values():
    vision_config = type("VisionCfg", (), {"image_size": 4, "patch_size": 2})()
    model = _fake_model(
        RBLNIdefics3ForConditionalGeneration,
        {"image_token_id": IMG, "vision_config": vision_config, "scale_factor": 2},
        lm_attr="text_model",
    )
    input_ids = torch.tensor([[1, IMG, 2, 0, 0, 0], [IMG, 3, IMG, 4, 5, 6], [7, 8, 9, 10, 11, 0]])
    pixel_values = torch.arange(3, dtype=torch.float).view(3, 1, 1, 1, 1)
    pixel_attention_mask = torch.arange(12).view(3, 1, 2, 2)
    kwargs = {
        "attention_mask": MASK.clone(),
        "pixel_values": pixel_values,
        "pixel_attention_mask": pixel_attention_mask,
    }

    sorted_ids, unsort_idx = model._sort_generation_inputs(input_ids, kwargs)

    assert torch.equal(sorted_ids, input_ids.index_select(0, SORT))
    assert torch.equal(kwargs["pixel_values"], pixel_values.index_select(0, SORT))
    assert torch.equal(kwargs["pixel_attention_mask"], pixel_attention_mask.index_select(0, SORT))
    assert kwargs["inputs_sorted"] is True
    assert torch.equal(model._unsort_generation_outputs(sorted_ids, unsort_idx), input_ids)


def test_idefics3_sorts_image_hidden_states():
    # (image_size // patch_size)**2 // scale_factor**2 = 1 placeholder token per patch image
    vision_config = type("VisionCfg", (), {"image_size": 4, "patch_size": 2})()
    model = _fake_model(
        RBLNIdefics3ForConditionalGeneration,
        {"image_token_id": IMG, "vision_config": vision_config, "scale_factor": 2},
        lm_attr="text_model",
    )
    input_ids = torch.tensor([[1, IMG, 2, 0, 0, 0], [IMG, 3, IMG, 4, 5, 6], [7, 8, 9, 10, 11, 0]])
    image_hidden_states = torch.arange(3, dtype=torch.float).view(3, 1, 1)
    kwargs = {"attention_mask": MASK.clone(), "image_hidden_states": image_hidden_states}

    sorted_ids, _ = model._sort_generation_inputs(input_ids, kwargs)

    assert torch.equal(sorted_ids, input_ids.index_select(0, SORT))
    assert torch.equal(kwargs["image_hidden_states"], _images_by_sample(image_hidden_states, [1, 2, 0], SORT.tolist()))


def test_paligemma_sorts_batch_first_pixel_values():
    model = _fake_model(RBLNPaliGemmaForConditionalGeneration, {"image_token_id": IMG})
    input_ids = torch.tensor([[IMG, 1, 2, 0, 0, 0], [IMG, 3, 4, 5, 6, 7], [IMG, 8, 9, 10, 11, 0]])
    pixel_values = torch.arange(3, dtype=torch.float).view(3, 1, 1, 1)
    kwargs = {"attention_mask": MASK.clone(), "pixel_values": pixel_values}

    sorted_ids, unsort_idx = model._sort_generation_inputs(input_ids, kwargs)

    assert torch.equal(sorted_ids, input_ids.index_select(0, SORT))
    assert torch.equal(kwargs["pixel_values"], pixel_values.index_select(0, SORT))
    assert kwargs["inputs_sorted"] is True
    assert torch.equal(model._unsort_generation_outputs(sorted_ids, unsort_idx), input_ids)


def test_forward_guard_requires_sorted_inputs():
    model = _fake_model(RBLNLlavaForConditionalGeneration, {"image_token_index": IMG, "image_seq_length": 2})
    batch_input = torch.zeros(2, 4, dtype=torch.long)
    with pytest.raises(RuntimeError, match="use_batch_attn_opt"):
        model._require_sorted_batch_inputs(batch_input, inputs_sorted=False)
    model._require_sorted_batch_inputs(batch_input, inputs_sorted=True)
    model._require_sorted_batch_inputs(batch_input[:1], inputs_sorted=False)

    disabled = _fake_model(
        RBLNLlavaForConditionalGeneration,
        {"image_token_index": IMG, "image_seq_length": 2},
        use_batch_attn_opt=None,
    )
    disabled._require_sorted_batch_inputs(batch_input, inputs_sorted=False)
