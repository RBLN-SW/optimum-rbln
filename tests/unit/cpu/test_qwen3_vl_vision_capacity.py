from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from optimum.rbln.transformers.models.qwen3_vl.modeling_qwen3_vl import RBLNQwen3VLVisionModel


def make_vision_model(height, width, buckets):
    hidden_states = torch.zeros(height * width, 4)
    model = SimpleNamespace(
        rbln_config=SimpleNamespace(dtype=torch.float32),
        patch_embed=torch.nn.Identity(),
        fast_pos_embed_interpolate=lambda grid: torch.zeros_like(hidden_states),
        spatial_merge_size=2,
        spatial_merge_unit=4,
        rotary_cos_table=torch.ones(max(height, width), 1),
        rotary_sin_table=torch.zeros(max(height, width), 1),
        max_seq_len=torch.tensor(buckets),
        deepstack_visual_indexes=[],
        _pad_hidden_states=RBLNQwen3VLVisionModel._pad_hidden_states,
        transformer=Mock(return_value=[torch.zeros(max(buckets) // 4, 4)]),
    )
    return model, hidden_states, torch.tensor([[1, height, width]])


@pytest.mark.parametrize(("height", "width", "buckets"), [(86, 128, [6400]), (14, 14, [64])])
def test_vision_capacity_error_explains_patch_count_and_remedy(height, width, buckets):
    model, hidden_states, grid = make_vision_model(height, width, buckets)

    with pytest.raises(ValueError, match="before spatial merging") as exc:
        RBLNQwen3VLVisionModel.forward(model, hidden_states, grid)

    message = str(exc.value)
    assert f"seq_len({height * width})" in message
    assert f"max_seq_len({buckets})" in message
    assert "before spatial merging" in message
    assert "Reduce the processed image/video resolution" in message
    assert "recompile the vision encoder with a larger max_seq_len" in message
    model.transformer.assert_not_called()


def test_vision_capacity_accepts_exact_bucket_size():
    model, hidden_states, grid = make_vision_model(16, 16, [64, 256])

    output, deepstack = RBLNQwen3VLVisionModel.forward(model, hidden_states, grid)

    assert output.shape == (64, 4)
    assert deepstack == []
    model.transformer.assert_called_once()
    assert model.transformer.call_args.args[0].shape[0] == 256
