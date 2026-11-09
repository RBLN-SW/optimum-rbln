import copy

import pytest
import torch
from diffusers.models.autoencoders.autoencoder_kandinsky6_sr import Kandinsky6SRCausalConv3d

from optimum.rbln.diffusers.models.autoencoders.autoencoder_kandinsky6_sr import (
    _Kandinsky6SRCausalConv3d,
    _SegmentCache,
)


@pytest.mark.parametrize(
    ("kernel", "stride", "lengths"),
    [
        (3, (1, 1, 1), (5, 4, 4)),
        (3, (1, 1, 1), (2, 1, 1, 1)),
        ((2, 1, 1), (1, 1, 1), (5, 4, 4)),
        ((2, 1, 1), (2, 1, 1), (5, 4, 4)),
    ],
)
@torch.no_grad()
def test_causal_convolution_preserves_segment_outputs_and_history(kernel, stride, lengths):
    torch.manual_seed(42)
    original = Kandinsky6SRCausalConv3d(4, 8, kernel, stride=stride).eval()
    compiled_form = _Kandinsky6SRCausalConv3d(copy.deepcopy(original)).eval()
    reference_cache, actual_cache = original.make_cache(), compiled_form.make_cache()
    for length in lengths:
        x = torch.randn(2, 4, length, 8, 12)
        expected = original(x, reference_cache)
        actual = compiled_form(x, actual_cache)
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
        hpad, wpad = original.height_pad, original.width_pad
        expected_history = reference_cache["padding"][..., hpad : -hpad or None, wpad : -wpad or None]
        torch.testing.assert_close(actual_cache["padding"], expected_history, rtol=0, atol=0)


@pytest.mark.parametrize("shape", [(1, 64, 2, 8, 12), (1, 16, 2, 8, 96), (1, 64, 2, 8, 128)])
def test_segment_cache_preserves_frames_across_layout_conversion(shape):
    cache = _SegmentCache(("conv", "padding"), shape)
    x = torch.arange(torch.tensor(shape).prod()).reshape(shape).float()
    stored = cache.to_cache(x)
    assert list(stored.shape) == cache.compile_shape
    assert stored.shape[2] == shape[2]
    torch.testing.assert_close(cache.from_cache(stored), x, rtol=0, atol=0)
