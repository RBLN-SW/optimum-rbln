import pytest
import torch
from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLVisionConfig
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionModel

from optimum.rbln import RBLNQwen3VLVisionModelConfig
from optimum.rbln.transformers.models.qwen3_vl.modeling_qwen3_vl import RBLNQwen3VLVisionModel
from optimum.rbln.transformers.models.qwen3_vl.qwen3_vl_architecture import Qwen3VLVisionModelWrapper


def tiny_vision():
    torch.manual_seed(7)
    config = Qwen3VLVisionConfig(
        hidden_size=16,
        num_heads=2,
        intermediate_size=32,
        depth=2,
        out_hidden_size=16,
        spatial_merge_size=2,
        num_position_embeddings=16,
        deepstack_visual_indexes=[0, 1],
    )
    config._attn_implementation = "sdpa"
    native = Qwen3VLVisionModel(config).eval()
    wrapper = Qwen3VLVisionModelWrapper(native, RBLNQwen3VLVisionModelConfig(dtype="float32", max_seq_len=16)).eval()
    hidden = torch.randn(4, 16)
    cos, sin = torch.ones(4, 8), torch.zeros(4, 8)
    return native, wrapper, hidden, cos, sin


@pytest.mark.parametrize("padding", [0, 4, 12])
def test_whole_vision_wrapper_preserves_native_valid_outputs_with_padding(padding):
    native, wrapper, hidden, cos, sin = tiny_vision()
    with torch.no_grad():
        expected_hidden = hidden
        expected_deepstack = []
        for index, block in enumerate(native.blocks):
            expected_hidden = block(
                expected_hidden, cu_seqlens=torch.tensor([0, 4], dtype=torch.int32), position_embeddings=(cos, sin)
            )
            expected_deepstack.append(native.deepstack_merger_list[index](expected_hidden))
        expected = (native.merger(expected_hidden), *expected_deepstack)
        padded, (padded_cos, padded_sin), mask, _ = RBLNQwen3VLVisionModel._pad_hidden_states(
            hidden, (cos, sin), 4 + padding
        )
        actual = wrapper(padded, mask, padded_cos[None, None], padded_sin[None, None])
    for tensor, reference in zip(actual, expected, strict=True):
        torch.testing.assert_close(tensor[:1], reference, rtol=1e-5, atol=1e-6)


def test_vision_export_preserves_binary_mask_for_masked_sdpa_compiler_path():
    _, wrapper, hidden, cos, sin = tiny_vision()
    padded, (cos, sin), mask, _ = RBLNQwen3VLVisionModel._pad_hidden_states(hidden, (cos, sin), 8)
    exported = torch.export.export(wrapper, (padded, mask, cos[None, None], sin[None, None]))
    attention = [
        node for node in exported.graph.nodes if node.target == torch.ops.aten.scaled_dot_product_attention.default
    ]
    assert len(attention) == 2
    # Binary masks select masked SDPA; floating additive masks select the unfused attention path.
    assert all(node.args[3].meta["val"].dtype == torch.bool for node in attention)
