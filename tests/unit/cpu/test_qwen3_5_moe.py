import copy

import torch
from torch import nn
from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import Qwen3_5MoeTextConfig
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeSparseMoeBlock as HFSparseMoeBlock

from optimum.rbln.transformers.models.qwen3_5_moe.qwen3_5_moe_architecture import Qwen3_5MoeSparseMoeBlock


def test_sparse_moe_block_matches_hf():
    # The RBLN block routes on the host (softmax of the top-k logits, scattered to [E, T]) and runs the
    # experts through custom_moe; HF routes softmax -> top-k -> renormalize and loops over the experts.
    torch.manual_seed(0)
    config = Qwen3_5MoeTextConfig(
        hidden_size=64,
        num_hidden_layers=1,
        num_experts=8,
        num_experts_per_tok=2,
        moe_intermediate_size=32,
        shared_expert_intermediate_size=48,
    )
    hf_block = HFSparseMoeBlock(config).eval()
    for param in hf_block.parameters():  # the router is zero-initialized, which would tie every expert
        nn.init.normal_(param, std=0.2)
    rbln_block = Qwen3_5MoeSparseMoeBlock(copy.deepcopy(hf_block)).eval()  # wrapping drops gate_up_proj

    hidden_states = torch.randn(2, 5, config.hidden_size)
    with torch.no_grad():
        expected = hf_block(hidden_states)
        actual = rbln_block(hidden_states)

    assert actual.shape == hidden_states.shape
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)
    assert hf_block.experts.gate_up_proj is not None  # the deepcopy, not the original, was split
