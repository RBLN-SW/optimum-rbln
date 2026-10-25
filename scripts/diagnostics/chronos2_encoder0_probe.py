"""Source-equivalent encoder0 boundary probe; compilation/execution require review."""

import hashlib

import torch
from torch import nn


OUTPUT_NAMES = ("block0_time", "block0_group", "block0_output", "block1_output", "block2_output")
INPUT_INFO = [
    ("hidden", [512, 577, 768], "float32"),
    ("attention_mask", [512, 1, 1, 577], "float32"),
    ("group_time_mask", [577, 1, 512, 512], "float32"),
]
ORIGINAL_FINAL_SHA256 = "6b0d3bcb1c1f9d7ae2d9284c9aa2db32cb2093eaaf230810e86c7e861f407c79"


def tensor_sha256(value):
    value = value.detach().cpu()
    digest = hashlib.sha256()
    rows = value.reshape(1) if value.ndim == 0 else value
    row_bytes = rows[0].numel() * rows.element_size()
    for chunk in rows.split(max(1, 8 * 1024 * 1024 // row_bytes)):
        digest.update(chunk.contiguous().numpy().tobytes())
    return digest.hexdigest()


class Encoder0BoundaryProbe(nn.Module):
    """One graph, original three trained blocks, five existing residual boundaries.

    The first block calls the same temporal/group/FFN modules in the native
    forward order. Blocks1/2 call their original forward methods. No attention
    weights, reduced batch, changed mask, host learned operation or new math is
    introduced. Extra returned tensors can affect compiler lowering; only a
    final-output byte-identity gate can make NPU observations attributable.
    """

    def __init__(self, layers):
        super().__init__()
        if len(layers) != 3:
            raise ValueError("Expected the original first three complete encoder blocks")
        self.layers = nn.ModuleList(layers)

    def forward(self, hidden, attention_mask, group_time_mask):
        positions = torch.arange(hidden.shape[1], dtype=torch.long, device=hidden.device).unsqueeze(0)
        first = self.layers[0]
        time0 = first.layer[0](hidden, position_ids=positions, attention_mask=attention_mask, output_attentions=False)[
            0
        ]
        group0 = first.layer[1](time0, attention_mask=group_time_mask, output_attentions=False)[0]
        block0 = first.layer[2](group0)
        block1 = self.layers[1](
            block0,
            position_ids=positions,
            attention_mask=attention_mask,
            group_time_mask=group_time_mask,
            output_attentions=False,
        ).hidden_states
        block2 = self.layers[2](
            block1,
            position_ids=positions,
            attention_mask=attention_mask,
            group_time_mask=group_time_mask,
            output_attentions=False,
        ).hidden_states
        return time0, group0, block0, block1, block2


def require_original_final_identity(outputs, expected_sha256=ORIGINAL_FINAL_SHA256, expected_shape=(512, 577, 768)):
    """Run before using any exposed intermediate for original-BAD attribution."""
    if not isinstance(outputs, (tuple, list)) or len(outputs) != len(OUTPUT_NAMES):
        raise ValueError("Invalid augmented output count; no boundary attribution is valid")
    if any(tuple(value.shape) != tuple(expected_shape) or value.dtype != torch.float32 for value in outputs):
        raise ValueError("Invalid augmented full native output interfaces")
    actual = tensor_sha256(outputs[-1])
    if actual != expected_sha256:
        raise ValueError(
            "INVALID PROBE: instrumentation changed original encoder0 final bytes; do not attribute boundaries"
        )
    return {"valid_for_original_encoder0_attribution": True, "final_sha256": actual}
