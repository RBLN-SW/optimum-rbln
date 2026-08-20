# Copyright 2025 Rebellions Inc. All rights reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""The pooler mask Gemma4 strips padding with, derived from the position ids."""

import pytest
import torch

from optimum.rbln.transformers.models.gemma4.modeling_gemma4 import (
    pooler_mask_from_position_ids,
)


def _position_ids(valid_patches: list[int], max_patches: int) -> torch.Tensor:
    ids = torch.full((len(valid_patches), max_patches, 2), -1, dtype=torch.int64)
    for row, n in enumerate(valid_patches):
        ids[row, :n] = torch.arange(n).unsqueeze(1).expand(n, 2)
    return ids


@pytest.mark.parametrize(
    ("valid_patches", "max_patches", "pooling", "expected"),
    [
        # the case observed in CI: 2394 of 2520 patches, 3x3 pooling -> 266 of 280
        pytest.param([2394], 2520, 3, [266], id="ci_case"),
        pytest.param([2394, 1800], 2520, 3, [266, 200], id="two_images"),
        pytest.param([2520], 2520, 3, [280], id="no_padding"),
        pytest.param([0], 2520, 3, [0], id="all_padding"),
    ],
)
def test_mask_counts_match_the_position_ids(valid_patches, max_patches, pooling, expected):
    max_soft_tokens = max_patches // (pooling * pooling)
    mask = pooler_mask_from_position_ids(_position_ids(valid_patches, max_patches), max_soft_tokens, pooling)
    assert mask.shape == (len(valid_patches), max_soft_tokens)
    assert mask.dtype == torch.bool
    assert mask.sum(dim=1).tolist() == expected


def test_valid_tokens_are_at_the_front():
    """The caller splits the flat output per image, so order is part of the contract."""
    mask = pooler_mask_from_position_ids(_position_ids([2394], 2520), 280, 3)
    assert mask[0, :266].all()
    assert not mask[0, 266:].any()
