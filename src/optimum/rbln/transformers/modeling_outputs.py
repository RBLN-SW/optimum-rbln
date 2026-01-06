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

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from transformers.modeling_outputs import ModelOutput

from ..configuration_utils import RBLNModelConfig


@dataclass
class RBLNDecoderOnlyOutput(ModelOutput):
    logits: torch.FloatTensor = None
    generate_idx: torch.Tensor = None
    padded_cache_lengths: int = None
    hidden_states: Tuple[torch.FloatTensor] = None


@dataclass
class RBLNGemma3ForCausalLMOutput(RBLNDecoderOnlyOutput):
    attention_mask: Optional[torch.Tensor] = None


@dataclass
class RBLNSeq2SeqTSDecoderOutput(ModelOutput):
    last_hidden_states: torch.FloatTensor = None
    params: Tuple[torch.FloatTensor] = None


def _validate_output_hidden_states(output_hidden_states: Optional[bool], rbln_config: RBLNModelConfig):
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else rbln_config.output_hidden_states
    )
    if output_hidden_states != rbln_config.output_hidden_states:
        raise ValueError(
            f"Variable output_hidden_states {output_hidden_states} is not equal to rbln_config.output_hidden_states {rbln_config.output_hidden_states} "
            f"Please compile again with the correct argument."
        )

    return output_hidden_states


def _validate_output_attentions(output_attentions: Optional[bool], rbln_config: RBLNModelConfig):
    output_attentions = output_attentions if output_attentions is not None else rbln_config.output_attentions
    if output_attentions != rbln_config.output_attentions:
        raise ValueError(
            f"Variable output_attentions {output_attentions} is not equal to rbln_config.output_attentions {rbln_config.output_attentions} "
            f"Please compile again with the correct argument."
        )
    return output_attentions
