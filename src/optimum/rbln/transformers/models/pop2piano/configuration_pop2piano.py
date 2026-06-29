# Copyright 2026 Rebellions Inc. All rights reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Any, Optional

from ..seq2seq import RBLNModelForSeq2SeqLMConfig


class RBLNPop2PianoForConditionalGenerationConfig(RBLNModelForSeq2SeqLMConfig):
    """
    Configuration class for RBLNPop2PianoForConditionalGeneration.

    Pop2Piano is a T5-based encoder-decoder model whose encoder consumes a
    log-mel spectrogram (`input_features`) rather than token ids, so the encoder
    is compiled with `use_inputs_embeds=True`.
    """

    support_paged_attention = False

    def __init__(self, enc_max_seq_len: Optional[int] = None, use_inputs_embeds: Optional[bool] = None, **kwargs: Any):
        # encoder consumes mel features (1 composer token + N mel frames); fix a
        # generous static length and pad/mask shorter segments. 256 covers down
        # to ~42 BPM for the default 2-bar window. `use_inputs_embeds` is always
        # True for Pop2Piano (accepted here so a serialized config reloads).
        super().__init__(enc_max_seq_len=enc_max_seq_len or 256, use_inputs_embeds=True, **kwargs)
