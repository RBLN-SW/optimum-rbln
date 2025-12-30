# Copyright 2022 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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

"""
Generation utilities for Whisper.
Modified from `transformers.models.whisper.generation_whisper.py`
"""

from typing import Any, Dict, Optional, Union

import torch
import transformers
from packaging import version
from transformers import GenerationMixin
from transformers.generation.configuration_utils import GenerationConfig
from transformers.modeling_outputs import ModelOutput
from transformers.models.whisper.generation_whisper import WhisperGenerationMixin


class RBLNWhisperGenerationMixin(WhisperGenerationMixin, GenerationMixin):
    def generate(
        self,
        input_features: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        generation_config: Optional[GenerationConfig] = None,
        return_segments: Optional[bool] = None,
        return_timestamps: Optional[bool] = None,
        return_token_timestamps: Optional[bool] = None,
        **kwargs,
    ) -> Union[ModelOutput, Dict[str, Any], torch.LongTensor]:
        """
        The generate function is utilized in its standard form as in the HuggingFace transformers library. User can use this function to generate text from the model.
        Check the [HuggingFace transformers documentation](https://huggingface.co/docs/transformers/v4.57.1/en/model_doc/whisper#transformers.WhisperForConditionalGeneration.generate) for more details.

        Args:
            input_features(torch.Tensor, optional): The input features to the model.
            attention_mask(torch.Tensor, optional): Attention mask needs to be passed when doing long-form transcription using a batch size > 1.
            generation_config(GenerationConfig, optional): The generation configuration to be used as base parametrization for the generation call. **kwargs passed to generate matching the attributes of generation_config will override them.
                If generation_config is not provided, the default will be used, which had the following loading priority: 1) from the generation_config.json model file, if it exists; 2) from the model configuration.
                Please note that unspecified parameters will inherit [GenerationConfig](https://huggingface.co/docs/transformers/v4.57.1/en/main_classes/text_generation#transformers.GenerationConfig)’s default values.
            return_segments(bool, optional): Whether to return segments.
            return_timestamps(bool, optional): Whether to return the timestamps with the text. For audios longer than 30 seconds, it is necessary to set return_timestamps=True.
            return_token_timestamps(bool, optional): Whether to return token timestamps.
            kwargs(dict[str, Any], optional): Additional arguments passed to the generate function.

        Returns:
            Transcribes or translates log-mel input features to a sequence of auto-regressively generated token ids.
        """
        if kwargs.get("num_beams", None) is not None:
            if kwargs.get("num_beams") != 1:
                raise ValueError(
                    "Beam search is not supported in RBLNWhisperGenerationMixin. "
                    "Received num_beams={num_beams}, but only num_beams=1 is allowed. "
                    "Please set num_beams=1 for greedy search or adjust your configuration."
                )

        return super().generate(
            input_features,
            attention_mask=attention_mask,
            generation_config=generation_config,
            return_segments=return_segments,
            return_timestamps=return_timestamps,
            return_token_timestamps=return_token_timestamps,
            **kwargs,
        )

    def _postprocess_outputs(
        self,
        seek_outputs,
        decoder_input_ids,
        return_token_timestamps,
        generation_config,
        is_shortform,
        seek,
        batch_idx_map,
    ):
        # remove all previously passed decoder input ids
        # should happen only if it is the first generated segment
        start_idx = decoder_input_ids.shape[-1]

        if isinstance(seek_outputs, torch.Tensor):
            return seek_outputs[:, start_idx:], seek_outputs

        if return_token_timestamps and not self.rbln_token_timestamps:
            raise RuntimeError(
                "To use .generate() with return_token_timestamps=True, the model must be compiled with rbln_token_timestamps=True. "
                "You can compile the model by calling .from_pretrained() with export=True and rbln_token_timestamps=True as keyword arguments, "
                "or you can generate with return_token_timestamps=False."
            )

        if return_token_timestamps and hasattr(generation_config, "alignment_heads"):
            num_frames = getattr(generation_config, "num_frames", None)

            if num_frames is not None:
                num_frames = num_frames - seek
                num_frames = num_frames[batch_idx_map]

            if version.parse(transformers.__version__) >= version.parse("4.46.0"):
                seek_outputs["token_timestamps"] = self._extract_token_timestamps(
                    seek_outputs,
                    generation_config.alignment_heads,
                    num_frames=num_frames,
                    num_input_ids=decoder_input_ids.shape[-1],
                )
            else:
                seek_outputs["token_timestamps"] = self._extract_token_timestamps(
                    seek_outputs,
                    generation_config.alignment_heads,
                    num_frames=num_frames,
                )
        seek_outputs["sequences"] = seek_outputs["sequences"][:, start_idx:]

        def split_by_batch_index(values, key, batch_idx):
            if key in ["scores", "encoder_attentions", "encoder_hidden_states", "logits"]:
                return [v[batch_idx].cpu() for v in values]
            if key in ["decoder_attentions", "decoder_hidden_states", "cross_attentions"]:
                return tuple(tuple(w[batch_idx][None].cpu() for w in v) for v in values)
            elif key == "past_key_values":
                # we don't save `past_key_values in rbln
                return None

            return values[batch_idx].cpu()

        sequence_tokens = seek_outputs["sequences"]

        valid_seekoutputs = []
        for k, v in seek_outputs.items():
            if v is not None and len(v) > 0 and v[0] is not None:
                valid_seekoutputs.append((k, v))
        seek_outputs = [
            {k: split_by_batch_index(v, k, i) for k, v in valid_seekoutputs} for i in range(sequence_tokens.shape[0])
        ]

        return sequence_tokens, seek_outputs
