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

import inspect
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Optional, Union

import torch
from transformers import AutoModelForSeq2SeqLM, Pop2PianoForConditionalGeneration
from transformers.generation.configuration_utils import GenerationConfig
from transformers.modeling_outputs import ModelOutput
from transformers.initialization import no_init_weights
from transformers.models.pop2piano.modeling_pop2piano import Pop2PianoConcatEmbeddingToMel

from ...models.seq2seq import RBLNModelForSeq2SeqLM
from .configuration_pop2piano import RBLNPop2PianoForConditionalGenerationConfig
from .pop2piano_architecture import Pop2PianoWrapper


if TYPE_CHECKING:
    from transformers import PreTrainedModel


class RBLNPop2PianoForConditionalGeneration(RBLNModelForSeq2SeqLM):
    """
    The Pop2Piano model with a language modeling head for conditional generation.

    Pop2Piano shares T5's encoder-decoder architecture, but the encoder consumes a
    log-mel spectrogram (`input_features`) instead of token ids, so it is compiled
    with `use_inputs_embeds=True` and fed the features directly as `inputs_embeds`.

    Because the number of beat-synced segments per song is dynamic while the
    compiled decoder batch size is fixed, generation iterates over segments with a
    batch size of 1.
    """

    auto_model_class = AutoModelForSeq2SeqLM
    support_causal_attn = False

    def __post_init__(self, **kwargs):
        super().__post_init__(**kwargs)
        # The composer conditioner (a small nn.Embedding + concat) runs on host;
        # its weights are persisted as a torch artifact, mirroring how decoder-only
        # models persist `embed_tokens` for the `use_inputs_embeds` path.
        artifacts = torch.load(
            Path(self.model_save_dir) / self.subfolder / "torch_artifacts.pth", weights_only=False
        )
        self.mel_conditioner = self._create_mel_conditioner()
        self.mel_conditioner.load_state_dict(artifacts["mel_conditioner"])

    def _create_mel_conditioner(self):
        with no_init_weights():
            return Pop2PianoConcatEmbeddingToMel(self.config)

    @classmethod
    def save_torch_artifacts(cls, model, save_dir_path, subfolder, rbln_config):
        torch.save(
            {"mel_conditioner": model.mel_conditioner.state_dict()},
            save_dir_path / subfolder / "torch_artifacts.pth",
        )

    @classmethod
    def _wrap_model_if_needed(
        cls, model: "PreTrainedModel", rbln_config: RBLNPop2PianoForConditionalGenerationConfig
    ):
        return Pop2PianoWrapper(
            model, enc_max_seq_len=rbln_config.enc_max_seq_len, dec_max_seq_len=rbln_config.dec_max_seq_len
        )

    def _validate_model_kwargs(self, model_kwargs):
        # `inputs_embeds` is consumed by the encoder via
        # `_prepare_encoder_decoder_kwargs_for_generation`; HF validation (which
        # receives a copy) would otherwise reject it as unused.
        model_kwargs.pop("inputs_embeds", None)
        super()._validate_model_kwargs(model_kwargs)

    def __getattr__(self, __name: str) -> Any:
        def redirect(func):
            return lambda *pargs, **kwargs: func(self, *pargs, **kwargs)

        val = getattr(Pop2PianoForConditionalGeneration, __name)

        if isinstance(val, Callable) and "self" in set(inspect.signature(val).parameters):
            return redirect(val)

        return val

    def generate(
        self,
        input_features: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        composer: str = "composer1",
        generation_config: Optional[GenerationConfig] = None,
        **kwargs,
    ) -> Union[ModelOutput, torch.LongTensor]:
        """
        Generates MIDI token ids for the piano cover of an audio clip.

        `input_features` is the log-mel spectrogram from `Pop2PianoProcessor`
        (shape `(num_segments, num_frames, d_model)`); the composer-conditioner
        token is prepended on host. Alternatively, pass already-conditioned
        `inputs_embeds` directly. The beat-synced segments are decoded in chunks
        of the compiled `batch_size` (the last chunk is zero-padded and its
        padding outputs discarded), and the per-segment token sequences are
        right-padded into a single `(num_segments, max_len)` tensor.
        """
        if inputs_embeds is None:
            if input_features is None:
                raise ValueError("Either `input_features` or `inputs_embeds` must be provided.")
            inputs_embeds, attention_mask = self.get_mel_conditioner_outputs(
                input_features=input_features,
                composer=composer,
                generation_config=generation_config or self.generation_config,
                attention_mask=attention_mask,
            )

        num_segments = inputs_embeds.shape[0]
        if attention_mask is None:
            attention_mask = torch.ones(inputs_embeds.shape[:2], dtype=torch.int64)

        batch_size = self.rbln_config.batch_size
        sequences = []
        for start in range(0, num_segments, batch_size):
            emb = inputs_embeds[start : start + batch_size]
            mask = attention_mask[start : start + batch_size]
            valid = emb.shape[0]
            if valid < batch_size:  # pad the final chunk up to the compiled batch size
                pad = batch_size - valid
                emb = torch.cat([emb, emb.new_zeros((pad, *emb.shape[1:]))], dim=0)
                mask = torch.cat([mask, mask.new_ones((pad, mask.shape[1]))], dim=0)
            out = super().generate(
                None,
                attention_mask=mask,
                generation_config=generation_config,
                inputs_embeds=emb,
                **kwargs,
            )
            sequences.extend(out[i] for i in range(valid))  # drop padding outputs

        pad_token_id = self.config.pad_token_id
        max_len = max(s.shape[0] for s in sequences)
        output = torch.full((num_segments, max_len), pad_token_id, dtype=torch.long)
        for i, s in enumerate(sequences):
            output[i, : s.shape[0]] = s

        return output
