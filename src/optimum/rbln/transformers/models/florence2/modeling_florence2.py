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

import torch
from transformers import AutoModelForImageTextToText
from transformers.generation.configuration_utils import GenerationMode

from ....configuration_utils import RBLNCompileConfig
from ..seq2seq import RBLNModelForSeq2SeqLM
from ..seq2seq.modeling_seq2seq import RBLNRuntimeEncoder
from .configuration_florence2 import RBLNFlorence2ForConditionalGenerationConfig
from .florence2_architecture import Florence2Wrapper


class RBLNRuntimeFlorence2Encoder(RBLNRuntimeEncoder):
    # HF generation validates pixel_values against the encoder's named arguments.
    def forward(self, input_ids, attention_mask, pixel_values, block_tables):
        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            block_tables=block_tables,
        )


class RBLNFlorence2ForConditionalGeneration(RBLNModelForSeq2SeqLM):
    """Florence-2 generation with a fused vision/text encoder and cached BART decoder.

    Uses native Transformers checkpoints such as ``florence-community/Florence-2-base``.
    Supports greedy generation and sampling with one image and one returned sequence per input.
    Set ``num_beams=1``; the checkpoint's default generation config uses beam search.
    """

    auto_model_class = AutoModelForImageTextToText
    base_model_prefix = "model"
    support_causal_attn = True
    _supported_generation_modes = (GenerationMode.GREEDY_SEARCH, GenerationMode.SAMPLE)

    @classmethod
    def _wrap_model_if_needed(cls, model, rbln_config: RBLNFlorence2ForConditionalGenerationConfig):
        return Florence2Wrapper(model, rbln_config.enc_max_seq_len, rbln_config.use_attention_mask)

    @classmethod
    def _update_rbln_config(cls, preprocessors, model, model_config, rbln_config):
        # Only the released checkpoints' 768x768 input resolution is supported.
        image_size = 768
        vision = model_config.vision_config
        feature_size = image_size
        for kernel, stride, padding in zip(vision.patch_size, vision.patch_stride, vision.patch_padding, strict=True):
            feature_size = (feature_size + 2 * padding - kernel) // stride + 1
        image_seq_len = feature_size**2 + 1
        rbln_config = super()._update_rbln_config(preprocessors, model, model_config.text_config, rbln_config)
        text_seq_len = rbln_config.enc_max_seq_len - image_seq_len
        if text_seq_len <= 0:
            raise ValueError("`enc_max_seq_len` must leave room for text tokens after the image features.")
        enc_input_info = list(rbln_config.compile_cfgs[0].input_info)
        enc_input_info[0] = ("input_ids", [1, text_seq_len], "int64")
        enc_input_info.insert(2, ("pixel_values", [1, 3, image_size, image_size], rbln_config.dtype))
        encoder_config = RBLNCompileConfig(compiled_model_name="encoder", input_info=enc_input_info)
        rbln_config.set_compile_cfgs([encoder_config, rbln_config.compile_cfgs[1]])
        return rbln_config

    def __post_init__(self, **kwargs):
        super().__post_init__(**kwargs)
        self.encoder = RBLNRuntimeFlorence2Encoder(runtime=self.model[0], main_input_name="input_ids")
        text_seq_len = self.rbln_config.compile_cfgs[0].input_info[0][1][1]
        self.image_seq_len = self.rbln_config.enc_max_seq_len - text_seq_len

    def _validate_generation_mode(self, generation_mode, generation_config, generation_mode_kwargs):
        if generation_config.num_beams > 1:
            raise ValueError(
                "Beam search is not supported for RBLNFlorence2ForConditionalGeneration "
                f"(num_beams={generation_config.num_beams}). Set `num_beams=1` for greedy generation or sampling."
            )
        super()._validate_generation_mode(generation_mode, generation_config, generation_mode_kwargs)

    def _prepare_encoder_decoder_kwargs_for_generation(
        self, inputs_tensor, model_kwargs, model_input_name=None, generation_config=None
    ):
        pixels = model_kwargs.pop("pixel_values").to(self.rbln_config.dtype)
        batch_size = inputs_tensor.shape[0]
        if batch_size != self.rbln_config.batch_size:
            raise ValueError("Input batch size does not match the compiled batch_size.")
        if pixels.shape[0] != batch_size:
            raise ValueError("Image batch size must match the number of input sequences.")
        image_tokens = inputs_tensor == self.config.image_token_id
        if not image_tokens[:, : self.image_seq_len].all() or (image_tokens.sum(1) != self.image_seq_len).any():
            raise ValueError("Expected one image-token prefix per input; use the native Florence-2 processor.")
        text_max_seq_len = self.rbln_config.enc_max_seq_len - self.image_seq_len
        attention_mask = model_kwargs.get("attention_mask", torch.ones_like(inputs_tensor))
        text_ids = inputs_tensor[:, self.image_seq_len :]
        pad_len = text_max_seq_len - text_ids.shape[1]
        if pad_len < 0:
            raise ValueError("Text input exceeds the compiled encoder length; increase rbln_enc_max_seq_len.")
        text_ids = torch.nn.functional.pad(text_ids, (0, pad_len), value=self.config.text_config.pad_token_id)
        attention_mask = torch.nn.functional.pad(attention_mask, (0, pad_len)).to(self.rbln_config.dtype)
        for index in range(batch_size):
            model_kwargs["encoder_outputs"] = self.encoder(
                input_ids=text_ids[index : index + 1].contiguous(),
                attention_mask=attention_mask[index : index + 1].contiguous(),
                pixel_values=pixels[index : index + 1].contiguous(),
                block_tables=torch.tensor([index], dtype=torch.int16),
            )
        model_kwargs["attention_mask"] = attention_mask
        return model_kwargs

    def _prepare_decoder_input_ids_for_generation(
        self, batch_size, model_input_name, model_kwargs, decoder_start_token_id, device=None
    ):
        decoder_ids, model_kwargs = super()._prepare_decoder_input_ids_for_generation(
            batch_size, model_input_name, model_kwargs, decoder_start_token_id, device
        )
        if decoder_ids.shape[1] != 1:
            raise NotImplementedError("Multi-token decoder prefill is not supported.")
        return decoder_ids, model_kwargs

    def _validate_generated_length(self, generation_config, input_ids_length, has_default_max_length):
        super()._validate_generated_length(generation_config, input_ids_length, has_default_max_length)
        if generation_config.max_length > self.rbln_config.dec_max_seq_len + 1:
            raise ValueError(
                "Requested generation exceeds the compiled decoder KV cache; increase rbln_dec_max_seq_len."
            )


__all__ = ["RBLNFlorence2ForConditionalGeneration"]
