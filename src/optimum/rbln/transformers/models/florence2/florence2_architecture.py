# Copyright 2026 Rebellions Inc. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0

from types import MethodType

import torch
from torch import nn

from ..bart.bart_architecture import (
    BartCrossAttention,
    BartDecoder,
    BartDecoderLayer,
    BartDecoderWrapper,
    BartForConditionalGeneration,
    BartSelfAttention,
    _encoder_layer_forward,
)
from ..seq2seq.seq2seq_architecture import Seq2SeqEncoderWrapper


class Florence2ImageEncoder(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.vision_tower = model.model.vision_tower
        self.projector = model.model.multi_modal_projector

    def forward(self, pixel_values):
        vision_hidden_states = self.vision_tower(pixel_values, return_dict=False)[0]
        return self.projector(vision_hidden_states)


class Florence2MultimodalEncoder(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.image_encoder = Florence2ImageEncoder(model)
        self.encoder = model.model.language_model.encoder

    def forward(self, input_ids, attention_mask, **kwargs):
        pixel_values, text_ids = input_ids
        image_features = self.image_encoder(pixel_values)
        text_features = self.encoder.embed_tokens(text_ids)
        inputs_embeds = torch.cat((image_features, text_features), dim=1)
        return self.encoder(inputs_embeds=inputs_embeds, attention_mask=attention_mask, **kwargs)


class Florence2EncoderWrapper(Seq2SeqEncoderWrapper):
    def __init__(self, model, enc_max_seq_len):
        super().__init__(model.model.language_model, enc_max_seq_len)
        self.encoder = Florence2MultimodalEncoder(model)

    def forward(self, input_ids, attention_mask, pixel_values, block_tables, *cross_key_values):
        # Pack both modalities to reuse the shared encoder's cross-KV update path.
        return super().forward((pixel_values, input_ids), attention_mask, block_tables, *cross_key_values)


class Florence2DecoderWrapper(BartDecoderWrapper):
    def __post_init__(self, model, **kwargs):
        self.config = model.config.text_config
        super().__post_init__(model, **kwargs)

    def convert_to_rbln_conditional_generation(self, model):
        decoder = model.model.language_model.get_decoder()
        new_layers = []
        for layer in decoder.layers:
            self_attn = BartSelfAttention(layer.self_attn, use_attention_mask=self.use_attention_mask)
            cross_attn = BartCrossAttention(layer.encoder_attn)
            new_layers.append(BartDecoderLayer(layer, self_attn, cross_attn))

        decoder_model = BartDecoder(decoder, new_layers)
        new_model = BartForConditionalGeneration(model, decoder_model)
        new_model.config = self.config
        return new_model


class Florence2Wrapper:
    def __init__(self, model, enc_max_seq_len, use_attention_mask):
        for layer in model.model.language_model.get_encoder().layers:
            layer.forward = MethodType(_encoder_layer_forward, layer)
        self.encoder = Florence2EncoderWrapper(model, enc_max_seq_len).eval()
        self.decoder = Florence2DecoderWrapper(model, use_attention_mask=use_attention_mask).eval()
