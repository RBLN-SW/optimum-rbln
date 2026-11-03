# Copyright 2026 Rebellions Inc. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0

import torch
from torch import nn

from ..bart.bart_architecture import BartWrapper


class Florence2LanguageAdapter(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.config = model.config.text_config
        self.model = model.model.language_model
        self.lm_head = model.lm_head

    def get_encoder(self):
        return self.model.get_encoder()

    def get_decoder(self):
        return self.model.get_decoder()


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


class Florence2EncoderWrapper(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, input_ids, attention_mask, pixel_values, block_tables, *cross_key_values):
        return self.encoder((pixel_values, input_ids), attention_mask, block_tables, *cross_key_values)


class Florence2Wrapper:
    def __init__(self, model, enc_max_seq_len, use_attention_mask):
        bart = BartWrapper(
            Florence2LanguageAdapter(model), enc_max_seq_len=enc_max_seq_len, use_attention_mask=use_attention_mask
        )
        bart.encoder.encoder = Florence2MultimodalEncoder(model)
        self.encoder = Florence2EncoderWrapper(bart.encoder).eval()
        self.decoder = bart.decoder.eval()
