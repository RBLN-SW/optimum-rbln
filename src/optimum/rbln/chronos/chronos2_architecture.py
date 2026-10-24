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

import copy

import torch
from chronos.chronos2.config import Chronos2ForecastingConfig
from chronos.chronos2.model import Chronos2Model
from chronos.chronos_bolt import InstanceNorm, Patch


class Chronos2Preprocessor:
    _validate_input = Chronos2Model._validate_input
    _prepare_patched_context = Chronos2Model._prepare_patched_context
    _prepare_patched_future = Chronos2Model._prepare_patched_future

    def __init__(self, config, context_length):
        self.chronos_config = Chronos2ForecastingConfig(**copy.deepcopy(config.chronos_config))
        if self.chronos_config.time_encoding_scale is None:
            self.chronos_config.time_encoding_scale = self.chronos_config.context_length
        self.chronos_config.context_length = context_length
        self.dtype = torch.float32
        self.device = torch.device("cpu")
        self.patch = Patch(self.chronos_config.input_patch_size, self.chronos_config.input_patch_stride)
        self.instance_norm = InstanceNorm(use_arcsinh=self.chronos_config.use_arcsinh)


class Chronos2ModelWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, patched_context, context_mask, patched_future, group_ids):
        batch = patched_context.shape[0]
        num_future = patched_future.shape[1]
        context_embeds = self.model.input_patch_embedding(patched_context)
        embeddings = [context_embeds]
        masks = [context_mask.to(context_embeds.dtype)]
        if self.model.chronos_config.use_reg_token:
            reg_ids = torch.full(
                (batch, 1), self.model.config.reg_token_id, dtype=torch.long, device=patched_context.device
            )
            embeddings.append(self.model.shared(reg_ids))
            masks.append(torch.ones((batch, 1), dtype=context_embeds.dtype, device=context_mask.device))
        embeddings.append(self.model.input_patch_embedding(patched_future))
        masks.append(torch.ones((batch, num_future), dtype=context_embeds.dtype, device=context_mask.device))
        hidden = self.model.encoder(
            inputs_embeds=torch.cat(embeddings, dim=1),
            attention_mask=torch.cat(masks, dim=1),
            group_ids=group_ids,
            output_attentions=False,
        ).last_hidden_state
        head = self.model.output_patch_embedding
        forecast = hidden[:, -num_future:]
        output = head.dropout(head.output_layer(head.act(head.hidden_layer(forecast))))
        residual = head.residual_layer(forecast)
        outputs: tuple[torch.Tensor, ...]
        if head.use_layer_norm:
            outputs = (head.layer_norm(output + residual),)
        else:
            outputs = (output, residual)
        outputs = tuple(
            value.reshape(batch, num_future, self.model.num_quantiles, self.model.chronos_config.output_patch_size)
            .permute(0, 2, 1, 3)
            .reshape(batch, self.model.num_quantiles, -1)
            for value in outputs
        )
        return outputs[0] if head.use_layer_norm else outputs
