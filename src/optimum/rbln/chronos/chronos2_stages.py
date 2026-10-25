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

import gc

import torch
from torch import nn

from ..configuration_utils import RBLNCompileConfig


class Chronos2Embedding(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.input_patch_embedding = model.input_patch_embedding
        self.shared = model.shared
        self.use_reg_token = model.chronos_config.use_reg_token
        self.reg_token_id = model.config.reg_token_id if self.use_reg_token else None
        self.encoder_dropout = model.encoder.dropout
        self.expand_time_mask = model.encoder._expand_and_invert_time_attention_mask
        self.expand_group_mask = model.encoder._construct_and_invert_group_time_mask

    def forward(self, patched_context, context_mask, patched_future, group_ids):
        context = self.input_patch_embedding(patched_context)
        embeddings = [context]
        masks = [context_mask.to(context.dtype)]
        batch = patched_context.shape[0]
        if self.use_reg_token:
            assert self.reg_token_id is not None
            ids = torch.full((batch, 1), self.reg_token_id, dtype=torch.long, device=patched_context.device)
            embeddings.append(self.shared(ids))
            masks.append(torch.ones((batch, 1), dtype=context.dtype, device=context.device))
        embeddings.append(self.input_patch_embedding(patched_future))
        masks.append(torch.ones((batch, patched_future.shape[1]), dtype=context.dtype, device=context.device))
        mask = torch.cat(masks, dim=1)
        return (
            self.encoder_dropout(torch.cat(embeddings, dim=1)),
            self.expand_time_mask(mask, context.dtype),
            self.expand_group_mask(group_ids, mask, context.dtype),
        )


class Chronos2Encoder(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.layers = nn.ModuleList(layers)

    def forward(self, hidden, attention_mask, group_time_mask):
        positions = torch.arange(hidden.shape[1], dtype=torch.long, device=hidden.device).unsqueeze(0)
        for layer in self.layers:
            hidden = layer(
                hidden,
                position_ids=positions,
                attention_mask=attention_mask,
                group_time_mask=group_time_mask,
                output_attentions=False,
            ).hidden_states
        return hidden


class Chronos2Forecast(nn.Module):
    def __init__(self, model, future_patches):
        super().__init__()
        self.layer_norm = model.encoder.final_layer_norm
        self.dropout = model.encoder.dropout
        self.head = model.output_patch_embedding
        self.future_patches = future_patches
        self.num_quantiles = model.num_quantiles
        self.output_patch_size = model.chronos_config.output_patch_size

    def forward(self, hidden):
        forecast = self.dropout(self.layer_norm(hidden))[:, -self.future_patches :]
        head = self.head
        output = head.dropout(head.output_layer(head.act(head.hidden_layer(forecast))))
        residual = head.residual_layer(forecast)
        values = (head.layer_norm(output + residual),) if head.use_layer_norm else (output, residual)
        values = tuple(
            value.reshape(hidden.shape[0], self.future_patches, self.num_quantiles, self.output_patch_size)
            .permute(0, 2, 1, 3)
            .reshape(hidden.shape[0], self.num_quantiles, -1)
            for value in values
        )
        return values[0] if head.use_layer_norm else values


def get_chronos2_stages(model, compile_config, chunk_size=3):
    """Return contiguous encoder jobs and exact full-batch compile interfaces."""
    layers = model.encoder.block
    if type(chunk_size) is not int or chunk_size < 1 or chunk_size > len(layers):
        raise ValueError("chunk_size must be a positive integer no larger than the encoder")
    inputs = compile_config.input_info
    batch, context_patches, _ = inputs[0][1]
    future_patches = inputs[2][1][1]
    tokens = context_patches + int(model.chronos_config.use_reg_token) + future_patches
    hidden = ("hidden", [batch, tokens, model.config.d_model], "float32")
    time_mask = ("attention_mask", [batch, 1, 1, tokens], "float32")
    group_mask = ("group_time_mask", [tokens, 1, batch, batch], "float32")
    jobs: list[tuple[str, nn.Module, list]] = [("embedding", Chronos2Embedding(model), inputs)]
    jobs.extend(
        (f"encoder_{start}", Chronos2Encoder(layers[start : start + chunk_size]), [hidden, time_mask, group_mask])
        for start in range(0, len(layers), chunk_size)
    )
    jobs.append(("forecast", Chronos2Forecast(model, future_patches), [hidden]))
    return [
        (
            job.eval(),
            RBLNCompileConfig(
                compiled_model_name=name,
                input_info=info,
                npu=compile_config.npu,
                num_devices=compile_config.num_devices,
            ),
        )
        for name, job, info in jobs
    ]


class Chronos2SequentialRuntime:
    """Run full-batch learned stages with one stage resident on the NPU at a time."""

    def __init__(self, compiled_models, rbln_config):
        self.compiled_models = compiled_models
        self.names = [cfg.compiled_model_name for cfg in rbln_config.compile_cfgs]
        expected = ["embedding", *[f"encoder_{i}" for i in range(0, 3 * (len(self.names) - 2), 3)], "forecast"]
        if self.names != expected or len(compiled_models) != len(self.names):
            raise ValueError("Expected every contiguous Chronos encoder stage")
        devices = [rbln_config.device_map[name] for name in self.names]
        if any(device != devices[0] for device in devices):
            raise ValueError("Chronos sequential stages must use the same device")
        self.device = devices[0]
        self.activate_profiler = rbln_config.activate_profiler
        self.timeout = rbln_config.timeout

    def _run(self, compiled, inputs):
        runtime = compiled.create_runtime(
            tensor_type="pt",
            device=self.device,
            activate_profiler=self.activate_profiler,
            timeout=self.timeout,
        )
        try:
            output = runtime(*inputs)
            # Runtime-owned storage must not escape beyond the stage lifetime.
            if isinstance(output, (list, tuple)):
                result = tuple(value.detach().cpu().clone() for value in output)
            else:
                result = output.detach().cpu().clone()
            del output
            return result
        finally:
            del runtime
            gc.collect()

    def __call__(self, *inputs):
        hidden, time_mask, group_mask = self._run(self.compiled_models[0], inputs)
        for compiled in self.compiled_models[1:-1]:
            hidden = self._run(compiled, (hidden, time_mask, group_mask))
            if isinstance(hidden, (list, tuple)):
                hidden = hidden[0]
        return self._run(self.compiled_models[-1], (hidden,))
