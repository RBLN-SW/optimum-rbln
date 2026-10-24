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

from copy import deepcopy
from threading import Lock, RLock
from weakref import WeakValueDictionary

import torch
from rebel.compile_context import CompileContext
from torch import nn
from transformers.image_utils import SizeDict

from ....configuration_utils import RBLNCompileConfig
from ...modeling_generic import RBLNImageModel, RBLNModelForImageClassification


class Dinov2EmbeddingsWrapper(nn.Module):
    def __init__(self, embeddings, image_size):
        super().__init__()
        self.embeddings = embeddings
        height, width = (image_size, image_size) if isinstance(image_size, int) else image_size
        patches = (height // embeddings.patch_size) * (width // embeddings.patch_size)
        dummy = embeddings.position_embeddings.new_empty(1, patches + 1, embeddings.position_embeddings.shape[-1])
        with torch.no_grad():
            position = embeddings.interpolate_pos_encoding(dummy, height, width).detach().clone()
        self.register_buffer("position", position)

    def forward(self, pixel_values, bool_masked_pos=None):
        source = self.embeddings
        embeddings = source.patch_embeddings(pixel_values.to(dtype=source.patch_embeddings.projection.weight.dtype))
        if bool_masked_pos is not None and source.use_mask_token:
            embeddings = torch.where(
                bool_masked_pos.unsqueeze(-1), source.mask_token.to(embeddings.dtype).unsqueeze(0), embeddings
            )
        cls_tokens = source.cls_token.expand(pixel_values.shape[0], -1, -1)
        return source.dropout(torch.cat((cls_tokens, embeddings), dim=1) + self.position)


class Dinov2Wrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, pixel_values):
        return self.model(
            pixel_values=pixel_values,
            output_hidden_states=False,
            output_attentions=False,
            return_dict=False,
        )


class Dinov2EmbeddingJob(nn.Module):
    def __init__(self, embeddings):
        super().__init__()
        self.embeddings = embeddings

    def forward(self, pixel_values, activation):
        return torch.ops.rbln_custom_ops.rbln_cache_update(
            activation,
            self.embeddings(pixel_values),
            torch.tensor(0, dtype=torch.int16),
            torch.tensor(0, dtype=torch.int16),
        )


class Dinov2EncoderJob(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.layers = nn.ModuleList(layers)

    def forward(self, activation, position):
        hidden = activation
        for layer in self.layers:
            hidden = layer(hidden)
        return torch.ops.rbln_custom_ops.rbln_cache_update(
            activation, hidden, position[0], torch.tensor(0, dtype=torch.int16)
        )


class Dinov2TailJob(nn.Module):
    def __init__(self, layernorm, classifier=None):
        super().__init__()
        self.layernorm = layernorm
        self.classifier = classifier

    def forward(self, activation, scale):
        hidden = self.layernorm(activation * scale[0])
        pooled = hidden[:, 0]
        if self.classifier is not None:
            return (self.classifier(torch.cat((pooled, hidden[:, 1:].mean(dim=1)), dim=-1)),)
        return hidden, pooled


# Reloads (including copies in different directories) can share the compiler session's
# device buffers. Keep creation and whole synchronous requests mutually exclusive.
_DINOV2_LOCKS = WeakValueDictionary()
_DINOV2_LOCKS_GUARD = Lock()


def _shared_job_lock(session, devices):
    key = (session, devices)
    with _DINOV2_LOCKS_GUARD:
        lock = _DINOV2_LOCKS.get(key)
        if lock is None:
            lock = RLock()
            _DINOV2_LOCKS[key] = lock
        return lock


def _job_devices(device, num_devices):
    # Match RBLNCompiledModel._default_args, including the default multi-device placement.
    if device is None or device == 0:
        return tuple(range(num_devices))
    devices = (device,) if isinstance(device, int) else tuple(device)
    if devices and all(d < 0 for d in devices) and len(devices) != num_devices:
        devices = (devices[0],) * num_devices
    return devices


class Dinov2RuntimeChain:
    def __init__(self, runtimes, lock):
        self.runtimes = runtimes
        self.lock = lock
        self.position = torch.zeros(1, dtype=torch.int16)
        self.scale = torch.ones(1, dtype=torch.float32)

    def __call__(self, pixel_values):
        with self.lock:
            self.runtimes[0](pixel_values)
            for runtime in self.runtimes[1:-1]:
                runtime(self.position)
            return self.runtimes[-1](self.scale)


class Dinov2Mixin:
    @classmethod
    def _wrap_model_if_needed(cls, model, rbln_config):
        model = deepcopy(model)
        backbone = getattr(model, "dinov2", model)
        backbone.embeddings = Dinov2EmbeddingsWrapper(
            backbone.embeddings, (rbln_config.image_height, rbln_config.image_width)
        )
        return Dinov2Wrapper(model).eval()

    @classmethod
    def _update_rbln_config(cls, preprocessors, model, model_config, rbln_config):
        if rbln_config.image_size is None:
            for processor in preprocessors or []:
                crop_size = getattr(processor, "crop_size", None)
                if getattr(processor, "do_center_crop", False) and isinstance(crop_size, (dict, SizeDict)):
                    if "height" in crop_size and "width" in crop_size:
                        rbln_config.image_size = (crop_size["height"], crop_size["width"])
                        break
        rbln_config = super()._update_rbln_config(preprocessors, model, model_config, rbln_config)
        chunk = rbln_config.encoder_chunk_size
        if chunk is not None:
            pixels = rbln_config.compile_cfgs[0].input_info
            tokens = (rbln_config.image_height // model_config.patch_size) * (
                rbln_config.image_width // model_config.patch_size
            ) + 1
            activation = ("activation", [rbln_config.batch_size, tokens, model_config.hidden_size], rbln_config.dtype)
            configs = [RBLNCompileConfig(compiled_model_name="embedding", input_info=[*pixels, activation])]
            configs.extend(
                RBLNCompileConfig(
                    compiled_model_name=f"encoder_{start}",
                    input_info=[activation, ("position", [1], "int16")],
                )
                for start in range(0, model_config.num_hidden_layers, chunk)
            )
            configs.append(
                RBLNCompileConfig(compiled_model_name="tail", input_info=[activation, ("scale", [1], "float32")])
            )
            rbln_config.set_compile_cfgs(configs)
        return rbln_config

    @classmethod
    @torch.inference_mode()
    def get_compiled_model(cls, model, rbln_config):
        if len(rbln_config.compile_cfgs) == 1:
            return super().get_compiled_model(model, rbln_config)
        wrapped = cls._wrap_model_if_needed(model, rbln_config).model
        backbone = getattr(wrapped, "dinov2", wrapped)
        layers = backbone.encoder.layer
        jobs = [Dinov2EmbeddingJob(backbone.embeddings)]
        # Saved compile names determine the contiguous layer ranges.
        starts = [int(cfg.compiled_model_name.removeprefix("encoder_")) for cfg in rbln_config.compile_cfgs[1:-1]]
        for start, end in zip(starts, [*starts[1:], len(layers)], strict=True):
            jobs.append(Dinov2EncoderJob(layers[start:end]))
        jobs.append(Dinov2TailJob(backbone.layernorm, getattr(wrapped, "classifier", None)))
        context = CompileContext(use_weight_sharing=False)
        inputs = rbln_config.compile_cfgs[0].get_dummy_inputs(fill=0)
        activation = inputs[-1]
        context.mark_static_address(activation, "activation")
        compiled = {}
        for job, cfg in zip(jobs, rbln_config.compile_cfgs, strict=True):
            inputs = cfg.get_dummy_inputs(fill=0, static_tensors={"activation": activation})
            if cfg.compiled_model_name == "tail":
                inputs[-1].fill_(1)
            compiled[cfg.compiled_model_name] = cls.compile(
                job.eval(),
                cfg,
                create_runtimes=rbln_config.create_runtimes,
                device=rbln_config.device,
                example_inputs=inputs,
                compile_context=context,
            )
        return compiled

    @classmethod
    def _create_runtimes(cls, compiled_models, rbln_config):
        if len(rbln_config.compile_cfgs) <= 1:
            return super()._create_runtimes(compiled_models, rbln_config)
        names = [cfg.compiled_model_name for cfg in rbln_config.compile_cfgs]
        if any(name not in rbln_config.device_map for name in names):
            cls._raise_missing_compiled_file_error(names)
        placements = [
            _job_devices(rbln_config.device_map[cfg.compiled_model_name], cfg.num_devices or 1)
            for cfg in rbln_config.compile_cfgs
        ]
        if any(placement != placements[0] for placement in placements):
            raise ValueError("All segmented DINO jobs must use the same device placement")
        sessions = [model.select_executor().get_context_id() for model in compiled_models]
        if any(session != sessions[0] for session in sessions):
            raise ValueError("All segmented DINO jobs must share one compile session")
        lock = _shared_job_lock(sessions[0], placements[0])
        with lock:
            runtimes = [
                model.create_runtime(
                    tensor_type="pt",
                    device=list(placement),
                    activate_profiler=rbln_config.activate_profiler,
                    timeout=rbln_config.timeout,
                )
                for model, placement in zip(compiled_models, placements, strict=True)
            ]
        return [Dinov2RuntimeChain(runtimes, lock)]


class RBLNDinov2Model(Dinov2Mixin, RBLNImageModel):
    pass


class RBLNDinov2ForImageClassification(Dinov2Mixin, RBLNModelForImageClassification):
    pass


__all__ = ["RBLNDinov2ForImageClassification", "RBLNDinov2Model"]
