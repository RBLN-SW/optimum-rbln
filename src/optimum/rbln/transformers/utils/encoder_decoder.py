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

"""Encoder-decoder models, whose encoder writes the cross-attention caches its decoder reads."""

from typing import TYPE_CHECKING

import torch

from ...utils.compiled_model import RBLNCompiledModel
from ...utils.runtime_utils import RBLNRuntime, create_runtimes, open_devices, zeroed_tensors


if TYPE_CHECKING:
    from ...configuration_utils import RBLNModelConfig
    from ...modeling import RBLNModel

COMPILED_MODEL_NAMES = ["encoder", "decoder"]


def is_cache(name: str) -> bool:
    return "key_value_states" in name


@torch.inference_mode()
def compile_encoder_decoder(
    cls: type["RBLNModel"], wrapped_model: torch.nn.Module, rbln_config: "RBLNModelConfig"
) -> dict[str, RBLNCompiledModel]:
    """Compiles the encoder, then the decoder with the caches they share laid out as the encoder
    lays them."""
    enc_compile_config, dec_compile_config = rbln_config.compile_cfgs[:2]
    encoder = cls.compile(
        wrapped_model.encoder,
        enc_compile_config,
        create_runtimes=rbln_config.create_runtimes,
        device=rbln_config.device,
    )
    asked = {a.name: a.type for a in encoder.functions[0].args if is_cache(a.name)}
    decoder = cls.compile(
        wrapped_model.decoder,
        dec_compile_config,
        create_runtimes=rbln_config.create_runtimes,
        device=rbln_config.device,
        asked=asked,
    )
    return {"encoder": encoder, "decoder": decoder}


def create_encoder_decoder_runtimes(
    cls: type["RBLNModel"], compiled_models: list[RBLNCompiledModel], rbln_config: "RBLNModelConfig"
) -> list[RBLNRuntime]:
    """Runtimes of the encoder and the decoder, which bind one tensor of each cache."""
    if any(model_name not in rbln_config.device_map for model_name in COMPILED_MODEL_NAMES):
        cls._raise_missing_compiled_file_error(COMPILED_MODEL_NAMES)
    decoder = compiled_models[1].functions[0]
    devices = open_devices(rbln_config.device_map["decoder"], decoder.num_devices, decoder.npu)
    caches = zeroed_tensors(decoder, [a.name for a in decoder.args if is_cache(a.name)], devices)
    return create_runtimes(
        compiled_models, [rbln_config.device_map[name] for name in COMPILED_MODEL_NAMES], tensors=caches
    )
