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

from typing import TYPE_CHECKING

from transformers.utils import _LazyModule

from ...utils.import_utils import define_import_structure


if TYPE_CHECKING:
    from .audio_spectrogram_transformer import *
    from .auto import *
    from .bart import *
    from .bert import *
    from .blip_2 import *
    from .clip import *
    from .colpali import *
    from .colqwen2 import *
    from .decoderonly import *
    from .depth_anything import *
    from .detr import *
    from .distilbert import *
    from .dpt import *
    from .exaone import *
    from .exaone4_5 import *
    from .gemma import *
    from .gemma2 import *
    from .gemma3 import *
    from .gemma4 import *
    from .gpt2 import *
    from .gpt_oss import *
    from .grounding_dino import *
    from .idefics3 import *
    from .llama import *
    from .llava import *
    from .llava_next import *
    from .midm import *
    from .mistral import *
    from .mixtral import *
    from .modernbert import *
    from .opt import *
    from .paligemma import *
    from .pegasus import *
    from .phi import *
    from .pixtral import *
    from .qwen2 import *
    from .qwen2_5_vl import *
    from .qwen2_moe import *
    from .qwen2_vl import *
    from .qwen3 import *
    from .qwen3_5 import *
    from .qwen3_moe import *
    from .qwen3_vl import *
    from .qwen3_vl_moe import *
    from .resnet import *
    from .roberta import *
    from .siglip import *
    from .swin import *
    from .t5 import *
    from .time_series_transformer import *
    from .vit import *
    from .wav2vec2 import *
    from .whisper import *
    from .xlm_roberta import *
else:
    import sys

    _file = globals()["__file__"]
    sys.modules[__name__] = _LazyModule(__name__, _file, define_import_structure(_file), module_spec=__spec__)
