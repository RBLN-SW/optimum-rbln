# Copyright 2026 Rebellions Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import inspect
from typing import TYPE_CHECKING, Any, Callable

import torch
from transformers import AutoModelForImageTextToText, PreTrainedModel, Qwen3VLMoeConfig
from transformers.initialization import no_init_weights
from transformers.models.qwen3_vl_moe.modeling_qwen3_vl_moe import (
    Qwen3VLMoeModel,
    Qwen3VLMoeTextRotaryEmbedding,
    Qwen3VLMoeVisionModel,
    Qwen3VLMoeVisionPatchEmbed,
    Qwen3VLMoeVisionRotaryEmbedding,
)

from ..decoderonly.decoderonly_runtime_utils import RBLNPageTableManager, RBLNRuntimeModel
from ..qwen3_vl.modeling_qwen3_vl import (
    RBLNQwen3VLForConditionalGeneration,
    RBLNQwen3VLModel,
    RBLNQwen3VLVisionModel,
)
from .configuration_qwen3_vl_moe import RBLNQwen3VLMoeVisionModelConfig
from .qwen3_vl_moe_architecture import Qwen3VLMoe_LanguageModelWrapper, Qwen3VLMoeVisionModelWrapper
from .qwen3_vl_moe_runtime_utils import RBLNQwen3VLMoeRuntimeModel


if TYPE_CHECKING:
    pass


class RBLNQwen3VLMoeVisionModel(RBLNQwen3VLVisionModel):
    def __post_init__(self, **kwargs):
        self.transformer = self.model[0]
        self.max_seq_lens = torch.tensor(sorted(self.rbln_config.max_seq_lens, reverse=False))
        config = self.config
        self.patch_size = config.patch_size
        self.spatial_merge_size = config.spatial_merge_size
        self.spatial_merge_unit = config.spatial_merge_size * config.spatial_merge_size

        head_dim = config.hidden_size // config.num_heads
        self.rotary_pos_emb = Qwen3VLMoeVisionRotaryEmbedding(head_dim // 2)
        self.deepstack_visual_indexes = config.deepstack_visual_indexes

        with no_init_weights():
            self.patch_embed = Qwen3VLMoeVisionPatchEmbed(config=config)
            self.pos_embed = torch.nn.Embedding(config.num_position_embeddings, config.hidden_size)

        self.num_grid_per_side = int(config.num_position_embeddings**0.5)

        artifacts = torch.load(self.model_save_dir / self.subfolder / "torch_artifacts.pth", weights_only=False)
        self.patch_embed.load_state_dict(artifacts["patch_embed"])
        self.pos_embed.load_state_dict(artifacts["pos_embed"])

    @classmethod
    def _wrap_model_if_needed(cls, model: "PreTrainedModel", rbln_config: RBLNQwen3VLMoeVisionModelConfig):
        return Qwen3VLMoeVisionModelWrapper(model, rbln_config).eval()

    def __getattr__(self, __name: str) -> Any:
        def redirect(func):
            return lambda *pargs, **kwargs: func(self, *pargs, **kwargs)

        val = getattr(Qwen3VLMoeVisionModel, __name)
        if isinstance(val, Callable) and "self" in set(inspect.signature(val).parameters):
            return redirect(val)
        return val


class RBLNQwen3VLMoeModel(RBLNQwen3VLModel):
    auto_model_class = AutoModelForImageTextToText
    _decoder_wrapper_cls = Qwen3VLMoe_LanguageModelWrapper
    _use_rotary_emb = False
    _rbln_submodule_prefix = "model"
    _rbln_submodules = [{"name": "visual"}]
    _config_class = Qwen3VLMoeConfig
    _rotary_emb_class = Qwen3VLMoeTextRotaryEmbedding
    _get_rope_index_func = Qwen3VLMoeModel.get_rope_index

    def setup_runtime(self):
        page_table_manager = RBLNPageTableManager(self.rbln_config)
        if self.rbln_config.use_position_ids:
            dec_attn_mask = torch.zeros(self.rbln_config.batch_size, self.rbln_config.max_seq_len, dtype=self.dtype)
        else:
            dec_attn_mask = torch.zeros(
                self.rbln_config.batch_size, 1, 1, self.rbln_config.max_seq_len, dtype=self.dtype
            )

        common_kwargs = {
            "main_input_name": "inputs_embeds" if self.rbln_config.use_inputs_embeds else "input_ids",
            "embed_tokens": self.embed_tokens,
            "dec_attn_mask": dec_attn_mask,
            "page_table_manager": page_table_manager,
            "rbln_config": self.rbln_config,
            "config": self.config.text_config,
        }

        self.prefill_decoder = RBLNQwen3VLMoeRuntimeModel(
            runtime=self.model[0],
            phase="prefill",
            batch_size=self.rbln_config.batch_size,
            logits_last_dim=self.logits_last_dim,
            num_deepstack_layers=self.num_deepstack_layers,
            **common_kwargs,
        )

        if self.can_generate():
            self.decoders = {}
            for i, batch_size in enumerate(self.rbln_config.decoder_batch_sizes):
                self.decoders[batch_size] = RBLNRuntimeModel(
                    runtime=self.model[i + 1],
                    phase="decode",
                    batch_size=batch_size,
                    **common_kwargs,
                )
            self.decoder = self.decoders[self.rbln_config.batch_size]


class RBLNQwen3VLMoeForConditionalGeneration(RBLNQwen3VLForConditionalGeneration):
    auto_model_class = AutoModelForImageTextToText
    _decoder_wrapper_cls = Qwen3VLMoe_LanguageModelWrapper
    _supports_non_fp32 = True
    _use_rotary_emb = False
    _rbln_submodule_prefix = "model"
    _rbln_submodules = [{"name": "visual"}]
