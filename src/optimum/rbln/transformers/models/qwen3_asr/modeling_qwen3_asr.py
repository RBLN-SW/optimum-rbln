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

import math
from typing import TYPE_CHECKING, Any, Union

import torch
from transformers import AutoModelForMultimodalLM, PretrainedConfig, PreTrainedModel
from transformers.models.qwen3_asr.feature_extraction_qwen3_asr import Qwen3ASRFeatureExtractor
from transformers.models.qwen3_asr.modeling_qwen3_asr import Qwen3ASREncoder

from ....configuration_utils import RBLNCompileConfig, RBLNModelConfig
from ....modeling import RBLNModel
from ..qwen3.modeling_qwen3 import RBLNQwen3ForCausalLM
from .configuration_qwen3_asr import RBLNQwen3ASREncoderConfig
from .qwen3_asr_architecture import Qwen3ASREncoderWrapper, Qwen3ASRLanguageModelWrapper, get_window_size


if TYPE_CHECKING:
    from transformers import AutoFeatureExtractor, AutoProcessor, AutoTokenizer


# Windows needed to cover one feature-extractor chunk (30s -> 390 tokens -> 4 windows).
def _default_num_windows(
    model_config: "PretrainedConfig",
    preprocessors: Union["AutoFeatureExtractor", "AutoProcessor", "AutoTokenizer"] | None,
) -> int:
    feature_extractor = None
    for preprocessor in preprocessors or []:
        candidate = getattr(preprocessor, "feature_extractor", preprocessor)
        if hasattr(candidate, "chunk_length"):
            feature_extractor = candidate
            break
    # Often exported without a preprocessor; let the feature extractor state its own defaults.
    if feature_extractor is None:
        feature_extractor = Qwen3ASRFeatureExtractor()

    frames_per_second = feature_extractor.sampling_rate / feature_extractor.hop_length
    frames_per_chunk = model_config.n_window * 2
    num_tokens = (
        math.ceil(feature_extractor.chunk_length * frames_per_second / frames_per_chunk)
        * model_config.max_position_embeddings
    )
    return math.ceil(num_tokens / get_window_size(model_config))


class RBLNQwen3ASREncoder(RBLNModel):
    """
    RBLN-optimized Qwen3-ASR audio encoder.

    This model inherits from [`RBLNModel`]. It implements the methods to convert and run
    pre-trained transformers based Qwen3-ASR audio encoder on RBLN devices by:

    - transferring the checkpoint weights of the original into an optimized RBLN graph,
    - compiling the resulting graph using the RBLN compiler.

    Converts log-mel audio features into audio embeddings in the language model's embedding
    space.
    """

    auto_model_class = None

    @property
    def chunk_len(self) -> int:
        return self.config.n_window * 2

    @property
    def window_size(self) -> int:
        return get_window_size(self.config)

    @property
    def chunks_per_window(self) -> int:
        return self.config.n_window_infer // self.chunk_len

    @classmethod
    def _wrap_model_if_needed(cls, model: "PreTrainedModel", rbln_config: RBLNQwen3ASREncoderConfig):
        return Qwen3ASREncoderWrapper(model, rbln_config).eval()

    @classmethod
    def _update_rbln_config(
        cls,
        preprocessors: Union["AutoFeatureExtractor", "AutoProcessor", "AutoTokenizer"] | None = None,
        model: PreTrainedModel | None = None,
        model_config: PretrainedConfig | None = None,
        rbln_config: RBLNQwen3ASREncoderConfig | None = None,
    ) -> RBLNQwen3ASREncoderConfig:
        if rbln_config.num_windows is None:
            rbln_config.num_windows = _default_num_windows(model_config, preprocessors)

        if rbln_config.num_windows < 1:
            raise ValueError(
                f"`num_windows` must be at least 1, got {rbln_config.num_windows}. It is the number of "
                f"{get_window_size(model_config)}-token attention windows the encoder graph handles per pass."
            )

        window = get_window_size(model_config)
        chunk_len = model_config.n_window * 2
        chunks_per_window = model_config.n_window_infer // chunk_len
        tokens_per_chunk = int(Qwen3ASREncoder._post_cnn_length(torch.tensor(chunk_len)))
        if chunks_per_window * tokens_per_chunk != window:
            raise ValueError(
                f"A window is {window} tokens but {chunks_per_window} chunks of "
                f"{tokens_per_chunk} post-convolution tokens is {chunks_per_window * tokens_per_chunk}. "
                "The graph reshapes chunks into windows, so the two must agree: "
                "`max_position_embeddings` has to equal the post-convolution length of an "
                "`n_window * 2` frame chunk."
            )

        input_info = [
            (
                "input_features",
                [rbln_config.num_windows * chunks_per_window, 1, model_config.num_mel_bins, chunk_len],
                rbln_config.dtype,
            ),
            ("attn_mask", [rbln_config.num_windows, 1, 1, window], rbln_config.dtype),
        ]
        rbln_config.set_compile_cfgs([RBLNCompileConfig(input_info=input_info)])
        return rbln_config

    def forward(self, input_features: torch.Tensor, input_features_mask: torch.Tensor) -> torch.Tensor:
        batch_size, num_mel_bins, padded_feature_length = input_features.shape
        if padded_feature_length % self.chunk_len != 0:
            raise ValueError(
                f"`padded_feature_length` ({padded_feature_length}) must be a multiple of "
                f"{self.chunk_len} mel frames; use the matching `Qwen3ASRFeatureExtractor`."
            )

        # One chunk per item of the convolution's batch, exactly as HF reshapes it.
        num_chunks = padded_feature_length // self.chunk_len
        chunked = (
            input_features.view(batch_size, num_mel_bins, num_chunks, self.chunk_len)
            .permute(0, 2, 1, 3)
            .reshape(batch_size * num_chunks, 1, num_mel_bins, self.chunk_len)
        )
        # The graph keeps every chunk's padding, so this decides which output rows survive.
        chunk_lengths = (
            input_features_mask.view(batch_size, num_chunks, self.chunk_len).sum(dim=-1).reshape(-1).to(torch.long)
        )
        valid_per_chunk = Qwen3ASREncoder._post_cnn_length(chunk_lengths)

        # Windows never span two samples, so each sample's chunks are padded up to a whole number
        # of windows before they are laid end to end.
        windows_per_sample = math.ceil(num_chunks / self.chunks_per_window)
        padded_chunks = windows_per_sample * self.chunks_per_window
        tokens_per_chunk = self.window_size // self.chunks_per_window

        chunk_buffer = chunked.new_zeros(batch_size * padded_chunks, 1, num_mel_bins, self.chunk_len)
        valid_buffer = torch.zeros(batch_size * padded_chunks, dtype=torch.long)
        for b in range(batch_size):
            src, dst = b * num_chunks, b * padded_chunks
            chunk_buffer[dst : dst + num_chunks] = chunked[src : src + num_chunks]
            valid_buffer[dst : dst + num_chunks] = valid_per_chunk[src : src + num_chunks]

        valid_mask = (torch.arange(tokens_per_chunk) < valid_buffer.unsqueeze(-1)).view(-1, self.window_size)

        # Windows are independent, so more of them than the graph holds just run in more passes.
        chunks_per_run = self.rbln_config.num_windows * self.chunks_per_window
        outputs = []
        for base in range(0, chunk_buffer.shape[0], chunks_per_run):
            run_chunks = chunk_buffer[base : base + chunks_per_run]
            run_mask = valid_mask[base // self.chunks_per_window : (base + chunks_per_run) // self.chunks_per_window]
            if run_chunks.shape[0] < chunks_per_run:
                pad = chunks_per_run - run_chunks.shape[0]
                run_chunks = torch.cat([run_chunks, run_chunks.new_zeros(pad, 1, num_mel_bins, self.chunk_len)])
                run_mask = torch.cat([run_mask, run_mask.new_zeros(pad // self.chunks_per_window, self.window_size)])

            # A fully padded window would leave softmax with an all-masked row; keeping one key
            # alive avoids the NaN, and `valid_mask` still discards the row afterwards.
            attn_mask = run_mask.clone()
            attn_mask[~attn_mask.any(dim=-1), 0] = True

            output = self.model[0](
                run_chunks.to(self.rbln_config.dtype),
                attn_mask.to(self.rbln_config.dtype)[:, None, None, :],
            )
            outputs.append(output[run_mask])

        return torch.cat(outputs, dim=0)


class RBLNQwen3ASRForConditionalGeneration(RBLNQwen3ForCausalLM):
    """
    RBLNQwen3ASRForConditionalGeneration is a speech recognition model that pairs a Qwen3-ASR audio
    encoder with a Qwen3 text decoder, optimized for RBLN NPUs.

    This model inherits from [`RBLNQwen3ForCausalLM`]. Check the superclass documentation for the generic methods the library implements for all its models.

    **Configuration:**
    This model uses [`RBLNQwen3ASRForConditionalGenerationConfig`] for configuration. When calling methods like `from_pretrained` or `from_model`,
    the `rbln_config` parameter should be an instance of [`RBLNQwen3ASRForConditionalGenerationConfig`] or a dictionary conforming to its structure.

    Examples:
        ```python
        from optimum.rbln import RBLNQwen3ASRForConditionalGeneration

        model = RBLNQwen3ASRForConditionalGeneration.from_pretrained(
            "Qwen/Qwen3-ASR-1.7B-hf",
            export=True,
            rbln_config={
                "max_seq_len": 8192,
                "batch_size": 1,
                "audio_tower": {"num_windows": 4},
            },
        )
        ```
    """

    auto_model_class = AutoModelForMultimodalLM
    _decoder_wrapper_cls = Qwen3ASRLanguageModelWrapper
    _rbln_submodules = [{"name": "audio_tower"}]
    _batch_sortable_kwargs = RBLNQwen3ForCausalLM._batch_sortable_kwargs + (
        "input_features",
        "input_features_mask",
    )

    def __post_init__(self, **kwargs: Any):
        super().__post_init__(**kwargs)
        self.audio_tower = self.rbln_submodules[0]

    @classmethod
    def _reconstruct_model_if_needed(cls, model: "PreTrainedModel"):
        # HF keeps the projector beside the audio tower but only ever runs the two together;
        # hand it over so they compile as one graph.
        model.model.audio_tower.multi_modal_projector = model.model.multi_modal_projector
        return model

    @classmethod
    def _update_submodule_rbln_config(
        cls,
        submodule_name: str,
        submodule_cls: type["RBLNModel"],
        model: "PreTrainedModel",
        submodule_config: "PretrainedConfig",
        submodule_rbln_config: RBLNModelConfig,
        preprocessors: Union["AutoFeatureExtractor", "AutoProcessor", "AutoTokenizer"] | None,
    ) -> RBLNModelConfig:
        if submodule_name == "audio_tower" and submodule_rbln_config.num_windows is None:
            submodule_rbln_config.num_windows = _default_num_windows(submodule_config, preprocessors)
        return submodule_rbln_config

    def _merge_audio_embeds(
        self,
        input_ids: torch.LongTensor,
        input_features: torch.Tensor,
        input_features_mask: torch.Tensor,
    ) -> torch.Tensor:
        inputs_embeds = self.embed_tokens(input_ids)
        audio_embeds = self.audio_tower(input_features, input_features_mask)

        audio_mask = input_ids == self.config.audio_token_id
        num_audio_tokens = int(audio_mask.sum())
        if num_audio_tokens != audio_embeds.shape[0]:
            raise ValueError(
                f"Audio placeholder tokens ({num_audio_tokens}) do not match the number of audio features "
                f"({audio_embeds.shape[0]}). Make sure the inputs come from the matching `Qwen3ASRProcessor`."
            )

        return inputs_embeds.masked_scatter(
            audio_mask.unsqueeze(-1).expand_as(inputs_embeds), audio_embeds.to(inputs_embeds.dtype)
        )

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        cache_position: torch.Tensor | None = None,
        input_features: torch.Tensor | None = None,
        input_features_mask: torch.Tensor | None = None,
        **kwargs: Any,
    ):
        # Prefill is the only step that sees audio; decode runs on `inputs_embeds` alone.
        if cache_position is None and input_features is not None:
            inputs_embeds = self._merge_audio_embeds(input_ids, input_features, input_features_mask)

        return super().forward(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            **kwargs,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.LongTensor,
        generate_idx: torch.Tensor | None = None,
        attention_mask: torch.LongTensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        padded_cache_lengths: torch.Tensor | None = None,
        inputs_sorted: bool = False,
        input_features: torch.Tensor | None = None,
        input_features_mask: torch.Tensor | None = None,
        **kwargs: Any,
    ):
        model_inputs = super().prepare_inputs_for_generation(
            input_ids,
            generate_idx=generate_idx,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            padded_cache_lengths=padded_cache_lengths,
            inputs_sorted=inputs_sorted,
            **kwargs,
        )
        if model_inputs.get("cache_position") is None:
            model_inputs.update(input_features=input_features, input_features_mask=input_features_mask)
        return model_inputs


__all__ = [
    "RBLNQwen3ASREncoder",
    "RBLNQwen3ASRForConditionalGeneration",
]
