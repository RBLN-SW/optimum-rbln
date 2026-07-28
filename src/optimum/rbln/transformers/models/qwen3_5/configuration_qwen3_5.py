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

from typing import Any

from ....configuration_utils import RBLNModelConfig
from ..decoderonly.configuration_decoderonly import RBLNDecoderOnlyModelConfig, RBLNDecoderOnlyModelForCausalLMConfig


class RBLNQwen3_5ForCausalLMConfig(RBLNDecoderOnlyModelForCausalLMConfig):
    """
    Configuration class for RBLN Qwen3.5 (text backbone) causal language models.

    Qwen3.5 is a hybrid decoder: most layers are `linear_attention` (GatedDeltaNet) and a
    minority are `full_attention` (gated softmax attention). Full-attention layers use the
    standard paged KV cache; linear-attention layers instead carry a `conv_state` and a
    `recurrent_state`. Which layers are linear is read directly from the HF `config.layer_types`
    (no RBLN-config field); this config extends `RBLNDecoderOnlyModelForCausalLMConfig` only with
    `gdn_chunk_size`.

    Example usage:
    ```python
    from optimum.rbln import RBLNQwen3_5ForCausalLM, RBLNQwen3_5ForCausalLMConfig

    config = RBLNQwen3_5ForCausalLMConfig(
        batch_size=1,
        max_seq_len=32768,
        tensor_parallel_size=4,
    )
    model = RBLNQwen3_5ForCausalLM.from_pretrained("Qwen/Qwen3.5-27B", export=True, rbln_config=config)
    ```
    """

    def __init__(
        self,
        gdn_chunk_size: int | None = None,
        **kwargs: Any,
    ):
        """
        Args:
            gdn_chunk_size (Optional[int]): GatedDeltaNet prefill sub-chunk size. Each prefill window
                is split into `prefill_chunk_size // gdn_chunk_size` sub-chunks processed by the chunked
                delta rule. Must divide `prefill_chunk_size`. `None` -> `prefill_chunk_size` (no split).
            kwargs: Additional arguments passed to `RBLNDecoderOnlyModelForCausalLMConfig`.
        """
        super().__init__(**kwargs)
        self.gdn_chunk_size = gdn_chunk_size


class RBLNQwen3_5TextModelConfig(RBLNDecoderOnlyModelConfig):
    """
    Configuration class for the bare RBLN Qwen3.5 text backbone (no LM head, text-only).

    Linear-attention layers are read from the HF `config.layer_types`; see
    `RBLNQwen3_5ForCausalLMConfig` for `gdn_chunk_size`.
    """

    def __init__(
        self,
        gdn_chunk_size: int | None = None,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        # GDN prefill sub-chunk size (must divide prefill_chunk_size). None -> = prefill_chunk_size
        # (n_chunks == 1, no sub-chunking). See rbln_chunk_gated_delta_rule.
        self.gdn_chunk_size = gdn_chunk_size


class RBLNQwen3_5VisionModelConfig(RBLNModelConfig):
    """Vision encoder config for Qwen3.5: per-image `max_seq_len`."""

    def __init__(self, max_seq_len: int | list[int] = None, batch_size: int = 1, **kwargs: Any):
        """
        Args:
            max_seq_len (Optional[Union[int, List[int]]]): Vision Transformer attention max sequence
                length(s) = number of (merged) patches per image/video. RBLN runs inference per image, so
                set this to the max expected resolution to bound compute. Required.
            batch_size (int): the vision encoder runs one image at a time (the parent config forces this
                by default).
            kwargs: Additional arguments passed to the parent RBLNModelConfig.

        Raises:
            ValueError: If `max_seq_len` is None or not provided, or if `batch_size` is not 1.
        """
        super().__init__(**kwargs)

        if batch_size != 1:
            raise ValueError(f"The Qwen3.5 vision encoder only supports batch_size=1, got {batch_size}.")
        self.batch_size = batch_size

        if max_seq_len is not None:
            if isinstance(max_seq_len, int):
                max_seq_len = [max_seq_len]
            elif isinstance(max_seq_len, list):
                max_seq_len.sort(reverse=True)
        else:
            raise ValueError("'max_seq_len' must be specified.")

        self.max_seq_len = max_seq_len


class RBLNQwen3_5ModelConfig(RBLNDecoderOnlyModelConfig):
    """
    Configuration for the bare Qwen3.5 model (vision encoder + hybrid text, no LM head).

    Qwen3.5 is natively vision-language, so this is the multimodal model config. Independent of the
    Qwen3-VL config (inherits `RBLNDecoderOnlyModelConfig` directly), carrying its own `visual`
    submodule handling plus the Qwen3.5-specific `gdn_chunk_size`. Which layers are linear is read
    from the HF `config.text_config.layer_types`. The vision encoder output is injected into
    `inputs_embeds` (`use_inputs_embeds=True`).
    """

    submodules = ["visual"]
    subclass_non_save_attributes = ["_load_visual_runtime", "memory_budget"]

    def __init__(
        self,
        gdn_chunk_size: int | None = None,
        visual: RBLNModelConfig | None = None,
        _load_visual_runtime: bool = True,
        **kwargs: Any,
    ):
        """
        Args:
            gdn_chunk_size (Optional[int]): GatedDeltaNet prefill sub-chunk size (must divide
                `prefill_chunk_size`). `None` -> `prefill_chunk_size` (no split). See rbln_chunk_gated_delta_rule.
            visual (Optional[RBLNModelConfig]): Configuration for the vision encoder submodule.
            _load_visual_runtime (bool): Whether to create the visual encoder runtime (False on
                decoder-only nodes in a disaggregated setup). Defaults to True.
            kwargs: Additional arguments passed to `RBLNDecoderOnlyModelConfig`.

        Raises:
            ValueError: If `use_inputs_embeds` is False.
        """
        super().__init__(**kwargs)
        if not getattr(self, "use_inputs_embeds", True):
            raise ValueError(
                "RBLNQwen3_5ModelConfig requires use_inputs_embeds=True. "
                "The visual encoder output must be injected into inputs_embeds."
            )
        # The vision encoder runs one image at a time, so force batch_size=1 on the submodule.
        self.visual = self.initialize_submodule_config(submodule_config=visual, force_kwargs=True, batch_size=1)
        self._load_visual_runtime = _load_visual_runtime
        self.gdn_chunk_size = gdn_chunk_size


class RBLNQwen3_5ForConditionalGenerationConfig(RBLNDecoderOnlyModelForCausalLMConfig):
    """
    Configuration for `RBLNQwen3_5ForConditionalGeneration` (vision-language).

    Qwen3.5 pairs a Qwen3-VL-style vision encoder (no deepstack) with the hybrid Qwen3.5
    text backbone (`linear_attention` GatedDeltaNet layers + `full_attention` gated layers).
    The vision encoder output is injected into `inputs_embeds` (`use_inputs_embeds=True`).
    Which layers are linear is read from the HF `config.text_config.layer_types` (no RBLN-config field).

    Independent of the Qwen3-VL config: inherits `RBLNDecoderOnlyModelForCausalLMConfig` directly
    (like the Qwen3-VL config does), carrying its own `visual` submodule handling plus the
    Qwen3.5-specific `gdn_chunk_size`.

    Example usage:
    ```python
    from optimum.rbln import RBLNQwen3_5ForConditionalGeneration

    model = RBLNQwen3_5ForConditionalGeneration.from_pretrained(
        "Qwen/Qwen3.5-...", export=True,
        rbln_config={"max_seq_len": 32768, "tensor_parallel_size": 4, "visual": {"max_seq_len": 6400}},
    )
    ```
    """

    submodules = ["visual"]
    subclass_non_save_attributes = ["_load_visual_runtime", "memory_budget"]

    def __init__(
        self,
        gdn_chunk_size: int | None = None,
        use_inputs_embeds: bool = True,
        visual: RBLNModelConfig | None = None,
        _load_visual_runtime: bool = True,
        **kwargs: Any,
    ):
        """
        Args:
            gdn_chunk_size (Optional[int]): GatedDeltaNet prefill sub-chunk size. Each prefill window is
                split into `prefill_chunk_size // gdn_chunk_size` sub-chunks. Must divide
                `prefill_chunk_size`. `None` -> `prefill_chunk_size` (no split). See rbln_chunk_gated_delta_rule.
            use_inputs_embeds (bool): Must be True — the vision encoder output is injected into inputs_embeds.
            visual (Optional[RBLNModelConfig]): Configuration for the vision encoder submodule.
            _load_visual_runtime (bool): Whether to create the visual encoder runtime. Set False on
                decoder-only nodes in a disaggregated setup (then pre-computed image_embeds must be fed to
                forward()). Defaults to True.
            kwargs: Additional arguments passed to `RBLNDecoderOnlyModelForCausalLMConfig`.

        Raises:
            ValueError: If `use_inputs_embeds` is False.
        """
        super().__init__(use_inputs_embeds=use_inputs_embeds, **kwargs)
        if not self.use_inputs_embeds:
            raise ValueError(
                "RBLNQwen3_5ForConditionalGenerationConfig requires use_inputs_embeds=True. "
                "The visual encoder output must be injected into inputs_embeds."
            )
        # The vision encoder runs one image at a time, so force batch_size=1 on the submodule.
        self.visual = self.initialize_submodule_config(submodule_config=visual, force_kwargs=True, batch_size=1)
        self._load_visual_runtime = _load_visual_runtime
        # GDN prefill sub-chunk size (must divide prefill_chunk_size). None -> = prefill_chunk_size
        # (n_chunks == 1, no sub-chunking). See rbln_chunk_gated_delta_rule.
        self.gdn_chunk_size = gdn_chunk_size
