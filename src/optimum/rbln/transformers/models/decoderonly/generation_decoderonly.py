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

from typing import TYPE_CHECKING, Any

import torch
from transformers import GenerationConfig
from transformers.generation.utils import GenerationMixin
from transformers.modeling_outputs import ModelOutput


if TYPE_CHECKING:
    from ...modeling_outputs import RBLNDecoderOnlyOutput


def _expand_batch_perm_idx(perm_idx: torch.Tensor, num_rows: int) -> torch.Tensor:
    # HF expands rows per sample via repeat_interleave (num_return_sequences), so the
    # permutation expands group-wise.
    batch_size = perm_idx.shape[0]
    n = num_rows // batch_size
    if n == 1:
        return perm_idx
    return (perm_idx[:, None] * n + torch.arange(n, device=perm_idx.device)).reshape(-1)


class RBLNDecoderOnlyGenerationMixin(GenerationMixin):
    _supports_cache_class = False  # Needed for GenerationMixin
    _is_stateful = False  # Needed for GenerationMixin
    _generate_batch_sortable_kwargs = ("attention_mask", "inputs_embeds", "token_type_ids", "lora_int_ids")

    def _reorder_cache(self, past_key_values, beam_idx):
        raise NotImplementedError

    @staticmethod
    def _unsort_generation_outputs(
        outputs: ModelOutput | torch.Tensor, unsort_idx: torch.Tensor
    ) -> ModelOutput | torch.Tensor:
        if isinstance(outputs, torch.Tensor):
            return outputs.index_select(0, _expand_batch_perm_idx(unsort_idx, outputs.shape[0]))

        num_rows = outputs.sequences.shape[0]
        idx = _expand_batch_perm_idx(unsort_idx, num_rows)

        def _unsort(value):
            if value is None:
                return None
            if isinstance(value, (tuple, list)):
                reordered = [_unsort(v) for v in value]
                return tuple(reordered) if isinstance(value, tuple) else reordered
            if isinstance(value, torch.Tensor) and value.dim() >= 1 and value.shape[0] == num_rows:
                return value.index_select(0, idx)
            return value

        for field in ("sequences", "scores", "logits", "attentions", "hidden_states"):
            value = getattr(outputs, field, None)
            if value is not None:
                setattr(outputs, field, _unsort(value))
        return outputs

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.LongTensor,
        generate_idx: torch.Tensor | None = None,
        attention_mask: torch.LongTensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        padded_cache_lengths: torch.Tensor | None = None,
        inputs_sorted: bool = False,
        **kwargs,
    ):
        # use_batch_attn_opt needs length-sorted batches: generate() sorts once upfront
        # (inputs_sorted=True); a bare forward() call sorts/unsorts per call instead.
        model_inputs = {}
        is_prefill_phase = generate_idx is None

        if is_prefill_phase:
            if attention_mask is not None:
                generate_idx = attention_mask.sum(dim=-1, keepdim=True).int()
            else:
                base = input_ids if input_ids is not None else inputs_embeds
                generate_idx = torch.full((base.shape[0], 1), base.shape[1], dtype=torch.int32, device=base.device)
            padded_cache_lengths = torch.zeros_like(generate_idx)
            cache_position = None
            position_ids = None
        else:
            if inputs_embeds is not None:
                # if `inputs_embeds` are passed, only use them in the 1st generation step for every prompt.
                inputs_embeds = None

            input_ids = input_ids[:, -1:]
            position_ids = generate_idx
            cache_position = generate_idx + padded_cache_lengths if padded_cache_lengths is not None else generate_idx
            generate_idx = generate_idx + 1
            model_inputs.update({"input_ids": input_ids})

        if inputs_embeds is not None:
            if self.rbln_config.use_inputs_embeds:
                model_inputs.update({"inputs_embeds": inputs_embeds})
            else:
                raise ValueError(
                    "The specifying inputs_embeds is only supported when using a compiled RBLN model with 'rbln_use_inputs_embeds' set to True."
                )
        else:
            model_inputs.update({"input_ids": input_ids})

        model_inputs.update(
            {
                "attention_mask": attention_mask,
                "cache_position": cache_position,
                "generate_idx": generate_idx,
                "position_ids": position_ids,
                "padded_cache_lengths": padded_cache_lengths,
                "inputs_sorted": inputs_sorted,
            }
        )

        return model_inputs

    def _update_model_kwargs_for_generation(
        self, outputs: "RBLNDecoderOnlyOutput", model_kwargs: dict[str, Any], **kwargs
    ) -> dict[str, Any]:
        # update generate_idx
        model_kwargs["generate_idx"] = outputs.generate_idx
        model_kwargs["padded_cache_lengths"] = outputs.padded_cache_lengths
        return model_kwargs

    def generate(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.LongTensor | None = None,
        generation_config: GenerationConfig | None = None,
        **kwargs,
    ) -> ModelOutput | torch.LongTensor:
        """
        The generate function is utilized in its standard form as in the HuggingFace transformers library. User can use this function to generate text from the model.
        Check the [HuggingFace transformers documentation](https://huggingface.co/docs/transformers/v4.57.1/en/main_classes/text_generation#transformers.GenerationMixin.generate) for more details.

        Args:
            input_ids (torch.LongTensor): The input ids to the model.
            attention_mask (torch.LongTensor, optional): The attention mask to the model.
            generation_config (GenerationConfig, optional): The generation configuration to be used as base parametrization for the generation call. **kwargs passed to generate matching the attributes of generation_config will override them.
                If generation_config is not provided, the default will be used, which had the following loading priority: 1) from the generation_config.json model file, if it exists; 2) from the model configuration.
                Please note that unspecified parameters will inherit [GenerationConfig](https://huggingface.co/docs/transformers/v4.57.1/en/main_classes/text_generation#transformers.GenerationConfig)’s default values.
            kwargs (dict[str, Any], optional): Additional arguments passed to the generate function. See the HuggingFace transformers documentation for more details.

        Returns:
            A ModelOutput (if return_dict_in_generate=True or when config.return_dict_in_generate=True) or a torch.LongTensor.
        """
        if generation_config is not None:
            kwargs["generation_config"] = generation_config
        if attention_mask is not None:
            kwargs["attention_mask"] = attention_mask

        input_ids, unsort_idx = self._sort_generation_inputs(input_ids, kwargs)
        outputs = super().generate(input_ids, **kwargs)
        if unsort_idx is not None:
            outputs = self._unsort_generation_outputs(outputs, unsort_idx)
        return outputs

    def _sort_generation_inputs(
        self, input_ids: torch.LongTensor | None, kwargs: dict
    ) -> tuple[torch.LongTensor | None, torch.Tensor | None]:
        batch_input = input_ids if input_ids is not None else kwargs.get("inputs_embeds")
        if (
            # getattr: multimodal top-level configs lack the field
            not getattr(self.rbln_config, "use_batch_attn_opt", None)
            or batch_input is None
            or batch_input.shape[0] <= 1
        ):
            return input_ids, None

        mask = kwargs.get("attention_mask")
        lengths = (
            mask.sum(dim=-1)
            if mask is not None
            else torch.full((batch_input.shape[0],), batch_input.shape[1], dtype=torch.long)
        )
        sort_idx = torch.argsort(lengths, descending=True)
        if input_ids is not None:
            input_ids = input_ids.index_select(0, sort_idx)
        for name in self._generate_batch_sortable_kwargs:
            value = kwargs.get(name)
            if isinstance(value, torch.Tensor) and value.dim() >= 1:
                kwargs[name] = value.index_select(0, sort_idx)
        kwargs["inputs_sorted"] = True
        return input_ids, torch.argsort(sort_idx)
