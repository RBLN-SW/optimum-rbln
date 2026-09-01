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

import torch

from ..models.decoderonly.generation_decoderonly import (
    RBLNDecoderOnlyGenerationMixin,
    _permute_flat_segments,
)


class RBLNMultimodalBatchSortMixin(RBLNDecoderOnlyGenerationMixin):
    # Composition VLMs keep a plain RBLNModelConfig at the top level; the batched-attention
    # contract (use_batch_attn_opt) lives on the inner language model's config.
    _lm_attr_name = "language_model"
    # generate kwargs stacked over images on dim 0, mapped to samples by placeholder-token
    # order in input_ids; batch-first extras go in _generate_batch_sortable_kwargs instead.
    _image_indexed_kwargs: tuple[str, ...] = ()

    @property
    def _batch_sort_enabled(self) -> bool:
        language_model = getattr(self, self._lm_attr_name, None)
        return bool(language_model is not None and getattr(language_model.rbln_config, "use_batch_attn_opt", None))

    @property
    def _image_token_id(self) -> int | None:
        for name in ("image_token_id", "image_token_index"):
            token_id = getattr(self.config, name, None)
            if token_id is not None:
                return token_id
        return None

    @property
    def _tokens_per_image(self) -> int | None:
        return None

    def _sort_generation_inputs(
        self, input_ids: torch.LongTensor | None, kwargs: dict
    ) -> tuple[torch.LongTensor | None, torch.Tensor | None]:
        orig_input_ids = input_ids
        input_ids, unsort_idx = super()._sort_generation_inputs(input_ids, kwargs)
        if unsort_idx is not None:
            self._sort_extra_generation_inputs(orig_input_ids, kwargs, torch.argsort(unsort_idx))
        return input_ids, unsort_idx

    def _sort_extra_generation_inputs(
        self, input_ids: torch.LongTensor | None, kwargs: dict, sort_idx: torch.Tensor
    ) -> None:
        self._permute_segment_kwargs(
            input_ids, kwargs, sort_idx, self._image_indexed_kwargs, self._image_token_id, self._tokens_per_image
        )

    def _permute_segment_kwargs(
        self,
        input_ids: torch.LongTensor | None,
        kwargs: dict,
        sort_idx: torch.Tensor,
        names: tuple[str, ...],
        token_id: int | None,
        tokens_per_segment: int | None,
        runs_per_segment: int = 1,
    ) -> None:
        tensors = {
            name: value for name in names if isinstance(value := kwargs.get(name), torch.Tensor) and value.shape[0] > 0
        }
        if not tensors:
            return
        num_segments = next(iter(tensors.values())).shape[0]
        if any(value.shape[0] != num_segments for value in tensors.values()):
            raise RuntimeError(
                f"Image-indexed inputs {tuple(tensors)} disagree on the number of images; "
                "cannot reorder them for batch-attention sorting."
            )
        seg_lens = self._segments_per_sample(input_ids, num_segments, token_id, tokens_per_segment, runs_per_segment)
        for name, value in tensors.items():
            kwargs[name] = _permute_flat_segments(value, seg_lens, sort_idx)

    def _segments_per_sample(
        self,
        input_ids: torch.LongTensor | None,
        num_segments: int,
        token_id: int | None,
        tokens_per_segment: int | None,
        runs_per_segment: int,
    ) -> list[int]:
        # map dim-0-stacked segments (images/videos) back to batch rows through the
        # placeholder tokens each sample carries
        if input_ids is not None and token_id is not None:
            is_placeholder = input_ids == token_id
            # each image (or video frame) expands to one contiguous placeholder run
            run_starts = is_placeholder.clone()
            run_starts[:, 1:] &= ~is_placeholder[:, :-1]
            run_counts = run_starts.sum(dim=1)
            if int(run_counts.sum()) == num_segments * runs_per_segment and bool(
                (run_counts % runs_per_segment == 0).all()
            ):
                return (run_counts // runs_per_segment).tolist()
            # adjacent placeholders merge runs; fixed-size expansions can still split by token count
            if tokens_per_segment:
                token_counts = is_placeholder.sum(dim=1)
                if int(token_counts.sum()) == num_segments * tokens_per_segment and bool(
                    (token_counts % tokens_per_segment == 0).all()
                ):
                    return (token_counts // tokens_per_segment).tolist()
        raise RuntimeError(
            "Cannot map image inputs to batch samples for the sorting `use_batch_attn_opt` requires. "
            "Pass inputs already sorted by sequence length (descending) with `inputs_sorted=True`."
        )

    def _require_sorted_batch_inputs(self, batch_input: torch.Tensor | None, inputs_sorted: bool) -> None:
        # the batched attention kernel reads a KV layout fixed by length-descending batch order;
        # generate() establishes it once up front, a bare multi-batch forward() cannot
        if inputs_sorted or not self._batch_sort_enabled:
            return
        if batch_input is not None and batch_input.shape[0] > 1:
            raise RuntimeError(
                "This model was compiled with `use_batch_attn_opt`, which requires batch inputs sorted by "
                "sequence length (descending). Use generate(), which sorts and unsorts automatically, or "
                "pass `inputs_sorted=True` with inputs already sorted."
            )
