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

from ..models.decoderonly.generation_decoderonly import RBLNDecoderOnlyGenerationMixin


_UNMAPPABLE_BATCH_SORT = (
    "Cannot map image inputs to batch samples for the sorting `requires_batch_sort` requires. "
    "Pass inputs already sorted by sequence length (descending) with `inputs_sorted=True`."
)


def _permute_flat_segments(tensor: torch.Tensor, seg_lens: list[int], perm_idx: torch.Tensor) -> torch.Tensor:
    segments = torch.split(tensor, seg_lens, dim=0)
    return torch.cat([segments[i] for i in perm_idx.tolist()], dim=0)


def _placeholder_run_counts(
    input_ids: torch.LongTensor | None, token_id: int | None, runs_per_segment: int = 1
) -> list[int]:
    # variable-size expansions: each segment is `runs_per_segment` contiguous placeholder runs
    if input_ids is None or token_id is None:
        raise RuntimeError(_UNMAPPABLE_BATCH_SORT)
    is_placeholder = input_ids == token_id
    run_starts = is_placeholder.clone()
    run_starts[:, 1:] &= ~is_placeholder[:, :-1]
    counts = run_starts.sum(dim=1)
    if runs_per_segment > 1:
        if not bool((counts % runs_per_segment == 0).all()):
            raise RuntimeError(_UNMAPPABLE_BATCH_SORT)
        counts = counts // runs_per_segment
    return counts.tolist()


def _placeholder_token_counts(
    input_ids: torch.LongTensor | None, token_id: int | None, tokens_per_segment: int | None
) -> list[int]:
    # fixed-size expansions: adjacent segments merge runs, so divide token totals instead
    if input_ids is None or token_id is None or not tokens_per_segment:
        raise RuntimeError(_UNMAPPABLE_BATCH_SORT)
    counts = (input_ids == token_id).sum(dim=1)
    if not bool((counts % tokens_per_segment == 0).all()):
        raise RuntimeError(_UNMAPPABLE_BATCH_SORT)
    return (counts // tokens_per_segment).tolist()


def _match_token_totals(
    input_ids: torch.LongTensor | None, token_id: int | None, tokens_per_segment: list[int]
) -> list[int]:
    # per-segment token counts known from side inputs (e.g. pixtral image_sizes),
    # greedily assigned in stacking order to each row's placeholder-token total
    if input_ids is None or token_id is None:
        raise RuntimeError(_UNMAPPABLE_BATCH_SORT)
    row_totals = (input_ids == token_id).sum(dim=1).tolist()
    counts, seg_idx = [], 0
    for total in row_totals:
        n = 0
        while total > 0:
            if seg_idx >= len(tokens_per_segment) or tokens_per_segment[seg_idx] > total:
                raise RuntimeError(_UNMAPPABLE_BATCH_SORT)
            total -= tokens_per_segment[seg_idx]
            seg_idx += 1
            n += 1
        counts.append(n)
    if seg_idx != len(tokens_per_segment):
        raise RuntimeError(_UNMAPPABLE_BATCH_SORT)
    return counts


class RBLNBatchSortGuardMixin:
    def _require_sorted_batch_inputs(self, batch_input: torch.Tensor | None, inputs_sorted: bool) -> None:
        # an unsorted direct multi-batch call would silently mis-lay the KV cache
        if inputs_sorted or not self._batch_sort_enabled:
            return
        if batch_input is not None and batch_input.shape[0] > 1:
            raise RuntimeError(
                "This model was compiled with `requires_batch_sort`, which requires batch inputs sorted by "
                "sequence length (descending). Use generate(), which sorts and unsorts automatically, or "
                "pass `inputs_sorted=True` with inputs already sorted."
            )


class RBLNMultimodalBatchSortMixin(RBLNBatchSortGuardMixin, RBLNDecoderOnlyGenerationMixin):
    _lm_attr_name = "language_model"
    # image-first kwargs, mapped to rows by _images_per_sample; batch-first extras go in
    # _batch_sortable_kwargs instead
    _image_indexed_kwargs: tuple[str, ...] = ()

    @property
    def _batch_sort_enabled(self) -> bool:
        language_model = getattr(self, self._lm_attr_name, None)
        return bool(language_model is not None and getattr(language_model.rbln_config, "requires_batch_sort", None))

    @property
    def _image_token_id(self) -> int | None:
        for name in ("image_token_id", "image_token_index"):
            token_id = getattr(self.config, name, None)
            if token_id is not None:
                return token_id
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
        tensors = self._collect_segment_kwargs(kwargs, self._image_indexed_kwargs)
        if tensors:
            self._permute_segment_kwargs(tensors, kwargs, sort_idx, self._images_per_sample(input_ids, kwargs))

    def _images_per_sample(self, input_ids: torch.LongTensor | None, kwargs: dict) -> list[int]:
        raise NotImplementedError(
            f"{type(self).__name__} declares _image_indexed_kwargs but no _images_per_sample mapping."
        )

    @staticmethod
    def _collect_segment_kwargs(kwargs: dict, names: tuple[str, ...]) -> dict[str, torch.Tensor]:
        tensors = {
            name: value for name in names if isinstance(value := kwargs.get(name), torch.Tensor) and value.shape[0] > 0
        }
        if len({value.shape[0] for value in tensors.values()}) > 1:
            raise RuntimeError(
                f"Image-indexed inputs {tuple(tensors)} disagree on the number of images; "
                "cannot reorder them for batch-attention sorting."
            )
        return tensors

    @staticmethod
    def _permute_segment_kwargs(
        tensors: dict[str, torch.Tensor], kwargs: dict, sort_idx: torch.Tensor, seg_lens: list[int]
    ) -> None:
        if sum(seg_lens) != next(iter(tensors.values())).shape[0]:
            raise RuntimeError(_UNMAPPABLE_BATCH_SORT)
        for name, value in tensors.items():
            kwargs[name] = _permute_flat_segments(value, seg_lens, sort_idx)


def _per_sample_patch_lens(grid_thw: torch.Tensor, rows_per_sample: list[int]) -> list[int]:
    # patches for grid row i = t*h*w; sum rows owned by each sample
    patches_per_row = grid_thw.prod(dim=-1).tolist()
    lens, row_idx = [], 0
    for n in rows_per_sample:
        lens.append(sum(patches_per_row[row_idx : row_idx + n]))
        row_idx += n
    return lens


class RBLNVisionBatchSortMixin(RBLNBatchSortGuardMixin):
    _video_grid_rows_are_chunks = False  # qwen3-style video grids are per temporal chunk
    _batch_sortable_kwargs = RBLNDecoderOnlyGenerationMixin._batch_sortable_kwargs + ("mm_token_type_ids",)
    _vision_sortable_kwargs = (
        "pixel_values",
        "pixel_values_videos",
        "image_grid_thw",
        "video_grid_thw",
        "second_per_grid_ts",
    )

    def _sort_generation_inputs(
        self, input_ids: torch.LongTensor | None, kwargs: dict
    ) -> tuple[torch.LongTensor | None, torch.Tensor | None]:
        presort_input_ids = input_ids
        presort_mask = kwargs.get("attention_mask")
        input_ids, unsort_idx = super()._sort_generation_inputs(input_ids, kwargs)
        if unsort_idx is None or not any(kwargs.get(k) is not None for k in self._vision_sortable_kwargs):
            return input_ids, unsort_idx
        if presort_input_ids is None:
            raise RuntimeError("Batch sorting flattened vision inputs requires input_ids.")

        # counts come from the pre-sort rows
        sort_idx = torch.argsort(unsort_idx)
        image_rows, video_rows = self._vision_grid_rows_per_sample(presort_input_ids, presort_mask, kwargs)
        self._sort_vision_kwargs(kwargs, sort_idx, image_rows, video_rows)
        return input_ids, unsort_idx

    def _vision_grid_rows_per_sample(
        self, input_ids: torch.LongTensor, attention_mask: torch.Tensor | None, kwargs: dict
    ) -> tuple[list[int], list[int]]:
        # same counting as _preprocess_prefill: vision_start marker followed by image/video token
        image_rows, video_rows = [], []
        video_grid_thw = kwargs.get("video_grid_thw")
        video_row_idx = 0
        for b_idx in range(input_ids.shape[0]):
            row = input_ids[b_idx]
            if attention_mask is not None:
                row = row[attention_mask[b_idx].bool()]
            vision_start_indices = torch.argwhere(row == self.config.vision_start_token_id).squeeze(1)
            vision_tokens = row[vision_start_indices + 1]
            image_rows.append(int((vision_tokens == self.config.image_token_id).sum().item()))
            video_nums = int((vision_tokens == self.config.video_token_id).sum().item())
            if self._video_grid_rows_are_chunks and video_grid_thw is not None:
                start_row = video_row_idx
                consumed_video_chunks = 0
                while video_row_idx < video_grid_thw.shape[0] and consumed_video_chunks < video_nums:
                    consumed_video_chunks += int(video_grid_thw[video_row_idx, 0].item())
                    video_row_idx += 1
                video_rows.append(video_row_idx - start_row)
            else:
                video_rows.append(video_nums)
        return image_rows, video_rows

    def _sort_vision_kwargs(
        self, kwargs: dict, sort_idx: torch.Tensor, image_rows: list[int], video_rows: list[int]
    ) -> None:
        for grid_key, pixel_key, seg_rows in (
            ("image_grid_thw", "pixel_values", image_rows),
            ("video_grid_thw", "pixel_values_videos", video_rows),
        ):
            grid = kwargs.get(grid_key)
            if grid is None:
                continue
            pixels = kwargs.get(pixel_key)
            if pixels is not None:
                kwargs[pixel_key] = _permute_flat_segments(pixels, _per_sample_patch_lens(grid, seg_rows), sort_idx)
            kwargs[grid_key] = _permute_flat_segments(grid, seg_rows, sort_idx)

        second_per_grid_ts = kwargs.get("second_per_grid_ts")
        if second_per_grid_ts is not None:
            if isinstance(second_per_grid_ts, torch.Tensor):
                kwargs["second_per_grid_ts"] = _permute_flat_segments(second_per_grid_ts, video_rows, sort_idx)
            else:
                bounds = [0]
                for n in video_rows:
                    bounds.append(bounds[-1] + n)
                kwargs["second_per_grid_ts"] = [
                    v for i in sort_idx.tolist() for v in second_per_grid_ts[bounds[i] : bounds[i + 1]]
                ]
