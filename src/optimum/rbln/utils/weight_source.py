# Copyright 2026 Rebellions Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open
from safetensors.torch import save_file


GENERATED_WEIGHT_FILENAME = "rbln_generated_tensors.safetensors"


def _tensor_identity(tensor: torch.Tensor) -> tuple:
    storage = tensor.untyped_storage()
    return (
        storage.data_ptr(),
        tensor.storage_offset(),
        tensor.numel(),
        str(tensor.dtype),
        tuple(tensor.shape),
        tuple(tensor.stride()),
    )


def build_decoder_weight_map(
    source_model: torch.nn.Module,
    wrapped_model: torch.nn.Module,
) -> tuple[dict[str, str], dict[str, str], dict[str, torch.Tensor]]:
    """Map compiler names to HF checkpoint keys and generated tensor keys."""
    source_names: dict[tuple, list[str]] = {}
    for name, tensor in source_model.state_dict(keep_vars=True).items():
        source_names.setdefault(_tensor_identity(tensor), []).append(name)

    name_map: dict[str, str] = {}
    generated_name_map: dict[str, str] = {}
    generated_state: dict[str, torch.Tensor] = {}
    generated_keys: dict[tuple, str] = {}
    named_tensors = [
        (name, f"p_{name.replace('.', '_')}", tensor)
        for name, tensor in wrapped_model.named_parameters(remove_duplicate=False)
    ] + [
        (name, f"b_{name.replace('.', '_')}", tensor)
        for name, tensor in wrapped_model.named_buffers(remove_duplicate=False)
    ]

    for module_name, traced_name, tensor in named_tensors:
        candidates = source_names.get(_tensor_identity(tensor))
        if candidates:
            for compiled_name in (module_name, traced_name):
                name_map[compiled_name] = candidates[0]
            continue

        identity = _tensor_identity(tensor)
        generated_key = generated_keys.get(identity)
        if generated_key is None:
            generated_key = f"tensor_{len(generated_state):04d}"
            generated_keys[identity] = generated_key
            generated_state[generated_key] = tensor.detach().cpu().contiguous().clone()
        for compiled_name in (module_name, traced_name):
            generated_name_map[compiled_name] = generated_key

    return name_map, generated_name_map, generated_state


def save_generated_weight_state(
    generated_state: dict[str, torch.Tensor],
    artifact_dir: Path,
) -> None:
    if not generated_state:
        return
    artifact_dir.mkdir(parents=True, exist_ok=True)
    save_file(generated_state, artifact_dir / GENERATED_WEIGHT_FILENAME)


def iter_decoder_weight_windows(
    weight_source: dict[str, Any],
    name_map: dict[str, str],
    generated_weight_map: dict[str, str],
    generated_weight_file: str | Path | None,
    windows: list[dict[str, Any]],
    *,
    token: bool | str | None = None,
    cache_dir: str | None = None,
    local_files_only: bool = False,
) -> Iterator[tuple[dict[str, Any], dict[str, torch.Tensor]]]:
    """Materialize only tensors needed by each weight-pool window."""
    files = _resolve_safetensors_files(
        weight_source,
        token=token,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
    )
    key_to_file: dict[str, Path] = {}
    for path in files:
        with safe_open(path, framework="pt", device="cpu") as checkpoint:
            for key in checkpoint.keys():
                if key in key_to_file:
                    raise ValueError(f"Duplicate checkpoint key across safetensors shards: {key}")
                key_to_file[key] = path

    for window in windows:
        requested_names = set(window["names"])
        hf_to_compiled: dict[str, list[str]] = {}
        generated_names = set()
        state_dict: dict[str, torch.Tensor] = {}

        for compiled_name in requested_names:
            hf_name = name_map.get(compiled_name)
            if hf_name is not None:
                if hf_name not in key_to_file:
                    raise ValueError(f"Hugging Face checkpoint is missing mapped weight: {hf_name}")
                hf_to_compiled.setdefault(hf_name, []).append(compiled_name)
                continue
            if compiled_name in generated_weight_map:
                generated_names.add(compiled_name)
                continue
            raise ValueError(f"No checkpoint or generated tensor mapping for: {compiled_name}")

        file_to_keys: dict[Path, list[str]] = {}
        for hf_name in hf_to_compiled:
            file_to_keys.setdefault(key_to_file[hf_name], []).append(hf_name)
        for path, hf_names in file_to_keys.items():
            with safe_open(path, framework="pt", device="cpu") as checkpoint:
                for hf_name in hf_names:
                    tensor = checkpoint.get_tensor(hf_name)
                    for compiled_name in hf_to_compiled[hf_name]:
                        state_dict[compiled_name] = tensor

        if generated_names:
            if generated_weight_file is None:
                raise ValueError("Generated weight map has no safetensors file.")
            state_dict.update(
                _load_generated_tensors(
                    generated_weight_file,
                    generated_weight_map,
                    generated_names,
                )
            )
        yield window, state_dict


def _load_generated_tensors(
    generated_weight_file: str | Path,
    generated_weight_map: dict[str, str],
    compiled_names: set[str],
) -> dict[str, torch.Tensor]:
    path = Path(generated_weight_file)
    if not path.is_file():
        raise FileNotFoundError(f"Generated tensor file not found: {path}")

    canonical_to_compiled: dict[str, list[str]] = {}
    for compiled_name in compiled_names:
        canonical_to_compiled.setdefault(generated_weight_map[compiled_name], []).append(compiled_name)

    state_dict = {}
    with safe_open(path, framework="pt", device="cpu") as tensors:
        available = set(tensors.keys())
        missing = sorted(set(canonical_to_compiled) - available)
        if missing:
            raise ValueError(f"Generated tensor file is missing key(s): {missing}")
        for canonical_name, aliases in canonical_to_compiled.items():
            tensor = tensors.get_tensor(canonical_name)
            for compiled_name in aliases:
                state_dict[compiled_name] = tensor
    return state_dict


def _resolve_safetensors_files(
    weight_source: dict[str, Any],
    *,
    token: bool | str | None,
    cache_dir: str | None,
    local_files_only: bool,
) -> list[Path]:
    model_id = weight_source["model_id"]
    subfolder = weight_source.get("subfolder", "")
    revision = weight_source.get("revision")
    variant = weight_source.get("variant")
    source_path = Path(model_id)

    if source_path.is_dir():
        root = source_path / subfolder
        files = _checkpoint_files(root, variant)
    else:
        pattern = f"{subfolder.rstrip('/')}/*.safetensors" if subfolder else "*.safetensors"
        index_name = f"model.safetensors.index.{variant}.json" if variant else "model.safetensors.index.json"
        index_pattern = f"{subfolder.rstrip('/')}/{index_name}" if subfolder else index_name
        snapshot = Path(
            snapshot_download(
                repo_id=model_id,
                revision=revision,
                token=token,
                cache_dir=cache_dir,
                local_files_only=local_files_only,
                allow_patterns=[pattern, index_pattern],
            )
        )
        files = _checkpoint_files(snapshot / subfolder, variant)

    if not files:
        raise FileNotFoundError(
            f"No safetensors checkpoint found for weight source {model_id!r}"
            + (f" in subfolder {subfolder!r}" if subfolder else "")
        )
    return files


def _checkpoint_files(root: Path, variant: str | None) -> list[Path]:
    index_name = f"model.safetensors.index.{variant}.json" if variant else "model.safetensors.index.json"
    index_path = root / index_name
    if index_path.is_file():
        with open(index_path) as index_file:
            index = json.load(index_file)
        filenames = sorted(set(index.get("weight_map", {}).values()))
        if not filenames:
            raise ValueError(f"Safetensors index has no weight_map entries: {index_path}")
        files = [root / filename for filename in filenames]
        missing = [str(path) for path in files if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"Safetensors index references missing shard(s): {missing}")
        return files

    canonical_name = f"model.{variant}.safetensors" if variant else "model.safetensors"
    canonical = root / canonical_name
    if canonical.is_file():
        return [canonical]
    candidates = sorted(root.glob("*.safetensors"))
    if len(candidates) > 1:
        raise ValueError(f"Multiple safetensors files found without {index_name}: {candidates}")
    return candidates
