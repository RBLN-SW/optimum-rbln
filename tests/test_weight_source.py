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

import ast
import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

from optimum.rbln import RBLNLlamaForCausalLMConfig
from optimum.rbln.modeling_base import RBLNBaseModel
from optimum.rbln.transformers.models.decoderonly.modeling_decoderonly import RBLNDecoderOnlyModel
from optimum.rbln.utils.weight_source import (
    GENERATED_WEIGHT_FILENAME,
    build_weight_map,
    iter_weight_windows,
    resolve_weight_index,
)


class _Source(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = torch.nn.Linear(4, 3, bias=False)


class _Wrapped(torch.nn.Module):
    def __init__(self, source):
        super().__init__()
        self.renamed_proj = source.proj
        self.scale = torch.nn.Parameter(torch.tensor(0.5))


class _ArtifactWriter(RBLNDecoderOnlyModel):
    @classmethod
    def _wrap_model_if_needed(cls, model, rbln_config):
        return _Wrapped(model)


class _FakeRuntime:
    def __init__(self, plan, required=None):
        self._plan = plan
        self._required = set(required) if required is not None else {n for w in plan for n in w["names"]}
        self.loaded: list[tuple[list[str], list[int]]] = []
        self.host_loaded: list[list[str]] = []

    def weight_load_plan(self):
        return [dict(window) for window in self._plan]

    def required_weight_names(self):
        return set(self._required)

    def load_weight_window(self, state_dict, graph_ids):
        self.loaded.append((sorted(state_dict), list(graph_ids)))

    def load_weights(self, state_dict, *, partial=False):
        assert partial
        self.host_loaded.append(sorted(state_dict))


def _window(buffer_id: int, names: list[str], graph_ids: list[int]) -> dict:
    return {
        "buffer_id": buffer_id,
        "node_id": 0,
        "start_offset": 0,
        "end_offset": 64,
        "names": names,
        "graph_ids": graph_ids,
    }


def test_build_weight_map_uses_storage_identity():
    source = _Source()
    wrapped = _Wrapped(source)

    name_map, generated_map, generated_state = build_weight_map(source, wrapped)

    assert name_map == {
        "renamed_proj.weight": "proj.weight",
        "p_renamed_proj_weight": "proj.weight",
    }
    assert generated_map == {"scale": "tensor_0000", "p_scale": "tensor_0000"}
    assert set(generated_state) == {"tensor_0000"}
    assert generated_state["tensor_0000"].item() == 0.5


def test_build_weight_map_captures_non_scalar_generated_tensor():
    source = _Source()
    wrapped = _Wrapped(source)
    wrapped.generated = torch.nn.Parameter(torch.ones(2))

    _, generated_map, generated_state = build_weight_map(source, wrapped)

    assert generated_map["generated"] == generated_map["p_generated"]
    assert torch.equal(generated_state[generated_map["generated"]], torch.ones(2))


def test_iter_weight_windows_reads_across_shards(tmp_path: Path):
    save_file({"first": torch.ones(2)}, tmp_path / "model-00001.safetensors")
    save_file({"second": torch.full((2,), 2.0)}, tmp_path / "model-00002.safetensors")
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": {
                    "first": "model-00001.safetensors",
                    "second": "model-00002.safetensors",
                }
            }
        )
    )
    windows = [{"graph_ids": [3, 7], "names": ["compiled.first", "compiled.second"]}]

    window, state_dict = next(
        iter_weight_windows(
            resolve_weight_index({"model_id": str(tmp_path)}),
            {"compiled.first": "first", "compiled.second": "second"},
            {},
            None,
            windows,
        )
    )

    assert window["graph_ids"] == [3, 7]
    assert set(state_dict) == {"compiled.first", "compiled.second"}
    assert torch.equal(state_dict["compiled.second"], torch.full((2,), 2.0))


def test_iter_weight_windows_loads_generated_safetensors(tmp_path: Path):
    save_file({"first": torch.ones(2)}, tmp_path / "model.safetensors")
    generated_path = tmp_path / GENERATED_WEIGHT_FILENAME
    save_file({"tensor_0000": torch.tensor(0.5)}, generated_path)
    windows = [{"graph_ids": [3], "names": ["compiled.first", "scale", "p_scale"]}]

    _, state_dict = next(
        iter_weight_windows(
            resolve_weight_index({"model_id": str(tmp_path)}),
            {"compiled.first": "first"},
            {"scale": "tensor_0000", "p_scale": "tensor_0000"},
            generated_path,
            windows,
        )
    )

    assert state_dict["scale"].item() == 0.5
    assert state_dict["scale"].data_ptr() == state_dict["p_scale"].data_ptr()


def test_resolve_weight_index_resolves_variant_index(tmp_path: Path):
    save_file({"first": torch.ones(2)}, tmp_path / "model-00001-of-00001.fp16.safetensors")
    (tmp_path / "model.safetensors.index.fp16.json").write_text(
        json.dumps({"weight_map": {"first": "model-00001-of-00001.fp16.safetensors"}})
    )

    _, state_dict = next(
        iter_weight_windows(
            resolve_weight_index({"model_id": str(tmp_path), "variant": "fp16"}),
            {"compiled.first": "first"},
            {},
            None,
            [{"graph_ids": [0], "names": ["compiled.first"]}],
        )
    )

    assert torch.equal(state_dict["compiled.first"], torch.ones(2))


def test_weight_free_artifacts_leave_config_json_serializable(tmp_path: Path):
    config = RBLNLlamaForCausalLMConfig(max_seq_len=128, weight_free=True)

    _ArtifactWriter._save_weight_free_artifacts(_Source(), tmp_path, config)
    config.save(tmp_path)

    assert (tmp_path / GENERATED_WEIGHT_FILENAME).is_file()
    assert config.generated_weight_map == {"scale": "tensor_0000", "p_scale": "tensor_0000"}


def test_iter_weight_windows_reports_missing_mapped_key(tmp_path: Path):
    save_file({"other": torch.ones(1)}, tmp_path / "model.safetensors")

    with pytest.raises(ValueError, match="missing mapped weight"):
        next(
            iter_weight_windows(
                resolve_weight_index({"model_id": str(tmp_path)}),
                {"renamed_proj.weight": "proj.weight"},
                {},
                None,
                [{"graph_ids": [0], "names": ["renamed_proj.weight"]}],
            )
        )


def test_config_serializes_weight_source_mapping():
    config = RBLNLlamaForCausalLMConfig(
        max_seq_len=128,
        weight_free=True,
        weight_source={"model_id": "org/model", "revision": "commit"},
        weight_name_map={"wrapped.weight": "model.weight"},
        generated_weight_map={"scale": "tensor_0000"},
    )

    serialized = config._prepare_for_serialization()
    assert serialized["weight_free"] is True
    assert serialized["weight_source"]["revision"] == "commit"
    assert serialized["weight_name_map"] == {"wrapped.weight": "model.weight"}
    assert serialized["generated_weight_map"] == {"scale": "tensor_0000"}


def test_weight_free_export_rejects_dtype_conversion():
    config = RBLNLlamaForCausalLMConfig(max_seq_len=128, weight_free=True)
    model = SimpleNamespace(config=SimpleNamespace(_commit_hash=None))

    with pytest.raises(ValueError, match="does not support dtype conversion"):
        RBLNBaseModel._update_weight_source_for_export(config, "org/model", model, "", None, None, torch.float32)


def test_weight_free_export_records_source_on_base_model():
    config = RBLNLlamaForCausalLMConfig(max_seq_len=128, weight_free=True)
    model = SimpleNamespace(config=SimpleNamespace(_commit_hash="abc123"))

    RBLNBaseModel._update_weight_source_for_export(config, "org/model", model, "sub", "main", "fp16", None)

    assert config.weight_source == {
        "model_id": "org/model",
        "subfolder": "sub",
        "revision": "abc123",
        "variant": "fp16",
    }


def test_group_runtimes_by_weight_pool_separates_independent_pools():
    prefill = _FakeRuntime([_window(0, ["shared.weight"], [1])])
    decode = _FakeRuntime([_window(0, ["shared.weight"], [2])])
    decoder = _FakeRuntime([_window(1, ["other.weight"], [5])])
    weightless = _FakeRuntime([])

    pools = RBLNBaseModel._group_runtimes_by_weight_pool(
        [(runtime, runtime.weight_load_plan()) for runtime in (prefill, decode, decoder, weightless)]
    )

    assert len(pools) == 2
    grouped = {tuple(plan[0]["names"]): [runtime for runtime, _ in group] for plan, group in pools}
    assert grouped[("shared.weight",)] == [prefill, decode]
    assert grouped[("other.weight",)] == [decoder]


def test_load_runtime_weights_feeds_each_pool_its_own_weights(tmp_path: Path):
    save_file({"a": torch.ones(2), "b": torch.full((2,), 2.0)}, tmp_path / "model.safetensors")
    encoder = _FakeRuntime([_window(0, ["enc.weight"], [1])])
    decoder = _FakeRuntime([_window(1, ["dec.weight"], [4])])
    decoder_sibling = _FakeRuntime([_window(1, ["dec.weight"], [9])])
    config = SimpleNamespace(
        create_runtimes=True,
        weight_free=True,
        weight_source={"model_id": str(tmp_path)},
        weight_name_map={"enc.weight": "a", "dec.weight": "b"},
        generated_weight_map={},
    )

    RBLNBaseModel._load_runtime_weights(
        models=[encoder, decoder, decoder_sibling],
        rbln_config=config,
        artifact_dir=tmp_path,
        token=None,
        cache_dir=None,
        local_files_only=True,
    )

    assert encoder.loaded == [(["enc.weight"], [1])]
    assert decoder.loaded == [(["dec.weight"], [4])]
    assert decoder_sibling.loaded == [(["dec.weight"], [9])]


def test_load_runtime_weights_supplies_host_only_weights(tmp_path: Path):
    save_file({"a": torch.ones(2), "h": torch.full((2,), 3.0)}, tmp_path / "model.safetensors")
    runtime = _FakeRuntime(
        [_window(0, ["dev.weight"], [1])],
        required={"dev.weight", "host.weight"},
    )
    config = SimpleNamespace(
        create_runtimes=True,
        weight_free=True,
        weight_source={"model_id": str(tmp_path)},
        weight_name_map={"dev.weight": "a", "host.weight": "h"},
        generated_weight_map={},
    )

    RBLNBaseModel._load_runtime_weights(
        models=[runtime],
        rbln_config=config,
        artifact_dir=tmp_path,
        token=None,
        cache_dir=None,
        local_files_only=True,
    )

    assert runtime.loaded == [(["dev.weight"], [1])]
    assert runtime.host_loaded == [["host.weight"]]


def test_compile_takes_weight_free_without_a_default():
    parameter = inspect.signature(RBLNBaseModel.compile).parameters["weight_free"]
    assert parameter.default is inspect.Parameter.empty


def test_every_compile_call_site_passes_weight_free():
    package_root = Path(RBLNBaseModel.__module__.replace(".", "/")).parent
    source_root = Path(inspect.getfile(RBLNBaseModel)).parent
    assert source_root.name == package_root.name

    missing = []
    for path in sorted(source_root.rglob("*.py")):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
                continue
            if node.func.attr != "compile" or not isinstance(node.func.value, ast.Name):
                continue
            if node.func.value.id not in ("cls", "self"):
                continue
            if not any(keyword.arg == "weight_free" for keyword in node.keywords):
                missing.append(f"{path.relative_to(source_root)}:{node.lineno}")

    assert missing == []
