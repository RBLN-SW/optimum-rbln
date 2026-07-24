import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

from optimum.rbln import RBLNLlamaForCausalLMConfig
from optimum.rbln.transformers.models.decoderonly.modeling_decoderonly import RBLNDecoderOnlyModel
from optimum.rbln.utils.weight_source import (
    GENERATED_WEIGHT_FILENAME,
    build_decoder_weight_map,
    iter_decoder_weight_windows,
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


def test_build_decoder_weight_map_uses_storage_identity():
    source = _Source()
    wrapped = _Wrapped(source)

    name_map, generated_map, generated_state = build_decoder_weight_map(source, wrapped)

    assert name_map == {
        "renamed_proj.weight": "proj.weight",
        "p_renamed_proj_weight": "proj.weight",
    }
    assert generated_map == {"scale": "tensor_0000", "p_scale": "tensor_0000"}
    assert set(generated_state) == {"tensor_0000"}
    assert generated_state["tensor_0000"].item() == 0.5


def test_build_decoder_weight_map_captures_non_scalar_generated_tensor():
    source = _Source()
    wrapped = _Wrapped(source)
    wrapped.generated = torch.nn.Parameter(torch.ones(2))

    _, generated_map, generated_state = build_decoder_weight_map(source, wrapped)

    assert generated_map["generated"] == generated_map["p_generated"]
    assert torch.equal(generated_state[generated_map["generated"]], torch.ones(2))


def test_iter_decoder_weight_windows_reads_across_shards(tmp_path: Path):
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
        iter_decoder_weight_windows(
            {"model_id": str(tmp_path)},
            {"compiled.first": "first", "compiled.second": "second"},
            {},
            None,
            windows,
        )
    )

    assert window["graph_ids"] == [3, 7]
    assert set(state_dict) == {"compiled.first", "compiled.second"}
    assert torch.equal(state_dict["compiled.second"], torch.full((2,), 2.0))


def test_iter_decoder_weight_windows_loads_generated_safetensors(tmp_path: Path):
    save_file({"first": torch.ones(2)}, tmp_path / "model.safetensors")
    generated_path = tmp_path / GENERATED_WEIGHT_FILENAME
    save_file({"tensor_0000": torch.tensor(0.5)}, generated_path)
    windows = [{"graph_ids": [3], "names": ["compiled.first", "scale", "p_scale"]}]

    _, state_dict = next(
        iter_decoder_weight_windows(
            {"model_id": str(tmp_path)},
            {"compiled.first": "first"},
            {"scale": "tensor_0000", "p_scale": "tensor_0000"},
            generated_path,
            windows,
        )
    )

    assert state_dict["scale"].item() == 0.5
    assert state_dict["scale"].data_ptr() == state_dict["p_scale"].data_ptr()


def test_iter_decoder_weight_windows_resolves_variant_index(tmp_path: Path):
    save_file({"first": torch.ones(2)}, tmp_path / "model-00001-of-00001.fp16.safetensors")
    (tmp_path / "model.safetensors.index.fp16.json").write_text(
        json.dumps({"weight_map": {"first": "model-00001-of-00001.fp16.safetensors"}})
    )

    _, state_dict = next(
        iter_decoder_weight_windows(
            {"model_id": str(tmp_path), "variant": "fp16"},
            {"compiled.first": "first"},
            {},
            None,
            [{"graph_ids": [0], "names": ["compiled.first"]}],
        )
    )

    assert torch.equal(state_dict["compiled.first"], torch.ones(2))


def test_weight_free_torch_artifacts_leave_config_json_serializable(tmp_path: Path):
    config = RBLNLlamaForCausalLMConfig(max_seq_len=128, weight_free=True)

    _ArtifactWriter.save_torch_artifacts(_Source(), tmp_path, "", config)
    config.save(tmp_path)

    assert (tmp_path / GENERATED_WEIGHT_FILENAME).is_file()
    assert config.generated_weight_map == {"scale": "tensor_0000", "p_scale": "tensor_0000"}


def test_iter_decoder_weight_windows_reports_missing_mapped_key(tmp_path: Path):
    save_file({"other": torch.ones(1)}, tmp_path / "model.safetensors")

    with pytest.raises(ValueError, match="missing mapped weight"):
        next(
            iter_decoder_weight_windows(
                {"model_id": str(tmp_path)},
                {"renamed_proj.weight": "proj.weight"},
                {},
                None,
                [{"graph_ids": [0], "names": ["renamed_proj.weight"]}],
            )
        )


def test_decoder_config_serializes_weight_source_mapping():
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
        _ArtifactWriter._update_weight_source_for_export(
            config,
            "org/model",
            model,
            "",
            None,
            None,
            torch.float32,
        )
