import json

import pytest
import torch
from transformers import PretrainedConfig

from optimum.rbln import RBLNMOSSImagePipeline, RBLNMOSSImageTransformer, RBLNMOSSImageTransformerConfig
from optimum.rbln.diffusers.models.transformers.transformer_moss_image import MOSSImageTransformerWrapper
from optimum.rbln.diffusers.pipelines.moss_image.pipeline_moss_image_upstream import (
    MOSSImagePipeline,
    MOSSImageTransformer,
)


@pytest.mark.parametrize("return_unused_kwargs", [False, True])
def test_load_config_preserves_metadata(tmp_path, return_unused_kwargs):
    config = {
        "_class_name": "MOSSImagePipeline",
        "_ignore_files": [],
        "scheduler": ["pipeline_moss_image", "MOSSImageScheduler"],
        "transformer": ["pipeline_moss_image", "MOSSImageTransformer"],
        "vae": ["diffusers", "AutoencoderKLFlux2"],
        "text_encoder": [None, None],
    }
    (tmp_path / "model_index.json").write_text(json.dumps(config))

    loaded = RBLNMOSSImagePipeline.load_config(tmp_path, return_unused_kwargs=return_unused_kwargs)
    if return_unused_kwargs:
        loaded, unused = loaded
        assert unused == {}

    assert loaded["_ignore_files"] == []
    assert loaded["vae"] == config["vae"]
    assert loaded["text_encoder"] == [None, None]
    for name in ("transformer", "scheduler"):
        assert loaded[name] == [MOSSImagePipeline.__module__, config[name][1]]


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("mask_kind", ["prefix", "scattered", "none"])
@pytest.mark.parametrize("skip_layers", [None, [0, 2], [0, 1, 2]])
def test_padded_transformer_matches_upstream(dtype, mask_kind, skip_layers):
    torch.manual_seed(42)
    model = (
        MOSSImageTransformer(
            dim=64, ffn_dim=128, num_heads=4, num_layers=3, text_dim=32, freq_dim=16, latent_channels=8
        )
        .to(dtype)
        .eval()
    )
    rbln_model = RBLNMOSSImageTransformer.__new__(RBLNMOSSImageTransformer)
    rbln_model.config = PretrainedConfig(**dict(model.config))
    rbln_model.rbln_config = RBLNMOSSImageTransformerConfig(
        batch_size=2, sample_size=(4, 6), max_sequence_length=8, dtype=str(dtype).removeprefix("torch.")
    )
    for name in ("time_embedding", "shared_block_modulation", "video_head_modulation", "rope"):
        setattr(rbln_model, name, getattr(model, name))
    rbln_model.model = [MOSSImageTransformerWrapper(model, 24)]

    latents = torch.randn(2, 8, 4, 6)
    embeddings = torch.randn(2, 8, 32).to(dtype)
    mask = None
    if mask_kind == "prefix":
        mask = torch.tensor([[1] * 5 + [0] * 3, [1] * 8], dtype=torch.bool)
    elif mask_kind == "scattered":
        mask = torch.tensor([[1, 0, 1, 1, 0, 0, 0, 1], [1] * 8], dtype=torch.bool)
    timestep = torch.tensor([999.0, 537.0])

    with torch.no_grad():
        expected = model(latents, timestep, embeddings, mask, skip_layers=skip_layers).sample
        actual = rbln_model(latents, timestep, embeddings, mask, skip_layers=skip_layers).sample

    atol, rtol = (1e-6, 1e-5) if dtype == torch.float32 else (0.016, 0.016)
    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)
