from types import SimpleNamespace
from unittest.mock import patch

import pytest
from diffusers import Flux2KleinPipeline

from optimum.rbln import RBLNFlux2KleinPipeline, RBLNFlux2KleinPipelineConfig


def test_pipeline_submodule_profiles_and_config_roundtrip(tmp_path):
    config = RBLNFlux2KleinPipelineConfig(
        npu="RBLN-CR13", batch_size=2, height=512, width=768, reference_image_count=2
    )
    assert config.transformer.batch_size == config.vae.batch_size == config.text_encoder.batch_size == 2
    assert config.transformer.image_size == config.vae.sample_size == (512, 768)
    assert config.transformer.prompt_embed_length == config.text_encoder.max_seq_len == 512
    assert config.transformer.reference_image_count == 2
    config.save(tmp_path)
    restored = RBLNFlux2KleinPipelineConfig.from_pretrained(tmp_path)
    assert restored.transformer.image_token_count == config.transformer.image_token_count


@pytest.mark.parametrize(
    "kwargs",
    [
        {"height": 512},
        {"max_sequence_length": 128},
        {"prompt": ["one", "two"]},
        {"num_images_per_prompt": 2},
        {"image": [object()]},
    ],
)
def test_pipeline_rejects_runtime_profile_mismatch(kwargs):
    config = RBLNFlux2KleinPipelineConfig(
        npu="RBLN-CR13",
    )
    pipe = SimpleNamespace(
        transformer=SimpleNamespace(rbln_config=config.transformer),
        text_encoder=SimpleNamespace(rbln_config=config.text_encoder),
    )
    with patch.object(Flux2KleinPipeline, "__call__", autospec=True) as native:
        # Preserve the original signature for the adapter's argument binding.
        native.__signature__ = RBLNFlux2KleinPipeline._native_call_signature
        with pytest.raises(ValueError, match="does not match"):
            RBLNFlux2KleinPipeline.__call__(pipe, **({"prompt": "one"} | kwargs))
        native.assert_not_called()
