import os
import shutil
import tempfile
from typing import Optional, Tuple

import pytest
import rebel
import torch

from optimum.rbln import (
    RBLNAutoConfig,
    RBLNAutoModel,
    RBLNCompileConfig,
    RBLNLlavaNextForConditionalGeneration,
    RBLNMistralForCausalLMConfig,
    RBLNModel,
    RBLNModelConfig,
    RBLNResNetForImageClassification,
    RBLNResNetForImageClassificationConfig,
    RBLNStableDiffusionPipeline,
)


@pytest.fixture
def model_id():
    return "hf-internal-testing/tiny-random-ResNetForImageClassification"


@pytest.fixture
def stable_diffusion_model():
    model = RBLNStableDiffusionPipeline.from_pretrained(
        "hf-internal-testing/tiny-sd-pipe",
        export=True,
        rbln_config={
            "unet": {
                "batch_size": 1,
                "npu": "RBLN-CA22",
                "create_runtimes": False,
            },
        },
    )
    return model


def test_stable_diffusion_config(stable_diffusion_model):
    model = stable_diffusion_model
    assert model is not None
    assert model.unet.rbln_config.batch_size == 1
    assert model.unet.rbln_config.npu == "RBLN-CA22"
    assert model.unet.rbln_config.create_runtimes is False
    assert model.unet.compiled_models[0]._meta["npu"] == "RBLN-CA22"

    npu = rebel.get_npu_name()
    assert model.text_encoder.compiled_models[0]._meta["npu"] == npu


def test_explicit_config_parameters(model_id):
    """Test loading model with explicit configuration parameters."""
    model = RBLNResNetForImageClassification.from_pretrained(
        model_id, rbln_image_size=224, rbln_batch_size=2, rbln_create_runtimes=False
    )
    assert model is not None
    assert hasattr(model, "rbln_config")
    # Config parameters should be applied correctly


def test_config_dict(model_id):
    """Test loading model with configuration passed as a dictionary."""
    rbln_config = {"create_runtimes": False, "image_size": 64}

    model = RBLNResNetForImageClassification.from_pretrained(model_id, rbln_config=rbln_config)
    assert model is not None
    assert hasattr(model, "rbln_config")
    assert model.rbln_config.image_size == 64
    # Config dict should be properly applied


def test_config_object_load(model_id, tmp_path):
    """Test loading rbln_config with various approaches: plain load, rbln_config dict, and rbln_ prefix."""
    # Create and save a base config
    base_config = RBLNResNetForImageClassificationConfig(image_size=128)
    compile_cfg = RBLNCompileConfig(input_info=[("pixel_values", (1, 3, 128, 128), "float32")])
    base_config.set_compile_cfgs([compile_cfg])
    base_config.freeze()
    config_path = tmp_path / "test_config.json"
    base_config.save(str(config_path))

    # Subtest 1: Plain load
    loaded_config = RBLNResNetForImageClassificationConfig.load(str(config_path))
    assert loaded_config.image_size == 128, "Plain load: image_size mismatch"

    # Subtest 2: Load with rbln_config dict
    loaded_config = RBLNResNetForImageClassificationConfig.load(
        str(config_path), rbln_config={"create_runtimes": False}
    )
    assert loaded_config.create_runtimes is False, "Load with rbln_config: create_runtimes mismatch"

    # Subtest 3: Load with rbln_ prefix
    loaded_config = RBLNResNetForImageClassificationConfig.load(str(config_path), rbln_create_runtimes=False)
    assert loaded_config.create_runtimes is False, "Load with rbln_ prefix: create_runtimes mismatch"

    # Subtest 4: Load with rbln_config object
    with pytest.raises(ValueError, match="Loading from an existing RBLNModelConfig instance is not supported"):
        loaded_config = RBLNResNetForImageClassificationConfig.load(
            str(config_path), rbln_config=RBLNResNetForImageClassificationConfig(create_runtimes=False)
        )
        assert loaded_config.create_runtimes is False, "Load with rbln_ prefix: create_runtimes mismatch"


def test_config_object(model_id):
    """Test loading model with a pre-configured RBLNResNetForImageClassificationConfig object."""
    config = RBLNResNetForImageClassificationConfig()
    config.create_runtimes = False
    config.image_size = 224

    # Properly set required compile configuration
    compile_cfg = RBLNCompileConfig(input_info=[("pixel_values", (1, 3, 224, 224), "float32")])
    config.set_compile_cfgs([compile_cfg])

    model = RBLNResNetForImageClassification.from_pretrained(model_id, rbln_config=config)
    assert model is not None
    assert hasattr(model, "rbln_config")
    # Pre-configured object should be properly applied


def test_mixed_config_approach_0(model_id):
    """Test that using both rbln_config class and rbln_ prefixed arguments."""
    config = RBLNResNetForImageClassificationConfig()
    config.create_runtimes = False

    # Properly set required compile configuration
    compile_cfg = RBLNCompileConfig(input_info=[("pixel_values", (1, 3, 224, 224), "float32")])
    config.set_compile_cfgs([compile_cfg])

    # Should raise ValueError: cannot use both rbln_config and rbln_ prefixed arguments
    with pytest.raises(ValueError, match="Cannot use both"):
        model = RBLNResNetForImageClassification.from_pretrained(  # noqa: F841
            model_id,
            export=True,
            rbln_config=config,
            rbln_image_size=128,
        )


def test_mixed_config_approach_1(model_id):
    """Test that using both rbln_config dict and rbln_ prefixed arguments raises ValueError."""
    # Create a base config dict
    config_dict = {"image_size": 224, "create_runtimes": False}

    # Should raise ValueError: cannot use both rbln_config and rbln_ prefixed arguments
    with pytest.raises(ValueError, match="Cannot use both"):
        model = RBLNResNetForImageClassification.from_pretrained(  # noqa: F841
            model_id,
            export=True,
            rbln_config=config_dict,
            rbln_batch_size=2,
        )


def test_mixed_config_approach_2():
    """Test that using both rbln_config dict and rbln_ prefixed arguments raises ValueError."""
    # Create a base config dict
    config_dict = {
        "language_model": {"max_seq_len": 32768, "use_inputs_embeds": True, "batch_size": 1},
    }

    # Should raise ValueError: cannot use both rbln_config and rbln_ prefixed arguments
    with pytest.raises(ValueError, match="Cannot use both"):
        model = RBLNLlavaNextForConditionalGeneration.from_pretrained(  # noqa: F841
            "trl-internal-testing/tiny-LlavaNextForConditionalGeneration",
            export=True,
            rbln_config=config_dict,
            rbln_language_model={"batch_size": 2},
        )


def test_submodule_config_dict():
    """Test loading submodule model with configuration passed as a dictionary."""
    model = RBLNLlavaNextForConditionalGeneration.from_pretrained(
        "trl-internal-testing/tiny-LlavaNextForConditionalGeneration",
        export=True,
        rbln_language_model={"max_seq_len": 16384, "use_inputs_embeds": True, "batch_size": 2},
    )
    assert model.rbln_config.language_model.max_seq_len == 16384
    assert model.rbln_config.language_model.batch_size == 2


def test_submodule_config_object():
    """Test loading submodule with a pre-configured RBLNMistralForCausalLMConfig object."""

    rbln_config = RBLNMistralForCausalLMConfig(max_seq_len=16384, use_inputs_embeds=True, batch_size=2)

    model = RBLNLlavaNextForConditionalGeneration.from_pretrained(
        "trl-internal-testing/tiny-LlavaNextForConditionalGeneration",
        export=True,
        rbln_language_model=rbln_config,
    )
    assert model.rbln_config.language_model.max_seq_len == 16384
    assert model.rbln_config.language_model.batch_size == 2


def test_config_persistence_after_reload(model_id, tmp_path):
    """Test that configuration values persist correctly after saving and reloading."""
    save_dir = tmp_path / "saved_model"
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)

    os.makedirs(save_dir, exist_ok=True)

    # Use distinctive values to ensure we can detect them
    original_model = RBLNResNetForImageClassification.from_pretrained(
        model_id, rbln_image_size=112, rbln_batch_size=3, rbln_create_runtimes=False
    )
    original_model.save_pretrained(save_dir)

    # Reload and check
    reloaded_model = RBLNResNetForImageClassification.from_pretrained(save_dir, export=False)

    # Assert specific expected values
    assert reloaded_model.rbln_config.image_size == 112, "image_size configuration was not preserved"
    assert reloaded_model.rbln_config.batch_size == 3, "batch_size configuration was not preserved"
    assert reloaded_model.rbln_config.create_runtimes is True, "create_runtimes configuration should not be preserved"

    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)


@pytest.mark.parametrize(
    "invalid_param",
    [
        {"rbln_nonexistent_param": "value"},
        {"rbln_image_size": "not_an_integer"},  # Type error
        {"rbln_batch_size": -1},  # Negative value
        {"rbln_tensor_parallel_size": 32},  # Tensor parallel size is not supported
        {"rbln_npu": "RBLN-Unknown"},  # NPU is not supported
        {"rbln_device": 32},  # Device is not supported
    ],
)
def test_invalid_config_parameters(model_id, invalid_param):
    """Test robust handling of various invalid configuration parameters."""
    # check invaild params
    if "rbln_tensor_parallel_size" in invalid_param:
        if rebel.device_count() <= invalid_param["rbln_tensor_parallel_size"]:
            pytest.skip("Sufficient devices for invalid tensor_parallel_size check")

    if "rbln_device" in invalid_param:
        if rebel.device_count() - 1 <= invalid_param["rbln_device"]:
            pytest.skip("Sufficient devices for invalid rbln_device check")

    with pytest.raises((ValueError, TypeError)):
        _ = RBLNResNetForImageClassification.from_pretrained(model_id, **invalid_param)


def test_custom_class(model_id):
    class RBLNResNetModel(RBLNModel):
        @classmethod
        def _update_rbln_config(cls, *, rbln_config=None, **kwargs):
            input_info = [
                (
                    "pixel_values",
                    [rbln_config.batch_size, 3, rbln_config.image_size[0], rbln_config.image_size[1]],
                    "float32",
                )
            ]

            # Configure compilation settings
            rbln_config.set_compile_cfgs([RBLNCompileConfig(input_info=input_info)])
            return rbln_config

    def forward(self, pixel_values, **kwargs):
        return self.model[0](pixel_values)

    class RBLNResNetModelConfig(RBLNModelConfig):
        def __init__(self, batch_size: int = None, image_size: Optional[Tuple[int, int]] = None, **kwargs):
            super().__init__(**kwargs)
            self.batch_size = batch_size or 1
            self.image_size = image_size or (64, 64)

    RBLNAutoModel.register(RBLNResNetModel)
    RBLNAutoConfig.register(RBLNResNetModelConfig)
    my_model = RBLNResNetModel.from_pretrained(model_id, rbln_device=-1)
    random_image_input = torch.randn(1, 3, 64, 64)
    _ = my_model(random_image_input)

    with tempfile.TemporaryDirectory() as tmp_dir:
        my_model.save_pretrained(tmp_dir)
        _ = RBLNResNetModel.from_pretrained(tmp_dir, export=False)


if __name__ == "__main__":
    pytest.main()
