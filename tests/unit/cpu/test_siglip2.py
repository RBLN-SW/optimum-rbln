import pytest
import torch
from transformers import Siglip2VisionConfig
from transformers.models.siglip2.modeling_siglip2 import Siglip2VisionEmbeddings


def native_embeddings(hidden_size=16):
    return (
        Siglip2VisionEmbeddings(
            Siglip2VisionConfig(hidden_size=hidden_size, num_patches=16, patch_size=2, num_channels=3)
        )
        .float()
        .eval()
    )


def test_packed_positions_preserve_all_geometries_and_native_padding():
    from optimum.rbln.transformers.models.siglip2.modeling_siglip2 import (
        Siglip2VisionEmbeddingsWrapper,
        patch_position_ids,
    )

    torch.manual_seed(42)
    source = native_embeddings()
    grids = [(h, w) for h in range(1, 10) for w in range(1, 9 // h + 1)]
    spatial_shapes = torch.tensor(grids)
    pixels = torch.randn(len(grids), 9, 12)
    wrapper = Siglip2VisionEmbeddingsWrapper(source, max_patches=9).eval()
    with torch.no_grad():
        expected = source(pixels, spatial_shapes)
        actual = wrapper(pixels, patch_position_ids(spatial_shapes, max_patches=9))
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_one_table_supports_changed_geometries_without_retracing():
    from optimum.rbln.transformers.models.siglip2.modeling_siglip2 import (
        Siglip2VisionEmbeddingsWrapper,
        patch_position_ids,
    )

    source = native_embeddings()
    wrapper = Siglip2VisionEmbeddingsWrapper(source, max_patches=9).eval()
    pixels = torch.randn(2, 9, 12)
    first_shapes = torch.tensor([[1, 9], [2, 4]])
    second_shapes = torch.tensor([[3, 3], [9, 1]])
    exported = torch.export.export(wrapper, (pixels, patch_position_ids(first_shapes, max_patches=9))).module()
    with torch.no_grad():
        for shapes in [first_shapes, second_shapes, first_shapes]:
            torch.testing.assert_close(
                exported(pixels, patch_position_ids(shapes, max_patches=9)),
                source(pixels, shapes),
                rtol=0,
                atol=0,
            )


@pytest.mark.parametrize("shapes", [[[0, 1]], [[1, 0]], [[-1, 2]], [[3, 4]], [[1, 10]], [[2, 2, 2]]])
def test_invalid_grids_are_rejected(shapes):
    from optimum.rbln.transformers.models.siglip2.modeling_siglip2 import patch_position_ids

    with pytest.raises(ValueError, match=r"grids|spatial_shapes"):
        patch_position_ids(torch.tensor(shapes), max_patches=9)


def test_grid_metadata_requires_integer_dtype():
    from optimum.rbln.transformers.models.siglip2.modeling_siglip2 import patch_position_ids

    with pytest.raises(ValueError, match="integer"):
        patch_position_ids(torch.tensor([[2.5, 3.0]]), max_patches=9)


@pytest.fixture
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def test_native_patch_budget_accepts_every_positive_grid(cpu_threads):
    from optimum.rbln.transformers.models.siglip2.modeling_siglip2 import (
        Siglip2VisionEmbeddingsWrapper,
        patch_position_ids,
    )

    source = (
        Siglip2VisionEmbeddings(Siglip2VisionConfig(hidden_size=16, num_patches=256, patch_size=2, num_channels=3))
        .float()
        .eval()
    )
    shapes = torch.tensor([(h, w) for h in range(1, 257) for w in range(1, 256 // h + 1)])
    pixels = torch.randn(len(shapes), 256, 12)
    wrapper = Siglip2VisionEmbeddingsWrapper(source, max_patches=256).eval()
    with torch.no_grad():
        torch.testing.assert_close(
            wrapper(pixels, patch_position_ids(shapes, max_patches=256)),
            source(pixels, shapes),
            rtol=0,
            atol=0,
        )


@pytest.mark.parametrize("return_dict", [False, True])
def test_full_model_wrapper_preserves_native_outputs_and_dynamic_geometry(return_dict):
    from transformers import Siglip2Config, Siglip2Model

    from optimum.rbln.transformers.models.siglip2.modeling_siglip2 import Siglip2Wrapper, patch_position_ids

    config = Siglip2Config(
        text_config={
            "vocab_size": 100,
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "max_position_embeddings": 16,
        },
        vision_config={
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_patches": 16,
            "patch_size": 2,
        },
    )
    torch.manual_seed(42)
    native = Siglip2Model(config).eval()
    wrapper = Siglip2Wrapper(native, max_patches=9).eval()
    ids = torch.randint(4, 100, (3, 16))
    text_mask = torch.ones_like(ids)
    text_mask[1, 7:] = 0
    pixels = torch.randn(2, 9, 12)
    with torch.no_grad():
        for shapes in [torch.tensor([[1, 9], [2, 4]]), torch.tensor([[3, 3], [9, 1]])]:
            mask = (torch.arange(9) < shapes.prod(-1, keepdim=True)).long()
            expected = native(
                input_ids=ids,
                pixel_values=pixels,
                attention_mask=text_mask,
                pixel_attention_mask=mask,
                spatial_shapes=shapes,
                return_dict=return_dict,
            )
            expected = expected.to_tuple() if return_dict else expected
            actual = wrapper(ids, pixels, text_mask, mask, patch_position_ids(shapes, max_patches=9))
            flattened = (*expected[:4], *expected[4], *expected[5])
            for a, e in zip(actual, flattened, strict=True):
                torch.testing.assert_close(a, e, rtol=0, atol=0)


@pytest.mark.parametrize("return_dict", [False, True])
def test_public_model_native_outputs_and_compile_limits(return_dict):
    from transformers import Siglip2Config, Siglip2Model

    from optimum.rbln import RBLNSiglip2Model, RBLNSiglip2ModelConfig

    native = Siglip2Model(
        Siglip2Config(
            text_config={
                "vocab_size": 100,
                "hidden_size": 32,
                "intermediate_size": 64,
                "num_hidden_layers": 1,
                "num_attention_heads": 4,
                "max_position_embeddings": 16,
            },
            vision_config={
                "hidden_size": 32,
                "intermediate_size": 64,
                "num_hidden_layers": 1,
                "num_attention_heads": 4,
                "num_patches": 16,
                "patch_size": 2,
            },
        )
    ).eval()
    config = RBLNSiglip2Model._update_rbln_config(
        None, native, native.config, RBLNSiglip2ModelConfig(batch_size=2, text_batch_size=3, max_patches=9)
    )
    assert config.max_seq_len == 16
    runtime = object.__new__(RBLNSiglip2Model)
    runtime.config = native.config
    runtime.rbln_config = config
    runtime.model = [RBLNSiglip2Model._wrap_model_if_needed(native, config)]
    inputs = {
        "input_ids": torch.randint(4, 100, (3, 16)),
        "pixel_values": torch.randn(2, 9, 12),
        "spatial_shapes": torch.tensor([[2, 4], [1, 9]]),
        "attention_mask": torch.ones((3, 16), dtype=torch.int64),
        "pixel_attention_mask": torch.tensor([[1] * 8 + [0], [1] * 9]),
        "position_ids": torch.arange(16).flip(0).unsqueeze(0),
    }
    with torch.no_grad():
        expected = native(**inputs, return_dict=return_dict)
        actual = runtime(**inputs, return_dict=return_dict)
    expected = expected.to_tuple() if return_dict else expected
    actual = actual.to_tuple() if return_dict else actual
    for a, e in zip((*actual[:4], *actual[4], *actual[5]), (*expected[:4], *expected[4], *expected[5]), strict=True):
        torch.testing.assert_close(a, e, rtol=0, atol=0)


@pytest.mark.parametrize("field", ["batch_size", "text_batch_size", "max_seq_len", "max_patches"])
@pytest.mark.parametrize("value", [0, -1, True, 1.5])
def test_config_rejects_invalid_capacities(field, value):
    from optimum.rbln import RBLNSiglip2ModelConfig

    with pytest.raises(ValueError, match=field):
        RBLNSiglip2ModelConfig(**{field: value})


def test_auto_model_discovers_siglip2(monkeypatch):
    from transformers import Siglip2Config, Siglip2Model

    from optimum.rbln import RBLNAutoModel, RBLNSiglip2Model

    native = Siglip2Model(
        Siglip2Config(
            text_config={
                "vocab_size": 20,
                "hidden_size": 16,
                "intermediate_size": 32,
                "num_hidden_layers": 1,
                "num_attention_heads": 2,
            },
            vision_config={
                "hidden_size": 16,
                "intermediate_size": 32,
                "num_hidden_layers": 1,
                "num_attention_heads": 2,
                "num_patches": 4,
            },
        )
    )
    monkeypatch.setattr(RBLNSiglip2Model, "from_model", classmethod(lambda cls, model, **kwargs: cls))
    assert RBLNAutoModel.from_model(native) is RBLNSiglip2Model
