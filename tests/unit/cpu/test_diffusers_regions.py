import copy
import pickle

import pytest
import rebel
import torch
from diffusers import (
    AutoencoderKL,
    AutoencoderKLCosmos,
    AutoencoderKLTemporalDecoder,
    ControlNetModel,
    CosmosTransformer3DModel,
    PriorTransformer,
    UNet2DConditionModel,
    UNetSpatioTemporalConditionModel,
    VQModel,
)
from diffusers.models.attention import BasicTransformerBlock, TemporalBasicTransformerBlock
from diffusers.models.autoencoders.autoencoder_kl_cosmos import CosmosCausalConv3d, CosmosResnetBlock3d
from diffusers.models.resnet import ResnetBlock2D, ResnetBlockCondNorm2D, SpatioTemporalResBlock
from diffusers.models.transformers.transformer_cosmos import CosmosTransformerBlock
from rebel.core.compilation._impl import clone_nn_module

from optimum.rbln import (
    RBLNAutoencoderKL,
    RBLNAutoencoderKLConfig,
    RBLNAutoencoderKLCosmos,
    RBLNAutoencoderKLCosmosConfig,
    RBLNAutoencoderKLTemporalDecoder,
    RBLNAutoencoderKLTemporalDecoderConfig,
    RBLNControlNetModel,
    RBLNControlNetModelConfig,
    RBLNCosmosTransformer3DModel,
    RBLNCosmosTransformer3DModelConfig,
    RBLNPriorTransformer,
    RBLNPriorTransformerConfig,
    RBLNStableDiffusionXLPipelineConfig,
    RBLNUNet2DConditionModel,
    RBLNUNet2DConditionModelConfig,
    RBLNUNetSpatioTemporalConditionModel,
    RBLNUNetSpatioTemporalConditionModelConfig,
    RBLNVQModel,
    RBLNVQModelConfig,
)
from optimum.rbln.diffusers.models.nested_region import DIFFUSERS_REGION_CLASSES, wrap_diffusers_modules
from optimum.rbln.utils.nested_compile_region import NestedCompileRegionMixin


@pytest.mark.parametrize(
    "config_cls",
    [
        RBLNAutoencoderKLConfig,
        RBLNAutoencoderKLCosmosConfig,
        RBLNAutoencoderKLTemporalDecoderConfig,
        RBLNControlNetModelConfig,
        RBLNCosmosTransformer3DModelConfig,
        RBLNPriorTransformerConfig,
        RBLNUNet2DConditionModelConfig,
        RBLNUNetSpatioTemporalConditionModelConfig,
        RBLNVQModelConfig,
    ],
)
@pytest.mark.parametrize("enabled", [False, True])
def test_region_config_roundtrip(config_cls, enabled, tmp_path):
    assert config_cls().nested_compile_region is False
    config, unused = config_cls.initialize_from_kwargs(rbln_nested_compile_region=enabled)
    assert unused == {}
    assert config.nested_compile_region is enabled
    config.save(tmp_path)
    assert "nested_compile_region" not in (tmp_path / "rbln_config.json").read_text()
    assert config_cls.from_pretrained(tmp_path).nested_compile_region is False
    assert config_cls.from_pretrained(tmp_path, rbln_nested_compile_region=enabled).nested_compile_region is enabled


def test_pipeline_regions_are_selected_per_component():
    config = RBLNStableDiffusionXLPipelineConfig(unet={"nested_compile_region": True})
    assert config.unet.nested_compile_region is True
    assert config.vae.nested_compile_region is False
    assert config.get_load_overrides()["unet"]["nested_compile_region"] is True


class RepeatedBlocks(torch.nn.Module):
    def __init__(self, block):
        super().__init__()
        self.blocks = torch.nn.ModuleList([block, copy.deepcopy(block), copy.deepcopy(block)])

    def forward(self, hidden_states, *args):
        for block in self.blocks:
            hidden_states = block(hidden_states, *args)
        return hidden_states


@pytest.mark.parametrize("kind", ["transformer", "temporal_transformer", "resnet", "temporal_resnet"])
def test_block_regions_preserve_weights_copies_and_export(kind):
    torch.manual_seed(0)
    if kind == "transformer":
        block = BasicTransformerBlock(dim=32, num_attention_heads=4, attention_head_dim=8)
        inputs = (torch.randn(1, 4, 32),)
    elif kind == "temporal_transformer":
        block = TemporalBasicTransformerBlock(
            dim=32, time_mix_inner_dim=32, num_attention_heads=4, attention_head_dim=8
        )
        inputs = (torch.randn(2, 4, 32), 2)
    elif kind == "resnet":
        block = ResnetBlock2D(in_channels=32, out_channels=32, temb_channels=32, groups=8)
        inputs = (torch.randn(1, 32, 4, 4), torch.randn(1, 32))
    else:
        block = SpatioTemporalResBlock(in_channels=32, out_channels=32, temb_channels=32)
        inputs = (torch.randn(2, 32, 4, 4), torch.randn(2, 32), torch.zeros(1, 2))
    model = RepeatedBlocks(block).eval()
    original_classes = [type(module) for module in model.modules()]
    parameters = dict(model.named_parameters())
    with torch.no_grad():
        expected = model(*inputs)
        wrapped = wrap_diffusers_modules(model, DIFFUSERS_REGION_CLASSES)
        assert [type(module) for module in model.modules()] == original_classes
        assert not set(wrapped.modules()) & set(model.modules())
        assert dict(wrapped.named_parameters()).keys() == parameters.keys()
        assert all(parameter is parameters[name] for name, parameter in wrapped.named_parameters())
        for restored in (copy.deepcopy(wrapped), pickle.loads(pickle.dumps(wrapped)), clone_nn_module(wrapped)):
            assert restored.state_dict().keys() == model.state_dict().keys()
            torch.testing.assert_close(restored(*inputs), expected)
        exported = torch.export.export(clone_nn_module(wrapped), inputs, strict=True)
        calls = [node for node in exported.graph.nodes if "invoke_subgraph" in str(node.target)]
        # Temporal attention changes strides on its first call; that one-off body is inlined by PyTorch.
        assert len(calls) == (2 if kind == "temporal_transformer" else 3)
        assert len({call.args[1] for call in calls}) == 1
        torch.testing.assert_close(exported.module()(*inputs), expected)
        assert [type(module) for module in model.modules()] == original_classes
        flat = torch.export.export(model, inputs, strict=True)
        assert not any("invoke_subgraph" in str(node.target) for node in flat.graph.nodes)
        torch.testing.assert_close(flat.module()(*inputs), expected)


def _check_compile_export(monkeypatch, model_cls, model, config, enabled):
    config = model_cls._update_rbln_config(None, model, model.config, config)
    original_modules = dict(model.named_modules())
    original_classes = {name: type(module) for name, module in original_modules.items()}
    references = []
    flat_config = copy.deepcopy(config)
    flat_config.nested_compile_region = False
    monkeypatch.setattr(rebel, "compile_from_torch", lambda wrapped, **kwargs: references.append(wrapped))
    model_cls.get_compiled_model(model, flat_config)
    exports = []
    compiled_wrappers = []

    def export(wrapped, **kwargs):
        assert kwargs.get("model_trace_method") == ("export_strict" if enabled else None)
        assert dict(model.named_modules()) == original_modules
        assert {name: type(module) for name, module in model.named_modules()} == original_classes
        reference = references[len(exports)]
        reference_parameters = dict(reference.named_parameters())
        assert wrapped.state_dict().keys() == reference.state_dict().keys()
        assert all(parameter is reference_parameters[name] for name, parameter in wrapped.named_parameters())
        if enabled:
            assert not set(wrapped.modules()) & set(model.modules())
        inputs = tuple(torch.randn(shape) for _, shape, _ in kwargs["input_info"])
        with torch.no_grad():
            expected = reference(*inputs)
            torch.testing.assert_close(wrapped(*inputs), expected)
            exported = torch.export.export(clone_nn_module(wrapped), inputs, strict=True)
            calls = [node for node in exported.graph.nodes if "invoke_subgraph" in str(node.target)]
            assert bool(calls) is enabled
            assert set(exported.graph_signature.inputs_to_parameters.values()) <= set(
                dict(wrapped.named_parameters(remove_duplicate=False))
            )
            torch.testing.assert_close(exported.module()(*inputs), expected)
        exports.append(exported)
        compiled_wrappers.append(wrapped)
        return exported

    monkeypatch.setattr(rebel, "compile_from_torch", export)
    model_cls.get_compiled_model(model, config)
    assert dict(model.named_modules()) == original_modules
    assert {name: type(module) for name, module in model.named_modules()} == original_classes
    # A later flat compile must not strip regions from a previously constructed wrapper.
    wrapper_classes = [[type(module) for module in wrapped.modules()] for wrapped in compiled_wrappers]
    monkeypatch.setattr(rebel, "compile_from_torch", lambda wrapped, **kwargs: None)
    model_cls.get_compiled_model(model, flat_config)
    assert [[type(module) for module in wrapped.modules()] for wrapped in compiled_wrappers] == wrapper_classes
    return exports


@pytest.mark.parametrize("enabled", [False, True])
def test_controlnet_compile_exports_regions(monkeypatch, enabled):
    model = ControlNetModel(
        in_channels=4,
        down_block_types=("CrossAttnDownBlock2D",),
        block_out_channels=(32,),
        layers_per_block=3,
        transformer_layers_per_block=2,
        cross_attention_dim=32,
        attention_head_dim=4,
        norm_num_groups=8,
        conditioning_embedding_out_channels=(16,),
    ).eval()
    for conv in [*model.controlnet_down_blocks, model.controlnet_mid_block]:
        torch.nn.init.normal_(conv.weight, std=0.1)
    config = RBLNControlNetModelConfig(
        unet_sample_size=(4, 4),
        vae_sample_size=(4, 4),
        max_seq_len=4,
        nested_compile_region=enabled,
        create_runtimes=False,
    )
    assert len(_check_compile_export(monkeypatch, RBLNControlNetModel, model, config, enabled)) == 1


@pytest.mark.parametrize("enabled", [False, True])
def test_svd_unet_compile_exports_regions(monkeypatch, enabled):
    model = UNetSpatioTemporalConditionModel(
        sample_size=4,
        in_channels=4,
        out_channels=4,
        down_block_types=("CrossAttnDownBlockSpatioTemporal",),
        up_block_types=("CrossAttnUpBlockSpatioTemporal",),
        block_out_channels=(32,),
        layers_per_block=2,
        transformer_layers_per_block=2,
        cross_attention_dim=32,
        num_attention_heads=(4,),
        num_frames=2,
        addition_time_embed_dim=8,
        projection_class_embeddings_input_dim=24,
    ).eval()
    config = RBLNUNetSpatioTemporalConditionModelConfig(
        sample_size=(4, 4), nested_compile_region=enabled, create_runtimes=False
    )
    assert len(_check_compile_export(monkeypatch, RBLNUNetSpatioTemporalConditionModel, model, config, enabled)) == 1


@pytest.mark.parametrize("enabled", [False, True])
def test_unet_compile_exports_regions(monkeypatch, enabled):
    model = UNet2DConditionModel(
        sample_size=4,
        in_channels=4,
        out_channels=4,
        down_block_types=("CrossAttnDownBlock2D",),
        up_block_types=("CrossAttnUpBlock2D",),
        block_out_channels=(32,),
        layers_per_block=2,
        transformer_layers_per_block=2,
        cross_attention_dim=32,
        attention_head_dim=4,
        norm_num_groups=8,
    ).eval()
    config = RBLNUNet2DConditionModelConfig(
        sample_size=4, max_seq_len=4, nested_compile_region=enabled, create_runtimes=False
    )
    assert len(_check_compile_export(monkeypatch, RBLNUNet2DConditionModel, model, config, enabled)) == 1


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("temporal", [False, True])
def test_vae_encoder_and_decoder_export_regions(monkeypatch, enabled, temporal):
    common = {"block_out_channels": (32,), "layers_per_block": 2, "sample_size": 4}
    if temporal:
        model = AutoencoderKLTemporalDecoder(**common).eval()
        model_cls = RBLNAutoencoderKLTemporalDecoder
        config = RBLNAutoencoderKLTemporalDecoderConfig(
            sample_size=(4, 4),
            decode_chunk_size=2,
            uses_encoder=True,
            nested_compile_region=enabled,
            create_runtimes=False,
        )
    else:
        model = AutoencoderKL(**common).eval()
        model_cls = RBLNAutoencoderKL
        config = RBLNAutoencoderKLConfig(
            sample_size=(4, 4), uses_encoder=True, nested_compile_region=enabled, create_runtimes=False
        )
    assert len(_check_compile_export(monkeypatch, model_cls, model, config, enabled)) == 2


@pytest.mark.parametrize("enabled", [False, True])
def test_prior_compile_exports_regions(monkeypatch, enabled):
    num_layers = 3
    model = PriorTransformer(
        num_attention_heads=2,
        attention_head_dim=8,
        num_layers=num_layers,
        embedding_dim=16,
        num_embeddings=4,
        additional_embeddings=4,
    ).eval()
    config = RBLNPriorTransformerConfig(nested_compile_region=enabled, create_runtimes=False)
    exports = _check_compile_export(monkeypatch, RBLNPriorTransformer, model, config, enabled)
    assert len(exports) == 1
    calls = [node for node in exports[0].graph.nodes if "invoke_subgraph" in str(node.target)]
    assert len(calls) == (num_layers if enabled else 0)


def _count_calls(exported):
    return len([node for node in exported.graph.nodes if "invoke_subgraph" in str(node.target)])


@pytest.mark.parametrize("enabled", [False, True])
def test_vqmodel_compile_exports_regions(monkeypatch, enabled):
    # norm_type="spatial" makes the decoder use ResnetBlockCondNorm2D (the Kandinsky movq layout).
    model = VQModel(
        in_channels=3,
        out_channels=3,
        down_block_types=("DownEncoderBlock2D", "DownEncoderBlock2D"),
        up_block_types=("UpDecoderBlock2D", "UpDecoderBlock2D"),
        block_out_channels=(32, 64),  # SpatialNorm hardcodes 32 groups
        layers_per_block=2,
        latent_channels=4,
        norm_num_groups=8,
        num_vq_embeddings=16,
        vq_embed_dim=4,
        norm_type="spatial",
    ).eval()
    encoder_blocks = sum(type(m) is ResnetBlock2D for m in model.encoder.modules())
    decoder_blocks = sum(type(m) is ResnetBlockCondNorm2D for m in model.decoder.modules())
    assert encoder_blocks > 0 and decoder_blocks > 0
    assert not any(type(m) is ResnetBlock2D for m in model.decoder.modules())
    config = RBLNVQModelConfig(
        sample_size=(8, 8), uses_encoder=True, nested_compile_region=enabled, create_runtimes=False
    )
    exports = _check_compile_export(monkeypatch, RBLNVQModel, model, config, enabled)
    assert [_count_calls(e) for e in exports] == ([encoder_blocks, decoder_blocks] if enabled else [0, 0])


@pytest.mark.parametrize("enabled", [False, True])
def test_cosmos_vae_compile_exports_regions(monkeypatch, enabled):
    model = AutoencoderKLCosmos(
        in_channels=3,
        out_channels=3,
        latent_channels=4,
        encoder_block_out_channels=(8, 8, 8),
        decode_block_out_channels=(8, 8, 8),
        attention_resolutions=(),
        resolution=8,
        num_layers=2,
        patch_size=1,
        patch_type="rearrange",
        spatial_compression_ratio=2,
        temporal_compression_ratio=2,
    ).eval()
    encoder_blocks = sum(type(m) is CosmosResnetBlock3d for m in model.encoder.modules())
    decoder_blocks = sum(type(m) is CosmosResnetBlock3d for m in model.decoder.modules())
    assert encoder_blocks > 0 and decoder_blocks > 0
    conv_forward = CosmosCausalConv3d.forward
    config = RBLNAutoencoderKLCosmosConfig(
        num_frames=3,
        height=8,
        width=8,
        num_channels_latents=4,
        vae_scale_factor_temporal=2,
        vae_scale_factor_spatial=2,
        uses_encoder=True,
        nested_compile_region=enabled,
        create_runtimes=False,
    )
    exports = _check_compile_export(monkeypatch, RBLNAutoencoderKLCosmos, model, config, enabled)
    # The compile-time CausalConv3d forward swap must be undone.
    assert CosmosCausalConv3d.forward is conv_forward
    assert [_count_calls(e) for e in exports] == ([encoder_blocks, decoder_blocks] if enabled else [0, 0])


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("family", ["cosmos1", "predict2", "predict2_5", "transfer2_5"])
def test_cosmos_compile_exports_regions(monkeypatch, enabled, family):
    transfer = family == "transfer2_5"
    model = CosmosTransformer3DModel(
        in_channels=4,
        out_channels=4,
        num_attention_heads=2,
        attention_head_dim=16,
        num_layers=3,
        text_embed_dim=16,
        adaln_lora_dim=8,
        max_size=(2, 2, 2),
        patch_size=(1, 1, 1),
        extra_pos_embed_type="learnable" if family == "cosmos1" else None,
        controlnet_block_every_n=2 if transfer else None,
        img_context_dim_in=16 if transfer else None,
        img_context_dim_out=32,
        img_context_num_tokens=2,
    ).eval()
    config = RBLNCosmosTransformer3DModelConfig(
        num_latent_frames=2,
        latent_height=2,
        latent_width=2,
        max_seq_len=4,
        embedding_dim=16,
        uses_per_frame_timestep=family in ("predict2_5", "transfer2_5"),
        nested_compile_region=enabled,
        create_runtimes=False,
    )
    exports = _check_compile_export(monkeypatch, RBLNCosmosTransformer3DModel, model, config, enabled)
    assert model.transformer_blocks[0].forward.__func__ is CosmosTransformerBlock.forward
    calls = [node for node in exports[0].graph.nodes if "invoke_subgraph" in str(node.target)]
    assert len(calls) == (3 if enabled else 0)


def test_module_wrappers_preserve_aliases_and_isolate_training_state():
    block = BasicTransformerBlock(dim=32, num_attention_heads=4, attention_head_dim=8)
    model = torch.nn.ModuleList([block, block])
    wrapped = wrap_diffusers_modules(model, DIFFUSERS_REGION_CLASSES).eval()
    assert wrapped[0] is wrapped[1]
    assert isinstance(wrapped[0], NestedCompileRegionMixin)
    assert type(model[0]) is BasicTransformerBlock
    assert all(module.training for module in model.modules())
    assert all(not module.training for module in wrapped.modules())
    assert all(a is b for a, b in zip(model.parameters(), wrapped.parameters(), strict=True))
