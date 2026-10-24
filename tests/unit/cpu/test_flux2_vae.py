import json

import torch
from diffusers import AutoencoderKLFlux2

from optimum.rbln import RBLNAutoencoderKLFlux2, RBLNAutoencoderKLFlux2Config
from optimum.rbln.diffusers.models.autoencoders.vae import _VAEDecoder, _VAEEncoder


def native_vae():
    torch.manual_seed(41)
    model = AutoencoderKLFlux2(
        block_out_channels=(16, 32),
        decoder_block_out_channels=(16, 32),
        down_block_types=("DownEncoderBlock2D",) * 2,
        up_block_types=("UpDecoderBlock2D",) * 2,
        layers_per_block=1,
        latent_channels=4,
        norm_num_groups=8,
        sample_size=32,
    ).eval()
    model.bn.running_mean.copy_(torch.linspace(-1, 1, 16))
    model.bn.running_var.copy_(torch.linspace(0.5, 1.5, 16))
    return model


def test_native_vae_encode_decode_wrappers_preserve_outputs():
    model = native_vae()
    image = torch.randn(1, 3, 32, 32)
    latents = torch.randn(2, 4, 16, 16)
    with torch.inference_mode():
        encoded = _VAEEncoder(model)(image)
        decoded = _VAEDecoder(model)(latents)[0]
        expected_moments = model.encode(image).latent_dist.parameters
        expected_image = model.decode(latents).sample
    torch.testing.assert_close(encoded, expected_moments, rtol=0, atol=0)
    torch.testing.assert_close(decoded, expected_image, rtol=0, atol=0)


def test_reference_encoder_batch_one_decoder_request_batch_and_bn_roundtrip(tmp_path):
    model = native_vae()
    config = RBLNAutoencoderKLFlux2Config(
        npu="RBLN-CR13", batch_size=2, sample_size=(32, 32), reference_image_size=(48, 32)
    )
    config = RBLNAutoencoderKLFlux2._update_rbln_config([], model, model.config, config)
    assert config.compile_cfgs[0].input_info[0][1] == [1, 3, 48, 32]
    assert config.compile_cfgs[1].input_info[0][1] == [2, 4, 16, 16]
    config.save(tmp_path)
    restored = RBLNAutoencoderKLFlux2Config(npu="RBLN-CR13", **json.loads((tmp_path / "rbln_config.json").read_text()))
    torch.testing.assert_close(torch.tensor(restored.bn_running_mean), model.bn.running_mean, rtol=0, atol=0)
    torch.testing.assert_close(torch.tensor(restored.bn_running_var), model.bn.running_var, rtol=0, atol=0)
