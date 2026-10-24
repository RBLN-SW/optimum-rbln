import numpy as np
import pytest
import torch
from diffusers import AutoencoderKLFlux2, FlowMatchEulerDiscreteScheduler, Flux2KleinPipeline, Flux2Transformer2DModel
from PIL import Image
from transformers import Qwen3Config, Qwen3ForCausalLM

from optimum.rbln import (
    RBLNAutoencoderKLFlux2,
    RBLNFlux2KleinPipeline,
    RBLNFlux2KleinPipelineConfig,
    RBLNFlux2KleinTextEncoder,
    RBLNFlux2Transformer2DModel,
)
from optimum.rbln.diffusers.models.autoencoders.vae import _VAEDecoder, _VAEEncoder
from optimum.rbln.diffusers.models.transformers.flux2_text_encoder import Flux2KleinTextEncoderWrapper
from optimum.rbln.diffusers.models.transformers.transformer_flux2 import Flux2TransformerWrapper


class Tokenizer:
    def apply_chat_template(self, messages, **kwargs):
        return messages[0]["content"]

    def __call__(self, text, max_length, **kwargs):
        length = min(len(text), max_length)
        ids = torch.zeros(1, max_length, dtype=torch.long)
        ids[0, :length] = torch.arange(length) % 63 + 1
        return {"input_ids": ids, "attention_mask": (ids != 0).long()}


class TorchRuntime:
    """Exercise the real public adapters with the CPU wrapper output ABI."""

    def __init__(self, module):
        self.module = module

    def __call__(self, *args, **kwargs):
        output = self.module(*args, **kwargs)
        return output[0] if isinstance(output, tuple) and len(output) == 1 else output


def adapter(cls, native, config, wrappers):
    config = cls._update_rbln_config([], native, native.config, config)
    config.freeze()
    return cls(models=[TorchRuntime(wrapper) for wrapper in wrappers], config=native.config, rbln_config=config)


@pytest.mark.parametrize(("batch", "references"), [(1, 0), (2, 0), (1, 1), (1, 2)])
def test_full_native_pipeline_and_public_adapters_preserve_native_outputs(batch, references):
    torch.manual_seed(43)
    text = Qwen3ForCausalLM(
        Qwen3Config(
            vocab_size=64,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=28,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=8,
        )
    ).eval()
    vae = AutoencoderKLFlux2(
        block_out_channels=(8, 8, 8, 8),
        decoder_block_out_channels=(8, 8, 8, 8),
        down_block_types=("DownEncoderBlock2D",) * 4,
        up_block_types=("UpDecoderBlock2D",) * 4,
        layers_per_block=1,
        latent_channels=4,
        norm_num_groups=4,
        sample_size=64,
    ).eval()
    vae.bn.running_mean.copy_(torch.linspace(-0.5, 0.5, 16))
    vae.bn.running_var.copy_(torch.linspace(0.5, 1.5, 16))
    transformer = Flux2Transformer2DModel(
        in_channels=16,
        out_channels=16,
        num_layers=1,
        num_single_layers=1,
        attention_head_dim=8,
        num_attention_heads=2,
        joint_attention_dim=48,
        timestep_guidance_channels=8,
        axes_dims_rope=(2, 2, 2, 2),
        guidance_embeds=False,
    ).eval()
    native = Flux2KleinPipeline(
        scheduler=FlowMatchEulerDiscreteScheduler(),
        vae=vae,
        text_encoder=text,
        tokenizer=Tokenizer(),
        transformer=transformer,
        is_distilled=True,
    )
    config = RBLNFlux2KleinPipelineConfig(
        npu="RBLN-CR13",
        batch_size=batch,
        height=64,
        width=64,
        max_seq_len=32,
        reference_image_count=references,
    )
    compiled = RBLNFlux2KleinPipeline(
        scheduler=FlowMatchEulerDiscreteScheduler(),
        tokenizer=Tokenizer(),
        is_distilled=True,
        vae=adapter(RBLNAutoencoderKLFlux2, vae, config.vae, [_VAEEncoder(vae), _VAEDecoder(vae)]),
        text_encoder=adapter(
            RBLNFlux2KleinTextEncoder, text, config.text_encoder, [Flux2KleinTextEncoderWrapper(text)]
        ),
        transformer=adapter(
            RBLNFlux2Transformer2DModel, transformer, config.transformer, [Flux2TransformerWrapper(transformer)]
        ),
    )
    prompts = ["short", "a longer mixed padded prompt"][:batch]
    image = [Image.fromarray(np.full((64, 64, 3), 80 + 20 * i, dtype=np.uint8)) for i in range(references)]
    kwargs = {
        "prompt": prompts,
        "image": image or None,
        "height": 64,
        "width": 64,
        "max_sequence_length": 32,
        "num_inference_steps": 2,
        "guidance_scale": 1.0,
        "output_type": "np",
    }
    latents = torch.randn(batch, 16, 4, 4)
    with torch.inference_mode():
        expected = native(**kwargs, latents=latents.clone()).images
        # RBLNRuntimeVAEEncoder materializes contiguous inputs. Compare against
        # that same CPU boundary too: native PIL preprocessing can yield a
        # channels-last view whose CPU convolution reduction order differs.
        handle = vae.encoder.register_forward_pre_hook(lambda module, args: (args[0].contiguous(),))
        contiguous_reference = native(**kwargs, latents=latents.clone()).images
        handle.remove()
        actual = compiled(**kwargs, latents=latents.clone()).images
        repeated = compiled(**kwargs, latents=latents.clone(), return_dict=False)[0]
    np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=3e-6)
    np.testing.assert_array_equal(actual, contiguous_reference)
    np.testing.assert_array_equal(repeated, actual)
