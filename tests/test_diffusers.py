import os
import shutil
import unittest

import torch
from diffusers import ControlNetModel
from PIL import Image

from optimum.rbln import (
    RBLNAutoPipelineForImage2Image,
    # RBLNAutoPipelineForInpainting, FIXME: add inpainting tests
    RBLNAutoPipelineForText2Image,
    RBLNCosmos2_5_PredictBasePipeline,
    RBLNCosmos2_5_TransferPipeline,
    RBLNCosmos2TextToImagePipeline,
    RBLNCosmos2VideoToWorldPipeline,
    RBLNKandinskyV22CombinedPipeline,
    RBLNKandinskyV22Img2ImgCombinedPipeline,
    RBLNQwenImage21Pipeline,
    RBLNStableDiffusion3Img2ImgPipeline,
    RBLNStableDiffusion3Pipeline,
    RBLNStableDiffusionControlNetPipeline,
    RBLNStableDiffusionImg2ImgPipeline,
    RBLNStableDiffusionPipeline,
    RBLNStableDiffusionXLControlNetPipeline,
    RBLNStableDiffusionXLPipeline,
    RBLNStableVideoDiffusionPipeline,
)

from .test_base import BaseHubTest, BaseTest


class TestSDModel(BaseTest.TestModel, BaseHubTest.TestHub):
    RBLN_AUTO_CLASS = RBLNAutoPipelineForText2Image
    RBLN_CLASS = RBLNStableDiffusionPipeline
    HF_MODEL_ID = "hf-internal-testing/tiny-sd-pipe"
    GENERATION_KWARGS = {
        "prompt": "an illustration of a cute white cat riding a black horse on mars",
        "num_inference_steps": 3,
        "generator": torch.manual_seed(42),
    }
    # Fix incorrect tiny-sd-pipe's vae config.json sample_size
    RBLN_CLASS_KWARGS = {
        "rbln_config": {
            "vae": {
                "sample_size": (64, 64),
            },
            "unet": {
                "batch_size": 2,
            },
        }
    }


class TestSDModelBatch(BaseTest.TestModel):
    RBLN_AUTO_CLASS = RBLNAutoPipelineForText2Image
    RBLN_CLASS = RBLNStableDiffusionPipeline
    HF_MODEL_ID = "hf-internal-testing/tiny-sd-pipe"
    GENERATION_KWARGS = {
        "prompt": ["an illustration of a cute white cat riding a black horse on mars"] * 2,
        "num_inference_steps": 3,
        "generator": torch.manual_seed(42),
        "guidance_scale": 0.0,
    }
    # Fix incorrect tiny-sd-pipe's vae config.json sample_size
    RBLN_CLASS_KWARGS = {
        "rbln_batch_size": 2,
        "rbln_config": {
            "vae": {
                "sample_size": (64, 64),
            },
        },
        "rbln_guidance_scale": 0.0,
    }


class TestSDXLModel(BaseTest.TestModel):
    RBLN_AUTO_CLASS = RBLNAutoPipelineForText2Image
    RBLN_CLASS = RBLNStableDiffusionXLPipeline
    HF_MODEL_ID = "hf-internal-testing/tiny-sdxl-pipe"
    GENERATION_KWARGS = {
        "prompt": "an illustration of a cute white cat riding a black horse on mars",
        "num_inference_steps": 3,
        "generator": torch.manual_seed(42),
    }
    # Fix incorrect tiny-sd-pipe-xl's vae config.json sample_size
    RBLN_CLASS_KWARGS = {
        "rbln_config": {
            "unet": {"batch_size": 2},
            "vae": {
                "sample_size": (64, 64),
            },
        }
    }


class TestSDImg2ImgModel(BaseTest.TestModel):
    RBLN_AUTO_CLASS = RBLNAutoPipelineForImage2Image
    RBLN_CLASS = RBLNStableDiffusionImg2ImgPipeline
    HF_MODEL_ID = "hf-internal-testing/tiny-sd-pipe"
    GENERATION_KWARGS = {
        "prompt": "an illustration of a cute white cat riding a black horse on mars",
        "num_inference_steps": 3,
        "strength": 0.75,
        "generator": torch.manual_seed(42),
        "image": torch.randn(1, 3, 64, 64, generator=torch.manual_seed(42)).clamp(0, 1),
    }
    # Fix incorrect tiny-sd-pipe's vae config.json sample_size
    RBLN_CLASS_KWARGS = {
        "rbln_config": {
            "vae": {
                "sample_size": (64, 64),
            },
            "unet": {
                "batch_size": 2,
            },
        },
        "rbln_img_width": 64,
        "rbln_img_height": 64,
    }


class TestSDControlNetModel(BaseTest.TestModel):
    RBLN_AUTO_CLASS = RBLNAutoPipelineForText2Image
    RBLN_CLASS = RBLNStableDiffusionControlNetPipeline
    HF_MODEL_ID = "hf-internal-testing/tiny-stable-diffusion-torch"
    CONTROLNET_ID = "hf-internal-testing/tiny-controlnet"

    GENERATION_KWARGS = {
        "prompt": "the mona lisa",
        "image": torch.randn(1, 3, 64, 64, generator=torch.manual_seed(42)),
    }
    RBLN_CLASS_KWARGS = {
        "rbln_img_width": 64,
        "rbln_img_height": 64,
        "rbln_config": {
            "controlnet": {
                "batch_size": 2,
            },
            "unet": {
                "batch_size": 2,
            },
        },
    }

    @classmethod
    def setUpClass(cls):
        controlnet = ControlNetModel.from_pretrained(cls.CONTROLNET_ID)
        cls.RBLN_CLASS_KWARGS["controlnet"] = controlnet
        return super().setUpClass()


class TestSDXLControlNetModel(BaseTest.TestModel):
    RBLN_AUTO_CLASS = RBLNAutoPipelineForText2Image
    RBLN_CLASS = RBLNStableDiffusionXLControlNetPipeline
    HF_MODEL_ID = "hf-internal-testing/tiny-sdxl-pipe"
    CONTROLNET_ID = "hf-internal-testing/tiny-controlnet-sdxl"

    GENERATION_KWARGS = {
        "prompt": "aerial view, a futuristic research complex in a bright foggy jungle, hard lighting",
        "image": torch.randn(1, 3, 64, 64, generator=torch.manual_seed(42)),
        "controlnet_conditioning_scale": 0.5,
    }
    RBLN_CLASS_KWARGS = {
        "rbln_img_width": 64,
        "rbln_img_height": 64,
        "rbln_config": {
            "unet": {
                "batch_size": 2,
            },
            "controlnet": {
                "batch_size": 2,
            },
        },
    }

    @classmethod
    def setUpClass(cls):
        controlnet = ControlNetModel.from_pretrained(cls.CONTROLNET_ID)
        cls.RBLN_CLASS_KWARGS["controlnet"] = controlnet
        return super().setUpClass()


class TestSD3Model(BaseTest.TestModel):
    RBLN_AUTO_CLASS = RBLNAutoPipelineForText2Image
    RBLN_CLASS = RBLNStableDiffusion3Pipeline
    HF_MODEL_ID = "hf-internal-testing/tiny-sd3-pipe"
    GENERATION_KWARGS = {
        "prompt": "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k",
        "num_inference_steps": 3,
        "generator": torch.manual_seed(42),
    }
    RBLN_CLASS_KWARGS = {
        "rbln_config": {
            "transformer": {
                "batch_size": 2,
            }
        }
    }


class TestSD3Img2ImgModel(BaseTest.TestModel):
    RBLN_AUTO_CLASS = RBLNAutoPipelineForImage2Image
    RBLN_CLASS = RBLNStableDiffusion3Img2ImgPipeline
    HF_MODEL_ID = "hf-internal-testing/tiny-sd3-pipe"
    GENERATION_KWARGS = {
        "prompt": "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k",
        "generator": torch.manual_seed(42),
        "strength": 0.95,
        "image": torch.randn(1, 3, 64, 64, generator=torch.manual_seed(42)),
    }
    RBLN_CLASS_KWARGS = {
        "rbln_config": {
            "image_size": (64, 64),
            "transformer": {"batch_size": 2},
        },
    }


class TestSDMultiControlNetModel(BaseTest.TestModel):
    RBLN_AUTO_CLASS = RBLNAutoPipelineForText2Image
    RBLN_CLASS = RBLNStableDiffusionControlNetPipeline
    HF_MODEL_ID = "hf-internal-testing/tiny-stable-diffusion-torch"
    CONTROLNET_ID = "hf-internal-testing/tiny-controlnet"

    GENERATION_KWARGS = {
        "prompt": "the mona lisa",
        "image": [
            torch.randn(1, 3, 64, 64, generator=torch.manual_seed(42)),
            torch.randn(1, 3, 64, 64, generator=torch.manual_seed(42)),
        ],
        "controlnet_conditioning_scale": [1.0, 0.8],
        "negative_prompt": "monochrome, lowres, bad anatomy, worst quality, low quality",
    }
    RBLN_CLASS_KWARGS = {
        "rbln_img_width": 64,
        "rbln_img_height": 64,
        "rbln_config": {
            "controlnet": {
                "batch_size": 2,
            },
            "unet": {
                "batch_size": 2,
            },
        },
    }

    @classmethod
    def setUpClass(cls):
        controlnet = ControlNetModel.from_pretrained(cls.CONTROLNET_ID)
        controlnet_1 = ControlNetModel.from_pretrained(cls.CONTROLNET_ID)
        controlnets = [controlnet, controlnet_1]
        cls.RBLN_CLASS_KWARGS["controlnet"] = controlnets
        return super().setUpClass()


class TestKandinskyV22Model(BaseTest.TestModel):
    RBLN_AUTO_CLASS = RBLNAutoPipelineForText2Image
    RBLN_CLASS = RBLNKandinskyV22CombinedPipeline
    HF_MODEL_ID = "hf-internal-testing/tiny-random-kandinsky-v22-decoder"
    GENERATION_KWARGS = {
        "prompt": "red cat, 4k photo",
        "generator": torch.manual_seed(42),
        "num_inference_steps": 3,
    }
    RBLN_CLASS_KWARGS = {
        "rbln_img_width": 64,
        "rbln_img_height": 64,
        "rbln_config": {
            "prior_pipe": {"prior": {"batch_size": 2}},
            "decoder_pipe": {"unet": {"batch_size": 2}},
        },
    }

    def test_complicate_config(self):
        rbln_config = {
            "prior_pipe": {
                "text_encoder": {
                    "batch_size": 2,
                },
            },
            "prior_prior": {
                "batch_size": 4,
            },
            "unet": {
                "batch_size": 2,
            },
            "batch_size": 1,
        }
        rbln_class_kwargs_copy = self.RBLN_CLASS_KWARGS.copy()
        rbln_class_kwargs_copy["rbln_config"] = rbln_config
        with self.subTest():
            _ = self.RBLN_CLASS.from_pretrained(model_id=self.HF_MODEL_ID, **rbln_class_kwargs_copy)
        with self.subTest():
            assert _.prior_text_encoder.rbln_config.batch_size == 2
            assert _.prior_prior.rbln_config.batch_size == 4
            assert _.unet.rbln_config.batch_size == 2


class TestKandinskyV22Img2ImgModel(BaseTest.TestModel):
    RBLN_AUTO_CLASS = RBLNAutoPipelineForImage2Image
    RBLN_CLASS = RBLNKandinskyV22Img2ImgCombinedPipeline
    HF_MODEL_ID = "hf-internal-testing/tiny-random-kandinsky-v22-decoder"

    from torchvision.transforms.functional import to_pil_image

    image = torch.randn(3, 64, 64)
    image = to_pil_image(image)
    GENERATION_KWARGS = {
        "prompt": "A red cartoon frog, 4k",
        "generator": torch.manual_seed(42),
        "prior_num_inference_steps": 10,
        "num_inference_steps": 10,
        "image": image,
    }
    RBLN_CLASS_KWARGS = {
        "rbln_img_width": 64,
        "rbln_img_height": 64,
        "rbln_config": {
            "prior_pipe": {"prior": {"batch_size": 2}},
            "decoder_pipe": {"unet": {"batch_size": 2}},
        },
    }


class TestSVDImg2VidModel(BaseTest.TestModel):
    RBLN_CLASS = RBLNStableVideoDiffusionPipeline
    # ref: https://github.com/huggingface/diffusers/blob/b88fef47851059ce32f161d17f00cd16d94af96a/tests/pipelines/stable_video_diffusion/test_stable_video_diffusion.py#L64
    HF_MODEL_ID = "seinpark/tiny-stable-video-diffusion-img2vid"

    GENERATION_KWARGS = {
        "num_inference_steps": 2,
        "generator": torch.manual_seed(42),
        "image": torch.randn(1, 3, 32, 32, generator=torch.manual_seed(42)).uniform_(0, 1),
        "num_frames": 2,
        "decode_chunk_size": 2,
        "output_type": "pt",
        "height": 32,
        "width": 32,
    }
    RBLN_CLASS_KWARGS = {
        "rbln_config": {
            "width": 32,
            "height": 32,
            "num_frames": 2,
            "decode_chunk_size": 2,
        },
    }


class _MockCosmosSafetyChecker:
    def to(self, device=None, dtype=None):
        return self

    def check_text_safety(self, prompt):
        return True

    def check_video_safety(self, frames):
        return frames


class _MockedSafetyCheckerMixin:
    """Compile the Cosmos pipelines against a stand-in safety checker."""

    # goes into every from_pretrained call of the base test (initial export and reloads)
    HF_CONFIG_KWARGS = {"safety_checker": _MockCosmosSafetyChecker()}

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        local_dir = cls.get_rbln_local_dir()
        if os.path.isdir(local_dir):
            cls.model.register_to_config(safety_checker=(None, None))
            cls.model.save_config(local_dir)


class TestCosmos2_5PredictModel(_MockedSafetyCheckerMixin, BaseTest.TestModel):
    RBLN_CLASS = RBLNCosmos2_5_PredictBasePipeline
    # tiny-random pipeline mirroring nvidia/Cosmos-Predict2.5-2B (diffusers/base/post-trained):
    # shrunk transformer/text encoder, real-architecture Wan VAE (the RBLN wrapper's cache
    # shapes are tied to it), real tokenizer/scheduler configs.
    HF_MODEL_ID = "rbln/tiny-cosmos-2.5-predict"
    GENERATION_KWARGS = {
        "prompt": "dance monkey",
        "num_inference_steps": 2,
        "generator": torch.manual_seed(42),
        # 32x32 hits a compiler assert ("Invalid SHM stride size") in the Wan VAE graphs;
        # 64x64 is the smallest verified resolution.
        "height": 64,
        "width": 64,
        "num_frames": 5,
        "output_type": "np",
    }
    RBLN_CLASS_KWARGS = {
        "export": True,
        "rbln_config": {
            "height": 64,
            "width": 64,
            "num_frames": 5,
        },
    }


class TestCosmos2Text2ImageModel(_MockedSafetyCheckerMixin, BaseTest.TestModel):
    RBLN_CLASS = RBLNCosmos2TextToImagePipeline
    # tiny-random pipeline mirroring nvidia/Cosmos-Predict2-2B-Text2Image (shrunk
    # transformer/T5, real-architecture Wan VAE, real tokenizer/scheduler configs)
    HF_MODEL_ID = "rbln/tiny-cosmos2-text2image"
    GENERATION_KWARGS = {
        "prompt": "dance monkey",
        "num_inference_steps": 2,
        "generator": torch.manual_seed(42),
        "height": 64,
        "width": 64,
        "output_type": "np",
    }
    RBLN_CLASS_KWARGS = {
        "export": True,
        "rbln_config": {
            "height": 64,
            "width": 64,
            "num_frames": 1,
        },
    }


class TestCosmos2_5TransferModel(_MockedSafetyCheckerMixin, BaseTest.TestModel):
    RBLN_CLASS = RBLNCosmos2_5_TransferPipeline
    # tiny-random Transfer2.5: Predict2.5 tiny base + a 2-block ControlNet, loaded from its own
    # repo like the official layout (pipeline with controlnet null + separate controlnet weights)
    HF_MODEL_ID = "rbln/tiny-cosmos-2.5-transfer"
    CONTROLNET_ID = "rbln/tiny-cosmos-2.5-transfer-controlnet-edge"
    GENERATION_KWARGS = {
        "prompt": "dance monkey",
        "num_inference_steps": 2,
        "generator": torch.manual_seed(42),
        "height": 64,
        "width": 64,
        "output_type": "np",
    }
    RBLN_CLASS_KWARGS = {
        "export": True,
        "rbln_config": {
            "height": 64,
            "width": 64,
            "num_frames": 9,  # the compiled chunk size; the output length follows `controls`
        },
    }

    @classmethod
    def setUpClass(cls):
        from diffusers.models.controlnets.controlnet_cosmos import CosmosControlNetModel

        cls.RBLN_CLASS_KWARGS["controlnet"] = CosmosControlNetModel.from_pretrained(cls.CONTROLNET_ID)
        return super().setUpClass()

    def get_inputs(self):
        inputs = dict(self.GENERATION_KWARGS)
        inputs["controls"] = [Image.new("RGB", (64, 64), color=(i * 20, 0, 0)) for i in range(9)]
        return inputs


class TestCosmos2Video2WorldModel(_MockedSafetyCheckerMixin, BaseTest.TestModel):
    RBLN_CLASS = RBLNCosmos2VideoToWorldPipeline
    # tiny-random pipeline mirroring nvidia/Cosmos-Predict2-2B-Video2World
    # (in_channels 17, rope_scale (1, 3, 3); otherwise same recipe as the t2i tiny)
    HF_MODEL_ID = "rbln/tiny-cosmos2-video2world"
    GENERATION_KWARGS = {
        "prompt": "dance monkey",
        "image": torch.randn(1, 3, 64, 64, generator=torch.manual_seed(42)).uniform_(0, 1),
        "num_inference_steps": 2,
        "generator": torch.manual_seed(42),
        "height": 64,
        "width": 64,
        "num_frames": 9,
        "output_type": "np",
    }
    RBLN_CLASS_KWARGS = {
        "export": True,
        "rbln_config": {
            "height": 64,
            "width": 64,
            "num_frames": 9,
        },
    }


def _save_tiny_qwenimage21_pipeline(save_dir: str):
    # Tiny random Qwen-Image 2.1: the real processor and scheduler, shrunk text encoder, transformer and VAE.
    from diffusers import (
        AutoencoderKLQwenImage21,
        FlowMatchEulerDiscreteScheduler,
        QwenImage21Pipeline,
        QwenImage21Transformer2DModel,
    )
    from transformers import Qwen3VLConfig, Qwen3VLForConditionalGeneration, Qwen3VLProcessor

    model_id = "Qwen/Qwen-Image-2.1-Turbo"
    torch.manual_seed(0)
    real = Qwen3VLConfig.from_pretrained(model_id, subfolder="text_encoder")
    text_config = real.text_config.to_dict()
    text_config.update(
        hidden_size=256, intermediate_size=512, num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1
    )
    vision_config = real.vision_config.to_dict()
    vision_config.update(
        depth=2, hidden_size=128, intermediate_size=256, num_heads=1, out_hidden_size=256, deepstack_visual_indexes=[0]
    )
    text_encoder = Qwen3VLForConditionalGeneration(
        Qwen3VLConfig(
            text_config=text_config,
            vision_config=vision_config,
            image_token_id=real.image_token_id,
            video_token_id=real.video_token_id,
            vision_start_token_id=real.vision_start_token_id,
            vision_end_token_id=real.vision_end_token_id,
        )
    )
    # Heads of 128 channels: smaller heads hit a compiler bug in the attention reshape.
    transformer = QwenImage21Transformer2DModel(
        num_layers=3,
        num_attention_heads=4,
        attention_head_dim=128,
        axes_dims_rope=(32, 48, 48),
        context_in_dim=256,
        in_channels=16,
        out_channels=16,
    )
    with torch.no_grad():
        for param in transformer.parameters():
            param.normal_(0, 0.02)  # The default init makes a random transformer diverge in bf16.
    vae = AutoencoderKLQwenImage21(
        base_dim=16,
        decoder_base_dim=16,
        z_dim=16,
        dim_mult=[1, 1, 1, 1, 1],
        num_res_blocks=1,
        latents_mean=[0.0] * 16,
        latents_std=[1.0] * 16,
    )
    QwenImage21Pipeline(
        scheduler=FlowMatchEulerDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler"),
        vae=vae,
        text_encoder=text_encoder,
        processor=Qwen3VLProcessor.from_pretrained(model_id, subfolder="processor"),
        transformer=transformer,
    ).save_pretrained(save_dir)


class TestQwenImage21Model(BaseTest.TestModel):
    RBLN_CLASS = RBLNQwenImage21Pipeline
    HF_MODEL_ID = "tiny-qwen-image-2.1"  # Built at setup by `_save_tiny_qwenimage21_pipeline`.
    GENERATION_KWARGS = {
        "prompt": "dance monkey",
        "num_inference_steps": 2,
        "generator": torch.manual_seed(42),
        "height": 256,
        "width": 256,
        "output_type": "np",
    }
    EDIT_KWARGS = {
        "prompt": "dance monkey",
        "image": Image.new("RGB", (256, 256), color=(120, 40, 200)),
        "num_inference_steps": 2,
        "generator": torch.manual_seed(42),
        "output_resolution": 256,  # The condition image and the output stay at 256x256.
        "output_type": "np",
    }
    RBLN_CLASS_KWARGS = {
        "export": True,
        "rbln_config": {
            "image_size": (256, 256),
            # Compiled for editing too, as by default. Text-to-image leaves the vision and VAE encoders unused.
            "condition_image_size": (256, 256),
            "max_seq_len": 128,
        },
    }

    @classmethod
    def setUpClass(cls):
        if not os.path.isdir(cls.HF_MODEL_ID):
            _save_tiny_qwenimage21_pipeline(cls.HF_MODEL_ID)
        return super().setUpClass()

    @classmethod
    def tearDownClass(cls):
        super().tearDownClass()
        shutil.rmtree(cls.HF_MODEL_ID, ignore_errors=True)

    def test_generate_edit(self):
        # The same compiled pipeline edits: the condition image goes through the vision and VAE encoders.
        image = self.model(**self.EDIT_KWARGS).images[0]
        assert image.shape[:2] == (256, 256)


if __name__ == "__main__":
    unittest.main()
