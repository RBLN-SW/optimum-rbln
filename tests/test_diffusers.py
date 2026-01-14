import unittest

import torch
from diffusers import ControlNetModel

from optimum.rbln import (
    RBLNAutoPipelineForImage2Image,
    # RBLNAutoPipelineForInpainting, FIXME: add inpainting tests
    RBLNAutoPipelineForText2Image,
    RBLNKandinskyV22CombinedPipeline,
    RBLNKandinskyV22Img2ImgCombinedPipeline,
    RBLNStableDiffusion3Img2ImgPipeline,
    RBLNStableDiffusion3Pipeline,
    RBLNStableDiffusionControlNetPipeline,
    RBLNStableDiffusionImg2ImgPipeline,
    RBLNStableDiffusionPipeline,
    RBLNStableDiffusionXLControlNetPipeline,
    RBLNStableDiffusionXLPipeline,
    RBLNStableVideoDiffusionPipeline,
    RBLNStableVideoDiffusionPipelineConfig,
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
        "rbln_config": {
            "vae": {
                "sample_size": (64, 64),
            },
            "batch_size": 2,
            "guidance_scale": 0.0,
        },
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
            "img_width": 64,
            "img_height": 64,
        },
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
        "rbln_config": {
            "img_width": 64,
            "img_height": 64,
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
        "rbln_config": {
            "img_width": 64,
            "img_height": 64,
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
        "rbln_config": {
            "img_width": 64,
            "img_height": 64,
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
        "rbln_config": {
            "img_width": 64,
            "img_height": 64,
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
        import copy

        rbln_class_kwargs_copy = copy.deepcopy(self.RBLN_CLASS_KWARGS)
        rbln_class_kwargs_copy["rbln_config"].update(rbln_config)
        model = self.RBLN_CLASS.from_pretrained(model_id=self.HF_MODEL_ID, **rbln_class_kwargs_copy)

        with self.subTest():
            self.assertEqual(model.prior_text_encoder.rbln_config.batch_size, 2)
            self.assertEqual(model.prior_prior.rbln_config.batch_size, 4)
            self.assertEqual(model.unet.rbln_config.batch_size, 2)


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
        "rbln_config": {
            "img_width": 64,
            "img_height": 64,
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

    def test_rbln_prefix_config(self):
        rbln_config = {
            "rbln_width": 32,
            "rbln_height": 32,
            "rbln_num_frames": 2,
            "rbln_decode_chunk_size": 2,
        }
        model = self.RBLN_CLASS.from_pretrained(
            self.HF_MODEL_ID,
            **rbln_config,
        )
        with self.subTest():
            assert model is not None

        with self.subTest():
            # optimum-rbln/src/optimum/rbln/diffusers/models/autoencoders/autoencoder_kl_temporal_decoder.py
            # chunk_frame() adjusts decode_chunk_size to the closest divisor of num_frames (excluding num_frames itself).
            # For num_frames=2, divisors are [1], so decode_chunk_size becomes 1.
            self.assertEqual(model.vae.rbln_config.decode_chunk_size, 1)
            self.assertEqual(model.vae.rbln_config.sample_size, (32, 32))
            self.assertEqual(model.unet.rbln_config.num_frames, 2)
            self.assertEqual(model.unet.rbln_config.sample_size, (16, 16))

    def test_rbln_config_object(self):
        config = RBLNStableVideoDiffusionPipelineConfig(
            width=32,
            height=32,
            num_frames=2,
            decode_chunk_size=2,
        )
        model = self.RBLN_CLASS.from_pretrained(
            self.HF_MODEL_ID,
            rbln_config=config,
        )
        with self.subTest():
            assert model is not None

        with self.subTest():
            # optimum-rbln/src/optimum/rbln/diffusers/models/autoencoders/autoencoder_kl_temporal_decoder.py
            # chunk_frame() adjusts decode_chunk_size to the closest divisor of num_frames (excluding num_frames itself).
            # For num_frames=2, divisors are [1], so decode_chunk_size becomes 1.
            self.assertEqual(model.vae.rbln_config.decode_chunk_size, 1)
            self.assertEqual(model.vae.rbln_config.sample_size, (32, 32))
            self.assertEqual(model.unet.rbln_config.num_frames, 2)
            self.assertEqual(model.unet.rbln_config.sample_size, (16, 16))


if __name__ == "__main__":
    unittest.main()
