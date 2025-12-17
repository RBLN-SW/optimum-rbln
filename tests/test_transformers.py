import json
import os
import shutil
import unittest

import torch
from PIL import Image
from rebel.core.exception import RBLNRuntimeError
from transformers import AutoConfig, T5EncoderModel

from optimum.rbln import (
    RBLNASTForAudioClassification,
    RBLNBartModel,
    RBLNBertForMaskedLM,
    RBLNBertForQuestionAnswering,
    RBLNBertModel,
    RBLNCLIPTextModel,
    RBLNColPaliForRetrieval,
    RBLNColQwen2ForRetrieval,
    RBLNDistilBertForQuestionAnswering,
    RBLNDPTForDepthEstimation,
    RBLNGroundingDinoForObjectDetection,
    RBLNPegasusModel,
    RBLNResNetForImageClassification,
    RBLNRobertaForMaskedLM,
    RBLNRobertaForSequenceClassification,
    RBLNT5EncoderModel,
    RBLNTimeSeriesTransformerForPrediction,
    RBLNViTForImageClassification,
    RBLNWav2Vec2ForCTC,
    RBLNWhisperForConditionalGeneration,
    RBLNXLMRobertaForSequenceClassification,
    RBLNXLMRobertaModel,
)
from optimum.rbln.transformers.models.auto.modeling_auto import (
    RBLNAutoModel,
    RBLNAutoModelForAudioClassification,
    RBLNAutoModelForCTC,
    RBLNAutoModelForDepthEstimation,
    RBLNAutoModelForImageClassification,
    RBLNAutoModelForMaskedLM,
    RBLNAutoModelForQuestionAnswering,
    RBLNAutoModelForSequenceClassification,
    RBLNAutoModelForSpeechSeq2Seq,
    RBLNAutoModelForTextEncoding,
    RBLNAutoModelForZeroShotObjectDetection,
)
from optimum.rbln.utils.runtime_utils import ContextRblnConfig
from optimum.rbln.utils.save_utils import maybe_load_preprocessors

from .test_base import BaseHubTest, BaseTest, TestLevel


RANDOM_INPUT_IDS = torch.randint(low=0, high=50, size=(1, 512), generator=torch.manual_seed(42), dtype=torch.int64)
RANDOM_ATTN_MASK = torch.randint(low=0, high=2, size=(1, 512), generator=torch.manual_seed(42), dtype=torch.int64)
RANDOM_TOKEN_TYPE_IDS = torch.randint(low=0, high=3, size=(1, 512), generator=torch.manual_seed(84), dtype=torch.int64)
RANDOM_AUDIO = torch.randn(size=(1, 160005), generator=torch.manual_seed(42), dtype=torch.float32)


class TestASTForAudioClassification(BaseTest.TestModel):
    RBLN_AUTO_CLASS = RBLNAutoModelForAudioClassification
    RBLN_CLASS = RBLNASTForAudioClassification
    # HF_MODEL_ID = "hf-internal-testing/tiny-random-ASTForAudioClassification"

    # FIXME:: Update to internal once enabled tiny model
    HF_MODEL_ID = "MIT/ast-finetuned-audioset-10-10-0.4593"
    GENERATION_KWARGS = {
        "input_values": torch.randn(size=(1, 1024, 128), generator=torch.manual_seed(42), dtype=torch.float32)
    }
    HF_CONFIG_KWARGS = {
        "num_hidden_layers": 1,
    }


class TestResNetModel(BaseTest.TestModel, BaseHubTest.TestHub):
    RBLN_AUTO_CLASS = RBLNAutoModelForImageClassification
    RBLN_CLASS = RBLNResNetForImageClassification
    HF_MODEL_ID = "hf-internal-testing/tiny-random-ResNetForImageClassification"
    GENERATION_KWARGS = {"pixel_values": torch.randn(1, 3, 224, 224, generator=torch.manual_seed(42))}
    TEST_LEVEL = TestLevel.ESSENTIAL

    def test_compile_from_model(self):
        if self.is_diffuser:
            pass  # TODO(diffuser from model)
        else:
            HF_CLASS = self.HF_AUTO_CLASS or self.HF_CLASS
            with ContextRblnConfig(device=self.DEVICE):
                preprocessors = maybe_load_preprocessors(self.HF_MODEL_ID)
                model = HF_CLASS.from_pretrained(
                    self.HF_MODEL_ID,
                    **self.HF_CONFIG_KWARGS,
                    **{
                        "torchscript": True,
                    },
                )
                _ = self.RBLN_CLASS.from_model(
                    model,
                    **self.RBLN_CLASS_KWARGS,
                    **self.HF_CONFIG_KWARGS,
                    preprocessors=preprocessors,  # For image_classification
                )

    def _inner_test_save_load(self, tmpdir):
        super()._inner_test_save_load(tmpdir)

        with self.assertRaises(RBLNRuntimeError):
            _ = self.RBLN_CLASS.from_pretrained(
                tmpdir,
                export=False,
                rbln_device=[0, 1, 2, 3],
            )

    def test_failed_to_create_runtime(self):
        with self.assertRaises(ValueError):
            _ = self.RBLN_CLASS.from_pretrained(
                self.HF_MODEL_ID,
                export=True,
                rbln_device=[0, 1, 2, 3],
            )


class TestBertModel(BaseTest.TestModel):
    RBLN_AUTO_CLASS = [RBLNAutoModelForTextEncoding, RBLNAutoModel]
    RBLN_CLASS = RBLNBertModel
    HF_MODEL_ID = "hf-tiny-model-private/tiny-random-BertModel"
    GENERATION_KWARGS = {
        "input_ids": RANDOM_INPUT_IDS,
        "attention_mask": RANDOM_ATTN_MASK,
        "token_type_ids": RANDOM_TOKEN_TYPE_IDS,
    }


class TestBertForQuestionAnswering(BaseTest.TestModel):
    RBLN_AUTO_CLASS = RBLNAutoModelForQuestionAnswering
    RBLN_CLASS = RBLNBertForQuestionAnswering
    HF_MODEL_ID = "hf-internal-testing/tiny-random-BertForQuestionAnswering"
    GENERATION_KWARGS = {
        "input_ids": RANDOM_INPUT_IDS,
        "attention_mask": RANDOM_ATTN_MASK,
        "token_type_ids": RANDOM_TOKEN_TYPE_IDS,
    }


class TestBertForMaskedLM(BaseTest.TestModel):
    RBLN_AUTO_CLASS = RBLNAutoModelForMaskedLM
    RBLN_CLASS = RBLNBertForMaskedLM
    HF_MODEL_ID = "hf-internal-testing/tiny-random-BertForMaskedLM"
    GENERATION_KWARGS = {
        "input_ids": RANDOM_INPUT_IDS,
        "attention_mask": RANDOM_ATTN_MASK,
        "token_type_ids": RANDOM_TOKEN_TYPE_IDS,
    }


class TestDPTModel(BaseTest.TestModel):
    RBLN_AUTO_CLASS = RBLNAutoModelForDepthEstimation
    RBLN_CLASS = RBLNDPTForDepthEstimation
    HF_MODEL_ID = "hf-internal-testing/tiny-random-DPTForDepthEstimation"
    GENERATION_KWARGS = {"pixel_values": torch.randn(1, 3, 32, 32, generator=torch.manual_seed(42))}


class TestT5EncoderModel(BaseTest.TestModel):
    RBLN_AUTO_CLASS = RBLNAutoModelForTextEncoding
    RBLN_CLASS = RBLNT5EncoderModel

    HF_MODEL_ID = "t5-small"
    GENERATION_KWARGS = {
        "input_ids": RANDOM_INPUT_IDS,
        "attention_mask": RANDOM_ATTN_MASK,
    }
    HF_CONFIG_KWARGS = {
        "num_layers": 1,
    }

    @classmethod
    def setUpClass(cls):
        REUSE_ARTIFACTS_PATH = os.environ.get("REUSE_ARTIFACTS_PATH", None)
        if REUSE_ARTIFACTS_PATH is None:
            if os.path.exists(cls.get_rbln_local_dir()):
                shutil.rmtree(cls.get_rbln_local_dir())

            t5_encoder_model = T5EncoderModel.from_pretrained(
                cls.HF_MODEL_ID, return_dict=False, **cls.HF_CONFIG_KWARGS
            )
            cls.model = cls.RBLN_CLASS.from_model(
                model=t5_encoder_model,
                model_save_dir=cls.get_rbln_local_dir(),
                **cls.RBLN_CLASS_KWARGS,
            )
        else:
            super().setUpClass()


class TestWhisperModel(BaseTest.TestModel):
    RBLN_AUTO_CLASS = RBLNAutoModelForSpeechSeq2Seq
    RBLN_CLASS = RBLNWhisperForConditionalGeneration
    HF_MODEL_ID = "openai/whisper-tiny"
    DEVICE = 0

    GENERATION_KWARGS = {
        "input_features": torch.randint(
            low=-1, high=1, size=(2, 80, 3000), generator=torch.manual_seed(42), dtype=torch.float32
        ),
        "max_new_tokens": 10,
    }
    HF_CONFIG_KWARGS = {
        "num_hidden_layers": 4,
        "encoder_layers": 4,
        "decoder_layers": 4,
    }
    RBLN_CLASS_KWARGS = {
        "rbln_token_timestamps": False,
        "rbln_batch_size": 2,
    }

    def test_generate(self):
        inputs = self.get_inputs()
        _ = self.model.generate(**inputs)
        # test_generate_language
        _ = self.model.generate(**inputs, language="en")
        # test_generate_language_auto_detect
        _ = self.model.generate(**inputs, language=None)[:, :5]

    def test_long_form_language_auto_detect_generate(self):
        inputs = self.get_inputs()

        inputs["input_features"] = torch.randint(
            low=-1, high=1, size=(2, 80, 3001), generator=torch.manual_seed(42), dtype=torch.float32
        )
        inputs["attention_mask"] = torch.ones(2, 3002, dtype=torch.int64)

        _ = self.model.generate(**inputs, temperature=0.0, language=None, return_timestamps=True)

    def test_long_form_generate(self):
        inputs = self.get_inputs()

        inputs["input_features"] = torch.randint(
            low=-1, high=1, size=(2, 80, 3001), generator=torch.manual_seed(42), dtype=torch.float32
        )
        inputs["attention_mask"] = torch.ones(2, 3002, dtype=torch.int64)

        _ = self.model.generate(**inputs, temperature=0.0, return_timestamps=True)

    def test_pipeline(self):
        import numpy as np
        from transformers import AutoProcessor, pipeline

        processor = AutoProcessor.from_pretrained(self.HF_MODEL_ID)

        pipe = pipeline(
            "automatic-speech-recognition",
            model=self.model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            chunk_length_s=30,
            return_timestamps=True,
            batch_size=2,
        )

        data = [np.random.rand(5000), np.random.rand(5000)]

        with torch.no_grad():
            _ = pipe(
                data,
                generate_kwargs={
                    "repetition_penalty": 1.3,
                    "num_beams": 1,
                },
                batch_size=2,
            )


class TestWhisperModel_TokenTimestamps(BaseTest.TestModel):
    RBLN_AUTO_CLASS = RBLNAutoModelForSpeechSeq2Seq
    RBLN_CLASS = RBLNWhisperForConditionalGeneration
    HF_MODEL_ID = "openai/whisper-tiny"

    GENERATION_KWARGS = {
        "input_features": torch.randint(
            low=0, high=50, size=(2, 80, 3000), generator=torch.manual_seed(42), dtype=torch.float32
        ),
        "max_new_tokens": 1,
        "return_token_timestamps": True,
        "return_timestamps": True,
        "generation_config": None,
    }

    HF_CONFIG_KWARGS = {
        "num_hidden_layers": 1,
        "encoder_layers": 1,
        "decoder_layers": 4,
    }

    RBLN_CLASS_KWARGS = {
        "rbln_token_timestamps": True,
        "rbln_batch_size": 2,
    }

    def test_generate(self):
        inputs = self.get_inputs()
        _ = self.model.generate(**inputs)

    def test_long_form_generate(self):
        inputs = self.get_inputs()

        inputs["input_features"] = torch.randint(
            low=0, high=50, size=(2, 80, 3001), generator=torch.manual_seed(42), dtype=torch.float32
        )
        inputs["attention_mask"] = torch.ones(2, 3002, dtype=torch.int64)

        _ = self.model.generate(**inputs, temperature=0.0)


class TestRBLNXLMRobertaForSequenceClassification(BaseTest.TestModel):
    RBLN_AUTO_CLASS = RBLNAutoModelForSequenceClassification
    RBLN_CLASS = RBLNXLMRobertaForSequenceClassification

    # FIXME:: Update to internal once enabled tiny model
    HF_MODEL_ID = "BAAI/bge-reranker-v2-m3"
    RBLN_CLASS_KWARGS = {"rbln_max_seq_len": 128}
    GENERATION_KWARGS = {
        "input_ids": torch.randint(low=0, high=50, size=(1, 128), generator=torch.manual_seed(42), dtype=torch.int64),
        "attention_mask": torch.randint(
            low=0, high=2, size=(1, 128), generator=torch.manual_seed(42), dtype=torch.int64
        ),
    }
    HF_CONFIG_KWARGS = {
        "num_hidden_layers": 1,
    }


class TestXLMRobertaModel(BaseTest.TestModel):
    RBLN_AUTO_CLASS = [RBLNAutoModelForTextEncoding, RBLNAutoModel]
    RBLN_CLASS = RBLNXLMRobertaModel
    TEST_LEVEL = TestLevel.FULL
    # HF_MODEL_ID = "hf-internal-testing/tiny-xlm-roberta"

    # FIXME:: Update to internal once enabled tiny model
    HF_MODEL_ID = "BAAI/bge-m3"
    RBLN_CLASS_KWARGS = {"rbln_max_seq_len": 128}
    GENERATION_KWARGS = {
        "input_ids": torch.randint(low=0, high=50, size=(1, 128), generator=torch.manual_seed(42), dtype=torch.int64),
        "attention_mask": torch.randint(
            low=0, high=2, size=(1, 128), generator=torch.manual_seed(42), dtype=torch.int64
        ),
    }
    HF_CONFIG_KWARGS = {
        "num_hidden_layers": 1,
        "vocab_size": 1024,
        "ignore_mismatched_sizes": True,
    }


class TestXLMRobertaModelWithTokenTypeIds(TestXLMRobertaModel):
    RBLN_CLASS_KWARGS = {
        "rbln_max_seq_len": 128,
        "rbln_model_input_names": ["input_ids", "attention_mask", "token_type_ids"],
    }
    GENERATION_KWARGS = {
        "input_ids": torch.randint(low=0, high=50, size=(1, 128), generator=torch.manual_seed(42), dtype=torch.int64),
        "attention_mask": torch.randint(
            low=0, high=2, size=(1, 128), generator=torch.manual_seed(42), dtype=torch.int64
        ),
        "token_type_ids": torch.randint(
            low=0, high=2, size=(1, 128), generator=torch.manual_seed(42), dtype=torch.int64
        ),
    }


class TestCLIPTextModel(BaseTest.TestModel):
    RBLN_AUTO_CLASS = RBLNAutoModelForTextEncoding
    RBLN_CLASS = RBLNCLIPTextModel
    HF_MODEL_ID = "peft-internal-testing/tiny-clip-text-2"
    GENERATION_KWARGS = {
        "input_ids": RANDOM_INPUT_IDS[:, :77],
        "attention_mask": RANDOM_ATTN_MASK[:, :77],
    }
    HF_CONFIG_KWARGS = {
        "num_hidden_layers": 1,
    }
    TEST_LEVEL = TestLevel.FULL


class TestColPaliModel(BaseTest.TestModel):
    RBLN_AUTO_CLASS = None
    RBLN_CLASS = RBLNColPaliForRetrieval
    HF_MODEL_ID = "thkim93/colpali-hf-1layer"
    RBLN_CLASS_KWARGS = {
        "rbln_config": {
            "vlm": {
                "language_model": {"prefill_chunk_size": 8192},
            }
        }
    }
    GENERATION_KWARGS = {
        "input_ids": torch.full((1, 1024), fill_value=257152, dtype=torch.int32),
        "attention_mask": torch.ones((1, 1024), dtype=torch.int32),
        "pixel_values": torch.randn(1, 3, 448, 448, generator=torch.manual_seed(42)),
    }


class TestColQwen2Model(BaseTest.TestModel):
    RBLN_AUTO_CLASS = None
    RBLN_CLASS = RBLNColQwen2ForRetrieval
    HF_MODEL_ID = "vidore/colqwen2-v1.0-hf"
    RBLN_CLASS_KWARGS = {
        "rbln_config": {
            "vlm": {
                "visual": {"max_seq_lens": 512},
                "tensor_parallel_size": 1,
                "max_seq_len": 32_768,
            }
        }
    }
    HF_CONFIG_KWARGS = {}  # Initialize empty to avoid sharing with other classes
    HF_CONFIG_KWARGS_PREPROCESSOR = {"max_pixels": 64 * 14 * 14}

    @classmethod
    def setUpClass(cls):
        config = AutoConfig.from_pretrained(cls.HF_MODEL_ID)

        # Reduce model size for faster testing
        vision_config = json.loads(config.vlm_config.vision_config.to_json_string())
        text_config = json.loads(config.vlm_config.text_config.to_json_string())
        vision_config["depth"] = 2
        vision_config["fullatt_block_indexes"] = [1]
        text_config["num_hidden_layers"] = 1
        text_config["layer_types"] = text_config["layer_types"][:1]

        cls.HF_CONFIG_KWARGS.update({"vlm_config": {"vision_config": vision_config, "text_config": text_config}})
        return super().setUpClass()

    def get_processor(self):
        from transformers import ColQwen2Processor

        processor = ColQwen2Processor.from_pretrained(self.HF_MODEL_ID)
        return processor

    def get_inputs(self):
        processor = self.get_processor()
        images = [
            Image.new("RGB", (64, 32), color="black"),
        ]
        inputs_image = processor(images=images)
        return inputs_image


class TestColQwen2Model_BFloat16(TestColQwen2Model):
    TEST_LEVEL = TestLevel.FULL
    HF_CONFIG_KWARGS = {
        "dtype": torch.bfloat16,
    }


class TestColQwen2Model_Auto(TestColQwen2Model):
    TEST_LEVEL = TestLevel.FULL
    HF_CONFIG_KWARGS = {
        "dtype": "auto",
    }


class TestColQwen2Model_Float32(TestColQwen2Model):
    TEST_LEVEL = TestLevel.FULL
    HF_CONFIG_KWARGS = {
        "dtype": torch.float32,
    }


class TestColQwen2_5Model(TestColQwen2Model):
    TEST_LEVEL = TestLevel.FULL
    HF_MODEL_ID = "Sahil-Kabir/colqwen2.5-v0.2-hf"


class TestWav2VecModel(BaseTest.TestModel):
    RBLN_AUTO_CLASS = RBLNAutoModelForCTC
    RBLN_CLASS = RBLNWav2Vec2ForCTC
    HF_MODEL_ID = "hf-internal-testing/tiny-random-Wav2Vec2ForCTC"
    GENERATION_KWARGS = {"input_values": RANDOM_AUDIO}
    RBLN_CLASS_KWARGS = {"rbln_max_seq_len": 160005}


class TestTimeSeriesTransformerForPrediction(BaseTest.TestModel):
    RBLN_AUTO_CLASS = None
    RBLN_CLASS = RBLNTimeSeriesTransformerForPrediction
    HF_MODEL_ID = "huggingface/time-series-transformer-tourism-monthly"
    GENERATION_KWARGS = {
        "static_categorical_features": torch.ones(1, 1, dtype=torch.long),
        "static_real_features": torch.ones(1, 1, dtype=torch.float),
        "past_time_features": torch.randn(1, 61, 2),
        "past_values": torch.randn(1, 61),
        "past_observed_mask": torch.ones(1, 61, dtype=torch.long),
        "future_time_features": torch.randn(1, 24, 2),
    }
    RBLN_CLASS_KWARGS = {"rbln_batch_size": 1, "rbln_num_parallel_samples": 100}
    DEVICE = 0

    def test_generate(self):
        REUSE_ARTIFACTS_PATH = os.environ.get("REUSE_ARTIFACTS_PATH", None)
        if REUSE_ARTIFACTS_PATH is not None:
            original_output_distribution = self.model._origin_model.output_distribution

            def patched_output_distribution(self, params, loc=None, scale=None, trailing_n=None):
                params = [p + 1e-6 for p in params]
                return original_output_distribution(params, loc=loc, scale=scale, trailing_n=trailing_n)

            self.model._origin_model.output_distribution = patched_output_distribution.__get__(
                self.model._origin_model
            )

        inputs = self.get_inputs()
        _ = self.model.generate(**inputs)

        if REUSE_ARTIFACTS_PATH is not None:
            self.model._origin_model.output_distribution = original_output_distribution


class TestBartModel(BaseTest.TestModel):
    RBLN_AUTO_CLASS = RBLNAutoModel
    RBLN_CLASS = RBLNBartModel
    HF_MODEL_ID = "hf-internal-testing/tiny-random-BartModel"
    RBLN_CLASS_KWARGS = {"rbln_max_seq_len": 100}
    GENERATION_KWARGS = {
        "input_ids": torch.randint(low=0, high=50, size=(1, 100), generator=torch.manual_seed(42), dtype=torch.int64),
        "attention_mask": torch.randint(
            low=0, high=2, size=(1, 100), generator=torch.manual_seed(42), dtype=torch.int64
        ),
    }


class TestPegasusModel(BaseTest.TestModel):
    RBLN_AUTO_CLASS = RBLNAutoModel
    RBLN_CLASS = RBLNPegasusModel
    HF_MODEL_ID = "hf-tiny-model-private/tiny-random-PegasusModel"
    RBLN_CLASS_KWARGS = {"rbln_max_seq_len": 100}
    GENERATION_KWARGS = {
        "input_ids": torch.randint(low=0, high=50, size=(1, 100), generator=torch.manual_seed(42), dtype=torch.int64),
        "attention_mask": torch.randint(
            low=0, high=2, size=(1, 100), generator=torch.manual_seed(42), dtype=torch.int64
        ),
        "decoder_input_ids": torch.randint(
            low=0, high=50, size=(1, 100), generator=torch.manual_seed(42), dtype=torch.int64
        ),
        "decoder_attention_mask": torch.randint(
            low=0, high=2, size=(1, 100), generator=torch.manual_seed(42), dtype=torch.int64
        ),
    }


class TestDistilBertForQuestionAnswering(BaseTest.TestModel):
    RBLN_AUTO_CLASS = RBLNAutoModelForQuestionAnswering
    RBLN_CLASS = RBLNDistilBertForQuestionAnswering
    HF_MODEL_ID = "hf-tiny-model-private/tiny-random-DistilBertForQuestionAnswering"
    GENERATION_KWARGS = {
        "input_ids": RANDOM_INPUT_IDS,
        "attention_mask": RANDOM_ATTN_MASK,
    }
    TEST_LEVEL = TestLevel.FULL


class TestRobertaForMaskedLM(BaseTest.TestModel):
    RBLN_AUTO_CLASS = RBLNAutoModelForMaskedLM
    RBLN_CLASS = RBLNRobertaForMaskedLM
    HF_MODEL_ID = "hf-tiny-model-private/tiny-random-RobertaForMaskedLM"
    GENERATION_KWARGS = {
        "input_ids": RANDOM_INPUT_IDS[:, :256],
        "attention_mask": RANDOM_ATTN_MASK[:, :256],
    }
    RBLN_CLASS_KWARGS = {"rbln_max_seq_len": 256, "rbln_batch_size": 1}
    HF_CONFIG_KWARGS = {
        "num_hidden_layers": 1,
    }
    TEST_LEVEL = TestLevel.FULL


class TestRobertaForSequenceClassification(BaseTest.TestModel):
    RBLN_AUTO_CLASS = RBLNAutoModelForSequenceClassification
    RBLN_CLASS = RBLNRobertaForSequenceClassification
    HF_MODEL_ID = "hf-tiny-model-private/tiny-random-RobertaForSequenceClassification"
    GENERATION_KWARGS = {
        "input_ids": RANDOM_INPUT_IDS[:, :256],
        "attention_mask": RANDOM_ATTN_MASK[:, :256],
    }
    RBLN_CLASS_KWARGS = {"rbln_max_seq_len": 256, "rbln_batch_size": 1}
    HF_CONFIG_KWARGS = {
        "num_hidden_layers": 1,
    }
    TEST_LEVEL = TestLevel.FULL


class TestViTForImageClassification(BaseTest.TestModel):
    RBLN_AUTO_CLASS = RBLNAutoModelForImageClassification
    RBLN_CLASS = RBLNViTForImageClassification
    HF_MODEL_ID = "WinKawaks/vit-tiny-patch16-224"
    GENERATION_KWARGS = {
        "pixel_values": torch.randn(1, 3, 224, 224, generator=torch.manual_seed(42)),
    }


class TestGroundingDinoModel(BaseTest.TestModel):
    RBLN_AUTO_CLASS = RBLNAutoModelForZeroShotObjectDetection
    RBLN_CLASS = RBLNGroundingDinoForObjectDetection
    HF_MODEL_ID = "IDEA-Research/grounding-dino-tiny"
    GENERATION_KWARGS = {
        "pixel_values": torch.randn(1, 3, 1333, 1333, generator=torch.manual_seed(42)),
        "input_ids": torch.randint(low=0, high=50, size=(1, 256), generator=torch.manual_seed(42), dtype=torch.int64),
    }
    TEST_LEVEL = TestLevel.FULL
    RBLN_CLASS_KWARGS = {
        "rbln_config": {
            "encoder": {"image_size": (1333, 1333)},
            "decoder": {"image_size": (1333, 1333)},
            "backbone": {"image_size": (1333, 1333)},
            "text_backbone": {
                "model_input_names": ["input_ids", "attention_mask", "token_type_ids", "position_ids"],
                "model_input_shapes": [(1, 256), (1, 256, 256), (1, 256), (1, 256)],
            },
        }
    }

    @classmethod
    def setUpClass(cls):
        from transformers import GroundingDinoConfig

        config = GroundingDinoConfig.from_pretrained(
            cls.HF_MODEL_ID,
            decoder_layers=1,
            decoder_attention_heads=2,
            encoder_layers=1,
            encoder_attention_heads=2,
            text_config={
                "num_attention_heads": 2,
                "num_hidden_layers": 1,
                "hidden_size": 32,
            },
            backbone_config={
                "attention_probs_dropout_prob": 0.0,
                "depths": [1, 1, 2, 1],
                "drop_path_rate": 0.1,
                "embed_dim": 12,
                "encoder_stride": 32,
                "hidden_act": "gelu",
                "hidden_dropout_prob": 0.0,
                "hidden_size": 48,
                "image_size": 224,
                "initializer_range": 0.02,
                "layer_norm_eps": 1e-05,
                "mlp_ratio": 4.0,
                "num_channels": 3,
                "num_heads": [1, 2, 3, 4],
                "num_layers": 4,
                "out_features": ["stage2", "stage3", "stage4"],
                "out_indices": [2, 3, 4],
                "patch_size": 4,
                "stage_names": ["stem", "stage1", "stage2", "stage3", "stage4"],
                "window_size": 7,
            },
        )
        cls.HF_CONFIG_KWARGS["config"] = config
        cls.HF_CONFIG_KWARGS["ignore_mismatched_sizes"] = True
        return super().setUpClass()


if __name__ == "__main__":
    unittest.main()
