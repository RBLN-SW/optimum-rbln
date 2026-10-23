import json
import os
import shutil
import unittest

import pytest
import torch
from PIL import Image
from transformers import (
    AutoConfig,
    AutoTokenizer,
    BertConfig,
    BertModel,
    T5EncoderModel,
    XLMRobertaConfig,
    XLMRobertaModel,
)

from optimum.rbln import (
    RBLNASTForAudioClassification,
    RBLNBartModel,
    RBLNBertForMaskedLM,
    RBLNBertForQuestionAnswering,
    RBLNBertModel,
    RBLNCLIPTextModel,
    RBLNColPaliForRetrieval,
    RBLNColQwen2ForRetrieval,
    RBLNDebertaV2ForSequenceClassification,
    RBLNDebertaV2Model,
    RBLNDistilBertForQuestionAnswering,
    RBLNDPTForDepthEstimation,
    RBLNGroundingDinoForObjectDetection,
    RBLNModernBertForMaskedLM,
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
                    torchscript=True,
                )
                _ = self.RBLN_CLASS.from_model(
                    model,
                    **self.RBLN_CLASS_KWARGS,
                    **self.HF_CONFIG_KWARGS,
                    preprocessors=preprocessors,  # For image_classification
                )

    def _inner_test_save_load(self, tmpdir):
        super()._inner_test_save_load(tmpdir)

        with pytest.raises(RuntimeError):
            _ = self.RBLN_CLASS.from_pretrained(
                tmpdir,
                export=False,
                rbln_device=[0, 1, 2, 3],
            )

    def test_failed_to_create_runtime(self):
        with pytest.raises(ValueError):  # noqa: PT011
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


class TestModernBertForMaskedLM(BaseTest.TestModel):
    RBLN_AUTO_CLASS = RBLNAutoModelForMaskedLM
    RBLN_CLASS = RBLNModernBertForMaskedLM
    HF_MODEL_ID = "hf-internal-testing/tiny-random-ModernBertForMaskedLM"
    # ModernBERT has no token_type_ids and forces the SDPA attention backend.
    GENERATION_KWARGS = {
        "input_ids": RANDOM_INPUT_IDS,
        "attention_mask": RANDOM_ATTN_MASK,
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


class TestDebertaV3Model(BaseTest.TestModel):
    RBLN_AUTO_CLASS = RBLNAutoModel
    RBLN_CLASS = RBLNDebertaV2Model
    HF_MODEL_ID = "microsoft/deberta-v3-base"
    HF_CONFIG_KWARGS = {"revision": "8ccc9b6f36199bec6961081d44eb72fb3f7353f3"}
    RBLN_CLASS_KWARGS = {
        "rbln_max_seq_len": 512,
        "rbln_batch_size": 1,
        "rbln_npu": os.environ.get("RBLN_TEST_NPU"),
    }
    GENERATION_KWARGS = {
        "input_ids": torch.full((1, 512), 4, dtype=torch.int64),
        "attention_mask": torch.ones((1, 512), dtype=torch.int64),
    }
    BATCH_SIZE = 1
    OUTPUT_NAME = "last_hidden_state"
    DEVICE = 0

    def test_numerical_parity(self):
        if os.environ.get("SAVE_ARTIFACTS_PATH") or os.environ.get("REUSE_ARTIFACTS_PATH"):
            self.skipTest("Artifact compatibility mode does not execute on a real NPU.")
        tokenizer = AutoTokenizer.from_pretrained(self.HF_MODEL_ID, **self.HF_CONFIG_KWARGS)
        reference, loading = self.get_hf_auto_class().from_pretrained(
            self.HF_MODEL_ID, **self.HF_CONFIG_KWARGS, output_loading_info=True
        )
        assert not loading.get("missing_keys")
        assert not loading.get("mismatched_keys")
        reference.float().eval()
        texts = [
            ("A man is eating pizza.", "A man eats something."),
            ("A black race car starts up in front of a crowd of people.", "A man is driving down a lonely road."),
            (
                "A long document discusses processors, language models, and parallel execution. " * 90,
                "The final paragraph.",
            ),
            ("", ""),
        ]
        dynamic = isinstance(self.model.rbln_config.max_seq_len, list)

        def compare(actual, expected, mask, case):
            assert actual.shape == expected.shape
            a, e = (actual[mask], expected[mask]) if self.OUTPUT_NAME == "last_hidden_state" else (actual, expected)
            pcc = torch.corrcoef(torch.stack([e.flatten(), a.flatten()]))[0, 1]
            nrmse = (a - e).square().mean().sqrt() / e.square().mean().sqrt().clamp_min(1e-8)
            print(
                "DEBERTA_PARITY "
                + json.dumps(
                    {
                        "test_class": type(self).__name__,
                        "case": case,
                        "npu": self.model.rbln_config.npu,
                        "reference_dtype": str(e.dtype),
                        "shape": list(actual.shape),
                        "active_tokens": mask.sum(-1).tolist(),
                        "pcc": float(pcc),
                        "normalized_rmse": float(nrmse),
                    }
                ),
                flush=True,
            )
            assert pcc >= 0.99
            if self.OUTPUT_NAME == "logits":
                assert nrmse <= 0.05
                assert torch.equal(actual.argmax(-1), expected.argmax(-1))

        with torch.no_grad():
            first_inputs = first_output = None
            for offset in range(0, len(texts), self.BATCH_SIZE):
                pairs = texts[offset : offset + self.BATCH_SIZE]
                inputs = tokenizer(
                    [pair[0] for pair in pairs],
                    [pair[1] for pair in pairs],
                    padding=True if dynamic else "max_length",
                    max_length=512,
                    truncation=True,
                    return_tensors="pt",
                )
                expected = getattr(reference(**inputs), self.OUTPUT_NAME)
                actual = getattr(self.model(**inputs), self.OUTPUT_NAME)
                compare(actual, expected, inputs["attention_mask"].bool(), f"paired-{offset}")
                torch.testing.assert_close(getattr(self.model(**inputs), self.OUTPUT_NAME), actual, rtol=0, atol=0)
                if first_inputs is None:
                    first_inputs, first_output = inputs, actual.clone()
            torch.testing.assert_close(
                getattr(self.model(**first_inputs), self.OUTPUT_NAME), first_output, rtol=0, atol=0
            )
            default_inputs = {"input_ids": first_inputs["input_ids"]}
            compare(
                getattr(self.model(**default_inputs), self.OUTPUT_NAME),
                getattr(reference(**default_inputs), self.OUTPUT_NAME),
                torch.ones_like(default_inputs["input_ids"], dtype=torch.bool),
                "default-inputs",
            )
            reloaded = self.RBLN_AUTO_CLASS.from_pretrained(self.get_rbln_local_dir(), rbln_device=self.DEVICE)
            torch.testing.assert_close(
                getattr(reloaded(**first_inputs), self.OUTPUT_NAME), first_output, rtol=0, atol=0
            )
            tuple_output = reloaded(**first_inputs, return_dict=False)
            assert len(tuple_output) == 1
            torch.testing.assert_close(tuple_output[0], first_output, rtol=0, atol=0)
            print(
                f"DEBERTA_STATE {type(self).__name__}: repeated requests, reload and tuple outputs passed", flush=True
            )


class TestDebertaV3ModelBatch2(TestDebertaV3Model):
    RBLN_CLASS_KWARGS = {**TestDebertaV3Model.RBLN_CLASS_KWARGS, "rbln_batch_size": 2}
    BATCH_SIZE = 2
    GENERATION_KWARGS = {name: tensor.repeat(2, 1) for name, tensor in TestDebertaV3Model.GENERATION_KWARGS.items()}


class TestDebertaV3ModelBucketing(TestDebertaV3Model):
    RBLN_CLASS_KWARGS = {**TestDebertaV3Model.RBLN_CLASS_KWARGS, "rbln_max_seq_len": [128, 512]}


class TestDebertaV3ForSequenceClassification(TestDebertaV3Model):
    RBLN_AUTO_CLASS = RBLNAutoModelForSequenceClassification
    RBLN_CLASS = RBLNDebertaV2ForSequenceClassification
    HF_MODEL_ID = "cross-encoder/nli-deberta-v3-base"
    HF_CONFIG_KWARGS = {"revision": "6c749ce3425cd33b46d187e45b92bbf96ee12ec7"}
    OUTPUT_NAME = "logits"


class TestDebertaV3ForSequenceClassificationBatch2(TestDebertaV3ForSequenceClassification):
    RBLN_CLASS_KWARGS = {**TestDebertaV3Model.RBLN_CLASS_KWARGS, "rbln_batch_size": 2}
    BATCH_SIZE = 2
    GENERATION_KWARGS = TestDebertaV3ModelBatch2.GENERATION_KWARGS


class TestDebertaV3ForSequenceClassificationBucketing(TestDebertaV3ForSequenceClassification):
    RBLN_CLASS_KWARGS = {**TestDebertaV3Model.RBLN_CLASS_KWARGS, "rbln_max_seq_len": [128, 512]}


class TestEncoderMaxSeqLenBucketing(unittest.TestCase):
    """Sequence-length bucketing for feature-extraction encoders.

    A single model is compiled for several ``max_seq_len`` values; the resulting buckets live in
    one compiled model and therefore share a single copy of the encoder weights. At inference the
    runtime routes each input to the smallest bucket that fits, and the forward pass pads up to that
    bucket and slices the outputs back. This test checks the routed outputs match the HF reference
    for the e5 (BERT) and bge-m3 (XLM-RoBERTa) families, using tiny random stand-ins.
    """

    BUCKETS = [32, 64, 128]
    TEST_SEQ_LENS = [20, 32, 100, 128]
    TEST_LEVEL = TestLevel.DEFAULT

    @classmethod
    def setUpClass(cls):
        env_coverage = TestLevel[os.environ.get("OPTIMUM_RBLN_TEST_LEVEL", "default").upper()]
        if env_coverage.value < cls.TEST_LEVEL.value:
            raise unittest.SkipTest(f"Skipped test : Test Coverage {env_coverage.name} < {cls.TEST_LEVEL.name}")

    def test_model_input_shapes_is_deprecated(self):
        from optimum.rbln.transformers.configuration_generic import RBLNTransformerEncoderConfig

        with self.assertLogs("optimum.rbln.utils.deprecation", level="WARNING") as logs:
            config = RBLNTransformerEncoderConfig(
                max_seq_len=[64, 128],
                model_input_shapes=[[1, 64], [1, 64]],
            )

        assert any("model_input_shapes" in msg for msg in logs.output)
        assert config.max_seq_len == [64, 128]
        assert not hasattr(config, "model_input_shapes")

    def _tiny_models(self):
        config_kwargs = {
            "vocab_size": 1024,
            "hidden_size": 64,
            "num_hidden_layers": 2,
            "num_attention_heads": 2,
            "intermediate_size": 128,
            "max_position_embeddings": 256,
        }
        return {
            "e5": (RBLNBertModel, BertModel(BertConfig(**config_kwargs)).eval()),
            "bge-m3": (RBLNXLMRobertaModel, XLMRobertaModel(XLMRobertaConfig(**config_kwargs)).eval()),
        }

    def test_bucketing_matches_reference(self):
        torch.manual_seed(42)
        for name, (rbln_cls, hf_model) in self._tiny_models().items():
            with self.subTest(model=name):
                # Reference outputs from the eager HF model, computed before compilation.
                inputs = {
                    seq_len: {
                        "input_ids": torch.randint(0, 1024, (1, seq_len), dtype=torch.int64),
                        "attention_mask": torch.ones(1, seq_len, dtype=torch.int64),
                    }
                    for seq_len in self.TEST_SEQ_LENS
                }
                references = {seq_len: hf_model(**args).last_hidden_state for seq_len, args in inputs.items()}

                save_dir = f"bucketing_{name.replace('-', '_')}-artifact"
                if os.path.exists(save_dir):
                    shutil.rmtree(save_dir)
                rbln_model = rbln_cls.from_model(
                    hf_model,
                    rbln_max_seq_len=self.BUCKETS,
                    rbln_device=0,
                    model_save_dir=save_dir,
                )
                try:
                    # One executor per bucket, all in a single compiled model (shared weights).
                    assert rbln_model.model[0].get_executor_count() == len(self.BUCKETS)

                    for seq_len, args in inputs.items():
                        output = rbln_model(**args, return_dict=False)[0]
                        reference = references[seq_len]
                        assert tuple(output.shape) == tuple(reference.shape)
                        max_diff = (output - reference).abs().max().item()
                        assert torch.allclose(output, reference, atol=0.02, rtol=0.01), (
                            f"{name} seq_len={seq_len}: outputs diverged (max abs diff {max_diff:.4g})"
                        )
                finally:
                    if os.path.exists(save_dir):
                        shutil.rmtree(save_dir)


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

    def _inner_test_save_load(self, tmpdir):
        super()._inner_test_save_load(tmpdir)
        self._inner_propagate_rbln_config(tmpdir)

    def _inner_propagate_rbln_config(self, tmpdir):
        with ContextRblnConfig(create_runtimes=False), self.subTest():
            rbln_config = {"vlm": {"vision_tower": {"device": 1}, "language_model": {"device": 2}}}
            model = self.RBLN_CLASS.from_pretrained(
                tmpdir,
                rbln_config=rbln_config,
            )
            assert model.rbln_config.vlm.vision_tower.device == 1
            assert model.rbln_config.vlm.language_model.device == 2


class TestColQwen2Model(BaseTest.TestModel):
    RBLN_AUTO_CLASS = None
    RBLN_CLASS = RBLNColQwen2ForRetrieval
    HF_MODEL_ID = "vidore/colqwen2-v1.0-hf"
    RBLN_CLASS_KWARGS = {
        "rbln_config": {
            "vlm": {
                "visual": {"max_seq_len": 512},
                "num_devices": 1,
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

    def _inner_test_save_load(self, tmpdir):
        super()._inner_test_save_load(tmpdir)
        self._inner_propagate_rbln_config(tmpdir)

    def _inner_propagate_rbln_config(self, tmpdir):
        with self.subTest():
            rbln_config = {"create_runtimes": False, "vlm": {"visual": {"device": 1}, "device": 2}}
            model = self.RBLN_CLASS.from_pretrained(
                tmpdir,
                rbln_config=rbln_config,
            )
            assert model.rbln_config.vlm.visual.device == 1
            assert model.rbln_config.vlm.device == 2
            assert not model.rbln_config.create_runtimes
            assert not model.rbln_config.vlm.create_runtimes


class TestColQwen2Model_BFloat16(TestColQwen2Model):
    HF_CONFIG_KWARGS = {
        "dtype": torch.bfloat16,
    }


class TestColQwen2Model_Auto(TestColQwen2Model):
    HF_CONFIG_KWARGS = {
        "dtype": "auto",
    }


class TestColQwen2Model_Float32(TestColQwen2Model):
    HF_CONFIG_KWARGS = {
        "dtype": torch.float32,
    }


class TestColQwen2_5Model(TestColQwen2Model):
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
                # The dummy device leaves whatever the buffer held (zeros, stale data, NaN bit
                # patterns), so make df, loc and scale finite and positive.
                params = [torch.nan_to_num(p, nan=0.0, posinf=0.0, neginf=0.0).abs() + 1e-6 for p in params]
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
    RBLN_CLASS_KWARGS = {
        "rbln_config": {
            "encoder": {"image_size": (1333, 1333)},
            "decoder": {"image_size": (1333, 1333)},
            "backbone": {"image_size": (1333, 1333)},
        }
    }

    @classmethod
    def setUpClass(cls):
        from transformers import GroundingDinoConfig

        config = GroundingDinoConfig.from_pretrained(
            cls.HF_MODEL_ID,
            # with `decoder_bbox_embed_share=True` v5 ties
            # `bbox_embed.0` into `bbox_embed.[1+]`, and its tie_weights
            # validation raises if no tie target exists.
            decoder_layers=2,
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
        cls.HF_CONFIG_KWARGS = {**cls.HF_CONFIG_KWARGS, "config": config, "ignore_mismatched_sizes": True}
        return super().setUpClass()

    def test_return_dict_false(self):
        inputs = self.get_inputs()
        tuple_output = self.model(**inputs, return_dict=False)
        dict_output = self.model(**inputs, return_dict=True)

        assert all(isinstance(output, torch.Tensor) for output in tuple_output)
        assert tuple_output[0].shape == dict_output.logits.shape
        assert tuple_output[1].shape == dict_output.pred_boxes.shape


if __name__ == "__main__":
    unittest.main()
