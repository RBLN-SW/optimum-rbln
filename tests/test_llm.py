import json
import os
import unittest
import warnings

import pytest
import torch
from PIL import Image
from transformers import AutoConfig, AutoProcessor, AutoTokenizer

from optimum.rbln import (
    RBLNAutoModel,
    RBLNAutoModelForCausalLM,
    RBLNAutoModelForImageTextToText,
    RBLNAutoModelForSeq2SeqLM,
    RBLNAutoModelForVision2Seq,
    RBLNBartForConditionalGeneration,
    RBLNBlip2ForConditionalGeneration,
    RBLNExaoneForCausalLM,
    RBLNGemma3ForCausalLM,
    RBLNGemma3ForConditionalGeneration,
    RBLNGPT2LMHeadModel,
    RBLNGPT2Model,
    RBLNIdefics3ForConditionalGeneration,
    RBLNLlamaForCausalLM,
    RBLNLlamaModel,
    RBLNLlavaForConditionalGeneration,
    RBLNLlavaNextForConditionalGeneration,
    RBLNLoRAAdapterConfig,
    RBLNMistralForCausalLM,
    RBLNMistralModel,
    RBLNOPTForCausalLM,
    RBLNOPTModel,
    RBLNPegasusForConditionalGeneration,
    RBLNPhiForCausalLM,
    RBLNPhiModel,
    RBLNQwen2_5_VLForConditionalGeneration,
    RBLNQwen2ForCausalLM,
    RBLNQwen2Model,
    RBLNQwen2MoeForCausalLM,
    RBLNQwen2VLForConditionalGeneration,
    RBLNQwen3ForCausalLM,
    RBLNQwen3Model,
    RBLNQwen3MoeForCausalLM,
    RBLNT5ForConditionalGeneration,
)

from .test_base import BaseTest, DisallowedTestBase, TestLevel


RANDOM_ATTN_MASK = torch.randint(low=0, high=2, size=(1, 512), generator=torch.manual_seed(42), dtype=torch.int64)
RANDOM_TOKEN_TYPE_IDS = torch.randint(low=0, high=3, size=(1, 512), generator=torch.manual_seed(84), dtype=torch.int64)
RANDOM_INPUT_FEATURES = torch.randint(
    low=0, high=50, size=(1, 80, 3000), generator=torch.manual_seed(42), dtype=torch.float32
)


class LLMTest:
    class TestLLM(BaseTest.TestModel):
        RBLN_AUTO_CLASS = RBLNAutoModelForCausalLM
        DEVICE = None  # Use device to run
        PROMPT = "Who are you?"
        IS_MULTIMODAL = False
        HF_CONFIG_KWARGS_PREPROCESSOR = {"padding_side": "left"}

        def get_tokenizer(self):
            PreProcessor = AutoProcessor if self.IS_MULTIMODAL else AutoTokenizer

            if getattr(self, "_tokenizer", None) is None:
                self._tokenizer = PreProcessor.from_pretrained(self.HF_MODEL_ID, **self.HF_CONFIG_KWARGS_PREPROCESSOR)
            return self._tokenizer

        def get_inputs(self):
            self.get_tokenizer().pad_token = self.get_tokenizer().eos_token
            inputs = self.get_tokenizer()(self.PROMPT, return_tensors="pt", padding=True)
            if self.model.can_generate():
                inputs["max_new_tokens"] = 20
                inputs["do_sample"] = False

            return inputs

        def postprocess(self, inputs, output):
            input_len = inputs["input_ids"].shape[-1]
            batch_size = inputs["input_ids"].shape[0]
            generated_texts = []
            for i in range(batch_size):
                input_len = inputs["input_ids"][i].shape[-1]
                generated_text = self.get_tokenizer().decode(
                    output[i][input_len:], skip_special_tokens=True, clean_up_tokenization_spaces=True
                )
                generated_texts.append(generated_text)
            if batch_size == 1:
                return generated_texts[0]

            return generated_texts

        def _test_output_hidden_states_generation(self):
            REUSE_ARTIFACTS_PATH = os.environ.get("REUSE_ARTIFACTS_PATH", None)

            inputs = self.get_inputs()
            if self.model.can_generate():
                inputs["max_new_tokens"] = 4
                inputs["do_sample"] = False
                inputs["return_dict_in_generate"] = True
                inputs["output_hidden_states"] = True
                output = self.model.generate(**inputs)
                if REUSE_ARTIFACTS_PATH is not None:
                    return

                # Check hidden states Shape and Type
                self.assertTrue(len(output.hidden_states) == inputs["max_new_tokens"])
                self.assertTrue(isinstance(output.hidden_states, tuple))
                num_hidden_layers = (
                    self.HF_CONFIG_KWARGS["num_hidden_layers"]
                    if "num_hidden_layers" in self.HF_CONFIG_KWARGS
                    else self.HF_CONFIG_KWARGS["text_config"]["num_hidden_layers"]
                )
                self.assertTrue(len(output.hidden_states[0]) == (num_hidden_layers + 1))
                self.assertTrue(isinstance(output.hidden_states[0], tuple))
                test_hidden_states = output.hidden_states[0][1]
            else:
                inputs["output_hidden_states"] = True
                output = self.model.forward(**inputs)
                if REUSE_ARTIFACTS_PATH is not None:
                    return
                # Check hidden states Shape and Type
                self.assertTrue(len(output.hidden_states) == (self.HF_CONFIG_KWARGS["num_hidden_layers"] + 1))
                self.assertTrue(isinstance(output.hidden_states, tuple))
                test_hidden_states = output.hidden_states[1]

                # Check last hidden state corresponds to the last hidden state of the prefill stage
                self.assertTrue(torch.allclose(output.last_hidden_state, output.hidden_states[-1]))

            # Check well-masked hidden states corresponds to attention mask
            for b_idx, bmask in enumerate(inputs.attention_mask):
                masked_indices = torch.where(bmask == 0)[0]
                unmasked_indices = torch.where(bmask == 1)[0]
                masked = torch.allclose(test_hidden_states[b_idx][masked_indices], torch.zeros([1]))
                # 1. all masked hidden states are zero
                self.assertTrue(masked)
                unmasked_tensor = test_hidden_states[b_idx][unmasked_indices]
                avg_unmasked_tensor = torch.mean(unmasked_tensor, dim=-1)
                approx_zero = torch.isclose(avg_unmasked_tensor, torch.zeros([1]), atol=1e-5)
                # 2. check if unmasked hidden states are not masked
                self.assertFalse(torch.any(approx_zero).item())

    class TestLLMWithoutLMHead(TestLLM):
        RBLN_AUTO_CLASS = RBLNAutoModel


class TestMistralForCausalLM(LLMTest.TestLLM):
    RBLN_CLASS = RBLNMistralForCausalLM
    HF_MODEL_ID = "openaccess-ai-collective/tiny-mistral"
    HF_CONFIG_KWARGS = {"num_hidden_layers": 1, "max_position_embeddings": 1024, "sliding_window": 512}


class TestMistralModel(LLMTest.TestLLMWithoutLMHead):
    RBLN_CLASS = RBLNMistralModel
    HF_MODEL_ID = "openaccess-ai-collective/tiny-mistral"
    HF_CONFIG_KWARGS = {"num_hidden_layers": 1, "max_position_embeddings": 1024, "sliding_window": 512}


class TestQwen2ForCausalLM(LLMTest.TestLLM):
    RBLN_CLASS = RBLNQwen2ForCausalLM
    HF_MODEL_ID = "Qwen/Qwen2-0.5B-Instruct"
    HF_CONFIG_KWARGS = {"num_hidden_layers": 1, "layer_types": ["full_attention"], "max_position_embeddings": 1024}


class TestQwen2Model(LLMTest.TestLLMWithoutLMHead):
    RBLN_CLASS = RBLNQwen2Model
    HF_MODEL_ID = "Qwen/Qwen2-0.5B-Instruct"
    HF_CONFIG_KWARGS = {"num_hidden_layers": 1, "layer_types": ["full_attention"], "max_position_embeddings": 1024}


class TestQwen2ForCausalLM_OutputHiddenStates(TestQwen2ForCausalLM):
    PROMPT = ["Who are you?", "What is the capital of France?"]
    RBLN_CLASS_KWARGS = {"rbln_config": {"output_hidden_states": True, "batch_size": 2}}

    def get_inputs(self):
        inputs = super().get_inputs()
        inputs["return_dict_in_generate"] = True
        inputs["output_hidden_states"] = True
        inputs["max_new_tokens"] = 4
        return inputs

    def test_generate(self):
        self._test_output_hidden_states_generation()


class TestQwen2Model_OutputHiddenStates(TestQwen2Model):
    PROMPT = ["Who are you?", "What is the capital of France?"]
    RBLN_CLASS_KWARGS = {"rbln_config": {"output_hidden_states": True, "batch_size": 2}}

    def test_generate(self):
        self._test_output_hidden_states_generation()


class TestQwen3ForCausalLM(LLMTest.TestLLM):
    RBLN_CLASS = RBLNQwen3ForCausalLM
    HF_MODEL_ID = "trl-internal-testing/tiny-Qwen3ForCausalLM"
    HF_CONFIG_KWARGS = {
        "num_hidden_layers": 1,
        "layer_types": ["full_attention"],
        "max_position_embeddings": 1024,
        "revision": "397c180b0ded9c45c33bbce7f88df86bb2d571d4",
    }


class TestQwen3ForCausalLM_UAM(TestQwen3ForCausalLM):
    RBLN_AUTO_CLASS = RBLNAutoModelForCausalLM
    RBLN_CLASS_KWARGS = {"rbln_config": {"use_attention_mask": True}}


class TestQwen3Model(LLMTest.TestLLMWithoutLMHead):
    RBLN_AUTO_CLASS = RBLNAutoModel
    RBLN_CLASS = RBLNQwen3Model
    HF_MODEL_ID = "trl-internal-testing/tiny-Qwen3ForCausalLM"
    HF_CONFIG_KWARGS = {"num_hidden_layers": 1, "layer_types": ["full_attention"], "max_position_embeddings": 1024}


class TestQwen2MoeForCausalLM(LLMTest.TestLLM):
    RBLN_CLASS = RBLNQwen2MoeForCausalLM
    # HF_MODEL_ID ="peft-internal-testing/tiny-random-qwen-1.5-MoE"
    HF_MODEL_ID = "Qwen/Qwen1.5-MoE-A2.7B"
    HF_CONFIG_KWARGS = {"num_hidden_layers": 1, "max_position_embeddings": 1024}
    TEST_LEVEL = TestLevel.FULL


class TestQwen3MoeForCausalLM(LLMTest.TestLLM):
    RBLN_CLASS = RBLNQwen3MoeForCausalLM
    HF_MODEL_ID = "katuni4ka/tiny-random-qwen3moe"

    @classmethod
    def setUpClass(cls):
        config = AutoConfig.from_pretrained(cls.HF_MODEL_ID)
        config.num_hidden_layers = 3
        config.max_position_embeddings = 4096
        config.hidden_size = 128
        cls.HF_CONFIG_KWARGS.update({"config": config, "ignore_mismatched_sizes": True})
        return super().setUpClass()


class TestQwen3Model_UAM(TestQwen3Model):
    RBLN_CLASS_KWARGS = {"rbln_config": {"use_attention_mask": True}}


class TestOPTForCausalLM(LLMTest.TestLLM):
    RBLN_CLASS = RBLNOPTForCausalLM
    HF_MODEL_ID = "facebook/opt-2.7b"
    HF_CONFIG_KWARGS = {"num_hidden_layers": 1, "max_position_embeddings": 2048}


class TestOPTModel(LLMTest.TestLLMWithoutLMHead):
    RBLN_CLASS = RBLNOPTModel
    HF_MODEL_ID = "facebook/opt-2.7b"
    HF_CONFIG_KWARGS = {"num_hidden_layers": 1, "max_position_embeddings": 2048}


class TestLlamaForCausalLM(LLMTest.TestLLM):
    RBLN_CLASS = RBLNLlamaForCausalLM
    HF_MODEL_ID = "afmck/testing-llama-tiny"
    TEST_LEVEL = TestLevel.ESSENTIAL
    HF_CONFIG_KWARGS = {"num_hidden_layers": 1, "max_position_embeddings": 1024}


class TestLlamaModel(LLMTest.TestLLMWithoutLMHead):
    RBLN_CLASS = RBLNLlamaModel
    HF_MODEL_ID = "afmck/testing-llama-tiny"
    HF_CONFIG_KWARGS = {"num_hidden_layers": 1, "max_position_embeddings": 1024}


class TestLlamaForCausalLM_Flash(LLMTest.TestLLM):
    RBLN_CLASS = RBLNLlamaForCausalLM
    HF_MODEL_ID = "afmck/testing-llama-tiny"
    TEST_LEVEL = TestLevel.ESSENTIAL
    HF_CONFIG_KWARGS = {"num_hidden_layers": 1, "max_position_embeddings": 8192}
    RBLN_CLASS_KWARGS = {"rbln_config": {"attn_impl": "flash_attn", "kvcache_partition_len": 4096}}


class TestLlamaModel_Flash(LLMTest.TestLLMWithoutLMHead):
    RBLN_CLASS = RBLNLlamaModel
    HF_MODEL_ID = "afmck/testing-llama-tiny"
    HF_CONFIG_KWARGS = {"num_hidden_layers": 1, "max_position_embeddings": 8192}
    RBLN_CLASS_KWARGS = {"rbln_config": {"attn_impl": "flash_attn", "kvcache_partition_len": 4096}}


class TestLlamaForCausalLM_Multibatch(TestLlamaForCausalLM):
    PROMPT = ["Who are you?", "What is the capital of France?", "What is the capital of Germany?"]
    RBLN_CLASS_KWARGS = {"rbln_config": {"batch_size": 3, "decoder_batch_sizes": [3, 2, 1]}}


class TestGPT2LMHeadModel(LLMTest.TestLLM):
    RBLN_CLASS = RBLNGPT2LMHeadModel
    HF_MODEL_ID = "openai-community/gpt2"
    HF_CONFIG_KWARGS = {"n_layer": 1, "max_position_embeddings": 1024}


class TestGPT2Model(LLMTest.TestLLMWithoutLMHead):
    RBLN_CLASS = RBLNGPT2Model
    HF_MODEL_ID = "openai-community/gpt2"
    HF_CONFIG_KWARGS = {"n_layer": 1, "max_position_embeddings": 1024}


class TestPhiForCausalLM(LLMTest.TestLLM):
    RBLN_CLASS = RBLNPhiForCausalLM

    # HF_MODEL_ID = "hf-internal-testing/tiny-random-PhiForCausalLM"
    HF_MODEL_ID = "microsoft/phi-2"
    HF_CONFIG_KWARGS = {"num_hidden_layers": 1, "max_position_embeddings": 1024, "trust_remote_code": True}


class TestPhiModel(LLMTest.TestLLMWithoutLMHead):
    RBLN_CLASS = RBLNPhiModel
    HF_MODEL_ID = "microsoft/phi-2"
    HF_CONFIG_KWARGS = {"num_hidden_layers": 1, "max_position_embeddings": 1024, "trust_remote_code": True}


class TestExaoneForCausalLM(LLMTest.TestLLM):
    RBLN_CLASS = RBLNExaoneForCausalLM
    # HF_MODEL_ID = "katuni4ka/tiny-random-exaone"
    HF_MODEL_ID = "LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct"
    HF_CONFIG_KWARGS = {"num_hidden_layers": 1, "max_position_embeddings": 1024, "trust_remote_code": True}


class TestT5Model(LLMTest.TestLLM):
    RBLN_AUTO_CLASS = RBLNAutoModelForSeq2SeqLM
    RBLN_CLASS = RBLNT5ForConditionalGeneration

    # HF_MODEL_ID = "hf-internal-testing/tiny-random-T5ForConditionalGeneration"
    # FIXME:: Update to internal once enabled tiny model
    HF_MODEL_ID = "t5-small"
    PROMPT = "summarize: studies have shown that owning a dog is good for you"
    RBLN_CLASS_KWARGS = {"rbln_config": {"enc_max_seq_len": 512, "dec_max_seq_len": 512}}
    HF_CONFIG_KWARGS = {"num_layers": 1}

    def get_inputs(self):
        inputs = self.get_tokenizer()(
            self.PROMPT, padding="max_length", max_length=512, truncation=True, return_tensors="pt"
        )
        inputs["max_new_tokens"] = 20
        inputs["do_sample"] = False
        inputs["num_beams"] = 1
        return inputs

    def postprocess(self, inputs, output):
        generated_text = self.get_tokenizer().decode(
            output[0], skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        return generated_text


class TestBartModel(LLMTest.TestLLM):
    RBLN_AUTO_CLASS = RBLNAutoModelForSeq2SeqLM
    RBLN_CLASS = RBLNBartForConditionalGeneration

    # HF_MODEL_ID = "sshleifer/bart-tiny-random"
    # FIXME:: Update to internal once enabled tiny model
    HF_MODEL_ID = "lucadiliello/bart-small"
    HF_CONFIG_KWARGS = {
        "num_hidden_layers": 1,
        "decoder_layers": 1,
        "encoder_layers": 1,
    }
    RBLN_CLASS_KWARGS = {"rbln_config": {"enc_max_seq_len": 512, "dec_max_seq_len": 512}}
    PROMPT = "summarize: studies have shown that owning a dog is good for you"
    TEST_LEVEL = TestLevel.ESSENTIAL

    def get_inputs(self):
        inputs = self.get_tokenizer()(
            self.PROMPT, padding="max_length", max_length=512, truncation=True, return_tensors="pt"
        )
        inputs["max_new_tokens"] = 20
        inputs["do_sample"] = False
        inputs["num_beams"] = 1
        return inputs

    def postprocess(self, inputs, output):
        generated_text = self.get_tokenizer().decode(
            output[0], skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        return generated_text

    def test_automap(self):
        # BartForConditionalGeneration -> RBLNBartForConditionalGeneration compile case
        with self.subTest():
            assert self.RBLN_CLASS == self.RBLN_AUTO_CLASS.get_rbln_cls(
                self.HF_MODEL_ID,
                **self.RBLN_CLASS_KWARGS,
                **self.HF_CONFIG_KWARGS,
            )

        # BartForConditionalGeneration -> RBLNBartModel compile case
        # Invoked rbln_class is different from config's architecture
        with self.subTest():
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")

                RBLNAutoModel.get_rbln_cls(self.HF_MODEL_ID)

                self.assertEqual(len(w), 1)
                self.assertTrue(issubclass(w[-1].category, UserWarning))
                self.assertIn("This mismatch could cause some operations", str(w[-1].message))

        # BartForConditionalGeneration -> RBLNBartForCausalLM compile case
        # RBLNBartForCausalLM is not yet supported in optimum.rbln
        with self.subTest():
            with pytest.raises(AttributeError):
                RBLNAutoModelForCausalLM.get_rbln_cls(self.HF_MODEL_ID)

        # RBLNBartForSeq2SeqLM -> RBLNBartForCausalLM load case
        with self.subTest():
            with pytest.raises(ValueError):
                _ = RBLNAutoModelForCausalLM.from_pretrained(
                    self.get_rbln_local_dir(),
                    export=False,
                    rbln_create_runtimes=False,
                    **self.HF_CONFIG_KWARGS,
                )


class TestLlavaForConditionalGeneration(LLMTest.TestLLM):
    RBLN_AUTO_CLASS = RBLNAutoModelForVision2Seq
    RBLN_CLASS = RBLNLlavaForConditionalGeneration
    HF_MODEL_ID = "trl-internal-testing/tiny-LlavaForConditionalGeneration"
    PROMPT = "[INST] <image>\nWhat’s shown in this image? [/INST]"
    RBLN_CLASS_KWARGS = {
        "rbln_config": {
            "vision_tower": {"output_hidden_states": True},
            "language_model": {"use_inputs_embeds": True},
        }
    }
    HF_CONFIG_KWARGS = {"revision": "8ab8bfc820a6bb9e0f8de1ac715f4b53db44e684"}
    HF_CONFIG_KWARGS_PREPROCESSOR = {"revision": "8ab8bfc820a6bb9e0f8de1ac715f4b53db44e684"}
    IS_MULTIMODAL = True

    def get_inputs(self):
        tokenizer = self.get_tokenizer()
        img_path = f"{os.path.dirname(__file__)}/../assets/rbln_logo.png"
        image = Image.open(img_path)
        inputs = tokenizer(images=[image], text=[self.PROMPT], return_tensors="pt", padding=True)
        inputs["max_new_tokens"] = 20
        inputs["do_sample"] = False
        return inputs

    def _inner_test_save_load(self, tmpdir):
        super()._inner_test_save_load(tmpdir)
        # Test loading from nested config
        _ = self.RBLN_CLASS.from_pretrained(
            tmpdir,
            export=False,
            rbln_config={
                "create_runtimes": False,
                "vision_tower": {"create_runtimes": False},
                "language_model": {"create_runtimes": False},
            },
            **self.HF_CONFIG_KWARGS,
        )


class TestPegasusModel(LLMTest.TestLLM):
    RBLN_AUTO_CLASS = RBLNAutoModelForSeq2SeqLM
    RBLN_CLASS = RBLNPegasusForConditionalGeneration

    # FIXME:: Update to internal once enabled tiny model
    # HF_MODEL_ID = "hf-tiny-model-private/tiny-random-PegasusForConditionalGeneration"
    HF_MODEL_ID = "google/pegasus-xsum"
    HF_CONFIG_KWARGS = {
        "num_hidden_layers": 1,
        "decoder_layers": 1,
        "encoder_layers": 1,
    }
    RBLN_CLASS_KWARGS = {"rbln_config": {"enc_max_seq_len": 512, "dec_max_seq_len": 512}}
    PROMPT = "summarize: studies have shown that owning a dog is good for you"
    TEST_LEVEL = TestLevel.ESSENTIAL

    def get_inputs(self):
        inputs = self.get_tokenizer()(
            self.PROMPT, padding="max_length", max_length=512, truncation=True, return_tensors="pt"
        )
        inputs["max_new_tokens"] = 20
        inputs["num_beams"] = 1
        inputs["do_sample"] = False
        return inputs

    def postprocess(self, inputs, output):
        generated_text = self.get_tokenizer().decode(
            output[0], skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        return generated_text


class TestLlavaNextForConditionalGeneration(LLMTest.TestLLM):
    RBLN_AUTO_CLASS = RBLNAutoModelForVision2Seq
    RBLN_CLASS = RBLNLlavaNextForConditionalGeneration
    HF_MODEL_ID = "trl-internal-testing/tiny-LlavaNextForConditionalGeneration"
    PROMPT = "[INST] <image>\nWhat’s shown in this image? [/INST]"
    RBLN_CLASS_KWARGS = {
        "rbln_config": {
            "language_model": {"use_inputs_embeds": True},
        }
    }
    HF_CONFIG_KWARGS = {"revision": "21948c1af6a0666e341b6403dc1cbbd5c8900e7d"}
    HF_CONFIG_KWARGS_PREPROCESSOR = {"revision": "21948c1af6a0666e341b6403dc1cbbd5c8900e7d"}
    IS_MULTIMODAL = True

    # override
    @classmethod
    def setUpClass(cls):
        config = AutoConfig.from_pretrained(cls.HF_MODEL_ID, revision=cls.HF_CONFIG_KWARGS["revision"])
        text_config = json.loads(config.text_config.to_json_string())
        text_config["num_hidden_layers"] = 1
        kwargs = {"text_config": text_config}
        cls.HF_CONFIG_KWARGS.update(kwargs)
        return super().setUpClass()

    def get_inputs(self):
        tokenizer = self.get_tokenizer()
        img_path = f"{os.path.dirname(__file__)}/../assets/rbln_logo.png"
        image = Image.open(img_path)
        inputs = tokenizer(images=[image], text=[self.PROMPT], return_tensors="pt", padding=True)
        inputs["max_new_tokens"] = 20
        inputs["do_sample"] = False
        return inputs

    def _inner_test_save_load(self, tmpdir):
        super()._inner_test_save_load(tmpdir)
        # Test loading from nested config
        _ = self.RBLN_CLASS.from_pretrained(
            tmpdir,
            export=False,
            rbln_config={"language_model": {"create_runtimes": False}},
            **self.HF_CONFIG_KWARGS,
        )

    def test_complicate_config(self):
        rbln_config = {
            "vision_tower": {
                "batch_size": 2,
                "create_runtimes": False,
            },
            "language_model": {
                "batch_size": 2,
                "create_runtimes": False,
            },
        }
        rbln_class_kwargs = {"rbln_config": rbln_config}

        with pytest.raises(
            ValueError, match="Parameter conflict for 'batch_size': submodule_config has 2, but kwargs has 1"
        ):
            _ = self.RBLN_CLASS.from_pretrained(model_id=self.HF_MODEL_ID, **rbln_class_kwargs)


class TestBlip2ForConditionalGeneration(LLMTest.TestLLM):
    RBLN_AUTO_CLASS = RBLNAutoModelForVision2Seq
    RBLN_CLASS = RBLNBlip2ForConditionalGeneration
    HF_MODEL_ID = "Salesforce/blip2-opt-2.7b"  # No tiny model yet.
    PROMPT = "Question: Describe this image? Answer:"
    RBLN_CLASS_KWARGS = {"rbln_config": {"language_model": {"use_inputs_embeds": True, "max_seq_len": 1024}}}
    HF_CONFIG_KWARGS = {}  # Initialize empty to avoid sharing with other classes
    IS_MULTIMODAL = True

    # override
    @classmethod
    def setUpClass(cls):
        config = AutoConfig.from_pretrained(cls.HF_MODEL_ID)

        text_config = json.loads(config.text_config.to_json_string())
        text_config["num_hidden_layers"] = 1
        kwargs = {"text_config": text_config}

        qformer_config = json.loads(config.qformer_config.to_json_string())
        qformer_config["num_hidden_layers"] = 1
        kwargs["qformer_config"] = qformer_config

        cls.HF_CONFIG_KWARGS.update(kwargs)
        return super().setUpClass()

    def get_inputs(self):
        tokenizer = self.get_tokenizer()
        img_path = f"{os.path.dirname(__file__)}/../assets/rbln_logo.png"
        image = Image.open(img_path)
        inputs = tokenizer(images=image, text=self.PROMPT, return_tensors="pt", padding=True)
        inputs["max_new_tokens"] = 20
        inputs["do_sample"] = False
        return inputs

    def _inner_test_save_load(self, tmpdir):
        super()._inner_test_save_load(tmpdir)
        # Test loading from nested config
        _ = self.RBLN_CLASS.from_pretrained(
            tmpdir,
            export=False,
            rbln_config={"language_model": {"create_runtimes": False}},
            **self.HF_CONFIG_KWARGS,
        )


class TestIdefics3ForConditionalGeneration(LLMTest.TestLLM):
    RBLN_AUTO_CLASS = RBLNAutoModelForVision2Seq
    RBLN_CLASS = RBLNIdefics3ForConditionalGeneration
    HF_MODEL_ID = "hf-internal-testing/tiny-random-Idefics3ForConditionalGeneration"
    PROMPT = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "Describe this image."}]}]
    RBLN_CLASS_KWARGS = {"rbln_config": {"text_model": {"use_inputs_embeds": True, "attn_impl": "flash_attn"}}}
    HF_CONFIG_KWARGS = {}  # Initialize empty to avoid sharing with other classes
    IS_MULTIMODAL = True

    @classmethod
    def setUpClass(cls):
        config = AutoConfig.from_pretrained(cls.HF_MODEL_ID)
        text_config = json.loads(config.text_config.to_json_string())
        text_config["num_hidden_layers"] = 1
        kwargs = {"text_config": text_config}
        cls.HF_CONFIG_KWARGS.update(kwargs)
        return super().setUpClass()

    def get_inputs(self):
        tokenizer = self.get_tokenizer()
        img_path = f"{os.path.dirname(__file__)}/../assets/rbln_logo.png"
        image = Image.open(img_path)
        text = tokenizer.apply_chat_template(self.PROMPT, add_generation_prompt=True)
        inputs = tokenizer(images=[image], text=[text], return_tensors="pt", padding=True)
        inputs["max_new_tokens"] = 20
        inputs["do_sample"] = False
        return inputs


class TestQwen2VLForConditionalGeneration(LLMTest.TestLLM):
    RBLN_AUTO_CLASS = RBLNAutoModelForVision2Seq
    RBLN_CLASS = RBLNQwen2VLForConditionalGeneration
    HF_MODEL_ID = "hf-internal-testing/tiny-random-Qwen2VLForConditionalGeneration"
    PROMPT = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>Describe this image.<|im_end|>\n<|im_start|>assistant\n"
    RBLN_CLASS_KWARGS = {
        "rbln_config": {
            "visual": {"max_seq_lens": 512},
            "tensor_parallel_size": 1,
            "max_seq_len": 32_768,
        }
    }
    HF_CONFIG_KWARGS = {"num_hidden_layers": 1}

    @classmethod
    def setUpClass(cls):
        config = AutoConfig.from_pretrained(cls.HF_MODEL_ID)
        vision_config = json.loads(config.vision_config.to_json_string())
        text_config = json.loads(config.text_config.to_json_string())
        text_config["num_hidden_layers"] = 1
        text_config["layer_types"] = text_config["layer_types"][:1]
        vision_config["depth"] = 2  # To make the test faster
        vision_config["fullatt_block_indexes"] = [1]
        kwargs = {"vision_config": vision_config, "text_config": text_config}
        cls.HF_CONFIG_KWARGS.update(kwargs)
        return super().setUpClass()

    @classmethod
    def get_tokenizer(cls):
        if getattr(cls, "_tokenizer", None) is None:
            cls._tokenizer = AutoProcessor.from_pretrained(cls.HF_MODEL_ID, max_pixels=64 * 14 * 14)
        return cls._tokenizer

    def get_inputs(self):
        tokenizer = self.get_tokenizer()
        img_path = f"{os.path.dirname(__file__)}/../assets/rbln_logo.png"
        image = Image.open(img_path)
        inputs = tokenizer(images=[image], text=[self.PROMPT], return_tensors="pt", padding=True)
        inputs["max_new_tokens"] = 20
        inputs["do_sample"] = False
        return inputs


class TestQwen2_5_VLForConditionalGeneration(LLMTest.TestLLM):
    RBLN_AUTO_CLASS = RBLNAutoModelForVision2Seq
    RBLN_CLASS = RBLNQwen2_5_VLForConditionalGeneration
    HF_MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"  # No tiny model yet.
    PROMPT = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>Describe this image.<|im_end|>\n<|im_start|>assistant\n"
    RBLN_CLASS_KWARGS = {
        "rbln_config": {
            "visual": {"max_seq_lens": 512},
            "tensor_parallel_size": 1,
            "kvcache_partition_len": 16_384,
            "max_seq_len": 32_768,
        }
    }
    HF_CONFIG_KWARGS = {"num_hidden_layers": 1}
    HF_CONFIG_KWARGS_PREPROCESSOR = {"max_pixels": 64 * 14 * 14}
    IS_MULTIMODAL = True

    @classmethod
    def setUpClass(cls):
        config = AutoConfig.from_pretrained(cls.HF_MODEL_ID)
        vision_config = json.loads(config.vision_config.to_json_string())
        text_config = json.loads(config.text_config.to_json_string())
        text_config["num_hidden_layers"] = 1
        text_config["layer_types"] = text_config["layer_types"][:1]
        vision_config["depth"] = 2
        vision_config["fullatt_block_indexes"] = [1]
        kwargs = {"vision_config": vision_config, "text_config": text_config}
        cls.HF_CONFIG_KWARGS.update(kwargs)
        return super().setUpClass()

    def get_inputs(self):
        tokenizer = self.get_tokenizer()
        img_path = f"{os.path.dirname(__file__)}/../assets/rbln_logo.png"
        image = Image.open(img_path)
        inputs = tokenizer(images=[image], text=[self.PROMPT], return_tensors="pt", padding=True)
        inputs["max_new_tokens"] = 20
        inputs["do_sample"] = False
        return inputs


class TestGemma3ForConditionalGeneration(LLMTest.TestLLM):
    RBLN_AUTO_CLASS = RBLNAutoModelForImageTextToText
    RBLN_CLASS = RBLNGemma3ForConditionalGeneration
    HF_MODEL_ID = "trl-internal-testing/tiny-Gemma3ForConditionalGeneration"
    PROMPT = "<bos><start_of_turn>user\n<start_of_image>Describe the image.<end_of_turn>\n<start_of_turn>model\n'"
    RBLN_CLASS_KWARGS = {"rbln_config": {"language_model": {"use_inputs_embeds": True, "kvcache_partition_len": 4096}}}
    HF_CONFIG_KWARGS = {"revision": "e1f4b0516ec80f86ed75c8cb1d45ede72526ad24"}
    HF_CONFIG_KWARGS_PREPROCESSOR = {"revision": "e1f4b0516ec80f86ed75c8cb1d45ede72526ad24"}
    TEST_LEVEL = TestLevel.FULL
    IS_MULTIMODAL = True

    # override
    @classmethod
    def setUpClass(cls):
        config = AutoConfig.from_pretrained(cls.HF_MODEL_ID, revision=cls.HF_CONFIG_KWARGS["revision"])
        text_config = json.loads(config.text_config.to_json_string())
        text_config["num_hidden_layers"] = 2
        text_config["layer_types"] = ["full_attention", "sliding_attention"]
        text_config["sliding_window_pattern"] = 2
        vision_config = json.loads(config.vision_config.to_json_string())
        vision_config["num_hidden_layers"] = 1
        vision_config["vision_use_head"] = False
        kwargs = {"text_config": text_config, "vision_config": vision_config}
        cls.HF_CONFIG_KWARGS.update(kwargs)
        return super().setUpClass()

    def get_inputs(self):
        tokenizer = self.get_tokenizer()
        img_path = f"{os.path.dirname(__file__)}/../assets/rbln_logo.png"
        image = Image.open(img_path)
        image = image.convert("RGB")
        inputs = tokenizer(images=[image], text=[self.PROMPT], return_tensors="pt", padding=True)
        inputs["max_new_tokens"] = 20
        inputs["do_sample"] = False
        return inputs


class TestGemma3ForConditionalGeneration_OutputHiddenStates(TestGemma3ForConditionalGeneration):
    RBLN_CLASS_KWARGS = {
        "rbln_config": {
            "language_model": {"use_inputs_embeds": True, "kvcache_partition_len": 4096, "output_hidden_states": True}
        }
    }

    def test_generate(self):
        self._test_output_hidden_states_generation()


class TestGemma3ForCausalLM(LLMTest.TestLLM):
    RBLN_CLASS = RBLNGemma3ForCausalLM
    HF_MODEL_ID = "google/gemma-3-1b-it"
    HF_CONFIG_KWARGS = {
        "trust_remote_code": True,
    }

    @classmethod
    def setUpClass(cls):
        hf_config = AutoConfig.from_pretrained(cls.HF_MODEL_ID)
        hf_config.num_hidden_layers = 2
        hf_config.layer_types = ["full_attention", "sliding_attention"]
        hf_config.sliding_window_pattern = 2
        hf_config.max_position_embeddings = 1024
        kwargs = {"config": hf_config}
        cls.HF_CONFIG_KWARGS.update(kwargs)
        return super().setUpClass()


class TestLlamaForCausalLM_fp8(LLMTest.TestLLM):
    RBLN_CLASS = RBLNLlamaForCausalLM
    HF_MODEL_ID = "RedHatAI/Meta-Llama-3-8B-Instruct-FP8-KV"  # No tiny model yet.
    HF_CONFIG_KWARGS = {"num_hidden_layers": 1}
    RBLN_CLASS_KWARGS = {
        "rbln_config": {
            "quantization": {"weights": "fp8", "kv_caches": "fp8"},
            "create_runtimes": False,
            "npu": "RBLN-CR03",
            "attn_impl": "flash_attn",
            "kvcache_partition_len": 4096,
            "max_seq_len": 8192,
            "tensor_parallel_size": 1,
        },
    }

    def test_generate(self):
        # Cannot generate output with fp8 quantization in ATOM™
        pass


class TestMultiLora(LLMTest.TestLLM):
    HF_MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"
    HF_CONFIG_KWARGS = {"num_hidden_layers": 1, "max_position_embeddings": 1024}
    RBLN_CLASS = RBLNLlamaForCausalLM
    RBLN_CLASS_KWARGS = {
        "rbln_config": {
            "max_seq_len": 1024,
            "lora_config": {
                "adapters": [
                    RBLNLoRAAdapterConfig(1, "nemoguard", "nvidia/llama-3.1-nemoguard-8b-topic-control"),
                    RBLNLoRAAdapterConfig(2, "abliterated", "reissbaker/llama-3.1-8b-abliterated-lora"),
                ]
            },
        }
    }

    def get_inputs(self):
        self.model.set_adapter(["abliterated"])
        return super().get_inputs()


class TestMultiLora_batch(LLMTest.TestLLM):
    PROMPT = ["Who are you?", "What is the capital of France?"]

    # Should check each output corresponds to each prompt
    EXPECTED_OUTPUT = [
        " bench_echointon Ebonylica Lennonnings909 norgeZN°Eusan倍oloadolen逸 Oaksodian surplusaniem",
        "/topicпідonus343../../../ Mund  Ont ReactionIPAچیIQUE beltーブ204umlu Cortexoisئةτερ",
    ]
    HF_MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"
    HF_CONFIG_KWARGS = {"num_hidden_layers": 1, "max_position_embeddings": 1024}
    RBLN_CLASS = RBLNLlamaForCausalLM
    RBLN_CLASS_KWARGS = {
        "rbln_config": {
            "batch_size": 2,
            "max_seq_len": 1024,
            "lora_config": {
                "adapters": [
                    RBLNLoRAAdapterConfig(1, "nemoguard", "nvidia/llama-3.1-nemoguard-8b-topic-control"),
                    RBLNLoRAAdapterConfig(2, "abliterated", "reissbaker/llama-3.1-8b-abliterated-lora"),
                ]
            },
        }
    }

    def get_inputs(self):
        self.model.set_adapter(["nemoguard", "abliterated"])
        return super().get_inputs()


class TestDisallowedLlama_1(DisallowedTestBase.DisallowedTest):
    # Too long sequence length
    RBLN_CLASS = RBLNLlamaForCausalLM
    HF_MODEL_ID = "afmck/testing-llama-tiny"
    HF_CONFIG_KWARGS = {"num_hidden_layers": 1, "max_position_embeddings": 32768 * 2}


class TestDisallowedLlama_2(DisallowedTestBase.DisallowedTest):
    # Flash attn : Not multiple
    RBLN_CLASS = RBLNLlamaForCausalLM
    HF_MODEL_ID = "afmck/testing-llama-tiny"
    HF_CONFIG_KWARGS = {"num_hidden_layers": 1, "max_position_embeddings": 8192}
    RBLN_CLASS_KWARGS = {"rbln_config": {"attn_impl": "flash_attn", "kvcache_partition_len": 8000}}


class TestDisallowedLlama_3(DisallowedTestBase.DisallowedTest):
    # Flash attn : too short partition
    RBLN_CLASS = RBLNLlamaForCausalLM
    HF_MODEL_ID = "afmck/testing-llama-tiny"
    HF_CONFIG_KWARGS = {"num_hidden_layers": 1, "max_position_embeddings": 8192}
    RBLN_CLASS_KWARGS = {"rbln_config": {"attn_impl": "flash_attn", "kvcache_partition_len": 1024}}


class TestDisallowedLlama_4(DisallowedTestBase.DisallowedTest):
    # Flash attn : too short max_seq_len
    RBLN_CLASS = RBLNLlamaForCausalLM
    HF_MODEL_ID = "afmck/testing-llama-tiny"
    HF_CONFIG_KWARGS = {"num_hidden_layers": 1, "max_position_embeddings": 2048}
    RBLN_CLASS_KWARGS = {"rbln_config": {"attn_impl": "flash_attn", "kvcache_partition_len": 1024}}


if __name__ == "__main__":
    unittest.main()
