import tempfile

import pytest
import torch
from transformers import (
    BartConfig,
    Florence2Config,
    Florence2ForConditionalGeneration,
    Florence2VisionConfig,
    GenerationConfig,
)

from optimum.rbln import (
    RBLNAutoModelForImageTextToText,
    RBLNFlorence2ForConditionalGeneration,
    RBLNFlorence2ForConditionalGenerationConfig,
)
from optimum.rbln.transformers.models.florence2.florence2_architecture import Florence2MultimodalEncoder


class TestFlorence2Encoder:
    @pytest.fixture(autouse=True)
    def initialize_model(self):
        torch.manual_seed(0)
        vision = Florence2VisionConfig(
            depths=[1, 1, 1, 1],
            embed_dim=[8, 16, 32, 64],
            num_heads=[1, 2, 4, 8],
            num_groups=[1, 2, 4, 8],
            window_size=2,
            projection_dim=32,
        )
        text = BartConfig(
            d_model=32,
            encoder_layers=1,
            decoder_layers=1,
            encoder_attention_heads=4,
            decoder_attention_heads=4,
            encoder_ffn_dim=64,
            decoder_ffn_dim=64,
            vocab_size=32,
            max_position_embeddings=64,
        )
        config = Florence2Config(vision_config=vision, text_config=text, image_token_id=31)
        config.architectures = ["Florence2ForConditionalGeneration"]
        config.vision_config._attn_implementation = "eager"
        config.text_config._attn_implementation = "eager"
        self.model = Florence2ForConditionalGeneration(config).eval()

    def test_auto_class_resolves_florence_generation(self):
        resolved = RBLNAutoModelForImageTextToText.get_rbln_cls("unused", export=True, config=self.model.config)
        assert resolved is RBLNFlorence2ForConditionalGeneration

    def test_fused_encoder_matches_native_image_placeholders(self):
        pixels = torch.randn(1, 3, 64, 64)
        text_ids = torch.tensor([[0, 4, 2, 1]])
        text_mask = torch.tensor([[1, 1, 1, 0]])
        with torch.inference_mode():
            features = self.model.get_image_features(pixels).pooler_output
            image_ids = torch.full((1, features.shape[1]), self.model.config.image_token_id)
            input_ids = torch.cat((image_ids, text_ids), dim=1)
            mask = torch.cat((torch.ones_like(image_ids), text_mask), dim=1)
            embeds = self.model.get_input_embeddings()(input_ids)
            embeds = embeds.masked_scatter((input_ids == self.model.config.image_token_id).unsqueeze(-1), features)
            expected = self.model.get_encoder()(inputs_embeds=embeds, attention_mask=mask)[0]
            actual = Florence2MultimodalEncoder(self.model)((pixels, text_ids), mask)[0]
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_compile_inputs_include_image_features_in_encoder_length(self):
        config = RBLNFlorence2ForConditionalGenerationConfig(image_size=64, enc_max_seq_len=64, dec_max_seq_len=64)
        config = RBLNFlorence2ForConditionalGeneration.update_rbln_config([], self.model, self.model.config, config)
        encoder = {name: (shape, dtype) for name, shape, dtype in config.compile_cfgs[0].input_info}
        decoder = {name: (shape, dtype) for name, shape, dtype in config.compile_cfgs[1].input_info}
        assert encoder["input_ids"] == ([1, 59], "int64")
        assert encoder["pixel_values"] == ([1, 3, 64, 64], "float32")
        assert encoder["attention_mask"] == ([1, 64], "float32")
        assert decoder["cross_key_value_states_0"][0] == [1, 4, 64, 8]
        with torch.inference_mode():
            features = self.model.get_image_features(torch.randn(1, 3, 64, 64)).pooler_output
        assert features.shape[1] == config.enc_max_seq_len - encoder["input_ids"][0][1]
        with tempfile.TemporaryDirectory() as directory:
            config.save(directory)
            restored = RBLNFlorence2ForConditionalGenerationConfig.from_pretrained(directory)
        assert restored.image_size == 64
        assert restored.compile_cfgs[0].input_info == config.compile_cfgs[0].input_info

    def test_encoder_length_must_leave_room_for_text(self):
        config = RBLNFlorence2ForConditionalGenerationConfig(image_size=256, enc_max_seq_len=64, dec_max_seq_len=64)
        with pytest.raises(ValueError, match="room for text"):
            RBLNFlorence2ForConditionalGeneration.update_rbln_config([], self.model, self.model.config, config)

    def runtime_model(self, batch_size=1):
        observed = {}

        def runtime(**kwargs):
            observed.update(kwargs)
            observed.setdefault("calls", []).append(kwargs)
            return torch.empty(0)

        config = RBLNFlorence2ForConditionalGenerationConfig(
            batch_size=batch_size, image_size=64, enc_max_seq_len=64, dec_max_seq_len=64
        )
        config = RBLNFlorence2ForConditionalGeneration.update_rbln_config([], self.model, self.model.config, config)
        model = object.__new__(RBLNFlorence2ForConditionalGeneration)
        model.config = self.model.config
        model.rbln_config = config
        model.generation_config = self.model.generation_config
        model.model = [runtime, runtime]
        model.__post_init__()
        return model, observed

    def test_generation_inputs_and_mask_reach_runtime(self):
        model, observed = self.runtime_model()
        ids = torch.tensor([[31, 31, 31, 31, 31, 0, 4, 2]])
        pixels = torch.randn(1, 3, 64, 64)
        kwargs = {"pixel_values": pixels, "attention_mask": torch.ones_like(ids)}
        model._validate_model_kwargs(dict(kwargs))
        result = model._prepare_encoder_decoder_kwargs_for_generation(ids, kwargs)
        assert observed["input_ids"].shape == (1, 59)
        assert observed["input_ids"][0, :3].tolist() == [0, 4, 2]
        assert observed["input_ids"][0, 3:].tolist() == [1] * 56
        assert observed["attention_mask"][0].tolist() == [1] * 8 + [0] * 56
        assert torch.equal(observed["pixel_values"], pixels)
        assert torch.equal(result["attention_mask"], observed["attention_mask"])
        assert "pixel_values" not in result

    def test_misplaced_image_prefix_fails_before_runtime(self):
        model, observed = self.runtime_model()
        ids = torch.tensor([[0, 31, 31, 31, 31, 31, 4, 2]])
        with pytest.raises(ValueError, match="image-token prefix"):
            model._prepare_encoder_decoder_kwargs_for_generation(ids, {"pixel_values": torch.randn(1, 3, 64, 64)})
        assert not observed

    def test_batched_encoder_preserves_image_text_mask_and_cache_rows(self):
        model, observed = self.runtime_model(batch_size=2)
        text_ids = torch.tensor([[0, 4, 2, 1], [0, 5, 6, 2]])
        ids = torch.cat((torch.full((2, 5), 31), text_ids), dim=1)
        mask = torch.tensor([[1, 1, 1, 1, 1, 1, 1, 1, 0], [1, 1, 1, 1, 1, 1, 1, 1, 1]])
        pixels = torch.arange(2, dtype=torch.float32).view(2, 1, 1, 1).expand(2, 3, 64, 64)
        result = model._prepare_encoder_decoder_kwargs_for_generation(
            ids, {"pixel_values": pixels, "attention_mask": mask}
        )

        assert len(observed["calls"]) == 2
        expected_mask = torch.nn.functional.pad(mask, (0, 55)).float()
        torch.testing.assert_close(result["attention_mask"], expected_mask)
        for index, call in enumerate(observed["calls"]):
            expected_ids = torch.nn.functional.pad(text_ids[index : index + 1], (0, 55), value=1)
            torch.testing.assert_close(call["input_ids"], expected_ids)
            torch.testing.assert_close(call["pixel_values"], pixels[index : index + 1])
            assert call["block_tables"].tolist() == [index]
            torch.testing.assert_close(call["attention_mask"], expected_mask[index : index + 1])

    def test_extra_images_fail_before_per_image_slicing(self):
        model, observed = self.runtime_model()
        ids = torch.tensor([[31, 31, 31, 31, 31, 0, 4, 2]])
        with pytest.raises(ValueError, match="Image batch size"):
            model._prepare_encoder_decoder_kwargs_for_generation(ids, {"pixel_values": torch.randn(2, 3, 64, 64)})
        assert not observed

    def test_input_batch_cannot_write_past_compiled_cross_cache(self):
        model, observed = self.runtime_model()
        ids = torch.tensor([[31, 31, 31, 31, 31, 0, 4, 2]]).repeat(2, 1)
        with pytest.raises(ValueError, match="compiled batch_size"):
            model._prepare_encoder_decoder_kwargs_for_generation(ids, {"pixel_values": torch.randn(2, 3, 64, 64)})
        assert not observed

    @pytest.mark.parametrize(
        ("options", "error", "message"),
        [
            ({"num_beams": 2}, ValueError, "generation mode"),
            ({"num_return_sequences": 2}, ValueError, "num_return_sequences"),
        ],
    )
    def test_unsupported_generation_mode_fails_before_runtime(self, options, error, message):
        model, observed = self.runtime_model()
        with pytest.raises(error, match=message):
            model.generate(torch.tensor([[0, 4, 2]]), pixel_values=torch.randn(1, 3, 64, 64), **options)
        assert not observed

    def test_generation_cannot_exceed_compiled_cache(self):
        model, observed = self.runtime_model()
        with pytest.raises(ValueError, match="KV cache"):
            model.generate(
                torch.tensor([[31, 31, 31, 31, 31, 0, 4, 2]]),
                pixel_values=torch.randn(1, 3, 64, 64),
                max_new_tokens=65,
            )
        assert "pixel_values" in observed

    @pytest.mark.parametrize("prefix", [[[4]], [[2, 4]]])
    def test_multi_token_effective_decoder_prefix_is_rejected(self, prefix):
        model, observed = self.runtime_model()
        with pytest.raises(NotImplementedError, match="decoder prefill"):
            model._prepare_decoder_input_ids_for_generation(
                1,
                "input_ids",
                {"decoder_input_ids": torch.tensor(prefix)},
                torch.tensor(2),
                device=torch.device("cpu"),
            )
        assert not observed

    def test_explicit_decoder_start_token_is_allowed(self):
        model, _ = self.runtime_model()
        ids, _ = model._prepare_decoder_input_ids_for_generation(
            1, "input_ids", {"decoder_input_ids": torch.tensor([[2]])}, torch.tensor(2), device=torch.device("cpu")
        )
        assert ids.tolist() == [[2]]

    @pytest.mark.parametrize(
        ("defaults", "error", "message"),
        [
            ({"num_beams": 3}, ValueError, "generation mode"),
            ({"num_beams": 1, "max_new_tokens": 65}, ValueError, "KV cache"),
        ],
    )
    def test_empty_generation_config_inherits_checkpoint_defaults(self, defaults, error, message):
        model, observed = self.runtime_model()
        model.generation_config.update(**defaults)
        ids = torch.tensor([[31, 31, 31, 31, 31, 0, 4, 2]])
        with pytest.raises(error, match=message):
            model.generate(ids, pixel_values=torch.randn(1, 3, 64, 64), generation_config=GenerationConfig())
        if "max_new_tokens" in defaults:
            assert "pixel_values" in observed
        else:
            assert not observed
