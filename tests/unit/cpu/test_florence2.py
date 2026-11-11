import pytest
import torch
from transformers import (
    BartConfig,
    Florence2Config,
    Florence2ForConditionalGeneration,
    Florence2VisionConfig,
)

from optimum.rbln import (
    RBLNFlorence2ForConditionalGeneration,
    RBLNFlorence2ForConditionalGenerationConfig,
)
from optimum.rbln.transformers.models.florence2.florence2_architecture import Florence2Wrapper


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
            max_position_embeddings=640,
        )
        config = Florence2Config(vision_config=vision, text_config=text, image_token_id=31)
        config.architectures = ["Florence2ForConditionalGeneration"]
        config.vision_config._attn_implementation = "eager"
        config.text_config._attn_implementation = "eager"
        self.model = Florence2ForConditionalGeneration(config).eval()

    def test_fused_encoder_matches_native_image_placeholders(self, monkeypatch):
        def update_cache(cache, values, index, axis):
            return cache.index_copy(axis.item(), index.reshape(1).long(), values)

        monkeypatch.setattr(torch.ops.rbln_custom_ops, "rbln_cache_update", update_cache)
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
            decoder = self.model.model.language_model.get_decoder()
            expected_kv = [
                projection(expected).view(1, input_ids.shape[1], 4, 8).transpose(1, 2)
                for layer in decoder.layers
                for projection in (layer.encoder_attn.k_proj, layer.encoder_attn.v_proj)
            ]
            caches = [torch.zeros(2, *values.shape[1:]) for values in expected_kv]
            wrapper = Florence2Wrapper(self.model, input_ids.shape[1], use_attention_mask=False)
            actual = wrapper.encoder(text_ids, mask, pixels, torch.tensor([1], dtype=torch.int16), *caches)
        for values, expected_values in zip(actual, expected_kv, strict=True):
            torch.testing.assert_close(values[1:2], expected_values, rtol=0, atol=0)
            torch.testing.assert_close(values[0], torch.zeros_like(values[0]), rtol=0, atol=0)

    def test_compile_inputs_include_image_features_in_encoder_length(self):
        config = RBLNFlorence2ForConditionalGenerationConfig(enc_max_seq_len=640, dec_max_seq_len=64)
        config = RBLNFlorence2ForConditionalGeneration.update_rbln_config([], self.model, self.model.config, config)
        encoder = {name: (shape, dtype) for name, shape, dtype in config.compile_cfgs[0].input_info}
        decoder = {name: (shape, dtype) for name, shape, dtype in config.compile_cfgs[1].input_info}
        assert encoder["input_ids"] == ([1, 63], "int64")
        assert encoder["pixel_values"] == ([1, 3, 768, 768], "float32")
        assert encoder["attention_mask"] == ([1, 640], "float32")
        assert decoder["cross_key_value_states_0"][0] == [1, 4, 640, 8]
        with torch.inference_mode():
            features = self.model.get_image_features(torch.randn(1, 3, 768, 768)).pooler_output
        assert features.shape[1] == config.enc_max_seq_len - encoder["input_ids"][0][1]

    def runtime_model(self, batch_size=1, dtype=torch.float32):
        self.model.to(dtype)
        observed = {}

        def runtime(**kwargs):
            observed.update(kwargs)
            observed.setdefault("calls", []).append({name: value.clone() for name, value in kwargs.items()})
            return torch.empty(0)

        config = RBLNFlorence2ForConditionalGenerationConfig(
            batch_size=batch_size, enc_max_seq_len=640, dec_max_seq_len=64
        )
        config = RBLNFlorence2ForConditionalGeneration.update_rbln_config([], self.model, self.model.config, config)
        model = object.__new__(RBLNFlorence2ForConditionalGeneration)
        model.config = self.model.config
        model.rbln_config = config
        model.generation_config = self.model.generation_config
        model.model = [runtime, runtime]
        model.__post_init__()
        return model, observed

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
    def test_batched_encoder_preserves_image_text_mask_and_cache_rows(self, dtype):
        model, observed = self.runtime_model(batch_size=2, dtype=dtype)
        text_ids = torch.tensor([[0, 4, 2, 1], [0, 5, 6, 2]])
        image_ids = torch.full((2, 577), 31)
        ids = torch.cat((image_ids, text_ids), dim=1)
        mask = torch.cat((torch.ones_like(image_ids), torch.tensor([[1, 1, 1, 0], [1, 1, 1, 1]])), dim=1)
        pixels = torch.arange(2, dtype=torch.float32).view(2, 1, 1, 1).expand(2, 3, 768, 768)
        kwargs = {"pixel_values": pixels, "attention_mask": mask}
        model._validate_model_kwargs(dict(kwargs))
        result = model._prepare_encoder_decoder_kwargs_for_generation(ids, kwargs)

        assert len(observed["calls"]) == 2
        assert "pixel_values" not in result
        expected_mask = torch.nn.functional.pad(mask, (0, 59)).to(dtype)
        torch.testing.assert_close(result["attention_mask"], expected_mask)
        for index, call in enumerate(observed["calls"]):
            expected_ids = torch.nn.functional.pad(text_ids[index : index + 1], (0, 59), value=1)
            torch.testing.assert_close(call["input_ids"], expected_ids)
            torch.testing.assert_close(call["pixel_values"], pixels[index : index + 1].to(dtype))
            assert call["block_tables"].tolist() == [index]
            assert call["block_tables"].dtype == torch.int16
            torch.testing.assert_close(call["attention_mask"], expected_mask[index : index + 1])

    def test_input_batch_cannot_write_past_compiled_cross_cache(self):
        model, observed = self.runtime_model()
        ids = torch.cat((torch.full((2, 577), 31), torch.tensor([[0, 4, 2]]).repeat(2, 1)), dim=1)
        with pytest.raises(ValueError, match="compiled batch_size"):
            model._prepare_encoder_decoder_kwargs_for_generation(ids, {"pixel_values": torch.randn(2, 3, 768, 768)})
        assert not observed
