from types import SimpleNamespace

import pytest
import torch
from transformers import BartConfig, BartForConditionalGeneration


@pytest.mark.parametrize("encoder_batch_size", [0, -1, True, 1.5, 3, 8])
def test_encoder_batch_must_be_a_positive_divisor(encoder_batch_size):
    from optimum.rbln import RBLNBartForConditionalGenerationConfig

    with pytest.raises(ValueError, match="encoder_batch_size"):
        RBLNBartForConditionalGenerationConfig(batch_size=4, encoder_batch_size=encoder_batch_size)


@pytest.mark.parametrize("encoder_batch_size,expected", [(None, 4), (1, 1), (2, 2), (4, 4)])
def test_bart_export_encoder_batch_and_decoder_cache_are_consistent(encoder_batch_size, expected):
    from optimum.rbln import RBLNBartForConditionalGeneration, RBLNBartForConditionalGenerationConfig

    config = BartConfig(
        encoder_layers=1,
        decoder_layers=1,
        d_model=16,
        encoder_attention_heads=2,
        decoder_attention_heads=2,
        encoder_ffn_dim=32,
        decoder_ffn_dim=32,
        vocab_size=32,
        max_position_embeddings=16,
    )
    cfg = RBLNBartForConditionalGeneration._update_rbln_config(
        [],
        model_config=config,
        rbln_config=RBLNBartForConditionalGenerationConfig(
            batch_size=4, encoder_batch_size=encoder_batch_size, enc_max_seq_len=8, dec_max_seq_len=8
        ),
    )
    inputs = {name: shape for name, shape, _ in cfg.compile_cfgs[0].input_info}
    assert cfg.encoder_batch_size == expected
    assert inputs["input_ids"] == inputs["attention_mask"] == [expected, 8]
    assert inputs["cross_key_value_states_0"] == [4, 2, 8, 8]


@pytest.mark.parametrize("encoder_batch_size,expected", [(None, 1), (2, 2)])
def test_t5_retains_serial_default_and_accepts_explicit_encoder_batch(encoder_batch_size, expected):
    from transformers import T5Config

    from optimum.rbln import RBLNT5ForConditionalGeneration, RBLNT5ForConditionalGenerationConfig

    cfg = RBLNT5ForConditionalGeneration._update_rbln_config(
        [],
        model_config=T5Config(num_layers=1, num_decoder_layers=1, d_model=16, num_heads=2, d_kv=8),
        rbln_config=RBLNT5ForConditionalGenerationConfig(
            batch_size=4, encoder_batch_size=encoder_batch_size, enc_max_seq_len=8, dec_max_seq_len=8
        ),
    )
    assert cfg.encoder_batch_size == expected
    inputs = {name: shape for name, shape, _ in cfg.compile_cfgs[0].input_info}
    assert inputs["input_ids"] == [expected, 8]


def test_legacy_bart_config_does_not_change_encoder_batch_when_loaded():
    from optimum.rbln import RBLNBartForConditionalGenerationConfig

    cfg = RBLNBartForConditionalGenerationConfig(batch_size=8)
    # The export path resolves the family default. Loading a saved config skips that path.
    assert cfg.encoder_batch_size is None


def test_existing_positional_config_arguments_keep_their_meaning():
    from optimum.rbln import RBLNBartForConditionalGenerationConfig

    cfg = RBLNBartForConditionalGenerationConfig(4, 8, 16, False, 4, 16)
    assert cfg.batch_size == 4
    assert cfg.enc_max_seq_len == 8
    assert cfg.dec_max_seq_len == 16
    assert cfg.encoder_batch_size is None


def test_batched_cross_attention_cache_writes_only_the_requested_rows(monkeypatch):
    from optimum.rbln.transformers.models.seq2seq.seq2seq_architecture import Seq2SeqEncoderWrapper

    torch.manual_seed(42)
    model = BartForConditionalGeneration(
        BartConfig(
            encoder_layers=1,
            decoder_layers=1,
            d_model=16,
            encoder_attention_heads=2,
            decoder_attention_heads=2,
            encoder_ffn_dim=32,
            decoder_ffn_dim=32,
            vocab_size=32,
            max_position_embeddings=16,
        )
    ).eval()
    wrapper = Seq2SeqEncoderWrapper(model, 5).eval()
    ids = torch.tensor([[2, 5, 6, 1, 1], [2, 7, 8, 9, 1]])
    mask = ids.ne(1).long()
    caches = [torch.full((4, 2, 5, 8), -1234.0) for _ in range(2)]

    def cache_update(cache, value, position, axis):
        assert int(axis) == 0
        result = cache.clone()
        result[int(position) : int(position) + value.shape[0]] = value
        return result

    monkeypatch.setattr(torch.ops.rbln_custom_ops, "rbln_cache_update", cache_update)
    with torch.no_grad():
        hidden = model.model.encoder(input_ids=ids, attention_mask=mask).last_hidden_state
        actual = wrapper(ids, mask, torch.tensor([1], dtype=torch.int16), *caches)
        for value, projection in zip(actual, [wrapper.cross_k_projects[0], wrapper.cross_v_projects[0]], strict=True):
            expected = projection(hidden).view(2, 5, 2, 8).transpose(1, 2)
            torch.testing.assert_close(value[1:3], expected)
            torch.testing.assert_close(value[0], caches[0][0])
            torch.testing.assert_close(value[3], caches[0][3])


@pytest.mark.parametrize(
    "encoder_batch_size,offsets", [(None, [0, 1, 2, 3]), (1, [0, 1, 2, 3]), (2, [0, 2]), (4, [0])]
)
def test_generation_keeps_padding_and_cache_row_order(encoder_batch_size, offsets):
    from optimum.rbln import RBLNBartForConditionalGeneration

    calls = []

    class Encoder:
        def forward(self, input_ids, attention_mask, block_tables, **kwargs):
            calls.append((input_ids.clone(), attention_mask.clone(), block_tables.tolist()))
            return SimpleNamespace(last_hidden_state=input_ids)

        __call__ = forward

    encoder = Encoder()
    model = SimpleNamespace(
        get_encoder=lambda: encoder,
        main_input_name="input_ids",
        config=SimpleNamespace(pad_token_id=1),
        rbln_config=SimpleNamespace(enc_max_seq_len=5, dtype=torch.float32, encoder_batch_size=encoder_batch_size),
    )
    ids = torch.arange(12).view(4, 3)
    result = RBLNBartForConditionalGeneration._prepare_encoder_decoder_kwargs_for_generation(
        model, ids, {"attention_mask": torch.ones_like(ids)}
    )
    assert [call[2][0] for call in calls] == offsets
    torch.testing.assert_close(torch.cat([call[0] for call in calls]), torch.nn.functional.pad(ids, (0, 2), value=1))
    torch.testing.assert_close(torch.cat([call[1] for call in calls]), result["attention_mask"].float())
