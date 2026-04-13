"""
Monkey-patched golden vs unpatched golden — pure CPU comparison.

No RBLN compilation. Tests whether the three compile-time patches
(_apply_rotary_emb_real, _modulate_no_where, _patch_rope_to_real)
produce numerically identical results to the original model.
"""

import copy
import types

import torch
from scipy.stats import pearsonr

MODEL_DIR = "/mnt/shared_data/.cache/pretrained_models/NC/VARCO_I2I_inference/models"


def compute_pearsonr(a: torch.Tensor, b: torch.Tensor) -> float:
    x = a.float().flatten().cpu().numpy()
    y = b.float().flatten().cpu().numpy()
    r, _ = pearsonr(x, y)
    return r


def main():
    import diffusers.models.transformers.transformer_qwenimage as _tq
    from diffusers.models.transformers.transformer_qwenimage import (
        QwenImageTransformer2DModel,
        QwenImageTransformerBlock,
    )
    from optimum.rbln.diffusers.models.transformers.transformer_qwenimage import (
        QwenImageTransformer2DModelWrapper,
        _apply_rotary_emb_real,
        _modulate_no_where,
        _patch_rope_to_real,
    )

    batch = 1
    height, width = 1024, 1024
    prompt_embed_length = 512
    vae_scale_factor = 8

    lat_h = 2 * (height // (vae_scale_factor * 2))
    lat_w = 2 * (width // (vae_scale_factor * 2))
    packed_h, packed_w = lat_h // 2, lat_w // 2

    img_shapes = [[(1, packed_h, packed_w), (1, packed_h, packed_w)]] * batch
    txt_seq_lens = [prompt_embed_length] * batch

    # ── Load inputs ──────────────────────────────────────────────────
    inputs = torch.load("/tmp/transformer_golden_inputs.pt")
    dummy_hs = inputs["hidden_states"]
    dummy_enc_raw = inputs["encoder_hidden_states"]
    dummy_t = torch.tensor([1.0] * batch)

    actual_enc_len = dummy_enc_raw.shape[1]
    if actual_enc_len < prompt_embed_length:
        pad_len = prompt_embed_length - actual_enc_len
        dummy_enc = torch.nn.functional.pad(dummy_enc_raw, (0, 0, 0, pad_len), value=0.0)
        enc_mask = torch.zeros(batch, prompt_embed_length, dtype=torch.bool)
        enc_mask[:, :actual_enc_len] = True
    else:
        dummy_enc = dummy_enc_raw
        enc_mask = None

    # ── Load model (1 layer for speed) ───────────────────────────────
    print("[1/3] Loading QwenImageTransformer2DModel (num_layers=1) ...")
    transformer = QwenImageTransformer2DModel.from_pretrained(
        MODEL_DIR, subfolder="transformer", torch_dtype=torch.float32,
        num_layers=3,
        zero_cond_t=True,
    )
    transformer.eval()

    # ── Run A: original (no patches) ─────────────────────────────────
    print("[2/3] Forward — original (no patches) ...")
    with torch.no_grad():
        out_original = transformer(
            hidden_states=dummy_hs,
            encoder_hidden_states=dummy_enc,
            encoder_hidden_states_mask=enc_mask,
            timestep=dummy_t,
            img_shapes=img_shapes,
            guidance=None,
            return_dict=False,
        )[0]

    # ── Run B: with all three monkey patches (direct call) ──────────
    print("[3/4] Forward — with monkey patches (direct) ...")

    original_rotary = _tq.apply_rotary_emb_qwen
    original_modulate = QwenImageTransformerBlock._modulate
    original_pos_forward = transformer.pos_embed.forward
    original_pos_freqs = transformer.pos_embed.pos_freqs
    original_neg_freqs = transformer.pos_embed.neg_freqs

    try:
        _tq.apply_rotary_emb_qwen = _apply_rotary_emb_real
        QwenImageTransformerBlock._modulate = _modulate_no_where
        _patch_rope_to_real(transformer, img_shapes, txt_seq_lens, dtype=torch.float32)

        with torch.no_grad():
            out_patched = transformer(
                hidden_states=dummy_hs,
                encoder_hidden_states=dummy_enc,
                encoder_hidden_states_mask=enc_mask,
                timestep=dummy_t,
                img_shapes=img_shapes,
                guidance=None,
                return_dict=False,
            )[0]
    finally:
        _tq.apply_rotary_emb_qwen = original_rotary
        QwenImageTransformerBlock._modulate = original_modulate
        transformer.pos_embed.forward = original_pos_forward
        transformer.pos_embed.pos_freqs = original_pos_freqs
        transformer.pos_embed.neg_freqs = original_neg_freqs

    # ── Run C: with all three monkey patches + wrapper ────────────
    print("[4/4] Forward — with monkey patches + wrapper ...")

    enc_mask_float = torch.ones(batch, prompt_embed_length, dtype=torch.float32)
    if actual_enc_len < prompt_embed_length:
        enc_mask_float[:, actual_enc_len:] = 0.0

    patched_model = copy.deepcopy(transformer)
    original_rotary2 = _tq.apply_rotary_emb_qwen
    original_modulate2 = QwenImageTransformerBlock._modulate
    try:
        _tq.apply_rotary_emb_qwen = _apply_rotary_emb_real
        QwenImageTransformerBlock._modulate = _modulate_no_where
        _patch_rope_to_real(patched_model, img_shapes, txt_seq_lens, dtype=torch.float32)
        wrapped = QwenImageTransformer2DModelWrapper(patched_model, img_shapes, txt_seq_lens).eval()
        with torch.no_grad():
            out_wrapped = wrapped(
                hidden_states=dummy_hs,
                encoder_hidden_states=dummy_enc,
                timestep=dummy_t,
                encoder_hidden_states_mask=enc_mask_float,
            )
        if isinstance(out_wrapped, tuple):
            out_wrapped = out_wrapped[0]
    finally:
        _tq.apply_rotary_emb_qwen = original_rotary2
        QwenImageTransformerBlock._modulate = original_modulate2

    # ── Compare ──────────────────────────────────────────────────────
    def report_pair(label, a, b):
        r = compute_pearsonr(a, b)
        diff = (a - b).abs()
        sa = a.abs().mean().item()
        sb = b.abs().mean().item()
        print(f"  {label}")
        print(f"    pearsonr     : {r:.10f}")
        print(f"    max abs diff : {diff.max().item():.6e}")
        print(f"    mean abs diff: {diff.mean().item():.6e}")
        print(f"    scale a/b    : {sa:.6e} / {sb:.6e}  ratio={sb / sa:.6f}")
        if r > 0.9999:
            tag = "PASS"
        elif r > 0.999:
            tag = "CLOSE"
        elif r > 0.99:
            tag = "WARNING"
        else:
            tag = "FAIL"
        print(f"    → {tag}")
        print()

    print()
    print("=" * 60)
    print("RESULTS (CPU only, no RBLN)")
    print("=" * 60)
    report_pair("A vs B : original  vs  patched (direct)", out_original, out_patched)
    report_pair("A vs C : original  vs  patched+wrapper",  out_original, out_wrapped)
    report_pair("B vs C : patched   vs  patched+wrapper",  out_patched,  out_wrapped)


if __name__ == "__main__":
    main()
