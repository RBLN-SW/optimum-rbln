"""Diagnostic-only fixed default-position RoPE cache; no learned operations move."""

import torch


class FixedModernBertRotary(torch.nn.Module):
    def __init__(self, rotary, max_seq_len=8192, dtype=torch.float32):
        super().__init__()
        self.max_seq_len = max_seq_len
        positions = torch.arange(max_seq_len, dtype=torch.int64).unsqueeze(0)
        dummy = torch.empty(1, max_seq_len, 1, dtype=dtype)
        for layer_type in rotary.layer_types:
            if rotary.config.rope_parameters[layer_type]["rope_type"] != "default":
                raise ValueError("This diagnostic cache requires default fixed RoPE")
            with torch.inference_mode():
                cos, sin = rotary(dummy, positions, layer_type)
            self.register_buffer(f"{layer_type}_cos", cos.clone(), persistent=False)
            self.register_buffer(f"{layer_type}_sin", sin.clone(), persistent=False)

    def forward(self, x, position_ids, layer_type):
        # The production model has no position_ids port and creates arange(seq_len).
        length = x.shape[1]
        if length > self.max_seq_len:
            raise ValueError("Sequence exceeds the fixed diagnostic cache")
        return (
            getattr(self, f"{layer_type}_cos")[:, :length].to(x.dtype),
            getattr(self, f"{layer_type}_sin")[:, :length].to(x.dtype),
        )


def install_fixed_rotary(model, max_seq_len=8192):
    model.rotary_emb = FixedModernBertRotary(model.rotary_emb, max_seq_len, model.dtype)
    return model
