"""
Toy compile tests for _apply_rotary_emb_real and _modulate_no_where in isolation.

Usage:
    python test_toy_ops.py                  # both
    python test_toy_ops.py --op rope        # RoPE only
    python test_toy_ops.py --op modulate    # modulate only
"""

import argparse
import rebel
import torch
import torch.nn as nn
from scipy.stats import pearsonr


def compute_pearsonr(a: torch.Tensor, b: torch.Tensor) -> float:
    x = a.float().flatten().cpu().numpy()
    y = b.float().flatten().cpu().numpy()
    r, _ = pearsonr(x, y)
    return r


# ── RoPE ──────────────────────────────────────────────────────────────

class RoPEModule(nn.Module):
    """Wraps the half-split RoPE as a compilable module.

    cos/sin are registered as buffers (compile-time constants).
    """

    def __init__(self, cos: torch.Tensor, sin: torch.Tensor):
        super().__init__()
        self.register_buffer("cos", cos)  # [1, S, 1, D]
        self.register_buffer("sin", sin)  # [1, S, 1, D]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        d = x.shape[-1]
        x1, x2 = x.split(d // 2, dim=-1)
        return x * self.cos + torch.cat([-x2, x1], dim=-1) * self.sin


def test_rope(batch=1, seq_len=512, num_heads=24, head_dim=128):
    print("\n" + "=" * 60)
    print("Toy compile: _apply_rotary_emb_real (half-split RoPE)")
    print("=" * 60)

    half_d = head_dim // 2
    freqs = torch.randn(seq_len, half_d)
    cos_half = torch.cos(freqs)
    sin_half = torch.sin(freqs)
    cos_full = torch.cat([cos_half, cos_half], dim=-1)  # [S, D]
    sin_full = torch.cat([sin_half, sin_half], dim=-1)  # [S, D]

    cos_buf = cos_full[None, :, None, :].float()  # [1, S, 1, D]
    sin_buf = sin_full[None, :, None, :].float()  # [1, S, 1, D]

    model = RoPEModule(cos_buf, sin_buf).eval()
    x = torch.randn(batch, seq_len, num_heads, head_dim)

    print("  Running PyTorch golden...")
    with torch.no_grad():
        golden = model(x)

    print("  Compiling with RBLN...")
    compiled = torch.compile(model, backend="rbln", dynamic=False)
    with torch.no_grad():
        result = compiled(x)

    r = compute_pearsonr(golden, result)
    scale_golden = golden.abs().mean().item()
    scale_result = result.abs().mean().item()
    max_diff = (golden - result).abs().max().item()
    status = "PASS" if r >= 0.99 else "FAIL"
    print(f"  [{status}] pearsonr = {r:.6f}")
    print(f"  scale golden={scale_golden:.6e}  rbln={scale_result:.6e}  ratio={scale_result/scale_golden:.6f}")
    print(f"  max abs diff = {max_diff:.6e}")
    return r


# ── Modulate ──────────────────────────────────────────────────────────

class ModulateModule(nn.Module):
    """Wraps _modulate_no_where as a compilable module."""

    _HIDDEN = 3072

    def forward(
        self,
        x: torch.Tensor,
        mod_params: torch.Tensor,
        index: torch.Tensor,
    ) -> tuple:
        shift, scale, gate = mod_params.chunk(3, dim=-1)

        n = shift.size(0) // 2
        idx = index.unsqueeze(-1).to(x.dtype)
        inv = 1 - idx
        shift = shift[:n].unsqueeze(1) * inv + shift[n:].unsqueeze(1) * idx
        scale = scale[:n].unsqueeze(1) * inv + scale[n:].unsqueeze(1) * idx
        gate = gate[:n].unsqueeze(1) * inv + gate[n:].unsqueeze(1) * idx

        return x * (1 + scale) + shift, gate


def test_modulate(batch=1, seq_len=8192, hidden=3072):
    print("\n" + "=" * 60)
    print("Toy compile: _modulate_no_where")
    print("=" * 60)

    model = ModulateModule().eval()

    x = torch.randn(batch, seq_len, hidden)
    mod_params = torch.randn(batch * 2, hidden * 3)
    index = torch.tensor([1] * batch)

    print("  Running PyTorch golden...")
    with torch.no_grad():
        golden_out, golden_gate = model(x, mod_params, index)

    print("  Compiling with RBLN...")
    compiled = torch.compile(model, backend="rbln", dynamic=False)
    with torch.no_grad():
        result_out, result_gate = compiled(x, mod_params, index)

    r_out = compute_pearsonr(golden_out, result_out)
    r_gate = compute_pearsonr(golden_gate, result_gate)
    s1 = "PASS" if r_out >= 0.99 else "FAIL"
    s2 = "PASS" if r_gate >= 0.99 else "FAIL"
    print(f"  [{s1}] output:  pearsonr = {r_out:.6f}")
    print(f"    scale golden={golden_out.abs().mean().item():.6e}  rbln={result_out.abs().mean().item():.6e}  ratio={result_out.abs().mean().item()/golden_out.abs().mean().item():.6f}")
    print(f"    max abs diff = {(golden_out - result_out).abs().max().item():.6e}")
    print(f"  [{s2}] gate:    pearsonr = {r_gate:.6f}")
    print(f"    scale golden={golden_gate.abs().mean().item():.6e}  rbln={result_gate.abs().mean().item():.6e}  ratio={result_gate.abs().mean().item()/golden_gate.abs().mean().item():.6f}")
    print(f"    max abs diff = {(golden_gate - result_gate).abs().max().item():.6e}")
    return r_out, r_gate


# ── Main ──────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--op", default="both", choices=["both", "rope", "modulate"])
    args = p.parse_args()

    if args.op in ("both", "rope"):
        test_rope()

    if args.op in ("both", "modulate"):
        test_modulate()


if __name__ == "__main__":
    main()
