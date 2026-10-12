"""Minimal repro: an `rbln_cache_update` placed before other compute can overwrite a static cache before
an earlier read of that cache in the same graph has run (write-after-read hazard).

This is the CogVideoX VAE decoder_n pattern: a causal conv reads its cache (cat(cache, x)), and the new
cache is the last frames of x. The new value does not depend on the old cache, so nothing in the data
flow orders the write after the read. In the VAE, writing each cache right after its conv (mid-graph)
corrupts the decoder_n output (per-frame r 0.003..0.99), while writing all caches at the graph end works.

Two graphs share one static cache (CompileContext.mark_static_address):
    W  : cache <- x0                               (like decoder_0)
    RW : out = conv(cat(cache, x1)); cache <- x1   (like decoder_n)
Run W(x0) then RW(x1); the expected out uses x0 from the cache. The RW write is placed either right after
the conv ("mid", followed by more compute) or at the end of the graph ("end").

    RBLN_DEVICES=8 python examples/cogvideo/repro_cache_update_war.py
"""

import sys

import rebel
import torch
from rebel.compile_context import CompileContext


C, T, H, W = 64, 2, 32, 64  # cache (1, C, T, H*W); H*W is a multiple of 64


def update(cache, item):
    position = torch.tensor(0, dtype=torch.int16)
    axis = torch.tensor(2, dtype=torch.int16)
    return torch.ops.rbln_custom_ops.rbln_cache_update(cache, item, position, axis)


class Writer(torch.nn.Module):
    def forward(self, x, cache):
        return x.sum(), update(cache, x.reshape(1, C, T, H * W))


class ReadWriter(torch.nn.Module):
    def __init__(self, mode: str):
        super().__init__()
        self.mode = mode
        torch.manual_seed(0)
        self.conv = torch.nn.Conv3d(C, C, kernel_size=(3, 3, 3), padding=(0, 1, 1))
        self.post = torch.nn.Conv3d(C, C, kernel_size=(1, 3, 3), padding=(0, 1, 1))

    def forward(self, x, cache):
        old = cache.reshape(1, C, T, H, W)
        h = self.conv(torch.cat([old, x], dim=2))
        new = x.reshape(1, C, T, H * W)
        if self.mode == "dep_mul0":  # make the written value depend on the read (may be folded away)
            new = new + old.reshape(1, C, T, H * W)[..., :1] * 0.0
        elif self.mode == "dep_slicecat":  # new cache = last T frames of cat(old, x), as in diffusers
            new = torch.cat([old, x], dim=2)[:, :, -T:].reshape(1, C, T, H * W)
        elif self.mode == "dep_h":  # the write waits for the conv output that read the cache
            new = new + h.mean() * 0.0
        elif self.mode == "dep_hslice":  # same, through a one-element slice of the conv output
            new = new + h[:, :1, :1, :1, :1].reshape(1, 1, 1, 1) * 0.0
        dummy = update(cache, new) if self.mode != "end" else None
        for _ in range(4):  # more compute after the write
            h = torch.nn.functional.silu(self.post(h))
        if self.mode == "end":
            dummy = update(cache, new)
        return h, dummy


def main(mode: str):
    torch.manual_seed(1)
    x0 = torch.randn(1, C, T, H, W)
    x1 = torch.randn(1, C, T, H, W)
    rw = ReadWriter(mode).eval()
    with torch.no_grad():
        expected = rw(x1, x0.reshape(1, C, T, H * W).clone())[0]  # CPU: op is a no-op, cache holds x0
        wrong = rw(x1, x1.reshape(1, C, T, H * W).clone())[0]  # what a write-before-read gives

    context = CompileContext(use_weight_sharing=True)
    cache = torch.zeros(1, C, T, H * W)
    context.mark_static_address(cache)
    info = [("x", [1, C, T, H, W], "float32"), ("cache", [1, C, T, H * W], "float32")]
    w = rebel.compile_from_torch(Writer().eval(), info, example_inputs=[x0, cache], compile_context=context)
    cache_rw = torch.zeros(1, C, T, H * W)
    rw_c = rebel.compile_from_torch(rw, info, example_inputs=[x1, cache], compile_context=context)
    del cache_rw
    rt_w = rebel.Runtime(w, tensor_type="pt")
    rt_rw = rebel.Runtime(rw_c, tensor_type="pt")
    rt_w(x0)
    out = rt_rw(x1)
    out = out[0] if isinstance(out, (list, tuple)) else out

    def r(a, b):
        a, b = a.flatten().double(), b.flatten().double()
        return torch.corrcoef(torch.stack([a, b]))[0, 1].item()

    print(f"[{mode}] r(out, expected={'cache=x0'})={r(out, expected):.6f}  r(out, cache=x1)={r(out, wrong):.6f}")


if __name__ == "__main__":
    for mode in sys.argv[1:] or ["end", "mid"]:
        main(mode)
