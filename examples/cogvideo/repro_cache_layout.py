"""Repro: static conv-cache layouts for the CogVideoX VAE, in the decoder_0 / decoder_n pattern.

Two graphs share one static cache buffer (CompileContext.mark_static_address), like decoder_0 / decoder_n:
    W  : out = conv(cat(x[:, :, :1], x[:, :, :1], x)); cache <- last 2 frames of the conv input
    RW : out = conv(cat(cache, x));                    cache <- last 2 frames of the conv input
The conv is diffusers' CogVideoXCausalConv3d. The cache item (n, c, 2, h, w) is stored with one of:
    pad            (n, c, d, h*w) with h*w zero-padded to a multiple of 64 (what the VAE uses)
    flat           (n, c, d, h*w), no padding
    chlast_view    (n, h*w, d, c) by a plain reshape (same bytes, axes relabeled)
    chlast_permute (n, h*w, d, c) by a real permute (n, c, d, h, w) -> (n, h, w, d, c)
Each variant compiles in its own process (a failed compile can abort it). With --run, the two graphs run
on the device (W(x0) then RW(x1)) and RW's output is compared with the CPU result.

    RBLN_DEVICES=26 python examples/cogvideo/repro_cache_layout.py --run
    python examples/cogvideo/repro_cache_layout.py --c 256 --h 240 --w 360 chlast_view

Limitation (2026-10-13): at 60x90 every layout compiles here, but in the read-write graph the cache stays a
regular runtime input ("The input(cache) is not specified"): this small graph's read path is left on the
host, so the cache is no longer a device-resident static buffer. At an aligned h*w (e.g. --h 64 --w 96) the
cache is static and RW matches the CPU (r=0.999991). Use the full VAE (verify_vae.py) to judge a layout.
"""

import argparse
import subprocess
import sys

import torch


LAYOUTS = ["pad", "flat", "chlast_view", "chlast_permute"]


def padded(hw: int) -> int:
    return -(-hw // 64) * 64


def buffer_shape(layout, c, h, w):
    if layout == "pad":
        return [1, c, 2, padded(h * w)]
    if layout == "flat":
        return [1, c, 2, h * w]
    return [1, h * w, 2, c]


def to_cache(layout, item):
    n, c, d, h, w = item.shape
    if layout == "chlast_view":
        return item.contiguous().reshape(n, h * w, d, c)
    if layout == "chlast_permute":
        return item.permute(0, 3, 4, 2, 1).reshape(n, h * w, d, c).contiguous()
    flat = item.reshape(n, c, d, h * w)
    if layout == "pad" and padded(h * w) != h * w:
        flat = torch.nn.functional.pad(flat, (0, padded(h * w) - h * w))
    return flat.contiguous()


def from_cache(layout, cache, c, h, w):
    n = cache.shape[0]
    if layout == "chlast_view":
        return cache.reshape(n, c, 2, h, w)
    if layout == "chlast_permute":
        return cache.reshape(n, h, w, 2, c).permute(0, 4, 3, 1, 2)
    return cache[..., : h * w].reshape(n, c, 2, h, w)


class Graph(torch.nn.Module):
    def __init__(self, conv, layout, c, h, w, reads_cache: bool):
        super().__init__()
        self.conv, self.layout, self.c, self.h, self.w, self.reads_cache = conv, layout, c, h, w, reads_cache

    def forward(self, x, cache):
        old = from_cache(self.layout, cache, self.c, self.h, self.w) if self.reads_cache else None
        out, new = self.conv(x, conv_cache=old)
        position = torch.tensor(0, dtype=torch.int16)
        axis = torch.tensor(2, dtype=torch.int16)
        item = to_cache(self.layout, new)
        return out, torch.ops.rbln_custom_ops.rbln_cache_update(cache, item, position, axis)


def run_one(layout, c, h, w, t, run):
    import rebel
    from diffusers.models.autoencoders.autoencoder_kl_cogvideox import CogVideoXCausalConv3d
    from rebel.compile_context import CompileContext

    torch.manual_seed(0)
    conv = CogVideoXCausalConv3d(c, c, kernel_size=3, pad_mode="first").eval()
    x0, x1 = torch.randn(1, c, t, h, w), torch.randn(1, c, t, h, w)
    with torch.no_grad():
        _, cache0 = conv(x0)
        expected, _ = conv(x1, conv_cache=cache0)

    context = CompileContext(use_weight_sharing=True)
    cache = torch.zeros(buffer_shape(layout, c, h, w))
    context.mark_static_address(cache)
    info = [("x", [1, c, t, h, w], "float32"), ("cache", buffer_shape(layout, c, h, w), "float32")]
    wr = rebel.compile_from_torch(
        Graph(conv, layout, c, h, w, False).eval(), info, example_inputs=[x0, cache], compile_context=context
    )
    rw = rebel.compile_from_torch(
        Graph(conv, layout, c, h, w, True).eval(), info, example_inputs=[x1, cache], compile_context=context
    )
    print(f"COMPILE OK {layout}", flush=True)
    if not run:
        return
    rt_w, rt_rw = rebel.Runtime(wr, tensor_type="pt"), rebel.Runtime(rw, tensor_type="pt")
    rt_w(x0)
    out = rt_rw(x1)
    out = out[0] if isinstance(out, (list, tuple)) else out
    a, b = out.flatten().double(), expected.flatten().double()
    r = torch.corrcoef(torch.stack([a, b]))[0, 1].item()
    print(f"RUN {layout}: r(out, CPU)={r:.6f} max|diff|={(a - b).abs().max().item():.4f}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("layouts", nargs="*", default=LAYOUTS)
    parser.add_argument("--c", type=int, default=512)
    parser.add_argument("--h", type=int, default=60)
    parser.add_argument("--w", type=int, default=90)
    parser.add_argument("--t", type=int, default=2, help="frames of x per call")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--child", action="store_true")
    args = parser.parse_args()

    if args.child:
        run_one(args.layouts[0], args.c, args.h, args.w, args.t, args.run)
        return
    print(f"c={args.c} h={args.h} w={args.w} (h*w={args.h * args.w}) t={args.t}")
    for layout in args.layouts:
        cmd = [sys.executable, __file__, layout, "--child", "--c", str(args.c), "--h", str(args.h)]
        cmd += ["--w", str(args.w), "--t", str(args.t)] + (["--run"] if args.run else [])
        proc = subprocess.run(cmd, capture_output=True, text=True)
        lines = [ln for ln in proc.stdout.splitlines() if ln.startswith(("COMPILE OK", "RUN"))]
        errors = [ln.strip() for ln in (proc.stdout + proc.stderr).splitlines() if "assert:" in ln or "error:" in ln]
        status = "PASS" if proc.returncode == 0 else f"FAIL (exit {proc.returncode})"
        print(f"{layout:15s} {status}  {' | '.join(lines)}")
        for err in errors[:2]:
            print(f"{'':16s}{err[:200]}")


if __name__ == "__main__":
    main()
