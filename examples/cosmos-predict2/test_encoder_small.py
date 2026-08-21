"""
Standalone Wan VAE **encoder** compile+run test at a small resolution (raw rebel, uses the shipping
_VAEWanEncoder0/_VAEWanEncoderN wrappers so it exercises the real encoder cache code).

Layout: caches are CDHW-flattened (n, d, c*h*w) -- folding C into the merged axis so the device does
NOT 64-block-pad the channel (avoids the AnnotatePhysicalView reconcile + SHM blow-up). idx0 (conv_in
cache, 3ch) is runtime I/O (channel-first), threaded E0->EN. idx1.. are shared static DRAM: E0 writes
them via rbln_cache_update, EN reads them back. Sharing is set up at compile time via one
CompileContext + mark_static_address reusing the SAME tensor objects for E0 and EN example inputs.

Encoder is causal-temporal: E0 encodes the first frame (-> 1 latent frame), EN encodes each following
4-frame chunk (-> 1 latent frame).

Run:
    python test_encoder_small.py [H] [W] [DEVICE]
    # default: python test_encoder_small.py 256 256 0

Same open issue as the decoder: EN READS the static caches, and on the current compiler read static
caches are demanded as runtime inputs (device=rbln) rather than auto-managed -- so EN's run may report
`The input(feat_cache_1) is not specified`. E0 (write-only) auto-manages fine. Compile of both graphs
and the E0 run work.
"""

import os
import sys

os.environ.setdefault("HF_HOME", "/mnt/shared_data/groups/sw_dev/.cache/huggingface")

import rebel
import torch
from diffusers.models.autoencoders.autoencoder_kl_wan import AutoencoderKLWan
from rebel.compile_context import CompileContext

from optimum.rbln.diffusers.models.autoencoders.autoencoder_kl_wan import (
    _VAEWanEncoder0,
    _VAEWanEncoderN,
    get_cache_size_enc,
)


H = int(sys.argv[1]) if len(sys.argv) > 1 else 256
W = int(sys.argv[2]) if len(sys.argv) > 2 else 256
DEV = int(sys.argv[3]) if len(sys.argv) > 3 else 0
CHUNK = 4  # EN encodes 4 input frames -> 1 latent frame

vae = AutoencoderKLWan.from_pretrained(
    "nvidia/Cosmos-Predict2-2B-Video2World", subfolder="vae", torch_dtype=torch.float32
).eval()
in_ch = vae.config.in_channels
cache_0, cache_n = get_cache_size_enc(H, W)  # per-slot (n,c,d,h,w); [0]=E0 (padded), [1]=EN
NC = len(cache_0)


def cache_shape(shape, is_idx0):
    n, c, d, h, w = shape
    if is_idx0:
        return [n, c, d, h, w]        # idx0: channel-first runtime I/O
    return [n, d, c * h * w]          # idx1..: cdhw static (n, d, c*h*w)


# Encoder E0 AND EN both take feat_cache_0 (idx0, channel-first I/O) + idx1..(cdhw static). NOTE: unlike
# the decoder (D0 drops idx0), the encoder E0 keeps feat_cache_0 as an input -- _VAEWanEncoder0.forward
# does `list(args)[1:]` to skip it, so it must be present or the whole arg list shifts by one.
e0_ii = [("x", [1, in_ch, 1, H, W], "float32")] + [
    (f"feat_cache_{i}", cache_shape(cache_0[i], i == 0), "float32") for i in range(NC)
]
en_ii = [("x", [1, in_ch, CHUNK, H, W], "float32")] + [
    (f"feat_cache_{i}", cache_shape(cache_n[i], i == 0), "float32") for i in range(NC)
]


def dummies(input_info, static=None):
    static = static or {}
    return tuple(static[n] if n in static else torch.zeros(*s, dtype=getattr(torch, dt)) for n, s, dt in input_info)


# shared context; idx1.. static, same tensor objects reused for E0 & EN (E0 writes -> EN reads).
ctx = CompileContext(use_weight_sharing=True)
e0x = dummies(e0_ii)
static_tensors = {}
for (name, _, _), t in zip(e0_ii, e0x):
    if "feat_cache" in name:
        static_tensors[name] = t
        ctx.mark_static_address(t, name)
enx = dummies(en_ii, static_tensors)
for (name, _, _), t in zip(en_ii, enx):
    if "feat_cache" in name and name != "feat_cache_0":
        ctx.mark_static_address(t, name)

print(f"H={H} W={W} in_ch={in_ch} enc_caches={NC} dev={DEV}", flush=True)
print("compiling E0 ...", flush=True)
ce0 = rebel.compile_from_torch(_VAEWanEncoder0(vae, H, W), input_info=e0_ii, example_inputs=e0x, compile_context=ctx)
print("compiling EN ...", flush=True)
cen = rebel.compile_from_torch(_VAEWanEncoderN(vae, H, W), input_info=en_ii, example_inputs=enx, compile_context=ctx)
print("both compiled OK", flush=True)

re0 = ce0.create_runtime(device=DEV, tensor_type="pt")
ren = cen.create_runtime(device=DEV, tensor_type="pt")
print("runtimes created", flush=True)
print(re0)
print(ren)

# ---- run + native reference (pearson) ----
from scipy.stats import pearsonr  # noqa: E402

torch.manual_seed(0)
num_frames = 1 + CHUNK  # E0(frame0) + EN(frames 1..4) = 2 latent frames
video = torch.randn(1, in_ch, num_frames, H, W)

# native chunked reference (diffusers encoder + quant_conv per chunk, same causal split)
conv_num = vae._cached_conv_counts["encoder"]
fc = [None] * conv_num
ref = []
with torch.no_grad():
    ci = [0]
    ref.append(vae.quant_conv(vae.encoder(video[:, :, :1], feat_cache=fc, feat_idx=ci)))
    ci = [0]
    ref.append(vae.quant_conv(vae.encoder(video[:, :, 1:1 + CHUNK], feat_cache=fc, feat_idx=ci)))
ref = torch.cat(ref, dim=2)

# rbln E0 then EN. E0's feat_cache_0 input is dead (list(args)[1:] skips it, idx0 output = feat_map[0])
# so it is DCE'd -> E0 runtime takes only x. idx1.. are static (E0 write-only auto-managed). EN then
# reads idx1.. (device=rbln) + takes feat_cache_0 (E0's fc0) as I/O input.
o0 = re0(video[:, :, :1])
print("E0 ran:", tuple(o0[0].shape), flush=True)
oN = ren(video[:, :, 1:1 + CHUNK], o0[1])
print("EN ran:", tuple(oN[0].shape), flush=True)
mine = torch.cat([o0[0], oN[0]], dim=2)

p = pearsonr(mine.flatten().float().numpy(), ref.flatten().float().numpy())[0]
print(f"[rbln vs native diffusers] pearson = {p:.6f}  maxabsdiff = {(mine - ref).abs().max().item():.3e}", flush=True)
