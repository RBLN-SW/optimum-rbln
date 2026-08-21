"""
Standalone Wan VAE **decoder** compile+run test at a small resolution (raw rebel, no optimum wrappers).

What it does:
  1. builds an explicit D0 (first_chunk) / DN (steady-state) forward over vae.decoder,
  2. caches are CDHW-flattened  (n, d, c*h*w)  -- folding C into the merged axis so the device does NOT
     64-block-pad the channel (that avoids the AnnotatePhysicalView reconcile and the SHM blow-up that
     channel-last / (n,c,d,h*w) layouts hit on the current compiler),
  3. idx0 (conv_in cache) is runtime I/O (channel-first), threaded D0->DN,
  4. idx1..31 are shared static DRAM: D0 writes them via rbln_cache_update, DN reads them back.
     Sharing is set up at compile time: ONE CompileContext + mark_static_address on the SAME tensor
     objects reused as D0's and DN's example inputs.

Run:
    python test_decoder_small.py [H] [W] [DEVICE]
    # default: python test_decoder_small.py 256 256 0

NOTE (current status): D0 compiles+runs and writes its caches. DN compiles, but at DN run the runtime
reports `The input(feat_cache_1) is not specified` -- i.e. the *read* static caches are being demanded
as runtime inputs instead of auto-managed (D0, which only WRITES them, auto-manages fine). This is the
open issue to look at. Everything up to that point (compile of both graphs, D0 run) works.
"""

import os
import sys

os.environ.setdefault("HF_HOME", "/mnt/shared_data/groups/sw_dev/.cache/huggingface")

import rebel
import torch
from diffusers.models.autoencoders.autoencoder_kl_wan import AutoencoderKLWan
from rebel.compile_context import CompileContext

from optimum.rbln.diffusers.models.autoencoders.autoencoder_kl_wan import get_cache_size_dec


H = int(sys.argv[1]) if len(sys.argv) > 1 else 256
W = int(sys.argv[2]) if len(sys.argv) > 2 else 256
DEV = int(sys.argv[3]) if len(sys.argv) > 3 else 0

CACHE_T = 2
POS = torch.tensor(0, dtype=torch.int16)
AXIS = torch.tensor(1, dtype=torch.int16)  # cache layout (n, d, c*h*w): the frame(D) axis is axis 1
NC = 32  # decoder cached-conv count (idx0 = conv_in I/O, idx1..31 = static)


# ----------------------------------------------------------------------------------------------------
# explicit decoder forward (mirrors diffusers WanDecoder3d); `cc` is the cached-conv hook, `resample`
# handles the upsample3d temporal cache.
# ----------------------------------------------------------------------------------------------------
def resblock_fwd(rb, x, cc):
    h = rb.conv_shortcut(x)
    x = rb.nonlinearity(rb.norm1(x)); x = cc(rb.conv1, x)
    x = rb.nonlinearity(rb.norm2(x)); x = rb.dropout(x); x = cc(rb.conv2, x)
    return x + h


def midblock_fwd(mid, x, cc):
    x = resblock_fwd(mid.resnets[0], x, cc)
    for attn, rb in zip(mid.attentions, mid.resnets[1:]):
        if attn is not None:
            x = attn(x)
        x = resblock_fwd(rb, x, cc)
    return x


def upblock_fwd(ub, x, cc, resample):
    for rb in ub.resnets:
        x = resblock_fwd(rb, x, cc)
    if ub.upsamplers is not None:
        x = resample(ub.upsamplers[0], x)
    return x


def decoder_fwd(dec, x, cc, resample):
    x = cc(dec.conv_in, x)
    x = midblock_fwd(dec.mid_block, x, cc)
    for ub in dec.up_blocks:
        x = upblock_fwd(ub, x, cc, resample)
    x = dec.nonlinearity(dec.norm_out(x))
    x = cc(dec.conv_out, x)
    return x


class Ctx:
    """Per-chunk cache state. D0: first_chunk=True (writes caches, no reads). DN: reads+writes."""

    def __init__(self, args, cache_dims, first_chunk, fc0_in):
        self.args = args            # static cache inputs (D0: idx1..; DN: idx0.. with idx0 channel-first)
        self.dims = cache_dims      # per-slot (n,c,d,h,w)
        self.first = first_chunk
        self.fc0_in = fc0_in        # idx0 runtime cache from previous chunk (DN only)
        self.feat_idx = [0]
        self.dummies = []           # rbln_cache_update dummies (returned so they are not DCE'd)
        self.new0 = None            # idx0 runtime output (channel-first) -> next chunk

    def _slot(self, idx):
        return self.args[idx] if not self.first else self.args[idx - 1]  # D0 dropped idx0 input

    def _to_cache(self, new):
        # channel-first (n,c,d,h,w) -> (n, d, c*h*w)
        n, c, d, h, w = new.shape
        return new.permute(0, 2, 1, 3, 4).reshape(n, d, c * h * w).contiguous()

    def _from_cache(self, idx):
        # (n, d, c*h*w) -> channel-first (n,c,d,h,w)
        n, c, _d, h, w = self.dims[idx]
        return self._slot(idx).reshape(n, CACHE_T, c, h, w).permute(0, 2, 1, 3, 4)

    def cc(self, conv, x):
        idx = self.feat_idx[0]; self.feat_idx[0] += 1
        if self.first:
            old = None                       # first chunk: WanCausalConv3d does its own causal padding
        elif idx == 0:
            old = self.fc0_in                # idx0 runtime I/O (channel-first)
        else:
            old = self._from_cache(idx)      # read static cache -> channel-first
        new = x[:, :, -CACHE_T:, :, :].clone()
        if new.shape[2] < 2 and old is not None:
            new = torch.cat([old[:, :, -1:, :, :], new], dim=2)
        out = conv(x, old)
        if self.dims[idx][2] == 2 and new.shape[2] == 1:
            new = torch.nn.functional.pad(new, (0, 0, 0, 0, 1, 0))  # pad D 1->2
        if idx == 0:
            self.new0 = new.contiguous()     # idx0 runtime output
        else:
            self.dummies.append(torch.ops.rbln_custom_ops.rbln_cache_update(self._slot(idx), self._to_cache(new), POS, AXIS))
        return out

    def resample(self, rs, x):
        b, c, t, h, w = x.size()
        if rs.mode == "upsample3d":
            idx = self.feat_idx[0]; self.feat_idx[0] += 1
            if self.first:
                S = self._slot(idx)
                self.dummies.append(torch.ops.rbln_custom_ops.rbln_cache_update(S, S * 0.0, POS, AXIS))  # "Rep" -> zeros
            else:
                old = self._from_cache(idx)
                cache_x = x[:, :, -CACHE_T:, :, :].clone()
                if cache_x.shape[2] < 2:
                    cache_x = torch.cat([torch.zeros_like(cache_x), cache_x], dim=2)
                x = rs.time_conv(x, old)
                self.dummies.append(torch.ops.rbln_custom_ops.rbln_cache_update(self._slot(idx), self._to_cache(cache_x), POS, AXIS))
                x = x.reshape(b, 2, c, t, h, w)
                x = torch.stack((x[:, 0], x[:, 1]), 3)
                x = x.reshape(b, c, t * 2, h, w)
        t = x.shape[2]
        x = x.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
        x = rs.resample(x)
        x = x.view(b, t, x.size(1), x.size(2), x.size(3)).permute(0, 2, 1, 3, 4)
        return x


class D0(torch.nn.Module):
    def __init__(s):
        super().__init__(); s.dec = vae.decoder; s.dims = get_cache_size_dec(H, W)[0]

    def forward(s, z, *args):
        ctx = Ctx(args, s.dims, True, None)
        out = decoder_fwd(s.dec, z, ctx.cc, ctx.resample)
        return (out, ctx.new0, ctx.dummies)


class DN(torch.nn.Module):
    def __init__(s):
        super().__init__(); s.dec = vae.decoder; s.dims = get_cache_size_dec(H, W)[1]

    def forward(s, z, *args):
        ctx = Ctx(args, s.dims, False, args[0])
        out = decoder_fwd(s.dec, z, ctx.cc, ctx.resample)
        return (out, ctx.new0, ctx.dummies)


# ----------------------------------------------------------------------------------------------------
vae = AutoencoderKLWan.from_pretrained(
    "nvidia/Cosmos-Predict2-2B-Video2World", subfolder="vae", torch_dtype=torch.float32
).eval()
zc = getattr(vae.config, "z_dim", 16)
_, dnc = get_cache_size_dec(H, W)  # steady-state cache shapes (n,c,d,h,w)
lh, lw = H // 8, W // 8


def cache_shape(shape, is_idx0):
    n, c, d, h, w = shape
    if is_idx0:
        return [n, c, d, h, w]        # idx0: channel-first runtime I/O
    return [n, d, c * h * w]          # idx1..: cdhw static (n, d, c*h*w)


# D0 inputs: z + idx1..31 (idx0 is a D0 OUTPUT). DN inputs: z + idx0(I/O) + idx1..31.
d0_ii = [("z", [1, zc, 1, lh, lw], "float32")] + [(f"feat_cache_{i}", cache_shape(dnc[i], False), "float32") for i in range(1, NC)]
dn_ii = [("z", [1, zc, 1, lh, lw], "float32")] + [(f"feat_cache_{i}", cache_shape(dnc[i], i == 0), "float32") for i in range(NC)]


def dummies(input_info, static=None):
    static = static or {}
    return tuple(static[n] if n in static else torch.zeros(*s, dtype=getattr(torch, dt)) for n, s, dt in input_info)


# One shared CompileContext; mark idx1..31 static and REUSE the same tensor objects for D0 and DN so
# both graphs bind those caches to the same device DRAM (D0 writes -> DN reads).
ctx = CompileContext(use_weight_sharing=True)
d0x = dummies(d0_ii)
static_tensors = {}
for (name, _, _), t in zip(d0_ii, d0x):
    if "feat_cache" in name:
        static_tensors[name] = t
        ctx.mark_static_address(t, name)
dnx = dummies(dn_ii, static_tensors)
for (name, _, _), t in zip(dn_ii, dnx):
    if "feat_cache" in name and name != "feat_cache_0":
        ctx.mark_static_address(t, name)

print(f"H={H} W={W} latent={lh}x{lw} dev={DEV}", flush=True)
print("compiling D0 ...", flush=True)
cd0 = rebel.compile_from_torch(D0(), input_info=d0_ii, example_inputs=d0x, compile_context=ctx)
print("compiling DN ...", flush=True)
cdn = rebel.compile_from_torch(DN(), input_info=dn_ii, example_inputs=dnx, compile_context=ctx)
print("both compiled OK", flush=True)

rd0 = cd0.create_runtime(device=DEV, tensor_type="pt")
rdn = cdn.create_runtime(device=DEV, tensor_type="pt")
print("runtimes created", flush=True)

# ---- run + native reference (pearson) ----
from scipy.stats import pearsonr  # noqa: E402

torch.manual_seed(0)
lat = torch.randn(1, zc, 3, lh, lw)
zpq = torch.nn.functional.conv3d(lat, vae.post_quant_conv.weight, vae.post_quant_conv.bias)

# native chunked reference
conv_num = vae._cached_conv_counts["decoder"]
fc = [None] * conv_num
ref = []
with torch.no_grad():
    for i in range(zpq.shape[2]):
        ci = [0]
        ref.append(vae.decoder(zpq[:, :, i:i + 1], feat_cache=fc, feat_idx=ci, first_chunk=(i == 0)))
ref = torch.cat(ref, dim=2)

# rbln D0 then DN chunks
o0 = rd0(zpq[:, :, :1])
print("D0 ran:", tuple(o0[0].shape), flush=True)
outs = [o0[0]]
fc0 = o0[1]
for i in range(1, zpq.shape[2]):
    on = rdn(zpq[:, :, i:i + 1], fc0)
    outs.append(on[0]); fc0 = on[1]
    print(f"DN[{i}] ran:", tuple(on[0].shape), flush=True)
mine = torch.cat(outs, dim=2)

p = pearsonr(mine.flatten().float().numpy(), ref.flatten().float().numpy())[0]
print(f"[rbln vs native diffusers] pearson = {p:.6f}  maxabsdiff = {(mine - ref).abs().max().item():.3e}", flush=True)
