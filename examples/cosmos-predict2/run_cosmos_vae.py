"""
Decoder-only Wan VAE decode via the shipping RBLNAutoencoderKLWan (cdhw cache layout).
Verifies correctness (pearson vs diffusers CPU decode) and checks the D0->DN cache handoff.

Usage:
    python run_cosmos_vae.py [DEVICE] [H] [W]
    # e.g. small (fits):  python run_cosmos_vae.py 0 256 256
    #      full res:      python run_cosmos_vae.py 0 704 1280
"""

import os
import sys

os.environ.setdefault("HF_HOME", "/mnt/shared_data/groups/sw_dev/.cache/huggingface")

import torch
from diffusers import AutoencoderKLWan
from scipy.stats import pearsonr

from optimum.rbln import RBLNAutoencoderKLWan
from optimum.rbln.diffusers.configurations.models.configuration_autoencoder_kl_wan import (
    RBLNAutoencoderKLWanConfig,
)


DEV = int(sys.argv[1]) if len(sys.argv) > 1 else 0
H = int(sys.argv[2]) if len(sys.argv) > 2 else 704
W = int(sys.argv[3]) if len(sys.argv) > 3 else 1280

model_id = "nvidia/Cosmos-Predict2-2B-Video2World"
vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float32).eval()

# Decoder-only. uses_encoder is force-True in the config __init__ (a FIXME `or True`); override on the
# instance so only the decoder compiles + builds runtimes.
cfg = RBLNAutoencoderKLWanConfig(
    device=DEV,
    create_runtimes=True,
    height=H,
    width=W,
    num_frames=93,
    vae_scale_factor_spatial=8,
    vae_scale_factor_temporal=4,
)
cfg.uses_encoder = False

model = RBLNAutoencoderKLWan.from_model(vae, rbln_config=cfg)

z_dim = getattr(vae.config, "z_dim", 16)
lh, lw = H // 8, W // 8
torch.manual_seed(0)
z = torch.randn(1, z_dim, 3, lh, lw)  # 3 latent frames = D0 + 2x DN (exercises the cache handoff)

print(f"decoding z={tuple(z.shape)} (z_dim={z_dim}, latent={lh}x{lw}) on device {DEV} ...", flush=True)
out = model.decode(z).sample
print("RBLN decoded:", tuple(out.shape), flush=True)

# reference: diffusers CPU decode of the same latent
with torch.no_grad():
    ref = vae.decode(z, return_dict=False)[0]
print("diffusers decoded:", tuple(ref.shape), flush=True)

p = pearsonr(out.flatten().float().numpy(), ref.flatten().float().numpy())[0]
maxdiff = (out - ref).abs().max().item()
print(f"[cdhw RBLN vs diffusers] pearson = {p:.6f}  maxabsdiff = {maxdiff:.3e}", flush=True)
