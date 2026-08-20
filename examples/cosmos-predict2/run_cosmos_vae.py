"""
Decoder-only, full-res (704x1280) Wan VAE decode via the shipping RBLNAutoencoderKLWan
(non-flush cache_update path). The DN chunk's Intermediate is ~12.4GB, so this OOMs at runtime
(SYS_ENOMEM, key=Intermediate, size~12400394368) -- the OOM we are trying to fix.

Only the decoder is compiled (encoder skipped) to keep the run fast and isolate the decoder OOM.

Usage:
    python run_cosmos_vae.py [DEVICE]     # e.g. python run_cosmos_vae.py 0
"""

import os
import sys

os.environ.setdefault("HF_HOME", "/mnt/shared_data/groups/sw_dev/.cache/huggingface")

import torch
from diffusers import AutoencoderKLWan

from optimum.rbln import RBLNAutoencoderKLWan
from optimum.rbln.diffusers.configurations.models.configuration_autoencoder_kl_wan import (
    RBLNAutoencoderKLWanConfig,
)

H, W = 704, 1280

# Load just the VAE (the full pipeline pulls a safety/guardrail model that fails to load here).
model_id = "nvidia/Cosmos-Predict2-2B-Video2World"
vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float32).eval()

# Decoder-only, full res. uses_encoder is force-True in the config __init__ (a FIXME `or True`);
# override it on the instance so only the decoder compiles + builds runtimes.
cfg = RBLNAutoencoderKLWanConfig(
    create_runtimes=True,
    height=H,
    width=W,
    num_frames=93,
    vae_scale_factor_spatial=8,
    vae_scale_factor_temporal=4,
)
cfg.uses_encoder = False

model = RBLNAutoencoderKLWan.from_model(vae, rbln_config=cfg)
print(model)

# Full-res decode -> DN chunk OOMs (~12.4GB Intermediate). 3 latent frames = D0 + 2x DN.
z_dim = getattr(vae.config, "z_dim", 16)
lh, lw = H // 8, W // 8
z = torch.randn(1, z_dim, 3, lh, lw)
print(f"decoding z={tuple(z.shape)} (z_dim={z_dim}, latent={lh}x{lw}) on device {DEV} ...", flush=True)
out = model.decode(z).sample
print("decoded:", tuple(out.shape))
