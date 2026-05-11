"""Subprocess worker for hybrid bf16/dlfloat compile of the Qwen2 visual tower.

Spawned by `hybrid_compile.run_hybrid_compile`. `RBLN_COMP_DTYPE` must already
be set in this subprocess's environment before Python starts — libtriton reads
it at init.

The visual tower is split into exactly two compile chunks per block, plus
the merger:

  kind=bf:
      block_bf_NN  — norm1 + attn + residual_1 + norm2 + fc1
                     outputs (h1, pre_act)
      merger       — full PatchMerger
  kind=dlf:
      block_dlf_NN — act + fc2 + residual_2

Only `block.mlp.fc2` (K=5120) needs dlfloat to keep image embedding
Pearson above 0.96; all other ops stay bfloat.
"""

import argparse
import json
import os
from pathlib import Path

import rebel
import torch
import torch.nn as nn
from transformers.modeling_utils import no_init_weights
from transformers.models.qwen2_vl.configuration_qwen2_vl import Qwen2VLVisionConfig
from transformers.models.qwen2_vl.modeling_qwen2_vl import (
    Qwen2VisionTransformerPretrainedModel,
)

from optimum.rbln.transformers.models.qwen2_vl.qwen2_vl_architecture import (
    VisionAttention,
)


class _DtypeOnlyConfig:
    """Minimal stand-in to satisfy VisionAttention's `rbln_config.dtype` access."""

    def __init__(self, dtype=torch.float32):
        self.dtype = dtype


class BlockBf(nn.Module):
    """Block forward up to and including fc1 (excludes act + fc2 + residual_2)."""

    def __init__(self, native_block):
        super().__init__()
        self.norm1 = native_block.norm1
        self.attn = VisionAttention(native_block.attn, _DtypeOnlyConfig())
        self.norm2 = native_block.norm2
        self.fc1 = native_block.mlp.fc1

    def forward(self, hidden_states, attn_masks, cos, sin):
        attn_out = self.attn(self.norm1(hidden_states), attn_masks, [cos, sin])
        h1 = hidden_states + attn_out
        pre_act = self.fc1(self.norm2(h1))
        return h1, pre_act


class BlockDlfFc2(nn.Module):
    """Block tail: act + fc2 + residual_2."""

    def __init__(self, native_block):
        super().__init__()
        self.act = native_block.mlp.act
        self.fc2 = native_block.mlp.fc2

    def forward(self, h1, pre_act):
        return h1 + self.fc2(self.act(pre_act))


class Merger(nn.Module):
    def __init__(self, native_merger):
        super().__init__()
        self.merger = native_merger

    def forward(self, hidden_states):
        return self.merger(hidden_states)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", choices=["bf", "dlf"], required=True)
    parser.add_argument("--state-dict", required=True)
    parser.add_argument("--config-json", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--seq-lens", required=True)
    args = parser.parse_args()

    sd = torch.load(args.state_dict, weights_only=False)
    with open(args.config_json) as f:
        cfg_dict = json.load(f)
    vision_config = Qwen2VLVisionConfig(**cfg_dict)

    with no_init_weights():
        native_visual = Qwen2VisionTransformerPretrainedModel(vision_config)
    native_visual.load_state_dict(sd, strict=False)
    native_visual.eval()

    H = vision_config.embed_dim
    heads = vision_config.num_heads
    head_dim = H // heads
    fc1_out = native_visual.blocks[0].mlp.fc1.out_features

    seq_lens = [int(x) for x in args.seq_lens.split(",") if x.strip()]
    out_dir = Path(args.out_dir)

    dt = os.environ.get("RBLN_COMP_DTYPE", "(unset)")
    print(f"[hybrid-worker] kind={args.kind} dtype={dt} out={out_dir}", flush=True)

    for seq_len in seq_lens:
        tag = f"S{seq_len}"

        if args.kind == "bf":
            bf_info = [
                ("hidden_states", [seq_len, H], torch.float32),
                ("attn_masks", [1, 1, seq_len, seq_len], torch.float32),
                ("cos", [1, 1, seq_len, head_dim], torch.float32),
                ("sin", [1, 1, seq_len, head_dim], torch.float32),
            ]
            for i, blk in enumerate(native_visual.blocks):
                out = out_dir / f"block_bf_{i:02d}_{tag}.rbln"
                print(f"  compile {out.name}", flush=True)
                cm = rebel.compile_from_torch(BlockBf(blk).eval(), input_info=bf_info)
                cm.save(str(out))
            merger_info = [("hidden_states", [seq_len, H], torch.float32)]
            out = out_dir / f"merger_{tag}.rbln"
            print(f"  compile {out.name}", flush=True)
            cm = rebel.compile_from_torch(
                Merger(native_visual.merger).eval(), input_info=merger_info
            )
            cm.save(str(out))
        else:  # dlf
            dlf_info = [
                ("h1", [seq_len, H], torch.float32),
                ("pre_act", [seq_len, fc1_out], torch.float32),
            ]
            for i, blk in enumerate(native_visual.blocks):
                out = out_dir / f"block_dlf_{i:02d}_{tag}.rbln"
                print(f"  compile {out.name}", flush=True)
                cm = rebel.compile_from_torch(
                    BlockDlfFc2(blk).eval(), input_info=dlf_info
                )
                cm.save(str(out))

    print(f"[hybrid-worker] kind={args.kind} done", flush=True)


if __name__ == "__main__":
    main()
