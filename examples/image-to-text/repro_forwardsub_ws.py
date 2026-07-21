"""REPRO: forward-substitution (I-A)^-1 in the GatedDeltaNet fails RTOSAWeightReusabilityCheck under
use_weight_sharing=True (the forced setting). Real Qwen3.5-0.8B, first 4 layers ([L,L,L,F]).

BACKGROUND
----------
The GatedDeltaNet prefill chunk rule needs (I - A)^-1 where A is a strictly-lower nilpotent SxS matrix
(S = prefill_chunk_size = 128). Two ways to compute it:

  * log-depth squaring  M=(I+A)(I+A^2)(I+A^4)...  -> lowers + weight-shares fine, BUT numerically WRONG at
    trained-weight scale: intermediate powers A^16/A^32 reach ~1e8 and must catastrophically cancel back
    to the true O(1e2) inverse; float32 can't preserve it (wrong even on CPU, cos 0.065 vs HF).

  * forward substitution (HF's method, `torch_chunk_gated_delta_rule`) -> NUMERICALLY CORRECT (cos 1.0 vs
    HF at trained scale), builds each row from earlier rows so values stay O(inverse). This is what is
    currently in
      optimum-rbln/.../qwen3_5/qwen3_5_architecture.py  ::  rbln_chunk_gated_delta_rule  (lines ~135-141)
        _S = A.shape[-1]
        attn = A
        for i in range(1, _S):
            row = attn[..., i, :i].clone()
            sub = attn[..., :i, :i].clone()
            attn[..., i, :i] = row + (row.unsqueeze(-1) * sub).sum(-2)   # <-- in-place row assignment
        attn = attn + eye

CONSTRAINTS (fixed): use_weight_sharing=True, prefill_chunk_size=128.

WHAT HAPPENS
------------
The in-place row assignment `attn[..., i, :i] = ...` lowers to concatenate/scatter ops. Compiling the full
hybrid model (prefill + decode graphs) with use_weight_sharing=True then fails:
    [E] RTOSAWeightReusabilityCheck pass failed
    [E] mlir::detail::OpToOpPassAdaptor pass failed
(plus earlier debug "basic check failed in contrib_concatenate"). The forward-sub is prefill-only, so its
concatenate/scatter-derived constants are absent from the decode graph -> the weight-sharing "gen mode"
check can't find them.

The isolated forward-sub inverse (single graph, no weight-sharing) compiles fine and is device-accurate
(cos 0.99998) — see iso_fwdsub.py. It is specifically the FULL hybrid model + weight-sharing that fails.

RUN
---
  ~/venv_optimum/bin/python ~/qwen35_tests/repro_forwardsub_ws.py
Expect: compile aborts with 'RTOSAWeightReusabilityCheck pass failed'. (Swap the forward-sub block back
to the squaring version in qwen3_5_architecture.py and it compiles — but is numerically wrong at scale.)
"""
import os
import traceback

os.environ.setdefault("RBLN_ENABLE_VALIDATION", "0")
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
import torch
import torch.nn as nn
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForConditionalGeneration

from optimum.rbln import RBLNQwen3_5ForConditionalGeneration


model_id = "Qwen/Qwen3.5-0.8B"
N = 4  # first 4 layers of the real checkpoint = [L, L, L, F]
print(f"loading {model_id} and slicing to first {N} layers ...", flush=True)
# hf = Qwen3_5ForConditionalGeneration.from_pretrained("Qwen/Qwen3.5-0.8B", torch_dtype=torch.float32).eval()
hf = Qwen3_5ForConditionalGeneration.from_pretrained(model_id, torch_dtype=torch.float32).eval()
dec = hf.model.language_model
dec.layers = nn.ModuleList(list(dec.layers)[:N])
for cfg in {id(c): c for c in [hf.config, hf.config.text_config, dec.config]}.values():
    if hasattr(cfg, "num_hidden_layers"):
        cfg.num_hidden_layers = N
    if getattr(cfg, "layer_types", None):
        cfg.layer_types = list(cfg.layer_types)[:N]
print("layer_types:", hf.config.text_config.layer_types, flush=True)
D = f"/tmp/test_{model_id.replace('/', '_')}_layers{N}_4hyb"
hf.save_pretrained(D)

print("\ncompiling (use_weight_sharing=True, prefill_chunk_size=128 — both defaults/forced) ...", flush=True)

model = RBLNQwen3_5ForConditionalGeneration.from_pretrained(
    D, export=True, rbln_config={"visual": {"max_seq_len": 512},"num_devices": 1, "max_seq_len": 8192, "kvcache_partition_len": 4096, "create_runtimes": False}) # todo : support RSD
model.save_pretrained(D+"_compiled")
print("\n=== COMPILE OK (unexpected — forward-sub was supposed to fail weight-sharing) ===", flush=True)