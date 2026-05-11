"""Hybrid bf16/dlfloat compile of the colqwen2 visual tower.

`RBLN_COMP_DTYPE` is a process-wide singleton (read by libtriton at init), so
two different compute dtypes cannot coexist in a single Python process. We
work around that by spawning two subprocesses — one with
`RBLN_COMP_DTYPE=bfloat`, one with `RBLN_COMP_DTYPE=dlfloat` — and stitching
their compiled chunks at inference time.

Per block, exactly two chunks are produced:

  block_bf_NN  (bf16)  — norm1 + attn(qkv+sdpa+proj) + residual_1 + norm2 + fc1
                          → (h1, pre_act)
  block_dlf_NN (dlf)   — act + fc2 + residual_2
                          → h_next

Plus a single `merger` (bf16) at the tower's tail. Only `block.mlp.fc2`
(K=5120 Linear, repeated 32×) is precision-sensitive in bfloat; everything
else stays bfloat.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

import torch


NUM_BLOCKS = 32  # Qwen2VL-2B visual has 32 blocks


# ---------------------------------------------------------------------------
# Orchestrator (called from RBLNQwen2VisionTransformerPretrainedModel)
# ---------------------------------------------------------------------------


def run_hybrid_compile(
    native_visual_state_dict: Dict[str, torch.Tensor],
    native_vision_config: Any,
    model_id: str,  # noqa: ARG001  (kept for back-compat with modeling caller)
    seq_lens: List[int],
    out_dir: Path,
) -> Dict[str, Any]:
    """Compile all bf + dlf sub-artifacts and return `{stem: RBLNCompiledModel}`."""
    import rebel

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sd_path = out_dir / "_native_visual_state_dict.pt"
    cfg_path = out_dir / "_native_vision_config.json"
    torch.save(native_visual_state_dict, sd_path)
    try:
        cfg_dict = native_vision_config.to_dict()
    except AttributeError:
        cfg_dict = dict(native_vision_config.__dict__)
    with open(cfg_path, "w") as f:
        json.dump(cfg_dict, f)

    seq_lens_str = ",".join(str(x) for x in seq_lens)
    worker_file = Path(__file__).with_name("_hybrid_compile_worker.py")

    def _spawn(kind: str, dtype: str):
        env = os.environ.copy()
        env["RBLN_COMP_DTYPE"] = dtype
        cmd = [
            sys.executable,
            str(worker_file),
            f"--kind={kind}",
            f"--state-dict={sd_path}",
            f"--config-json={cfg_path}",
            f"--out-dir={out_dir}",
            f"--seq-lens={seq_lens_str}",
        ]
        attempts = int(os.environ.get("RBLN_VIT_HYBRID_COMPILE_RETRIES", "8"))
        # Native compiler abort (-6/-11) is non-deterministic in current build.
        transient_rcs = {-6, -11}
        for attempt in range(attempts):
            print(
                f"[hybrid-orchestrator] subprocess kind={kind} dtype={dtype} "
                f"(attempt {attempt + 1}/{attempts})",
                flush=True,
            )
            r = subprocess.run(cmd, env=env, capture_output=True, text=True)
            if r.returncode == 0:
                if r.stdout:
                    print(r.stdout, flush=True)
                return
            stderr = r.stderr or ""
            tail = "\n".join(stderr.splitlines()[-30:])
            print(
                f"[hybrid-orchestrator] subprocess failed (rc={r.returncode}):\n{tail}",
                flush=True,
            )
            native_crash = (
                r.returncode in transient_rcs
                or "double free" in stderr
                or "corruption" in stderr
                or "Segmentation fault" in stderr
            )
            if native_crash:
                print("[hybrid-orchestrator] transient native crash — retrying …", flush=True)
                continue
            break
        raise RuntimeError(f"hybrid compile subprocess (kind={kind}) failed")

    _spawn("bf", "bfloat")
    _spawn("dlf", "dlfloat")

    compiled: Dict[str, Any] = {}
    for p in sorted(out_dir.glob("*.rbln")):
        compiled[p.stem] = rebel.RBLNCompiledModel(p)
    print(f"[hybrid-orchestrator] loaded {len(compiled)} compiled artifacts", flush=True)
    return compiled


# ---------------------------------------------------------------------------
# Inference (called from RBLNQwen2VisionTransformerPretrainedModel.forward)
# ---------------------------------------------------------------------------


def load_hybrid_runtimes(
    compile_dir: Path,
    device_map: Dict[str, Any],
    seq_lens: List[int],  # noqa: ARG001  (kept for back-compat)
    activate_profiler: bool = False,
    timeout: int | None = None,
) -> Dict[str, Any]:
    """Load one `rebel.Runtime` per saved .rbln, keyed by file stem."""
    import rebel

    compile_dir = Path(compile_dir)
    runtimes: Dict[str, Any] = {}
    for p in sorted(compile_dir.glob("*.rbln")):
        key = p.stem
        runtimes[key] = rebel.Runtime(
            str(p),
            tensor_type="pt",
            device=device_map.get(key, 0),
            activate_profiler=activate_profiler,
            timeout=timeout,
        )
    print(f"[hybrid-runtime] loaded {len(runtimes)} runtimes from {compile_dir}", flush=True)
    return runtimes


def _as_tensor(x):
    import numpy as np

    if isinstance(x, (list, tuple)):
        x = x[0]
    if not torch.is_tensor(x):
        x = torch.tensor(np.asarray(x))
    return x


def hybrid_forward_image(
    runtimes: Dict[str, Any],
    seq_len: int,
    hidden_state_padded: torch.Tensor,  # [seq_len, embed_dim]
    full_attn_masks: torch.Tensor,      # [1, 1, seq_len, seq_len]  1/0
    cos: torch.Tensor,                  # [1, 1, seq_len, head_dim]
    sin: torch.Tensor,                  # [1, 1, seq_len, head_dim]
) -> torch.Tensor:
    """Chain: block_bf_i → block_dlf_i for each block, then merger."""
    # The compiled BlockBf wrapper expects an additive logit mask
    # (-inf at masked positions, 0 elsewhere); transform on CPU so the
    # device graph sees a uniform input contract.
    additive_mask = (1.0 - full_attn_masks.to(torch.float32)) * torch.finfo(torch.float32).min
    tag = f"S{seq_len}"

    h = hidden_state_padded
    for i in range(NUM_BLOCKS):
        bf_rt = runtimes[f"block_bf_{i:02d}_{tag}"]
        dlf_rt = runtimes[f"block_dlf_{i:02d}_{tag}"]
        bf_out = bf_rt.run(h, additive_mask, cos, sin)
        if not isinstance(bf_out, (list, tuple)) or len(bf_out) != 2:
            raise RuntimeError(f"block_bf_{i:02d}_{tag} expected to return (h1, pre_act)")
        h1 = _as_tensor(bf_out[0]).float()
        pre_act = _as_tensor(bf_out[1]).float()
        h = _as_tensor(dlf_rt.run(h1, pre_act)).float()

    return _as_tensor(runtimes[f"merger_{tag}"].run(h)).float()
