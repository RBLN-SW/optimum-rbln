"""Minimal repro: `rbln_cache_update` on a static (n, c, d, h*w) cache fails to compile for some shapes.

The CogVideoX VAE decoder keeps its temporal conv caches as static device buffers, flattened like the
Wan VAE to (n, c, d, h*w). At 480x720 the 60x90 latent-resolution caches become (1, 512, 2, 5400) and
the compiler rejects them:

    'rblnTensor.nn.contrib_custom_cache_update' op wrong shape for custom op for running on device
    (Partition.cpp verifyCustomOpsDevRunnable -> RBLNCompileError: [DEVICE_GRAPH_CONVERSION])

This script compiles one cache update per shape (each in its own process, since a failed compile can
abort the process) and reports which ones pass.

    python examples/cogvideo/repro_cache_update_shape.py                    # default shape list
    python examples/cogvideo/repro_cache_update_shape.py x 1,512,2,21632    # custom shapes

Result (rebel 0.12.1.dev99+g18b188f8, 2026-10-12): the flat h*w axis must be a multiple of 64.
    FAIL (1, 512, 2, 5400)    partition: "wrong shape for custom op for running on device"
    FAIL (1, 16, 2, 5400)     same
    FAIL (1, 512, 2, 21600)   SIGABRT: "a memory.store still writes a DRAM-view concatenate;
                              FoldMemoryTransfer must fold every such store into chunks"
    FAIL (1, 512, 2, 60, 90)  unflattened 5-D cache, same partition error
    PASS (1, 512, 2, 5408) (1, 512, 2, 5440) (1, 512, 2, 5632) (1, 512, 2, 21632)
    PASS (1, 256, 2, 86400) (1, 128, 2, 345600) (1, 16, 2, 5408)
Workaround in autoencoder_kl_cogvideox.py: zero-pad the flat axis to a multiple of 64 (_CACHE_HW_ALIGN).
"""

import sys

import rebel
import torch
from rebel.compile_context import CompileContext


class CacheWrite(torch.nn.Module):
    def forward(self, x, cache):
        y = x * 2.0  # some device compute that produces the cache item
        position = torch.tensor(0, dtype=torch.int16)
        axis = torch.tensor(2, dtype=torch.int16)
        return y.sum(), torch.ops.rbln_custom_ops.rbln_cache_update(cache, y, position, axis)


def try_shape(shape) -> bool:
    context = CompileContext(use_weight_sharing=True)
    cache = torch.zeros(shape)
    context.mark_static_address(cache)
    try:
        rebel.compile_from_torch(
            CacheWrite().eval(),
            input_info=[("x", list(shape), "float32"), ("cache", list(shape), "float32")],
            example_inputs=[torch.zeros(shape), cache],
            compile_context=context,
        )
        return True
    except Exception as e:  # noqa: BLE001
        print(f"   -> {type(e).__name__}: {str(e).splitlines()[0][:120]}")
        return False


SHAPES = [
    (1, 512, 2, 5400),  # 60x90 latent, as in the VAE (fails)
    (1, 512, 2, 21600),  # 120x180 (passes in the VAE)
    (1, 16, 2, 5400),  # conv_in channels at 60x90
    (1, 512, 2, 5408),  # 5400 padded to a multiple of 16
    (1, 512, 2, 5440),  # 5400 padded to a multiple of 32/64
    (1, 512, 2, 5632),  # 5400 padded to a multiple of 256/512
    (1, 512, 2, 60, 90),  # no flatten
]

if __name__ == "__main__":
    if len(sys.argv) == 2:
        # child: compile one shape; a failed compile can abort the process, so each shape runs alone
        sys.exit(0 if try_shape(tuple(int(v) for v in sys.argv[1].split(","))) else 1)

    import subprocess

    shapes = [tuple(int(v) for v in s.split(",")) for s in sys.argv[2:]] or SHAPES
    results = {}
    for shape in shapes:
        arg = ",".join(str(v) for v in shape)
        proc = subprocess.run([sys.executable, __file__, arg], capture_output=True, text=True)
        results[shape] = proc.returncode == 0
        print(f"{'PASS' if results[shape] else 'FAIL'}  {shape}  (exit {proc.returncode})", flush=True)
