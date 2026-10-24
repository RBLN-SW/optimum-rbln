"""Load one full-batch Chronos candidate stage at a time in scheduler diagnostics."""

import gc
import time
from contextlib import contextmanager
from pathlib import Path


@contextmanager
def load_stage(path, device):
    """Own and release one independent compiled model and synchronous runtime."""
    import rebel

    compiled = rebel.RBLNCompiledModel(path)
    runtime = None
    try:
        runtime = compiled.create_runtime(tensor_type="pt", device=device)
        yield runtime
    finally:
        if runtime is not None:
            del runtime
        del compiled
        gc.collect()


def clone_outputs(output):
    if isinstance(output, (tuple, list)):
        return tuple(value.detach().cpu().clone() for value in output)
    return output.detach().cpu().clone()


class SequentialRuntimeChain:
    """Preserve full hidden/mask tensors while keeping only one device job loaded.

    Every call loads each stage afresh. The loader is a context manager so its
    resources are destroyed before the next stage opens. Outputs are cloned on
    CPU before unloading; the final output/residual tuple is left separate for
    the product wrapper's original FP32 summation and inverse normalization.
    """

    def __init__(self, root, stages, device, event, loader=load_stage):
        self.root = Path(root)
        self.stages = stages
        self.device = device
        self.event = event
        self.loader = loader
        self.calls = 0

    def _run(self, stage, inputs):
        name = stage["name"]
        context = self.loader(self.root / stage["file"], self.device)
        started = time.perf_counter()
        with context as runtime:
            loaded = time.perf_counter()
            self.event(
                {
                    "call": self.calls,
                    "stage": name,
                    "phase": "cold_load",
                    "host_wall_seconds": loaded - started,
                    "alloc_per_node": stage["alloc_per_node"],
                }
            )
            # No runtime-owned tensor escapes into the next stage.
            output = runtime(*inputs)
            frozen = clone_outputs(output)
            del output
            finished = time.perf_counter()
            self.event(
                {
                    "call": self.calls,
                    "stage": name,
                    "phase": "stage_inference_and_host_copy",
                    "host_wall_seconds": finished - loaded,
                }
            )
            # Drop this local reference before the loader's finally block.
            del runtime
        self.event(
            {
                "call": self.calls,
                "stage": name,
                "phase": "unload",
                "host_wall_seconds": time.perf_counter() - finished,
            }
        )
        return frozen

    def __call__(self, *inputs):
        self.calls += 1
        started = time.perf_counter()
        hidden, time_mask, group_mask = self._run(self.stages[0], inputs)
        for stage in self.stages[1:-1]:
            hidden = self._run(stage, (hidden, time_mask, group_mask))
            if isinstance(hidden, (list, tuple)):
                hidden = hidden[0]
        result = self._run(self.stages[-1], (hidden,))
        self.event({"call": self.calls, "phase": "request", "host_wall_seconds": time.perf_counter() - started})
        return result
