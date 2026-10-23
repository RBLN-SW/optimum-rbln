import gc
import hashlib
import importlib.metadata
import json
from pathlib import Path

import rebel
import torch
import torch.nn.functional as F


class Consumer(torch.nn.Module):
    def __init__(self, kind):
        super().__init__()
        self.kind = kind

    def forward(self, query):
        padded = F.pad(query, (0, 56))
        if self.kind == "softmax":
            return padded.softmax(-1)
        if self.kind == "norm":
            return F.layer_norm(padded, (64,))
        if self.kind == "sum":
            return padded.sum(-1)
        return padded


def main():
    root = Path("debug-chronos2-core-probe/pad-consumer")
    root.mkdir(parents=True, exist_ok=True)
    assert importlib.metadata.version("rebel-compiler") == "0.12.1.dev319+g8185335c"
    torch.set_num_threads(4)
    reports = []
    for sequence in (96, 576):
        query = torch.eye(8).expand(sequence, 12, 8, 8).contiguous()
        for kind in ("pad", "sum", "softmax", "norm"):
            name = f"{kind}-{sequence}"
            model = Consumer(kind).eval()
            reference = model(query)
            compiled = rebel.compile_from_torch(
                model, input_info=[("query", list(query.shape), "float32")], npu="RBLN-CA25", num_devices=1
            )
            compiled.save(str(root / f"{name}.rbln"))
            runtime = compiled.create_runtime(tensor_type="pt", device=0)
            first = None
            rows = []
            captured = False
            with torch.no_grad():
                for iteration in range(100):
                    output = runtime(query).clone()
                    if first is None:
                        first = output.clone()
                    row = {
                        "iteration": iteration,
                        "sha256": hashlib.sha256(output.numpy().tobytes()).hexdigest(),
                        "delta_first": (output - first).abs().max().item(),
                        "max_error": (output - reference).abs().max().item(),
                        "nrmse": ((output - reference).square().mean() / reference.square().mean()).sqrt().item(),
                    }
                    rows.append(row)
                    if not captured and (row["delta_first"] or row["nrmse"] > 0.01):
                        torch.save({"query": query, "output": output, "reference": reference}, root / f"{name}-bad.pt")
                        captured = True
                    if iteration % 10 == 0:
                        print("PAD_CONSUMER", name, json.dumps(row), flush=True)
            assert torch.equal(query, torch.eye(8).expand(sequence, 12, 8, 8))
            reports.append({"name": name, "rows": rows})
            (root / "results.json").write_text(json.dumps(reports, indent=2))
            del runtime, compiled, output, first, reference
            gc.collect()


if __name__ == "__main__":
    main()
