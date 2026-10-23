import gc
import hashlib
import importlib.metadata
import json
from pathlib import Path

import rebel
import torch
import torch.nn.functional as F


class Product(torch.nn.Module):
    def __init__(self, pad_query=False, pad_value=False):
        super().__init__()
        self.pad_query = pad_query
        self.pad_value = pad_value

    def forward(self, query, value):
        if self.pad_query:
            query = F.pad(query, (0, 56))
        if self.pad_value:
            value = F.pad(value, (0, 0, 0, 56))
        return query @ value


def main():
    root = Path("debug-chronos2-core-probe/pv-input-isolation")
    root.mkdir(parents=True, exist_ok=True)
    assert importlib.metadata.version("rebel-compiler") == "0.12.1.dev319+g8185335c"
    torch.set_num_threads(4)
    reports = []
    for name in ("float32", "float16", "padded64", "query_padded64", "value_padded64"):
        generator = torch.Generator().manual_seed(42)
        value = torch.randn(576, 12, 8, 64, generator=generator)
        query = torch.eye(8).expand(576, 12, 8, 8).contiguous()
        if name == "float16":
            query, value = query.half(), value.half()
        elif name == "padded64":
            query, value = F.pad(query, (0, 56)), F.pad(value, (0, 0, 0, 56))
        elif name == "query_padded64":
            query = F.pad(query, (0, 56))
        elif name == "value_padded64":
            value = F.pad(value, (0, 0, 0, 56))
        model = Product(pad_query=name == "value_padded64", pad_value=name == "query_padded64").eval()
        reference = model(query.float(), value.float())
        dtype = "float16" if name == "float16" else "float32"
        inputs = [query, value]
        hashes = [hashlib.sha256(x.numpy().tobytes()).hexdigest() for x in inputs]
        with torch.no_grad():
            compiled = rebel.compile_from_torch(
                model,
                input_info=[("query", list(query.shape), dtype), ("value", list(value.shape), dtype)],
                npu="RBLN-CA25",
                num_devices=1,
            )
            compiled.save(str(root / f"{name}.rbln"))
            runtime = compiled.create_runtime(tensor_type="pt", device=0)
            rows = []
            first = None
            captured = False
            for iteration in range(100):
                output = runtime(*inputs).clone()
                if first is None:
                    first = output.clone()
                row = {
                    "iteration": iteration,
                    "sha256": hashlib.sha256(output.numpy().tobytes()).hexdigest(),
                    "delta_first": (output.float() - first.float()).abs().max().item(),
                    "nrmse": ((output.float() - reference).square().mean() / reference.square().mean()).sqrt().item(),
                }
                rows.append(row)
                if row["nrmse"] > 0.01 and not captured:
                    torch.save({"output": output, "first": first, "reference": reference}, root / f"{name}-bad.pt")
                    captured = True
                if iteration % 10 == 0:
                    print("PV_INPUT_ISOLATION", name, json.dumps(row), flush=True)
            assert hashes == [hashlib.sha256(x.numpy().tobytes()).hexdigest() for x in inputs]
            reports.append({"name": name, "input_hashes": hashes, "rows": rows})
            (root / "results.json").write_text(json.dumps(reports, indent=2))
        del runtime, compiled, inputs, query, value, reference, output, first
        gc.collect()


if __name__ == "__main__":
    main()
