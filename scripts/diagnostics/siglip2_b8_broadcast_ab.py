"""Compare frozen compiler-only broadcast repair on a scheduler-owned device."""

import hashlib
import json
from pathlib import Path

import rebel
import torch

from .siglip2_b8_logits import metrics


def main():
    source = Path(__file__).parent / "fixtures" / "siglip2-b8-broadcast-ab"
    out = Path("debug-siglip2-b8-ab")
    out.mkdir(exist_ok=True)
    manifest = json.loads((source / "manifest.json").read_text())
    for name, info in manifest.items():
        assert hashlib.sha256((source / name).read_bytes()).hexdigest() == info["sha256"]
    fixture = torch.load(source / "fixture.pt", map_location="cpu", weights_only=True)
    text, image = fixture["text"], fixture["image"]
    expected = (text @ image.T) * fixture["logit_scale"].exp() + fixture["logit_bias"]
    results = []
    first = None
    for label in ("before", "after", "before"):
        compiled = rebel.RBLNCompiledModel(source / f"{label}.rbln")
        runtime = compiled.create_runtime(tensor_type="pt", device=0)
        actual = runtime(text, image).clone()
        repeated = runtime(text, image).clone()
        row = dict(metrics(actual, expected), label=label, repeat_equal=torch.equal(actual, repeated))
        torch.save({"actual": actual, "expected": expected, "repeated": repeated}, out / f"{label}-{len(results)}.pt")
        results.append(row)
        (out / "results.json").write_text(json.dumps(results, indent=2))
        print(json.dumps(row), flush=True)
        assert row["repeat_equal"]
        assert row["finite"]
        if label == "before":
            assert row["nrmse"] > 0.9
            assert torch.count_nonzero(actual[:, 1:]) == 0
            if first is None:
                first = actual
            else:
                assert torch.equal(first, actual)
        else:
            assert row["nrmse"] <= 0.05
            assert row["pcc"] >= 0.99
        del runtime, compiled


if __name__ == "__main__":
    main()
