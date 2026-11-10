"""CFG-019 summary over config_cfg019_submodules.py output: classes, mapping, helper calls, annotation split by area."""
import json, sys, collections
rows = [x for x in json.load(sys.stdin) if x["submodules"]]
print("classes", len(rows), "has_map", sum(x["has_map"] for x in rows),
      "init_calls", sum(x["n_init_calls"] > 0 for x in rows))
c = collections.Counter()
for x in rows:
    area = "diffusers" if "/diffusers/" in x["path"] else "transformers"
    for v in x["ann"].values():
        g = v is not None and v.replace(" ", "").replace("'", "") in ("RBLNModelConfig|None", "Optional[RBLNModelConfig]")
        c[(area, "generic" if g else "concrete")] += 1
print(dict(c))
