"""Leaf RBLN model classes (RBLN<X> where <X> is a transformers/diffusers class) and their RBLN<X>Config must be
reachable as optimum.rbln.<name>; for transformers leaves whose HF name sits in an HF auto *_MAPPING_NAMES, the
RBLNAuto path (convert_hf_to_rbln_model_name -> getattr(optimum.rbln)) must resolve. usage: PYTHONPATH=<src> python3 this.py <pkg_root>"""
import ast, sys, pathlib, json, warnings, importlib
warnings.filterwarnings("ignore")
import transformers, diffusers
import optimum.rbln as o
root = pathlib.Path(sys.argv[1])
leaves = {}
for p in sorted(root.rglob("*.py")):
    for n in ast.parse(p.read_text()).body:
        if isinstance(n, ast.ClassDef) and n.name.startswith("RBLN") and not n.name.endswith("Config"):
            hf = n.name[4:]
            if hasattr(transformers, hf) or hasattr(diffusers, hf):
                leaves[n.name] = (str(p).split("src/")[-1], n.lineno)
ok, bad = 0, []
for name, (path, line) in sorted(leaves.items()):
    miss = [x for x in (name, name + "Config") if not hasattr(o, x)]
    if miss: bad.append({"path": "src/" + path, "line": line, "snippet": f"class {name}", "note": "not exported: " + ",".join(miss)})
    else: ok += 1
print(json.dumps({"leaves": len(leaves), "ok": ok, "bad": bad}, indent=1))
