"""API-015/PR17: each leaf RBLN<X> class lives in optimum/rbln/<lib>/models/<dir>/ where <dir> is the directory of HF's
<X> (transformers.models.<dir>.* / diffusers.models|pipelines.<...>). usage: PYTHONPATH=<src> python3 this.py <pkg_root>"""
import ast, sys, pathlib, json, warnings; warnings.filterwarnings("ignore")
import transformers, diffusers
root = pathlib.Path(sys.argv[1]); ok, bad = 0, []
for p in sorted(root.rglob("*.py")):
    rel = str(p).split("src/")[-1]; lines = p.read_text().splitlines()
    for n in ast.parse(p.read_text()).body:
        if not (isinstance(n, ast.ClassDef) and n.name.startswith("RBLN")) or n.name.startswith("RBLNAuto"): continue
        x = n.name[4:]
        if x.endswith("Config"): continue
        lib = transformers if hasattr(transformers, x) else diffusers if hasattr(diffusers, x) else None
        if lib is None: continue
        try: hfmod = getattr(lib, x).__module__
        except Exception: continue
        parts = hfmod.split(".")
        if lib is transformers:
            if "models" not in parts: continue
            want = parts[parts.index("models") + 1]
            have = rel.split("/transformers/models/")[1].split("/")[0] if "/transformers/models/" in rel else None
        else:
            want = "/".join(parts[1:-1])
            have = "/".join(rel.split("/diffusers/")[1].split("/")[:-1]) if "/diffusers/" in rel else None
        if want == have: ok += 1
        else: bad.append({"path": "src/" + rel, "line": n.lineno, "snippet": lines[n.lineno-1].strip()[:120], "note": f"HF {x} lives in {hfmod}"})
print(json.dumps({"ok": ok, "bad": bad}, indent=1))
