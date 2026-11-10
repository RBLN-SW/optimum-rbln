"""List signatures of contract methods across src tree. usage: python3 lifecycle_api_sigs.py <src_root> [method ...]"""
import ast, sys, pathlib, collections
root = pathlib.Path(sys.argv[1])
methods = sys.argv[2:] or ["get_compiled_model", "get_input_info", "_update_rbln_config", "_wrap_model_if_needed", "_update_submodule_config", "_update_submodule_rbln_config"]
out = collections.defaultdict(list)
for p in sorted(root.rglob("*.py")):
    t = ast.parse(p.read_text())
    for c in ast.walk(t):
        if isinstance(c, ast.ClassDef):
            for f in c.body:
                if isinstance(f, (ast.FunctionDef, ast.AsyncFunctionDef)) and f.name in methods:
                    a = f.args
                    names = [x.arg for x in a.posonlyargs + a.args]
                    kwo = [x.arg for x in a.kwonlyargs]
                    sig = ",".join(names) + ("|*" + a.vararg.arg if a.vararg else "") + ("|kwonly:" + ",".join(kwo) if kwo else "") + ("|**" + a.kwarg.arg if a.kwarg else "")
                    out[f.name].append((str(p.relative_to(root.parent.parent.parent) if 'src' in str(root) else p), f.lineno, c.name, sig))
for m, rows in out.items():
    cnt = collections.Counter(r[3] for r in rows)
    print(f"== {m}: {len(rows)} defs")
    for s, n in cnt.most_common():
        print(f"  {n:3d}  {s}")
    if "-v" in sys.argv[0:0]: pass
import json
json.dump(out, open("/dev/stderr","w") if False else open(sys.argv[0]+".json","w"))
