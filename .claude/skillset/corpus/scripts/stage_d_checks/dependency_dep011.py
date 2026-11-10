"""DEP-011: every torch.load call (incl. partial(torch.load, ...)) passes weights_only explicitly.
Usage: python3 dependency_dep011.py [ROOT]  (scans src/, tests/, examples/, scripts/ if present)"""
import ast, pathlib, sys
root = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else ".")
ok, bad = [], []
def is_torch_load(f):
    return isinstance(f, ast.Attribute) and f.attr == "load" and isinstance(f.value, ast.Name) and f.value.id == "torch"
for d in ("src", "tests", "examples", "scripts"):
    for p in sorted((root / d).rglob("*.py")):
        src = p.read_text(); lines = src.splitlines()
        for n in ast.walk(ast.parse(src)):
            if not isinstance(n, ast.Call): continue
            target = None
            if is_torch_load(n.func): target = n
            elif getattr(n.func, "id", getattr(n.func, "attr", None)) == "partial" and n.args and is_torch_load(n.args[0]): target = n
            if target is None: continue
            rec = (str(p.relative_to(root)), n.lineno, lines[n.lineno - 1].strip()[:120])
            (ok if any(k.arg == "weights_only" for k in n.keywords) else bad).append(rec)
print("compliant", len(ok), "violations", len(bad))
for r in ok: print("OK ", *r)
for r in bad: print("BAD", *r)
