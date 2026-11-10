"""DEP-008: every third-party package imported by src/ is declared in pyproject.toml (default deps or an optional extra);
packages outside the default deps must be imported lazily (inside a function / try / availability guard), not at module top level.
Usage: python3 dependency_dep008.py [ROOT]"""
import ast, pathlib, re, sys, tomllib
root = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else ".")
pp = tomllib.loads((root / "pyproject.toml").read_text())
norm = lambda s: re.split(r"[<>=!~;\[ ]", s.strip())[0].lower().replace("_", "-")
default = {norm(d) for d in pp["project"].get("dependencies", [])}
optional = {norm(d) for g in pp["project"].get("optional-dependencies", {}).values() for d in g}
groups = {norm(d) for g in pp.get("dependency-groups", {}).values() for d in g if isinstance(d, str)}
# import name -> distribution name
ALIAS = {"PIL": "pillow", "cv2": "opencv-python", "skimage": "scikit-image", "sklearn": "scikit-learn", "yaml": "pyyaml", "huggingface_hub": "huggingface-hub", "rebel": "rebel-compiler", "google": "protobuf"}
# provided transitively by a declared default dep (torch/transformers/diffusers/accelerate) -- not counted
TRANSITIVE = {"numpy", "huggingface-hub", "safetensors", "filelock", "regex", "requests", "tqdm", "pyyaml", "pillow", "typing-extensions", "jinja2", "sympy", "networkx", "fsspec", "psutil", "importlib-metadata"}
stdlib = set(sys.stdlib_module_names) | {"optimum", "__future__"}
uses = {}
for p in sorted((root / "src/optimum/rbln").rglob("*.py")):
    tree = ast.parse(p.read_text())
    top_nodes = set(id(n) for n in tree.body)
    # module-level imports also include those under top-level if/try? treat try/if guarded as lazy
    for n in ast.walk(tree):
        if isinstance(n, (ast.Import, ast.ImportFrom)):
            if isinstance(n, ast.ImportFrom) and (n.level or not n.module): continue
            mods = [a.name for a in n.names] if isinstance(n, ast.Import) else [n.module]
            for m in mods:
                top = m.split(".")[0]
                if top in stdlib: continue
                dist = ALIAS.get(top, top).lower().replace("_", "-")
                uses.setdefault(dist, []).append((str(p.relative_to(root)), n.lineno, id(n) in top_nodes))
ok, bad = [], []
for dist, sites in sorted(uses.items()):
    eager = [s for s in sites if s[2]]
    if dist in default or dist in TRANSITIVE:
        ok.append(dist); continue
    if dist == "rebel-compiler":  # installed from private index, deliberately not in pyproject
        ok.append(dist); continue
    declared = dist in optional
    for s in (sites if not declared else eager):
        bad.append((s[0], s[1], dist, ("undeclared in pyproject" if not declared else "optional dep imported at module top level") + (" (top-level import)" if s[2] else " (lazy import)")))
    if not bad or bad[-1][2] != dist: ok.append(dist)
print("default", sorted(default)); print("optional", sorted(optional))
print("compliant packages", len(ok), ok)
print("violation sites", len(bad), "packages", sorted({b[2] for b in bad}))
for b in bad: print("BAD", *b)
