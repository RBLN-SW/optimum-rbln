"""API-003: find references to RBLN API that no longer exists.
 1) in src: `<ImportedRBLNClass>.<attr>` attribute accesses where the class (resolved by import) has no such attr
 2) in README.md / examples/ / docs (*.md, *.py): every `RBLN\w+` identifier and `<RBLNClass>.<attr>` must resolve
usage: PYTHONPATH=<src> python3 this.py <repo_root>"""
import ast, sys, pathlib, re, json, warnings, importlib, inspect; warnings.filterwarnings("ignore")
import optimum.rbln as o
repo = pathlib.Path(sys.argv[1]); pkg = repo / "src/optimum/rbln"
def resolve(name):
    try: return getattr(o, name)
    except Exception: return None
sites, bad = 0, []
for p in sorted(pkg.rglob("*.py")):
    t = ast.parse(p.read_text()); lines = p.read_text().splitlines()
    imported = {}
    for n in ast.walk(t):
        if isinstance(n, ast.ImportFrom):
            for a in n.names:
                if a.name.startswith("RBLN"): imported[a.asname or a.name] = a.name
    for n in ast.walk(t):
        if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name) and n.value.id in imported:
            c = resolve(imported[n.value.id])
            if c is None or not inspect.isclass(c): continue
            sites += 1
            if not hasattr(c, n.attr):
                bad.append({"path": str(p.relative_to(repo)), "line": n.lineno, "snippet": lines[n.lineno-1].strip()[:120], "note": f"{imported[n.value.id]} has no attribute {n.attr}"})
docs = [repo / "README.md"] + sorted((repo / "examples").rglob("*")) + sorted((repo / "docs").rglob("*") if (repo / "docs").exists() else [])
dsites = 0
for p in docs:
    if not p.is_file() or p.suffix not in (".md", ".py", ".ipynb"): continue
    text = p.read_text(errors="ignore"); lines = text.splitlines()
    local = set(re.findall(r"^class (RBLN\w+)", text, re.M))  # classes the example defines itself
    for i, l in enumerate(lines, 1):
        for m in re.finditer(r"\b(RBLN[A-Za-z0-9_]+)(?:\.([A-Za-z_]\w*))?", l):
            name, attr = m.group(1), m.group(2)
            dsites += 1
            c = resolve(name)
            if name in local: continue
            if c is None: bad.append({"path": str(p.relative_to(repo)), "line": i, "snippet": l.strip()[:120], "note": f"{name} not exported by optimum.rbln"})
            elif attr and inspect.isclass(c) and not hasattr(c, attr): bad.append({"path": str(p.relative_to(repo)), "line": i, "snippet": l.strip()[:120], "note": f"{name}.{attr} does not exist"})
print(json.dumps({"src_sites": sites, "doc_sites": dsites, "bad": bad}, indent=1))
