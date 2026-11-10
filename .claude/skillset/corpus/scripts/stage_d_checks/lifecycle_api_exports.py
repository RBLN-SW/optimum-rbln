"""API-014 checks. usage: python3 lifecycle_api_exports.py <pkg_root e.g. src/optimum/rbln>
 A) every top-level public `class RBLN*` in a non-__init__ module is listed in that module's literal __all__
 B) every package __init__.py (except the root) matches the fixed lazy template modulo the `from .x import *` lines
 C) (optional, needs import) every RBLN* model/config class defined in the tree is reachable as optimum.rbln.<name>"""
import ast, sys, pathlib, re, json
root = pathlib.Path(sys.argv[1]); rel = lambda p: str(p).split("src/", 1)[-1] if "src/" in str(p) else str(p)
A_ok, A_bad, defined = 0, [], {}
for p in sorted(root.rglob("*.py")):
    if p.name == "__init__.py": continue
    t = ast.parse(p.read_text())
    allv = None
    for n in t.body:
        if isinstance(n, ast.Assign) and any(isinstance(x, ast.Name) and x.id == "__all__" for x in n.targets):
            allv = set(ast.literal_eval(n.value))
    for n in t.body:
        if isinstance(n, ast.ClassDef) and n.name.startswith("RBLN"):
            defined[n.name] = (rel(p), n.lineno)
            if allv is not None and n.name in allv: A_ok += 1
            else: A_bad.append({"path": "src/" + rel(p), "line": n.lineno, "snippet": f"class {n.name}", "note": "no __all__ in module" if allv is None else "class not listed in module __all__"})
B_ok, B_bad = 0, []
def norm(src):
    src = re.sub(r"\A(#.*\n|\n)*", "", src)                      # license header
    src = re.sub(r"^\s+from \.[\w.]+ import \*\n", "", src, flags=re.M)  # TYPE_CHECKING star imports
    src = re.sub(r"from (\.+)utils\.import_utils", "from .utils.import_utils", src)
    return src.strip()
inits = [p for p in sorted(root.rglob("__init__.py")) if p.parent != root]
canon = {}
for p in inits: canon.setdefault(norm(p.read_text()), []).append(p)
template = max(canon, key=lambda k: len(canon[k]))
for k, ps in canon.items():
    for p in ps:
        if k == template: B_ok += 1
        else:
            first = next((l for l in (k.splitlines()) if l not in template.splitlines()), k.splitlines()[0] if k else "")
            B_bad.append({"path": "src/" + rel(p), "line": None, "snippet": first[:120], "note": "package __init__ deviates from fixed lazy template"})
res = {"A_ok": A_ok, "A_bad": A_bad, "B_ok": B_ok, "B_bad": B_bad, "n_template_variants": len(canon)}
if "--import" in sys.argv:
    import warnings; warnings.filterwarnings("ignore"); import optimum.rbln as o
    C_ok, C_bad = 0, []
    for name, (path, line) in sorted(defined.items()):
        try: getattr(o, name); C_ok += 1
        except Exception as e: C_bad.append({"path": "src/" + path, "line": line, "snippet": f"class {name}", "note": "not reachable as optimum.rbln." + name})
    res.update(C_ok=C_ok, C_bad=C_bad)
print(json.dumps(res, indent=1))
