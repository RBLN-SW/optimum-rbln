"""API-015: find compatibility shims. usage: python3 lifecycle_api_shims.py <pkg_root>
 SHIM_MODULE  - non-__init__ module whose top level is only imports/__all__/docstring/warnings (no class/def/logic)
 CLASS_ALIAS  - module-level `RBLNX = RBLNY` / `X = Y` where Y is a class name (CamelCase) -> alias instead of subclass
 SUBCLASS_ALIAS - `class RBLNX(RBLNY): pass` with an empty body (docstring/pass only) -> naming alias
 DEPRECATED_PATH - module mentioning 'deprecated' + 'moved'/'shim'/'backward compat' in comments or strings"""
import ast, sys, pathlib, re, json, io, tokenize
root = pathlib.Path(sys.argv[1]); rows = {"SHIM_MODULE": [], "CLASS_ALIAS": [], "SUBCLASS_ALIAS": [], "DEPRECATED_PATH": []}
nmods = 0
for p in sorted(root.rglob("*.py")):
    src = p.read_text(); t = ast.parse(src); r = "src/" + str(p).split("src/")[-1]; lines = src.splitlines()
    if p.name != "__init__.py":
        nmods += 1
        body = [n for n in t.body if not (isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant))]
        if body and all(isinstance(n, (ast.Import, ast.ImportFrom)) or (isinstance(n, ast.Assign) and any(getattr(x, "id", "") == "__all__" for x in n.targets)) for n in body):
            rows["SHIM_MODULE"].append({"path": r, "line": 1, "snippet": lines[body[0].lineno - 1][:120], "note": "module only re-imports names"})
    for n in t.body:
        if isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name) and isinstance(n.value, (ast.Name, ast.Attribute)):
            tgt = n.targets[0].id; val = n.value.id if isinstance(n.value, ast.Name) else n.value.attr
            if re.match(r"^_?[A-Z][a-z]+[A-Za-z0-9_]*$", tgt) and re.match(r"^[A-Z][a-z]+[A-Za-z0-9_]*$", val) and not tgt.isupper():
                rows["CLASS_ALIAS"].append({"path": r, "line": n.lineno, "snippet": lines[n.lineno - 1].strip()[:120], "note": "module-level class alias"})
        if isinstance(n, ast.ClassDef) and n.name.startswith("RBLN"):
            b = [x for x in n.body if not (isinstance(x, ast.Expr) and isinstance(x.value, ast.Constant)) and not isinstance(x, ast.Pass)]
            if not b:
                rows["SUBCLASS_ALIAS"].append({"path": r, "line": n.lineno, "snippet": lines[n.lineno - 1].strip()[:120], "note": "empty-body subclass (name alias)"})
    for tok in tokenize.generate_tokens(io.StringIO(src).readline):
        if tok.type in (tokenize.COMMENT, tokenize.STRING) and re.search(r"backward[- ]compat|\bshim\b|kept for compat|moved to|deprecated alias", tok.string, re.I):
            rows["DEPRECATED_PATH"].append({"path": r, "line": tok.start[0], "snippet": lines[tok.start[0] - 1].strip()[:120], "note": "compat/shim wording"})
print(json.dumps({"modules": nmods, **{k: v for k, v in rows.items()}}, indent=1))
