# (1) pass-through overrides: method body (minus docstring) is only `[return] super().<same>(<same params forwarded>)`
# (2) private (_name) module-level functions / methods / classes never referenced by name anywhere else in src+tests+examples
import ast, sys, pathlib, re, collections
repo = pathlib.Path(sys.argv[1]); root = repo / "src/optimum/rbln"
files = sorted(root.rglob("*.py"))
corpus_files = files + sorted((repo / "tests").rglob("*.py")) + sorted((repo / "examples").rglob("*.py"))
text = {f: f.read_text() for f in corpus_files}
passthrough = []; documented_pt = 0; methods = 0
def forwarded(fn, call):
    a = fn.args
    params = [x.arg for x in a.posonlyargs + a.args][1:]  # drop self/cls
    kwonly = [x.arg for x in a.kwonlyargs]
    pos = [ast.unparse(x) for x in call.args]
    kws = {k.arg: ast.unparse(k.value) for k in call.keywords if k.arg}
    star = [ast.unparse(x) for x in call.args if isinstance(x, ast.Starred)]
    dstar = [ast.unparse(k.value) for k in call.keywords if k.arg is None]
    names = [p for p in pos if not p.startswith("*")]
    exp = params + kwonly
    got = names + [k for k, v in kws.items() if v == k]
    ok_va = (a.vararg is None and not star) or (a.vararg and star == ["*" + a.vararg.arg])
    ok_kw = (a.kwarg is None and not dstar) or (a.kwarg and dstar == [a.kwarg.arg])
    return sorted(got) == sorted(exp) and all(v == k for k, v in kws.items()) and ok_va and ok_kw
for f in files:
    t = ast.parse(text[f]); rel = str(f.relative_to(repo))
    for cls in [n for n in ast.walk(t) if isinstance(n, ast.ClassDef)]:
        for fn in cls.body:
            if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)): continue
            methods += 1
            body = fn.body
            has_doc = bool(body) and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) and isinstance(body[0].value.value, str)
            if has_doc: body = body[1:]
            if len(body) != 1 or fn.decorator_list and any(ast.unparse(d) not in ("classmethod", "staticmethod") for d in fn.decorator_list): continue
            st = body[0]
            call = st.value if isinstance(st, (ast.Return, ast.Expr)) else None
            if not (isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute) and call.func.attr == fn.name and isinstance(call.func.value, ast.Call) and isinstance(call.func.value.func, ast.Name) and call.func.value.func.id == "super"): continue
            if not forwarded(fn, call): continue
            if has_doc: documented_pt += 1; continue
            passthrough.append((rel, fn.lineno, f"{cls.name}.{fn.name}"))
private_unused = []; private_total = 0
for f in files:
    t = ast.parse(text[f]); rel = str(f.relative_to(repo))
    cands = [(n.name, n.lineno) for n in t.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name.startswith("_") and not n.name.startswith("__")]
    for cls in [n for n in t.body if isinstance(n, ast.ClassDef)]:
        cands += [(m.name, m.lineno) for m in cls.body if isinstance(m, ast.FunctionDef) and m.name.startswith("_") and not m.name.startswith("__")]
    for name, ln in cands:
        private_total += 1
        pat = re.compile(r"\b" + re.escape(name) + r"\b")
        refs = sum(len(pat.findall(s)) for s in text.values())
        if refs <= 1: private_unused.append((rel, ln, name))
print(f"methods {methods}; pass-through overrides without docstring {len(passthrough)}; with docstring (doc-only override) {documented_pt}")
for x in passthrough: print("PASSTHROUGH", *x)
print(f"private defs {private_total}; referenced nowhere else {len(private_unused)}")
for x in private_unused: print("UNUSED_PRIVATE", *x)
