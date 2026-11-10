import ast, sys, pathlib, collections
root = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "src/optimum/rbln")
legacy = {"Optional", "List", "Dict", "Union", "Tuple", "Set", "Type", "FrozenSet"}
cnt = collections.Counter(); sites = []
future = []; implicit = []; native_none = 0; native_generic = 0; ann_params = 0
def is_none_union(a):
    # annotation already allows None?
    s = ast.unparse(a)
    return "None" in s or "Optional" in s or s in ("Any", "typing.Any", "object") or "Any" == s
for f in sorted(root.rglob("*.py")):
    src = f.read_text(); t = ast.parse(src)
    for n in ast.walk(t):
        if isinstance(n, ast.ImportFrom) and n.module == "__future__" and any(a.name == "annotations" for a in n.names):
            future.append(str(f))
        if isinstance(n, ast.Subscript):
            v = n.value
            name = v.id if isinstance(v, ast.Name) else (v.attr if isinstance(v, ast.Attribute) and isinstance(v.value, ast.Name) and v.value.id == "typing" else None)
            if name in legacy:
                cnt[name] += 1; sites.append((str(f), n.lineno, name))
            elif isinstance(v, ast.Name) and v.id in ("list", "dict", "tuple", "set", "type"):
                native_generic += 1
        if isinstance(n, ast.BinOp) and isinstance(n.op, ast.BitOr) and (isinstance(n.right, ast.Constant) and n.right.value is None or isinstance(n.left, ast.Constant) and n.left.value is None):
            native_none += 1
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
            a = n.args
            pos = a.posonlyargs + a.args
            defs = [None] * (len(pos) - len(a.defaults)) + list(a.defaults)
            pairs = list(zip(pos, defs)) + list(zip(a.kwonlyargs, a.kw_defaults))
            for arg, d in pairs:
                if arg.annotation is not None:
                    ann_params += 1
                    if isinstance(d, ast.Constant) and d.value is None and not is_none_union(arg.annotation):
                        implicit.append((str(f), arg.lineno, f"{arg.arg}: {ast.unparse(arg.annotation)} = None"))
print("legacy typing subscripts:", dict(cnt), "total", sum(cnt.values()))
print("native X|None:", native_none, "native generics:", native_generic)
print("from __future__ import annotations files:", len(future))
print("annotated params:", ann_params, "implicit Optional:", len(implicit))
for s in sites[:30]: print("LEGACY", *s)
for s in implicit[:40]: print("IMPLICIT", *s)
for s in future: print("FUTURE", s)
