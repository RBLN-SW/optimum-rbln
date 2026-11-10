import ast, sys, pathlib
root = pathlib.Path(sys.argv[1])
quoted_union = 0; legacy_fwd = 0; none_default = 0; mixed = 0; ex=[]
for f in sorted(root.rglob("*.py")):
    t = ast.parse(f.read_text())
    for n in ast.walk(t):
        if isinstance(n, ast.arg) and n.annotation is not None:
            a = n.annotation
            if isinstance(a, ast.Constant) and isinstance(a.value, str) and "| None" in a.value: quoted_union += 1
            if isinstance(a, ast.BinOp) and any(isinstance(x, ast.Constant) and isinstance(x.value, str) for x in ast.walk(a)): mixed += 1; ex.append((f, n.lineno, ast.unparse(a)))
            if isinstance(a, ast.Subscript) and isinstance(a.value, ast.Name) and a.value.id in ("Optional","Union"): legacy_fwd += 1
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
            a = n.args; pos = a.posonlyargs + a.args
            defs = [None]*(len(pos)-len(a.defaults)) + list(a.defaults)
            for arg, d in list(zip(pos, defs)) + list(zip(a.kwonlyargs, a.kw_defaults)):
                if arg.annotation is not None and isinstance(d, ast.Constant) and d.value is None: none_default += 1
print("param annotations: quoted 'X | None' strings:", quoted_union, " Optional/Union[...] params:", legacy_fwd, " BinOp with str operand:", mixed, " annotated None-default params:", none_default)
for e in ex[:5]: print(*e)
