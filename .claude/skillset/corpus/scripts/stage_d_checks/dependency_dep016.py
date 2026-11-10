"""DEP-016 (proxy): matmul sites (torch.matmul/bmm/baddbmm/einsum, `@`) in src model code; a site counts as 'explicitly aligned'
if any operand expression contains .to(...)/.type(...)/.float()/.half()/.bfloat16() or the result of such a call bound on one of the 3 preceding lines.
Static dtype inference is impossible, so this is a proxy only. Usage: python3 dependency_dep016.py [ROOT]"""
import ast, pathlib, re, sys
root = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else ".")
CAST = re.compile(r"\.(to|type|float|half|bfloat16|type_as)\(")
FN = {"matmul", "bmm", "baddbmm", "einsum", "mm"}
cast_ok, plain = [], []
for p in sorted((root / "src/optimum/rbln").rglob("*.py")):
    src = p.read_text(); lines = src.splitlines()
    for n in ast.walk(ast.parse(src)):
        ops = None
        if isinstance(n, ast.BinOp) and isinstance(n.op, ast.MatMult): ops = [n.left, n.right]
        elif isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr in FN and getattr(n.func.value, "id", "") == "torch": ops = n.args
        if ops is None: continue
        seg = " ".join(ast.unparse(o) for o in ops)
        names = {x.id for o in ops for x in ast.walk(o) if isinstance(x, ast.Name)}
        prev = "\n".join(lines[max(0, n.lineno - 4): n.lineno - 1])
        bound = any(re.search(rf"\b{re.escape(v)}\s*=.*{CAST.pattern}", prev) for v in names)
        rec = (str(p.relative_to(root)), n.lineno, lines[n.lineno - 1].strip()[:120])
        (cast_ok if CAST.search(seg) or bound else plain).append(rec)
print("matmul sites", len(cast_ok) + len(plain), "explicit-cast", len(cast_ok), "no-visible-cast", len(plain))
for r in cast_ok: print("CAST", *r)
for r in plain: print("PLAIN", *r)
