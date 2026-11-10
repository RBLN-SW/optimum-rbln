import ast, sys, pathlib, collections
root = pathlib.Path(sys.argv[1])
c = collections.Counter(); files=set(); ex=[]
for f in sorted(root.rglob("*.py")):
    t = ast.parse(f.read_text())
    for n in ast.walk(t):
        if isinstance(n, ast.Subscript) and isinstance(n.value, ast.Name) and n.value.id in ("Optional","Union","List","Dict","Tuple"):
            has_str = any(isinstance(x, ast.Constant) and isinstance(x.value, str) for x in ast.walk(n.slice))
            c[(n.value.id, has_str)] += 1; files.add(str(f))
            if not has_str: ex.append((str(f), n.lineno, ast.unparse(n)[:100]))
print(dict(c), "files", len(files))
for e in ex[:20]: print("NOSTR", *e)
