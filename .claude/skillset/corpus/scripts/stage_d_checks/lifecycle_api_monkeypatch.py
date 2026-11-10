"""Find global monkey-patch sites: attribute assignment / setattr whose root name is an imported module or imported
class (i.e. state shared beyond the current object). For each, report whether it sits inside a try whose finally
(or a context manager __exit__) restores it. usage: python3 lifecycle_api_monkeypatch.py <src_root>"""
import ast, sys, pathlib
root = pathlib.Path(sys.argv[1])

def root_name(n):
    while isinstance(n, (ast.Attribute, ast.Subscript)):
        n = n.value
    return n.id if isinstance(n, ast.Name) else None

def dotted(n):
    parts = []
    while isinstance(n, ast.Attribute):
        parts.append(n.attr); n = n.value
    if isinstance(n, ast.Name): parts.append(n.id)
    return ".".join(reversed(parts))

rows = []
for p in sorted(root.rglob("*.py")):
    src = p.read_text(); t = ast.parse(src); lines = src.splitlines()
    imported = set()
    for n in ast.walk(t):
        if isinstance(n, ast.Import):
            for a in n.names: imported.add((a.asname or a.name).split(".")[0])
        elif isinstance(n, ast.ImportFrom):
            for a in n.names: imported.add(a.asname or a.name)
    parents = {}
    for n in ast.walk(t):
        for c in ast.iter_child_nodes(n): parents[c] = n
    def in_finally(node):
        cur = node
        while cur in parents:
            par = parents[cur]
            if isinstance(par, ast.Try) and cur in par.finalbody: return "finally-restore"
            if isinstance(par, ast.Try) and cur in par.body and par.finalbody: return "try-body"
            cur = par
        return None
    for n in ast.walk(t):
        targets = []
        if isinstance(n, (ast.Assign, ast.AugAssign, ast.AnnAssign)):
            ts = n.targets if isinstance(n, ast.Assign) else [n.target]
            for tg in ts:
                for e in (tg.elts if isinstance(tg, ast.Tuple) else [tg]):
                    if isinstance(e, ast.Attribute) and root_name(e) in imported: targets.append(dotted(e))
        elif isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "setattr" and n.args:
            if root_name(n.args[0]) in imported: targets.append("setattr(" + dotted(n.args[0]) + ")")
        for d in targets:
            rows.append((str(p.relative_to(root.parent.parent)), n.lineno, d, in_finally(n), lines[n.lineno-1].strip()[:120]))
for r in rows: print(*r, sep=" | ")
print("TOTAL", len(rows))
