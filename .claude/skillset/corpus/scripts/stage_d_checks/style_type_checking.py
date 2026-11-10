import ast, sys, pathlib, collections
root = pathlib.Path(sys.argv[1]); base = pathlib.Path(sys.argv[2])
def is_tc(test):
    return (isinstance(test, ast.Name) and test.id == "TYPE_CHECKING") or (isinstance(test, ast.Attribute) and test.attr == "TYPE_CHECKING")
stats = collections.Counter(); viol = []; origin = collections.Counter()
for f in sorted(p for p in root.rglob("*.py") if p.name != "__init__.py"):
    src = f.read_text(); t = ast.parse(src); rel = str(f.relative_to(base))
    has_future = any(isinstance(n, ast.ImportFrom) and n.module == "__future__" for n in t.body)
    tc_blocks = [n for n in ast.walk(t) if isinstance(n, ast.If) and is_tc(n.test)]
    if not tc_blocks: continue
    stats["modules"] += 1; stats["blocks"] += len(tc_blocks)
    tc_nodes = set(); tc_names = {}
    for b in tc_blocks:
        for n in (x for st in b.body for x in ast.walk(st)):
            tc_nodes.add(id(n))
            if isinstance(n, (ast.Import, ast.ImportFrom)):
                for a in n.names:
                    nm = (a.asname or a.name).split(".")[0]
                    tc_names[nm] = n.lineno
                    mod = getattr(n, "module", None) or a.name
                    origin["first_party" if (getattr(n, "level", 0) > 0 or mod.startswith("optimum.rbln")) else "third_party"] += 1
    stats["tc_names"] += len(tc_names)
    # runtime imports at module level (outside TC blocks), excluding else: branch of TC if
    rt = {}; rt_local = {}
    toplevel = {id(x) for x in t.body}
    for n in ast.walk(t):
        if isinstance(n, (ast.Import, ast.ImportFrom)) and id(n) not in tc_nodes:
            for a in n.names:
                (rt if id(n) in toplevel else rt_local).setdefault((a.asname or a.name).split(".")[0], n.lineno)
    for nm, ln in tc_names.items():
        if nm in rt:
            viol.append((rel, ln, nm, "duplicate: also imported at module level line %d" % rt[nm]))
        elif nm in rt_local:
            stats["local_runtime_reimport"] += 1
    # annotation nodes
    ann_ids = set(); ann_roots = []
    for n in ast.walk(t):
        anns = []
        if isinstance(n, ast.arg) and n.annotation is not None: anns.append(n.annotation)
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.returns is not None: anns.append(n.returns)
        if isinstance(n, ast.AnnAssign): anns.append(n.annotation)
        for a in anns:
            ann_roots.append(a)
            for m in ast.walk(a): ann_ids.add(id(m))
    used_str = collections.Counter(); used_unquoted_ann = collections.Counter(); used_runtime = collections.defaultdict(list)
    for a in ann_roots:
        for m in ast.walk(a):
            if isinstance(m, ast.Constant) and isinstance(m.value, str):
                try:
                    for x in ast.walk(ast.parse(m.value, mode="eval")):
                        if isinstance(x, ast.Name) and x.id in tc_names: used_str[x.id] += 1
                except SyntaxError: pass
            if isinstance(m, ast.Name) and m.id in tc_names: used_unquoted_ann[m.id] += 1
    for n in ast.walk(t):
        if isinstance(n, ast.Name) and n.id in tc_names and id(n) not in ann_ids and id(n) not in tc_nodes:
            used_runtime[n.id].append(n.lineno)
        # cast("X", ...) string uses
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "cast" and n.args and isinstance(n.args[0], ast.Constant) and isinstance(n.args[0].value, str):
            for x in ast.walk(ast.parse(n.args[0].value, mode="eval")):
                if isinstance(x, ast.Name) and x.id in tc_names: used_str[x.id] += 1
    for nm, ln in tc_names.items():
        if used_runtime[nm] and nm not in rt:
            viol.append((rel, used_runtime[nm][0], nm, "TC name used at runtime (outside annotations)"))
        if used_unquoted_ann[nm] and not has_future:
            # check whether unquoted use sits in a local variable annotation (not evaluated) - report anyway
            viol.append((rel, ln, nm, f"unquoted annotation use x{used_unquoted_ann[nm]} without __future__"))
        if not used_str[nm] and not used_unquoted_ann[nm] and not used_runtime[nm] and not any(isinstance(c, ast.Constant) and isinstance(c.value, str) and nm in c.value for c in ast.walk(t)):
            stats["unused_tc_names"] += 1; viol.append((rel, ln, nm, "TC-imported name never used"))
        if used_str[nm] and not used_runtime[nm] and not (used_unquoted_ann[nm] and not has_future):
            stats["ok_names"] += 1
print(dict(stats), dict(origin), "violations", len(viol))
for v in viol: print(*v, sep=" | ")
