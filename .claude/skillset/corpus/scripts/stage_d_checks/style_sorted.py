# Check sortedness of: __all__, _import_structure (keys + value lists), TYPE_CHECKING star-import order in __init__,
# and any other module-level list/tuple/set of >=3 str constants or dict with >=3 str keys ("registries").
import ast, sys, pathlib, collections, re
root = pathlib.Path(sys.argv[1]); base = pathlib.Path(sys.argv[2])
def isort_key(s):  # ruff RUF022 "isort-style": SCREAMING_SNAKE, then CamelCase, then rest; natural-ish
    kind = 0 if s.isupper() and len(s) > 1 else (1 if s[:1].isupper() else 2)
    return (kind, s.lower(), s)
def strs(node):
    if isinstance(node, (ast.List, ast.Tuple, ast.Set)) and node.elts and all(isinstance(e, ast.Constant) and isinstance(e.value, str) for e in node.elts):
        return [e.value for e in node.elts]
    return None
res = collections.Counter(); viol = []
def check(cat, rel, ln, items, name):
    ok = items == sorted(items) or items == sorted(items, key=isort_key) or items == sorted(items, key=str.lower)
    res[(cat, ok)] += 1
    if not ok:
        first = next(i for i, (a, b) in enumerate(zip(items, sorted(items, key=isort_key))) if a != b)
        viol.append((cat, rel, ln, name, f"first out-of-order at idx {first}: {items[first]!r}"))
for f in sorted(root.rglob("*.py")):
    t = ast.parse(f.read_text()); rel = str(f.relative_to(base))
    for st in t.body:
        if isinstance(st, ast.If) and isinstance(st.test, ast.Name) and st.test.id == "TYPE_CHECKING" and f.name == "__init__.py":
            mods = [s.module for s in st.body if isinstance(s, ast.ImportFrom) and any(a.name == "*" for a in s.names)]
            if len(mods) >= 2: check("init_star_imports", rel, st.lineno, mods, "TYPE_CHECKING")
        if not isinstance(st, (ast.Assign, ast.AnnAssign)): continue
        tgt = st.targets[0] if isinstance(st, ast.Assign) else st.target
        name = tgt.id if isinstance(tgt, ast.Name) else ast.unparse(tgt)
        v = st.value
        if v is None: continue
        if name == "__all__" and strs(v) is not None:
            check("__all__", rel, st.lineno, strs(v), name); continue
        if name == "_import_structure" and isinstance(v, ast.Dict):
            keys = [k.value for k in v.keys if isinstance(k, ast.Constant)]
            check("_import_structure_keys", rel, st.lineno, keys, name)
            for k, val in zip(v.keys, v.values):
                if strs(val) and len(strs(val)) >= 2: check("_import_structure_values", rel, val.lineno, strs(val), f"{name}[{k.value!r}]")
            continue
        if strs(v) and len(strs(v)) >= 3: check("other_str_list", rel, st.lineno, strs(v), name)
        if isinstance(v, ast.Dict) and len(v.keys) >= 3 and all(isinstance(k, ast.Constant) and isinstance(k.value, str) for k in v.keys):
            check("other_str_dict_keys", rel, st.lineno, [k.value for k in v.keys], name)
for (cat, ok), n in sorted(res.items()): print(cat, "sorted" if ok else "UNSORTED", n)
for x in viol: print(*x, sep=" | ")
