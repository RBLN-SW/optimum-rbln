"""CFG-019: for every RBLN*Config class that declares `submodules`, check
(a) submodule_config_classes mapping, (b) __init__ param annotation for each submodule,
(c) how dict->config conversion happens (initialize_submodule_config call in __init__ = manual)."""
import ast, glob, json, sys
root = sys.argv[1] if len(sys.argv) > 1 else "."
rows = []
for f in sorted(glob.glob(f"{root}/src/optimum/rbln/**/configuration_*.py", recursive=True)):
    tree = ast.parse(open(f).read())
    for node in tree.body:
        if not isinstance(node, ast.ClassDef):
            continue
        subs = None; has_map = False; init = None
        for b in node.body:
            if isinstance(b, (ast.Assign, ast.AnnAssign)):
                tgt = b.targets[0] if isinstance(b, ast.Assign) else b.target
                if isinstance(tgt, ast.Name) and tgt.id == "submodules":
                    try: subs = ast.literal_eval(b.value)
                    except Exception: subs = "<expr>"
                if isinstance(tgt, ast.Name) and tgt.id == "submodule_config_classes":
                    has_map = True
            if isinstance(b, ast.FunctionDef) and b.name == "__init__":
                init = b
        if subs is None:
            continue
        ann = {}; calls = set(); manual_isinstance_dict = 0
        if init is not None:
            for a in init.args.args + init.args.kwonlyargs:
                if isinstance(subs, list) and a.arg in subs:
                    ann[a.arg] = ast.unparse(a.annotation) if a.annotation else None
            for n in ast.walk(init):
                if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "initialize_submodule_config":
                    calls.add(n.lineno)
                if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "isinstance" and len(n.args) == 2 and ast.unparse(n.args[1]) == "dict":
                    manual_isinstance_dict += 1
        rows.append(dict(path=f[len(root)+1:], cls=node.name, line=node.lineno, submodules=subs,
                         has_map=has_map, has_init=init is not None, n_init_calls=len(calls),
                         manual_isinstance_dict=manual_isinstance_dict, ann=ann))
json.dump(rows, sys.stdout, indent=1)
