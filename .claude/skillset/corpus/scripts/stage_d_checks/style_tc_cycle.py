# For each first-party import inside `if TYPE_CHECKING:`, check whether a module-level import
# of the target would create a cycle (target transitively imports the source at module level).
import ast, sys, pathlib, collections
src_root = pathlib.Path(sys.argv[1])  # path to .../src
pkg = "optimum.rbln"
mods = {}
for f in (src_root / "optimum/rbln").rglob("*.py"):
    parts = list(f.relative_to(src_root).with_suffix("").parts)
    is_pkg = parts[-1] == "__init__"
    if is_pkg: parts = parts[:-1]
    mods[".".join(parts)] = (f, is_pkg)
def is_tc(test):
    return (isinstance(test, ast.Name) and test.id == "TYPE_CHECKING") or (isinstance(test, ast.Attribute) and test.attr == "TYPE_CHECKING")
def resolve(mod, is_pkg, node, alias=None):
    if getattr(node, "level", 0):
        base = mod.split(".") if is_pkg else mod.split(".")[:-1]
        base = base[: len(base) - (node.level - 1)]
        tgt = ".".join(base + ([node.module] if node.module else []))
    else:
        tgt = node.module if isinstance(node, ast.ImportFrom) else alias
    out = []
    if tgt and tgt.startswith(pkg):
        if isinstance(node, ast.ImportFrom):
            for a in node.names:
                sub = f"{tgt}.{a.name}"
                out.append(sub if sub in mods else tgt)
        else:
            out.append(tgt)
    return out
graph = collections.defaultdict(set); tc_edges = []
for m, (f, is_pkg) in mods.items():
    t = ast.parse(f.read_text())
    tc_ids = set()
    for n in ast.walk(t):
        if isinstance(n, ast.If) and is_tc(n.test):
            for st in n.body:
                for x in ast.walk(st):
                    tc_ids.add(id(x))
                    if isinstance(x, ast.ImportFrom) and not is_pkg:
                        for tgt in resolve(m, is_pkg, x):
                            tc_edges.append((m, tgt, x.lineno))
    # module-level runtime imports (top-level statements + else branches of TC ifs + try blocks), skip function bodies
    def visit(stmts):
        for st in stmts:
            if isinstance(st, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                if isinstance(st, ast.ClassDef): visit(st.body)
                continue
            if isinstance(st, ast.If) and is_tc(st.test):
                visit(st.orelse); continue
            if isinstance(st, (ast.Import, ast.ImportFrom)):
                aliases = [a.name for a in st.names] if isinstance(st, ast.Import) else [None]
                for al in aliases:
                    for tgt in resolve(m, is_pkg, st, al):
                        graph[m].add(tgt)
                        # importing a submodule also executes parent packages
                        p = tgt.rsplit(".", 1)[0]
                        while p.startswith(pkg) and p != tgt:
                            graph[m].add(p);
                            if "." not in p: break
                            p = p.rsplit(".", 1)[0]
            for fld in ("body", "orelse", "handlers", "finalbody"):
                v = getattr(st, fld, None)
                if isinstance(v, list) and v and isinstance(v[0], ast.stmt): visit(v)
                if fld == "handlers" and v:
                    for h in v: visit(h.body)
    visit(t.body)
def reaches(a, b):
    seen = set(); st = [a]
    while st:
        x = st.pop()
        if x == b: return True
        if x in seen: continue
        seen.add(x); st.extend(graph.get(x, ()))
    return False
cyc = 0; nocyc = []
seen_pairs = set()
for m, tgt, ln in tc_edges:
    if (m, tgt) in seen_pairs: continue
    seen_pairs.add((m, tgt))
    if reaches(tgt, m): cyc += 1
    else: nocyc.append((m, tgt, ln))
print("first-party TYPE_CHECKING edges:", len(seen_pairs), "would-cycle:", cyc, "no-cycle:", len(nocyc))
for x in nocyc: print("NOCYCLE", *x)
