"""TEST-019 check.
(A) in-place writes to an UPPER_CASE class-level dict through cls./self.
    (X.update/setdefault/pop/clear, X[k]=, X[k].update, del X[k]).
    compliant: target dict is declared in the enclosing class body AND the write is in setUpClass.
    violation: target is inherited (not declared in this class) OR the write happens outside setUpClass
               (i.e. in a test method, so it persists across the class's tests).
(B) RBLN_CLASS.from_pretrained(...) calls that COMPILE from the hub id (first arg / model_id is HF_MODEL_ID)
    inside a test class: compliant if HF_CONFIG_KWARGS is splatted into the call.
Usage: python3 test_doc_test019.py <repo_root> [git-ref]   (git-ref reads tests via `git show`)
"""
import ast, pathlib, subprocess, sys
root = pathlib.Path(sys.argv[1]); ref = sys.argv[2] if len(sys.argv) > 2 else None
files = ["tests/test_base.py", "tests/test_transformers.py", "tests/test_llm.py", "tests/test_diffusers.py", "tests/test_config.py"]
def read(f):
    if ref: return subprocess.run(["git", "-C", str(root), "show", f"{ref}:{f}"], capture_output=True, text=True).stdout
    return (root / f).read_text()
classes = {}
def collect(body, f, prefix=""):
    for n in body:
        if isinstance(n, ast.ClassDef):
            own = {t.id for s in n.body if isinstance(s, (ast.Assign,)) for t in s.targets if isinstance(t, ast.Name)}
            own |= {s.target.id for s in n.body if isinstance(s, ast.AnnAssign) and isinstance(s.target, ast.Name)}
            classes[prefix + n.name] = dict(node=n, file=f, own=own, bases=[ast.unparse(b) for b in n.bases])
            collect(n.body, f, prefix + n.name + ".")
srcs = {f: read(f) for f in files}
for f, s in srcs.items(): collect(ast.parse(s).body, f)
def owner(cname, attr, seen=()):
    c = classes.get(cname) or classes.get(cname.split(".")[-1])
    if not c: return None
    if attr in c["own"]: return cname
    for b in c["bases"]:
        r = owner(b, attr)
        if r: return r
    return None
MUT = {"update", "setdefault", "pop", "clear", "popitem"}
def root_attr(e):
    # returns (cls|self, ATTR) if e is cls.ATTR / self.ATTR[...]...
    while isinstance(e, ast.Subscript): e = e.value
    if isinstance(e, ast.Attribute) and isinstance(e.value, ast.Name) and e.value.id in ("cls", "self") and e.attr.isupper():
        return e.attr
    return None
A_ok, A_bad, B_ok, B_bad, B_other = [], [], [], [], 0
for cname, c in classes.items():
    lines = srcs[c["file"]].splitlines()
    for m in c["node"].body:
        if not isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef)): continue
        for n in ast.walk(m):
            tgt = None
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr in MUT:
                tgt = root_attr(n.func.value)
            elif isinstance(n, (ast.Assign, ast.AugAssign, ast.Delete)):
                ts = n.targets if isinstance(n, (ast.Assign, ast.Delete)) else [n.target]
                for t in ts:
                    if isinstance(t, ast.Subscript): tgt = root_attr(t) or tgt
            if tgt:
                declared_here = tgt in c["own"]
                ok = declared_here and m.name == "setUpClass"
                why = "" if ok else ("inherited dict " + str(owner(cname, tgt)) if not declared_here else f"mutated in {m.name}, persists across tests")
                (A_ok if ok else A_bad).append((c["file"], n.lineno, lines[n.lineno - 1].strip(), cname, why))
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "from_pretrained":
                recv = ast.unparse(n.func.value)
                if not (recv.endswith("RBLN_CLASS") or recv.startswith("RBLN")): continue
                args = [ast.unparse(a) for a in n.args] + [ast.unparse(k.value) for k in n.keywords if k.arg == "model_id"]
                if not any("HF_MODEL_ID" in a for a in args):
                    B_other += 1; continue
                has = any(k.arg is None and "HF_CONFIG_KWARGS" in ast.unparse(k.value) for k in n.keywords)
                (B_ok if has else B_bad).append((c["file"], n.lineno, lines[n.lineno - 1].strip(), cname, "compiles HF_MODEL_ID without HF_CONFIG_KWARGS"))
print(f"A in-place writes: ok={len(A_ok)} bad={len(A_bad)}")
for v in A_bad: print("  A", *v, sep=" | ")
print(f"B compile-from-hub from_pretrained: ok={len(B_ok)} bad={len(B_bad)} (load-from-dir calls skipped: {B_other})")
for v in B_bad: print("  B", *v, sep=" | ")
