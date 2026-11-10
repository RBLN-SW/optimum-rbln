"""DOC-012: markdown links inside docstrings (module/class/function) of src/optimum/rbln.
Relative target (not http/https/#/mailto) = violation. docs.rbln.ai targets are resolved against an
rbln_sdk_doc checkout (argv[2]) docs/ tree: <path>.html -> <path>.md / <path>.en.md / <path>/index*.md.
Also lists docs.rbln.ai URLs found in non-docstring string literals (runtime messages) for information.
Usage: python3 test_doc_doc012.py <repo_root> <sdk_doc_root> [git-ref]"""
import ast, pathlib, re, subprocess, sys
root = pathlib.Path(sys.argv[1]); sdk = pathlib.Path(sys.argv[2]); ref = sys.argv[3] if len(sys.argv) > 3 else None
if ref:
    paths = subprocess.run(["git", "-C", str(root), "ls-tree", "-r", "--name-only", ref, "src/optimum/rbln"], capture_output=True, text=True).stdout.split()
    read = lambda p: subprocess.run(["git", "-C", str(root), "show", f"{ref}:{p}"], capture_output=True, text=True).stdout
else:
    paths = [str(p.relative_to(root)) for p in (root / "src/optimum/rbln").rglob("*.py")]
    read = lambda p: (root / p).read_text()
LINK = re.compile(r"\[[^\]]*\]\(([^)\s]+)\)")
URL = re.compile(r"https?://docs\.rbln\.ai/[^\s)\"'`>]+")
def exists(url):
    path = re.sub(r"^https?://docs\.rbln\.ai/(latest/)?", "", url).split("#")[0].rstrip("/.")
    base = path[:-5] if path.endswith(".html") else path
    cands = [f"{base}.md", f"{base}.en.md", f"{base}/index.md", f"{base}/index.en.md"]
    return any((sdk / "docs" / c).exists() for c in cands)
ok, bad, info = [], [], []
for p in sorted(paths):
    if not p.endswith(".py"): continue
    src = read(p); tree = ast.parse(src)
    doc_nodes = set()
    for n in ast.walk(tree):
        if isinstance(n, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) and n.body \
           and isinstance(n.body[0], ast.Expr) and isinstance(n.body[0].value, ast.Constant) and isinstance(n.body[0].value.value, str):
            d = n.body[0].value; doc_nodes.add(id(d))
            for m in LINK.finditer(d.value):
                t = m.group(1)
                line = d.lineno + d.value[: m.start()].count("\n")
                if not re.match(r"(https?://|#|mailto:)", t):
                    bad.append((p, line, m.group(0), "relative link in docstring"))
                elif "docs.rbln.ai" in t and not exists(t):
                    bad.append((p, line, m.group(0), "docs.rbln.ai path not in SDK docs tree"))
                else:
                    ok.append((p, line, t))
    for n in ast.walk(tree):
        if isinstance(n, ast.Constant) and isinstance(n.value, str) and id(n) not in doc_nodes:
            for u in URL.findall(n.value):
                info.append((p, n.lineno, u, "exists" if exists(u) else "MISSING in SDK docs"))
print(f"docstring links ok={len(ok)} bad={len(bad)}")
for b in bad: print("BAD", *b, sep=" | ")
print("-- docs.rbln.ai URLs in non-docstring strings (info):")
for i in info: print("INFO", *i, sep=" | ")
