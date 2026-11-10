import ast, sys, pathlib
root = pathlib.Path(sys.argv[1]); base = pathlib.Path(sys.argv[2]) if len(sys.argv) > 2 else root
good_files = 0; bad = []; files_with_logging = 0
for f in sorted(p for p in root.rglob("*.py") if p.as_posix().rsplit("src/optimum/rbln/",1)[-1] != "utils/logging.py"):
    src = f.read_text(); t = ast.parse(src); rel = f.relative_to(base); lines = src.splitlines()
    uses_ours = False; local_bad = []
    for n in ast.walk(t):
        if isinstance(n, ast.ImportFrom) and n.module and n.module.endswith("utils.logging") and any(a.name == "get_logger" for a in n.names):
            uses_ours = True
        if isinstance(n, ast.ImportFrom) and n.module and "logging" in n.module and n.level == 0 and n.module.startswith(("transformers", "diffusers")) :
            local_bad.append((n.lineno, "imports HF logging"))
        if isinstance(n, ast.Import) and any(a.name in ("logging", "warnings") for a in n.names):
            local_bad.append((n.lineno, f"import {n.names[0].name}"))
        if isinstance(n, ast.ImportFrom) and n.level == 0 and n.module in ("logging", "warnings"):
            local_bad.append((n.lineno, f"from {n.module} import"))
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and isinstance(n.func.value, ast.Name) and (n.func.value.id, n.func.attr) in {("logging", "getLogger"), ("warnings", "warn"), ("logging", "warning"), ("logging", "info")}:
            local_bad.append((n.lineno, f"{n.func.value.id}.{n.func.attr}()"))
    if uses_ours: good_files += 1
    for ln, why in local_bad: bad.append((str(rel), ln, lines[ln-1].strip()[:120], why))
print("files using utils.logging.get_logger:", good_files, " stdlib/HF logging|warnings sites:", len(bad), " files:", len({b[0] for b in bad}))
for b in bad: print(*b, sep=" | ")
