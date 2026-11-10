"""DEP-001: deprecations use deprecate_kwarg/deprecate_method with a removal version > current version.
Usage (repo root): python3 dependency_dep001.py [CURRENT_VERSION]  (default: git describe --tags, 'v' stripped)"""
import ast, pathlib, re, subprocess, sys
from packaging.version import Version

cur = sys.argv[1] if len(sys.argv) > 1 else subprocess.check_output(["git", "describe", "--tags"], text=True).strip().lstrip("v")
cur = Version(Version(cur.split("-")[0]).base_version)
DECOS = {"deprecate_kwarg", "deprecate_method"}
LOG = {"warning", "warning_once", "warn", "info", "error"}
ok, bad = [], []
for p in sorted(pathlib.Path("src/optimum/rbln").rglob("*.py")):
    if p.name == "deprecation.py":
        continue
    src = p.read_text()
    lines = src.splitlines()
    for n in ast.walk(ast.parse(src)):
        if isinstance(n, ast.Call):
            fn = n.func
            name = fn.attr if isinstance(fn, ast.Attribute) else getattr(fn, "id", None)
            if name in DECOS:
                v = next((k.value.value for k in n.keywords if k.arg == "version"), None)
                if v is None and n.args:
                    v = n.args[0].value if name == "deprecate_method" else (n.args[1].value if len(n.args) > 1 else None)
                rec = (str(p), n.lineno, lines[n.lineno - 1].strip())
                if v is None or Version(v) <= cur:
                    bad.append(rec + (f"removal version {v} <= current {cur}; should be removed",))
                else:
                    ok.append(rec)
            elif name in LOG or (isinstance(fn, ast.Attribute) and fn.attr == "warn" and getattr(fn.value, "id", "") == "warnings"):
                txt = " ".join(c.value for a in n.args for c in ast.walk(a) if isinstance(c, ast.Constant) and isinstance(c.value, str))
                if re.search(r"deprecat", txt, re.I):
                    # upstream-copied HF messages that reference HF library versions are out of scope
                    if re.search(r"v4\.\d+", txt):
                        continue
                    bad.append((str(p), n.lineno, lines[n.lineno - 1].strip(), "ad-hoc deprecation warning, no decorator/removal version"))
print("current", cur)
print("compliant", len(ok))
for r in ok: print("OK ", *r)
print("violations", len(bad))
for r in bad: print("BAD", *r)
