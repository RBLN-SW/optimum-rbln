"""DOC-011: Google-style docstring declaration shape over every public class/function/method in src/optimum/rbln.
Parses each docstring with griffe's Google parser (what mkdocstrings uses) and classifies:
  NAME  - a documented parameter whose parsed name is not an identifier (declaration split across lines /
          name not ending at first ':')  -> violation
  RET   - a Returns section that parses into >1 rows while the description continues on a same-indent line
          (function return annotation is not a tuple) -> violation
  DEF   - a parameter description that states a default without the 'Defaults to' wording -> violation
Usage: <python-with-griffe> test_doc_doc011.py <repo_root>"""
import logging, re, sys, pathlib, collections
import griffe
root = pathlib.Path(sys.argv[1]).resolve()
recs = []
class H(logging.Handler):
    def emit(self, r): recs.append(r.getMessage())
for name in ("griffe", "_griffe", "griffe.docstrings"):
    lg = logging.getLogger(name); lg.addHandler(H()); lg.setLevel(logging.DEBUG); lg.propagate = False
pkg = griffe.load("optimum.rbln", search_paths=[str(root / "src")], allow_inspection=False, docstring_parser=None)
objs = []
def walk(o, public=True):
    for n, m in o.members.items():
        if m.is_alias: continue
        pub = public and (not n.startswith("_") or (m.kind.value != "module" and n.startswith("__") and n.endswith("__")))
        if m.is_module: walk(m, pub); continue
        if m.kind.value in ("class", "function"):
            if pub: objs.append(m)
            if m.is_class: walk(m, pub)
walk(pkg)
IDENT = re.compile(r"^\*{0,2}[A-Za-z_]\w*$")
DEFAULT_ANY = re.compile(r"\bdefaults? to\b|by default|\bdefault\s*[=:]|\bdefault (?:value )?is\b|\(default\b|will default to", re.I)
DEFAULTS_TO = re.compile(r"\bdefaults to\b", re.I)
stats = collections.Counter(); viol = []
for o in objs:
    if not o.docstring: continue
    stats["documented"] += 1
    del recs[:]
    secs = griffe.parse_google(o.docstring)
    rel = str(pathlib.Path(o.filepath).resolve().relative_to(root))
    doc_lines = o.docstring.value.splitlines()
    def lineof(token):
        for i, l in enumerate(doc_lines):
            if token and token.split("\n")[0][:40] in l: return o.docstring.lineno + i, l.strip()
        return o.docstring.lineno, ""
    for s in secs:
        k = s.kind.value
        if k in ("parameters", "other parameters", "attributes"):
            for p in s.value:
                stats["params"] += 1
                if not IDENT.match(p.name):
                    ln, sn = lineof(p.name); viol.append(("NAME", rel, ln, sn, f"{o.path}: parsed param name '{p.name[:30]}'"))
                else:
                    stats["params_ok_name"] += 1
                d = p.description or ""
                if DEFAULT_ANY.search(d):
                    stats["params_mention_default"] += 1
                    if DEFAULTS_TO.search(d): stats["defaults_to"] += 1
                    else:
                        ln, sn = lineof(re.search(r".*default.*", d, re.I|re.S).group(0).strip())
                        viol.append(("DEF", rel, ln, sn, f"{o.path}.{p.name}: default not phrased 'Defaults to'"))
        if k == "returns":
            stats["returns_sections"] += 1
            rows = s.value
            ret_ann = str(getattr(o, "returns", "") or "")
            if len(rows) > 1 and not ret_ann.lower().startswith(("tuple", "typing.tuple")):
                ln, sn = lineof(rows[1].description or str(rows[1].name))
                viol.append(("RET", rel, ln, sn, f"{o.path}: Returns parsed into {len(rows)} rows"))
            else:
                stats["returns_ok"] += 1
print(dict(stats), "public objects:", len(objs))
by = collections.Counter(v[0] for v in viol); print("violations by kind:", dict(by))
for v in viol: print(*v, sep=" | ")
