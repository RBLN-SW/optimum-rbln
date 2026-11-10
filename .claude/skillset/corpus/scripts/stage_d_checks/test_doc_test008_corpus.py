# Merged PRs since 2025-01 that add a new public model export (touch src/optimum/rbln/__init__.py with
# a "model" title or the 'New Model Support' box ticked) and whether they also touch tests/test_*.py
import json, sys, re
rows = []
for l in open(sys.argv[1]):
    d = json.loads(l)
    if not d.get("mergedAt") or d["mergedAt"] < "2025-01-01": continue
    files = [f if isinstance(f, str) else f.get("path") for f in d["files"]]
    body = d.get("body") or ""
    new_model = re.search(r"\[x\]\s*New Model Support", body, re.I) or re.match(r"model(\(|:)", d["title"])
    if not new_model or "src/optimum/rbln/__init__.py" not in files: continue
    rows.append((d["number"], d["title"][:70], any(re.match(r"tests/test_\w+\.py", f) for f in files)))
ok = [r for r in rows if r[2]]; bad = [r for r in rows if not r[2]]
print(f"new-model PRs: {len(rows)} with tests: {len(ok)} without: {len(bad)}")
for r in bad: print("NO-TEST", *r, sep=" | ")
