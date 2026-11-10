"""DEP-033: bot-generated dependency PRs (Renovate / create-pull-request) merged since 2025-01 must be merged by a human, not via auto-merge.
Usage: python3 dependency_dep033.py <prs.jsonl>   (needs gh auth, read-only)"""
import json, subprocess, sys
rows = []
for l in open(sys.argv[1]):
    p = json.loads(l)
    b = p.get("body") or ""
    bot = ("renovatebot/renovate" in b) or ("peter-evans/create-pull-request" in b)
    if bot and p.get("mergedAt") and p["mergedAt"] >= "2025-01-01":
        rows.append(p)
ok, bad = [], []
for p in sorted(rows, key=lambda x: x["number"]):
    n = p["number"]
    merged_by = subprocess.check_output(["gh", "pr", "view", str(n), "-R", "rebellions-sw/optimum-rbln", "--json", "mergedBy", "-q", ".mergedBy.login"], text=True).strip()
    ev = subprocess.check_output(["gh", "api", "--paginate", f"repos/rebellions-sw/optimum-rbln/issues/{n}/timeline", "-q", ".[].event"], text=True).split()
    auto = "auto_merge_enabled" in ev or "auto_squash_enabled" in ev or "auto_rebase_enabled" in ev
    kind = "renovate" if "renovatebot" in (p.get("body") or "") else "create-pull-request"
    rec = (n, kind, merged_by, auto, p["title"][:70])
    (bad if auto or "bot" in merged_by.lower() else ok).append(rec)
print("compliant", len(ok), "violations", len(bad))
for r in ok: print("OK ", *r)
for r in bad: print("BAD", *r)
