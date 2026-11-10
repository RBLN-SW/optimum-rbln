"""Concatenate per-area merged YAML, validate schema + quotes vs corpus, compute convergence, write rules_v0.yaml + report."""
import json, sys, os, glob, collections, yaml
S = sys.argv[1]
bodies = []
for f in ("review_comments.jsonl", "prs.jsonl", "issue_comments.jsonl"):
    for l in open(os.path.join(S, "corpus", f)):
        bodies.append(json.loads(l).get("body") or "")
corpus = "\n".join(bodies)
titles = []
for l in open(os.path.join(S, "corpus", "prs.jsonl")):
    j = json.loads(l); titles.append(j.get("title") or ""); titles.extend(j.get("files") or [])
weak_corpus = "\n".join(titles)
REQ = ["id","statement","area","kind","status","confidence","n_prs","n_users","sources","evidence","merged_from"]
rules, problems = [], []
for f in sorted(glob.glob(os.path.join(S, "stage_c", "merged", "*.yaml"))):
    try: items = yaml.safe_load(open(f))
    except Exception as e: problems.append(f"{f}: YAML error {e}"); continue
    if not isinstance(items, list): problems.append(f"{f}: not a list"); continue
    for r in items:
        for k in REQ:
            if k not in r: problems.append(f"{r.get('id','?')}: missing {k}")
        ev_ok = []
        for e in r.get("evidence", []) or []:
            q = e.get("quote") or ""
            if q in corpus: ev_ok.append(e)
            elif q in weak_corpus: e["weak"] = True; ev_ok.append(e)   # PR title or file path, not prose
            else: problems.append(f"{r.get('id')}: quote not in corpus: {q[:60]!r}")
        r["evidence"] = ev_ok
        if not ev_ok: problems.append(f"{r.get('id')}: NO verified evidence -> dropped"); continue
        # recompute counts from evidence actually kept
        r["n_prs"] = len({e.get("pr") for e in ev_ok}); r["n_users"] = len({e.get("user") for e in ev_ok if e.get("user")})
        if all(e.get("weak") for e in ev_ok):
            r["status"] = "candidate" if r["status"] != "interview" else "interview"; r["confidence"] = "low"; r["note"] = "title/path-only evidence"
        for e in ev_ok:
            if e.get("source") in ("review_comment",): e["source"] = "review_comments"
            if e.get("source") in ("pr_body",): e["source"] = "pr_bodies"
            if e.get("source") in ("conversation_comment",): e["source"] = "conversation_comments"
        rules.append(r)
ids = collections.Counter(r["id"] for r in rules)
for i, n in ids.items():
    if n > 1: problems.append(f"duplicate id {i} x{n}")
# convergence by source order
order = ["review_comments", "pr_bodies", "conversation_comments"]
new_at = collections.Counter()
for r in rules:
    srcs = [s for s in order if s in (r.get("sources") or [])]
    new_at[srcs[0] if srcs else "unknown"] += 1
out = os.path.join(S, "stage_c", "rules_v0.yaml")
yaml.safe_dump(rules, open(out, "w"), allow_unicode=True, sort_keys=False, width=120)
rep = []
rep.append(f"rules: {len(rules)}  problems: {len(problems)}")
rep.append("by area: " + json.dumps(collections.Counter(r['area'] for r in rules).most_common(), ensure_ascii=False))
rep.append("by status: " + json.dumps(collections.Counter(r['status'] for r in rules)))
rep.append("by confidence: " + json.dumps(collections.Counter(r['confidence'] for r in rules)))
rep.append("by kind: " + json.dumps(collections.Counter(r['kind'] for r in rules)))
rep.append("new rules at source (order rc→pr→ic): " + json.dumps(new_at))
rep.append("merged_from total: " + str(sum(int(r.get('merged_from') or 0) for r in rules)))
rep.append("area×status: " + json.dumps(collections.Counter((r['area'], r['status']) for r in rules).most_common(), ensure_ascii=False))
print("\n".join(rep)); open(os.path.join(S, "stage_c", "finalize_report.txt"), "w").write("\n".join(rep) + "\n\nPROBLEMS\n" + "\n".join(problems))
print("--- problems (first 30) ---"); print("\n".join(problems[:30]))
