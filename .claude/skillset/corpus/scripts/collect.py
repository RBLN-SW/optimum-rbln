"""Collect stage-B outputs + pilot into one candidates file; verify every quote against the corpus; print stats."""
import json, glob, os, sys, collections
S = sys.argv[1]
corpus_txt = ""
for f in ("review_comments.jsonl", "prs.jsonl", "issue_comments.jsonl"):
    corpus_txt += open(os.path.join(S, "corpus", f)).read()
# jsonl stores bodies JSON-escaped; also keep an unescaped copy
unesc = []
for f in ("review_comments.jsonl", "prs.jsonl", "issue_comments.jsonl"):
    for l in open(os.path.join(S, "corpus", f)):
        unesc.append(json.loads(l).get("body") or "")
unesc = "\n".join(unesc)
def found(q): return bool(q) and (q in unesc or q in corpus_txt)

cands, procs, decs, debs = [], [], [], []
bad = 0
# pilot (review comments)
pilot = json.load(open(os.path.join(S, "pilot", "rules_from_review_comments.json")))
for c in pilot["clusters"]:
    ev = [dict(source="review_comment", pr=e["pr"], user=e.get("user"), quote=e["quote"]) for e in c["evidence"]]
    cands.append(dict(src="review_comments", candidate=c["rule"], area=c["area"], kind=c["kind"], polarity="do",
                      open_question=c.get("open_question"), evidence=ev))
for d in pilot.get("unresolved_debates", []):
    debs.append(dict(src="review_comments", **d))
# stage B
for f in sorted(glob.glob(os.path.join(S, "stage_b", "out", "*.json"))):
    try: d = json.load(open(f))
    except Exception as e: print("BAD JSON", f, e); continue
    src = "pr_bodies" if os.path.basename(f).startswith("pr_") else "conversation_comments"
    for c in d.get("candidates", []):
        c["src"] = src
        for e in c.get("evidence", []):
            e["source"] = "pr_body" if src == "pr_bodies" else "conversation_comment"
            if not found(e.get("quote")): bad += 1; e["unverified"] = True
        cands.append(c)
    for p in d.get("procedures", []): p["src"] = src; procs.append(p)
    for x in d.get("decisions", []): x["src"] = src; decs.append(x)
    for x in d.get("debates", []): x["src"] = src; debs.append(x)
out = dict(candidates=cands, procedures=procs, decisions=decs, debates=debs)
json.dump(out, open(os.path.join(S, "stage_c", "all_candidates.json"), "w"), ensure_ascii=False, indent=1)
print("candidates", len(cands), "procedures", len(procs), "decisions", len(decs), "debates", len(debs), "unverified quotes", bad)
print("by src:", collections.Counter(c["src"] for c in cands))
print("by area:", collections.Counter(c.get("area") for c in cands).most_common())
print("by kind:", collections.Counter(c.get("kind") for c in cands))
print("chars:", os.path.getsize(os.path.join(S, "stage_c", "all_candidates.json")))
