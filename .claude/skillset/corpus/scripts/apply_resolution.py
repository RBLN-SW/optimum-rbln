"""Apply a resolution file (merge/unpair/reject/restate) to rules.yaml, shortened.json and interview_options.json."""
import json, sys, yaml, datetime
S, W, RES = sys.argv[1], sys.argv[2], sys.argv[3]
RY = f"{W}/.claude/skillset/rules.yaml"
text = open(RY).read(); hdr = "".join(l for l in text.splitlines(True) if l.startswith("#"))
rules = yaml.safe_load(text); byid = {r["id"]: r for r in rules}
SH = f"{S}/stage_e/shortened.json"; short = {x["id"]: x for x in json.load(open(SH))}
OP = f"{S}/stage_e/interview_options.json"; opts = json.load(open(OP))
res = json.load(open(RES)); today = str(datetime.date.today()); log = []
def recount(r):
    ev = r.get("evidence") or []; strong = {e.get("pr") for e in ev if not e.get("weak")}
    r["n_prs"] = len({e.get("pr") for e in ev}); r["n_users"] = len({e.get("user") for e in ev if e.get("user")})
    v = (r.get("grounding") or {}).get("verdict")
    if v in ("verified", "aspirational", "rejected") and r["status"] != "interview": r["status"] = v
    elif r["status"] not in ("rejected", "verified", "aspirational"): r["status"] = "supported" if len(strong) >= 2 else "candidate"
    r["confidence"] = "high" if (r["n_prs"] >= 3 and r["n_users"] >= 2) else "medium" if r["n_prs"] >= 2 else "low"
def clear_pair(r):
    for k in ("contradicts", "open_question"): r.pop(k, None)
    if r["status"] == "interview": r["status"] = "supported"
    opts["rules"].pop(r["id"], None)
    if r["id"] in short: short[r["id"]]["question"] = None
for m in res.get("merge", []):
    k = byid.get(m["keep"])
    if not k: log.append(f"skip merge: {m['keep']} missing"); continue
    clear_pair(k); prev = [k["statement"]]
    for f in m.get("fold", []):
        src = byid.pop(f, None)
        if not src: log.append(f"skip fold {f}"); continue
        seen = {e["quote"] for e in k["evidence"]}
        k["evidence"] += [e for e in src["evidence"] if e["quote"] not in seen]
        k["merged_from"] = int(k.get("merged_from") or 0) + int(src.get("merged_from") or 0)
        k.setdefault("folded_ids", []).append(f); prev.append(src["statement"])
        short.pop(f, None); opts["rules"].pop(f, None)
    k["evidence"] = k["evidence"][:6]
    k["statement"] = m["short"]; k["detail"] = m.get("detail"); k["previous_statements"] = (k.get("previous_statements") or []) + prev
    k["resolution"] = {"date": today, "action": "merge", "why": m.get("why"), "code": m.get("code")}
    short[k["id"]] = {"id": k["id"], "short": m["short"], "detail": m.get("detail"), "question": None}
    recount(k); log.append(f"merge {k['id']} <- {m.get('fold')} -> {k['status']}")
for u in res.get("unpair", []):
    for i in u["ids"]:
        r = byid.get(i)
        if not r: continue
        clear_pair(r); r["resolution"] = {"date": today, "action": "unpair", "why": u.get("why")}; recount(r); log.append(f"unpair {i} -> {r['status']}")
for x in res.get("reject", []):
    r = byid.get(x["id"])
    if not r: continue
    clear_pair(r); r["status"] = "rejected"; r["rejected_why"] = x["why"]
    if x.get("refined"): r["refined_statement"] = x["refined"]
    r["resolution"] = {"date": today, "action": "reject", "why": x["why"]}; log.append(f"reject {x['id']}")
for x in res.get("restate", []):
    r = byid.get(x["id"])
    if not r: continue
    r.setdefault("previous_statements", []).append(r["statement"]); r["statement"] = x["short"]; r["detail"] = x.get("detail")
    short[r["id"]] = {"id": r["id"], "short": x["short"], "detail": x.get("detail"), "question": None}
    if x.get("verified"):
        r["grounding"] = {"verdict": "verified", "reason": f"dev 코드에서 확인: {x['verified']}", "date": today, "batch": "manual", "measured_on": "origin/dev 74e97fa4"}
        r["status"] = "verified"
    log.append(f"restate {x['id']} -> {r['status']}")
# debates: point rule_ids at survivors
alias = {f: r["id"] for r in byid.values() for f in (r.get("folded_ids") or [])}
for d in opts.get("debates", []):
    d["rule_ids"] = sorted({alias.get(i, i) for i in (d.get("rule_ids") or []) if alias.get(i, i) in byid})
final = [r for r in rules if r["id"] in byid]
open(RY, "w").write(hdr + yaml.safe_dump(final, allow_unicode=True, sort_keys=False, width=110))
json.dump(list(short.values()), open(SH, "w"), ensure_ascii=False, indent=1)
json.dump(opts, open(OP, "w"), ensure_ascii=False, indent=1)
for l in log: print(l)
from collections import Counter
print("status:", dict(Counter(r["status"] for r in final)), "| rules:", len(final))
