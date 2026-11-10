"""Apply dedupe.json to rules_v0.yaml -> final rules.yaml (+ dropped list + report)."""
import json, sys, os, collections, yaml, datetime
S, OUT = sys.argv[1], sys.argv[2]
rules = {r["id"]: r for r in yaml.safe_load(open(f"{S}/stage_c/rules_v0.yaml"))}
d = json.load(open(f"{S}/stage_c/dedupe.json"))
ORDER = ["review_comments", "pr_bodies", "conversation_comments"]
def recount(r):
    ev = r["evidence"]
    r["n_prs"] = len({e.get("pr") for e in ev}); r["n_users"] = len({e.get("user") for e in ev if e.get("user")})
    r["sources"] = [s for s in ORDER if any(e.get("source") == s for e in ev)]
    strong = [e for e in ev if not e.get("weak")]
    if r["status"] != "interview":
        r["status"] = "supported" if len({e.get("pr") for e in strong}) >= 2 else "candidate"
    r["confidence"] = "high" if (r["n_prs"] >= 3 and r["n_users"] >= 2) else "medium" if r["n_prs"] >= 2 else "low"
    if not strong: r["confidence"] = "low"
dropped, folded = [], {}
for m in d.get("merges", []):
    k = m["keep"]
    if k not in rules: continue
    keep = rules[k]
    for f in m.get("fold", []):
        if f not in rules or f == k: continue
        src = rules.pop(f)
        seen = {e["quote"] for e in keep["evidence"]}
        keep["evidence"] += [e for e in src["evidence"] if e["quote"] not in seen]
        keep["merged_from"] = int(keep.get("merged_from") or 0) + int(src.get("merged_from") or 0)
        if src["status"] == "interview": keep["status"] = "interview"
        if src.get("open_question") and not keep.get("open_question"): keep["open_question"] = src["open_question"]
        keep.setdefault("folded_ids", []).append(f); folded[f] = k
    if m.get("statement"): keep["statement"] = m["statement"]
    if m.get("area"): keep["area"] = m["area"]
    keep["evidence"] = keep["evidence"][:6]
    recount(keep)
for x in d.get("reassign", []):
    if x["id"] in rules: rules[x["id"]]["area"] = x["area"]
for x in d.get("not_a_rule", []):
    i = folded.get(x["id"], x["id"])
    if x["id"] in rules and x["id"] not in folded:
        r = rules.pop(x["id"]); r["dropped_why"] = x.get("why"); dropped.append(r)
for c in d.get("contradictions", []):
    ids = [folded.get(i, i) for i in c.get("ids", [])]
    for i in ids:
        if i in rules:
            rules[i]["status"] = "interview"
            if not rules[i].get("open_question"): rules[i]["open_question"] = c.get("question")
            rules[i]["contradicts"] = [j for j in ids if j != i]
for r in rules.values(): recount(r) if r["status"] != "interview" else None
AREA_ORDER = ["rbln_config","multimodal/submodule","modeling_base/lifecycle","decoderonly","attention/kvcache","diffusers","api/naming","testing","docstring/docs","dependency/version","pr_process","style/other"]
final = sorted(rules.values(), key=lambda r: (AREA_ORDER.index(r["area"]) if r["area"] in AREA_ORDER else 99, {"interview":0,"supported":1,"candidate":2}[r["status"]], -r["n_prs"], r["id"]))
hdr = f"""# optimum-rbln rules DB v0 — generated {datetime.date.today()} by the stage A→C mining pipeline (see PLAN.md §3).
# status: supported (≥2 distinct PRs with prose evidence) | candidate (1 PR, or title/path-only evidence) | interview (contradictory evidence or open question)
# Every evidence.quote is a verbatim substring of the PR/review/comment corpus; `weak: true` marks a PR title or file path rather than prose.
# `check`, `violations_known`, `owner` are filled by the code-grounding (D) and interview (E) stages. Ids are stable provenance keys; do not renumber.
"""
open(OUT, "w").write(hdr + yaml.safe_dump(final, allow_unicode=True, sort_keys=False, width=110))
yaml.safe_dump(dropped, open(f"{S}/stage_c/dropped.yaml", "w"), allow_unicode=True, sort_keys=False)
C = collections.Counter
rep = {
 "rules": len(final), "dropped_not_a_rule": len(dropped), "folded": len(folded),
 "by_status": dict(C(r["status"] for r in final)), "by_confidence": dict(C(r["confidence"] for r in final)),
 "by_kind": dict(C(r["kind"] for r in final)), "by_area": dict(C(r["area"] for r in final).most_common()),
 "area_x_status": {a: dict(C(r["status"] for r in final if r["area"] == a)) for a in AREA_ORDER},
 "new_at_source": dict(C(next((s for s in ORDER if s in r["sources"]), "unknown") for r in final)),
 "supported_or_interview": sum(1 for r in final if r["status"] != "candidate"),
 "mechanical_supported": sum(1 for r in final if r["kind"] == "mechanical" and r["status"] == "supported"),
}
json.dump(rep, open(f"{S}/stage_c/final_report.json", "w"), ensure_ascii=False, indent=1)
print(json.dumps(rep, ensure_ascii=False, indent=1))
