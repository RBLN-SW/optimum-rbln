"""rules.yaml + shortened.json (+ interview_options.json) -> artifact/review_data.json for the review page."""
import yaml, json, sys, os, re, collections
W, S = sys.argv[1], sys.argv[2]
rules = yaml.safe_load(open(f"{W}/.claude/skillset/rules.yaml"))
short = {x["id"]: x for x in json.load(open(f"{S}/stage_e/shortened.json"))}
opt_path = f"{S}/stage_e/interview_options.json"
opts = json.load(open(opt_path)) if os.path.exists(opt_path) else {"rules": {}, "debates": []}
AREAS = [("rbln_config", "rbln_config"), ("multimodal/submodule", "멀티모달 · 서브모듈"), ("modeling_base/lifecycle", "모델 베이스 · 라이프사이클"),
         ("decoderonly", "decoderonly"), ("attention/kvcache", "attention · KV cache"), ("diffusers", "diffusers"), ("api/naming", "공개 API · 이름"),
         ("testing", "테스트"), ("docstring/docs", "docstring · 문서"), ("dependency/version", "의존성 · 버전"), ("pr_process", "PR · CI · 브랜치"), ("style/other", "코드 스타일 · 기타")]
NORM = re.compile(r"should|must|never|always|instead|rather|prefer|avoid|don'?t|do not|raise|해야|않|말고|대신|항상|금지|원칙|규칙|통일|일관|합니다|하세요|필요|좋을 것|권장|지양", re.I)
def clean(q, n=160):
    q = re.sub(r"\s+", " ", q).strip()
    if len(q) <= n: return q
    cut = q[:n]; i = max(cut.rfind(". "), cut.rfind("다. "), cut.rfind("요. "))
    return (cut[:i+1] if i > n // 2 else cut[:n-1].rstrip() + "…")
def ev_score(e):
    q = e.get("quote") or ""
    return (len(NORM.findall(q)) > 0) * 2 + min(len(q), 110) / 110 - (len(q) < 30) * 2 - (e.get("weak") and 5 or 0)
def pick_ev(r):
    ev = sorted(r.get("evidence", []), key=ev_score, reverse=True); seen_pr, seen_user, out = set(), set(), []
    for e in ev:
        if e.get("pr") in seen_pr: continue
        if out and e.get("user") in seen_user and any(x.get("pr") not in seen_pr and x.get("user") not in seen_user and ev_score(x) > ev_score(e) - 1 for x in ev): continue
        seen_pr.add(e.get("pr")); seen_user.add(e.get("user"))
        out.append({"pr": e.get("pr"), "user": (e.get("user") or "").replace("rebel-", ""), "quote": clean(e.get("quote") or ""), "src": e.get("source")})
        if len(out) == 2: break
    return out
VERDICT_KO = {"verified": "코드 일치", "aspirational": "코드 불일치", "interview": "코드도 갈림", "rejected": "폐기 제안", "unverifiable": "검사 불가"}
items = []
for r in rules:
    if r["status"] == "candidate": continue
    s = short.get(r["id"], {}); o = opts["rules"].get(r["id"])
    g = r.get("grounding"); c = r.get("compliance") or {}
    kind = "choice" if r["status"] == "interview" else "confirm"
    items.append({
        "id": r["id"], "area": r["area"], "kind": kind, "status": r["status"],
        "rule": s.get("short") or r["statement"], "detail": s.get("detail"), "full": r["statement"],
        "evidence": pick_ev(r), "n_prs": r.get("n_prs"), "n_users": r.get("n_users"),
        "code": ({"verdict": g["verdict"], "verdict_ko": VERDICT_KO.get(g["verdict"], g["verdict"]), "compliant": c.get("compliant"), "violations": c.get("violations_total"),
                  "reason": clean(g.get("reason") or "", 220), "violations_known": [f"{v.get('path')}" + (f":{v['line']}" if v.get("line") else "") for v in (r.get("violations_known") or [])[:8]]} if g else None),
        "question": (o or {}).get("question") or s.get("question") if kind == "choice" else None,
        "options": (o or {}).get("options") if kind == "choice" else None,
        "pair": (o or {}).get("pair") if kind == "choice" else None,
        "refined": r.get("refined_statement"),
    })
debates = []
for d in opts.get("debates", []):
    debates.append({"id": d["id"], "area": "debates", "topic_area": d.get("area") or "style/other", "kind": "choice", "status": "debate", "rule": d.get("topic"), "detail": None,
                    "evidence": [{"pr": p, "user": "", "quote": "", "src": "debate"} for p in (d.get("prs") or [])[:3]],
                    "question": d.get("question"), "options": d.get("options"), "rule_ids": d.get("rule_ids") or [], "code": None})
# pair groups: union-find over `pair` links so one answer covers the whole group
parent = {}
def find(x):
    parent.setdefault(x, x)
    while parent[x] != x: parent[x] = parent[parent[x]]; x = parent[x]
    return x
for rid, o in opts["rules"].items():
    if o.get("pair"): parent[find(rid)] = find(o["pair"])
groups = collections.defaultdict(list)
for rid in opts["rules"]: groups[find(rid)].append(rid)
area_of = {i["id"]: i["area"] for i in items}
for i in items:
    if i["kind"] == "choice" and i["id"] in parent:
        g = sorted(groups[find(i["id"])])
        if len(g) > 1: i["group"] = [{"id": x, "area": area_of.get(x)} for x in g]
# examples per option (stage F): by own id, else by a group member, else by identical question+options
import glob
ex = {}
for f in glob.glob(f"{S}/stage_f/out/batch_*.json"):
    try:
        for o in json.load(open(f)): ex[o["id"]] = {"lang": o.get("lang") or "text", "intro": o.get("intro"), "examples": o.get("examples") or []}
    except Exception as e: print("bad examples file", f, e)
bykey = {}
for i in items + debates:
    if i["kind"] != "choice": continue
    if i["id"] in ex: bykey.setdefault((i.get("question"), tuple(i.get("options") or [])), ex[i["id"]])
for i in items + debates:
    if i["kind"] != "choice": continue
    e = ex.get(i["id"]) or next((ex[g["id"]] for g in (i.get("group") or []) if g["id"] in ex), None) \
        or next((ex[r] for r in (i.get("rule_ids") or []) if r in ex), None) or bykey.get((i.get("question"), tuple(i.get("options") or [])))
    if e and len(e["examples"]) == len(i.get("options") or []): i["example"] = e
print("choice items with example:", sum(1 for i in items + debates if i.get("example")), "/", sum(1 for i in items + debates if i["kind"] == "choice"))
for f in glob.glob(f"{W}/.claude/skillset/mining/explain/*.json"):
    try:
        e = json.load(open(f)); tgt = next((i for i in items + debates if i["id"] == e["id"]), None)
        if tgt: tgt["explain"] = {k: e.get(k) for k in ("summary", "examples", "evidence", "verified")}
    except Exception as err: print("bad explain file", f, err)
en = {}
for f in glob.glob(f"{S}/stage_h/out/batch_*.json"):
    try:
        for o in json.load(open(f)): en[o["id"]] = o
    except Exception as err: print("bad en file", f, err)
for i in items + debates:
    o = en.get(i["id"])
    if o: i["rule_en"] = o.get("rule_en"); i["options_en"] = o.get("options_en")
print("with English rule:", sum(1 for i in items + debates if i.get("rule_en") or i.get("options_en")))
data = {"generated": "2026-10-23", "areas": [{"key": k, "label": l} for k, l in AREAS], "items": items, "debates": debates,
        "counts": dict(collections.Counter(i["status"] for i in items))}
json.dump(data, open(f"{S}/artifact/review_data.json", "w"), ensure_ascii=False, separators=(",", ":"))
print("items", len(items), "debates", len(debates), "bytes", os.path.getsize(f"{S}/artifact/review_data.json"), "choice items with options", sum(1 for i in items if i["kind"] == "choice" and i.get("options")))
