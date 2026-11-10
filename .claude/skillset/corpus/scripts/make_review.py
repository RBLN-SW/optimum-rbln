"""rules.yaml (+ shortened.json) -> human review sheets under .claude/skillset/review/."""
import yaml, json, sys, os, re, collections, datetime
W, S = sys.argv[1], sys.argv[2]
OUT = f"{W}/.claude/skillset/review"; os.makedirs(OUT, exist_ok=True)
rules = yaml.safe_load(open(f"{W}/.claude/skillset/rules.yaml"))
short = {x["id"]: x for x in json.load(open(f"{S}/stage_e/shortened.json"))}
REPO = "https://github.com/rebellions-sw/optimum-rbln/pull/"
AREAS = [("rbln_config", "rbln_config (설정 객체)"), ("multimodal/submodule", "멀티모달 · 서브모듈"), ("modeling_base/lifecycle", "모델 베이스 · 라이프사이클"),
         ("decoderonly", "decoderonly"), ("attention/kvcache", "attention · KV cache"), ("diffusers", "diffusers"), ("api/naming", "공개 API · 이름"),
         ("testing", "테스트"), ("docstring/docs", "docstring · 문서"), ("dependency/version", "의존성 · 버전"), ("pr_process", "PR · CI · 브랜치 운영"), ("style/other", "코드 스타일 · 기타")]
STATUS_KO = {"verified": "코드 일치", "aspirational": "코드 불일치(의도는 분명)", "interview": "결정 필요", "supported": "판단형", "rejected": "폐기 제안", "candidate": "근거 1건"}
def clean(q, n=130):
    q = re.sub(r"\s+", " ", q.replace("`", "'")).strip()
    if len(q) <= n: return q
    cut = q[:n]; i = max(cut.rfind(". "), cut.rfind("다. "), cut.rfind("요. "))
    return (cut[:i+1] if i > n // 2 else cut[:n-1].rstrip() + "…")
NORM = re.compile(r"should|must|never|always|instead|rather|prefer|avoid|don'?t|do not|raise|해야|않|말고|대신|항상|금지|원칙|규칙|통일|일관|합니다|하세요|필요|좋을 것|권장|지양", re.I)
def ev_score(e):
    q = e.get("quote") or ""
    return (len(NORM.findall(q)) > 0) * 2 + min(len(q), 110) / 110 - (len(q) < 30) * 2 - (e.get("weak") and 5 or 0)
def ev_lines(r):
    ev = sorted(r.get("evidence", []), key=ev_score, reverse=True)
    seen_pr, seen_user, out = set(), set(), []
    # best quote first; second quote from a different PR, preferring a different user
    for e in ev:
        if e.get("pr") in seen_pr: continue
        if out and e.get("user") in seen_user and any(x.get("pr") not in seen_pr and x.get("user") not in seen_user and ev_score(x) > ev_score(e) - 1 for x in ev): continue
        seen_pr.add(e.get("pr")); seen_user.add(e.get("user"))
        who = f" ({e['user'].replace('rebel-','')})" if e.get("user") else ""
        out.append(f"[#{e['pr']}]({REPO}{e['pr']}){who}: “{clean(e['quote'])}”")
        if len(out) == 2: break
    more = r.get("n_prs", 0) - len(out)
    return out, more
def code_line(r):
    g = r.get("grounding"); c = r.get("compliance") or {}
    if not g: return None
    v = g["verdict"]; base = f"**코드 대조** {STATUS_KO.get(v, v)} — 준수 {c.get('compliant')} / 위반 {c.get('violations_total')}"
    if v in ("aspirational", "rejected", "interview") and g.get("reason"): base += f". {clean(g['reason'], 160)}"
    return base
def rule_block(r):
    s = short.get(r["id"], {}); st = r["status"]
    L = [f"### {r['id']} · {STATUS_KO.get(st, st)}", "", f"**룰** {s.get('short') or r['statement']}"]
    if s.get("detail"): L.append(f"**조건** {s['detail']}")
    ev, more = ev_lines(r)
    L.append("**근거** " + ev[0] if ev else "**근거** (없음)")
    for x in ev[1:]: L.append(f"　　　{x}")
    if more > 0: L.append(f"　　　외 PR {more}건")
    cl = code_line(r)
    if cl: L.append(cl)
    if st == "interview" and (s.get("question") or r.get("open_question")): L.append(f"**질문** {s.get('question') or r['open_question']}")
    L.append("**응답** [ ] 맞다 　[ ] 틀리다 　[ ] 조건부로 맞다 　[ ] 필요 없는 룰이다 　[ ] 모른다 　[ ] 기억나지 않는다 　· 메모: ")
    L.append("")
    return L
index = [f"# optimum-rbln 룰 리뷰 시트\n", f"생성일 {datetime.date.today()}. 출처: PR 804건·리뷰 코멘트 1,037건·대화 코멘트 375건에서 채굴한 뒤 코드(dev tip)와 대조한 룰. 각 룰은 **룰 · 근거(원문 인용, PR 링크) · 코드 대조 · 응답** 네 줄이다. 판단형은 코드로 검사할 수 없어 대조를 건너뛴 룰이다. 리뷰 방법은 [README.md](README.md).\n",
         "| 영역 | 파일 | 결정 필요 | 코드 일치 | 코드 불일치 | 판단형 | 폐기 제안 | 합계 |", "|---|---|---|---|---|---|---|---|"]
ORDER = {"interview": 0, "aspirational": 1, "verified": 2, "supported": 3, "rejected": 4}
tot = collections.Counter()
for area, label in AREAS:
    rs = [r for r in rules if r["area"] == area and r["status"] != "candidate"]
    if not rs: continue
    rs.sort(key=lambda r: (ORDER.get(r["status"], 9), -r.get("n_prs", 0), r["id"]))
    c = collections.Counter(r["status"] for r in rs); tot.update(c)
    fn = area.replace("/", "_") + ".md"
    L = [f"# {label} — 룰 {len(rs)}개\n", f"결정 필요 {c.get('interview',0)} · 코드 일치 {c.get('verified',0)} · 코드 불일치 {c.get('aspirational',0)} · 판단형 {c.get('supported',0)} · 폐기 제안 {c.get('rejected',0)}. 순서는 결정이 필요한 것부터. 각 룰의 **응답** 줄에서 하나를 `[x]`로 표시하고, 틀리거나 조건부면 **메모**에 한 줄 적어 주세요.\n"]
    cur = None
    for r in rs:
        if r["status"] != cur:
            cur = r["status"]; L.append(f"## {STATUS_KO[cur]}\n")
        L += rule_block(r)
    open(f"{OUT}/{fn}", "w").write("\n".join(L))
    index.append(f"| {label} | [{fn}]({fn}) | {c.get('interview',0)} | {c.get('verified',0)} | {c.get('aspirational',0)} | {c.get('supported',0)} | {c.get('rejected',0)} | {len(rs)} |")
index.append(f"| **합계** | | {tot.get('interview',0)} | {tot.get('verified',0)} | {tot.get('aspirational',0)} | {tot.get('supported',0)} | {tot.get('rejected',0)} | {sum(tot.values())} |")
# candidates appendix: one line each
cands = [r for r in rules if r["status"] == "candidate"]
A = [f"# 부록 — 근거가 PR 1건뿐인 후보 {len(cands)}개\n", "리뷰 대상이 아닙니다. 팀이 \"이건 실제 룰이다\"라고 지목하면 본 시트로 올립니다. 한 줄씩만 적습니다.\n"]
for area, label in AREAS:
    cs = [r for r in cands if r["area"] == area]
    if not cs: continue
    A.append(f"## {label}\n")
    for r in cs:
        e = (r.get("evidence") or [{}])[0]; pr = e.get("pr")
        A.append(f"- `{r['id']}` {clean(r['statement'], 110)}" + (f" — [#{pr}]({REPO}{pr})" if pr else ""))
    A.append("")
open(f"{OUT}/appendix_candidates.md", "w").write("\n".join(A))
open(f"{OUT}/INDEX.md", "w").write("\n".join(index) + "\n")
print(json.dumps(dict(tot), ensure_ascii=False), "candidates", len(cands))
