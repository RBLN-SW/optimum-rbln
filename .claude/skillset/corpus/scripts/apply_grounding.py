"""Apply stage-D results (stage_d/out/*.json) to rules.yaml; write mining/grounding_report.md + mining/violations.md."""
import json, sys, os, glob, collections, yaml, datetime
S, W = sys.argv[1], sys.argv[2]
RY = f"{W}/.claude/skillset/rules.yaml"
text = open(RY).read()
hdr = "".join(l for l in text.splitlines(True) if l.startswith("#"))
rules = yaml.safe_load(text); byid = {r["id"]: r for r in rules}
results, problems = {}, []
REF = {"batch_config.json": "main 88ea9736 (2026-07-31) — superseded by out_dev", "batch_lifecycle_api.json": "origin/dev ae60d5fd (2026-10-08)",
       "batch_test_doc.json": "main 88ea9736 — superseded by out_dev", "batch_dependency.json": "dev tip 74e97fa4 (2026-10-21)",
       "batch_process.json": "PR corpus 2025-01..2026-10 + main .github (CI-dependent ids superseded by out_dev)", "batch_style.json": "dev tip 74e97fa4 (2026-10-21)"}
REF_DEV = "dev tip 74e97fa4 (2026-10-21)"
for f in sorted(glob.glob(f"{S}/stage_d/out/batch_*.json")) + sorted(glob.glob(f"{S}/stage_d/out_dev/batch_*.json")):
    try: items = json.load(open(f))
    except Exception as e: problems.append(f"{f}: {e}"); continue
    for it in items:
        if it.get("id") not in byid: problems.append(f"{f}: unknown id {it.get('id')}"); continue
        tag = os.path.basename(f) + (" [dev]" if "/out_dev/" in f else "")
        it["_ref"] = REF_DEV if "/out_dev/" in f else REF.get(os.path.basename(f), "?")
        results[it["id"]] = (tag, it)   # out_dev loaded last, so it overrides
VALID = {"verified", "aspirational", "interview", "rejected", "unverifiable"}
for rid, (src, it) in results.items():
    r = byid[rid]; v = it.get("verdict")
    if v not in VALID: problems.append(f"{rid}: bad verdict {v}"); continue
    chk = it.get("check") or {}
    r["check"] = {"kind": chk.get("kind"), "description": chk.get("description"), "command": chk.get("command")}
    r["compliance"] = {"compliant": it.get("compliant"), "violations_total": it.get("violations_total")}
    r["violations_known"] = [{"path": x.get("path"), "line": x.get("line"), "snippet": x.get("snippet"), "note": x.get("note")} for x in (it.get("violations") or [])][:25]
    r["grounding"] = {"verdict": v, "reason": it.get("verdict_reason"), "date": str(datetime.date.today()), "batch": src, "measured_on": it.get("_ref")}
    if it.get("rule_refinement"): r["refined_statement"] = it["rule_refinement"]
    if it.get("suggested_test"): r["suggested_test"] = it["suggested_test"]
    if v == "unverifiable":
        r["kind"] = "judgment"            # the check could not be written; status stays 'supported'
    elif v == "interview":
        r["status"] = "interview"
        if not r.get("open_question"): r["open_question"] = it.get("verdict_reason")
    else:
        r["status"] = v                   # verified | aspirational | rejected
# header: add the new states
hdr = hdr.replace("# status: supported (≥2 distinct PRs with prose evidence) | candidate (1 PR, or title/path-only evidence) | interview (contradictory evidence or open question)",
 "# status: supported (≥2 distinct PRs with prose evidence) | candidate (1 PR, or title/path-only evidence) | interview (contradictory evidence or open question)\n#         after code grounding (stage D): verified (code follows it; violations are tech debt) | aspirational (team intent, code does not yet follow) | rejected (code consistently differs and evidence is thin/stale)")
open(RY, "w").write(hdr + yaml.safe_dump(rules, allow_unicode=True, sort_keys=False, width=110))
# reports
C = collections.Counter
ver = C(it["verdict"] for _, it in results.values())
L = [f"# 2단계 코드 대조 결과 (stage D)\n", f"작성일 {datetime.date.today()}. 대상: mechanical+supported 룰 {len(results)}개. 각 룰을 재현 가능한 코드 질의로 번역해 현재 코드의 준수/위반을 셌다. 질의는 `rules.yaml`의 `check.command`에 있고, 위반 위치는 `violations_known`에 있다.\n",
     "## 1. 판정 집계\n", "| 판정 | 수 | 뜻 |\n|---|---|---|",
     f"| verified | {ver.get('verified',0)} | 코드가 따른다. 남은 위반은 기술부채 |",
     f"| aspirational | {ver.get('aspirational',0)} | 팀 의도는 분명하나 코드가 아직 안 따른다 |",
     f"| interview | {ver.get('interview',0)} | 코드에 두 패턴이 공존하거나 적용 경계를 코드로 정할 수 없다 |",
     f"| rejected | {ver.get('rejected',0)} | 코드가 일관되게 반대로 하고 근거가 얇거나 낡았다 |",
     f"| unverifiable | {ver.get('unverifiable',0)} | 검사로 번역 불가 → kind를 judgment로 바꿈 |", ""]
L.append("## 2. 룰별 결과\n"); L.append("| id | 영역 | 판정 | 준수 | 위반 | 검사 종류 | 룰 |\n|---|---|---|---|---|---|---|")
for rid, (src, it) in sorted(results.items(), key=lambda kv: ({"verified":0,"aspirational":1,"interview":2,"rejected":3,"unverifiable":4}[kv[1][1]["verdict"]], kv[0])):
    r = byid[rid]; L.append(f"| {rid} | {r['area']} | {it['verdict']} | {it.get('compliant')} | {it.get('violations_total')} | {(it.get('check') or {}).get('kind')} | {r['statement'][:80]} |")
L.append("\n## 3. 판정 사유와 보정된 룰\n")
for rid, (src, it) in sorted(results.items()):
    r = byid[rid]
    L.append(f"### {rid} — {it['verdict']}\n")
    L.append(f"- 룰: {r['statement']}")
    L.append(f"- 검사: {(it.get('check') or {}).get('description')}")
    L.append(f"- 사유: {it.get('verdict_reason')}")
    if it.get("rule_refinement"): L.append(f"- **보정된 룰:** {it['rule_refinement']}")
    if it.get("suggested_test"): L.append(f"- 테스트 제안: {it['suggested_test']}")
    L.append("")
open(f"{W}/.claude/skillset/mining/grounding_report.md", "w").write("\n".join(L) + "\n")
V = [f"# 알려진 위반 목록 (stage D, {datetime.date.today()})\n", "룰별로 현재 코드에서 룰을 어기는 위치. 상태가 verified인 룰의 위반은 기술부채 이슈 후보이고, aspirational인 룰의 위반은 룰 채택 시 함께 고쳐야 할 범위다.\n"]
for rid, (src, it) in sorted(results.items(), key=lambda kv: -(kv[1][1].get("violations_total") or 0)):
    vs = it.get("violations") or []
    if not vs: continue
    r = byid[rid]
    V.append(f"## {rid} ({it['verdict']}) — 위반 {it.get('violations_total')} / 준수 {it.get('compliant')}\n"); V.append(f"{r['statement']}\n")
    for x in vs[:25]:
        loc = f"{x.get('path')}" + (f":{x.get('line')}" if x.get("line") else "")
        V.append(f"- `{loc}` — {x.get('note')}" + (f"\n    ```\n    {x.get('snippet')}\n    ```" if x.get("snippet") else ""))
    if (it.get("violations_total") or 0) > len(vs): V.append(f"- … 외 {it['violations_total'] - len(vs)}건 (검사 명령으로 전체 목록 재현 가능)")
    V.append("")
open(f"{W}/.claude/skillset/mining/violations.md", "w").write("\n".join(V) + "\n")
print(json.dumps({"applied": len(results), "verdicts": dict(ver), "problems": problems}, ensure_ascii=False, indent=1))
