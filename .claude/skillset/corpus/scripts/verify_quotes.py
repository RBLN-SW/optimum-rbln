import json, sys
# usage: verify_quotes.py <extraction.json> <chunk.txt>  -> prints mismatching quotes, exit 1 if any
d = json.load(open(sys.argv[1])); txt = open(sys.argv[2]).read()
bad = 0
def chk(q, where):
    global bad
    if q and q not in txt:
        bad += 1; print("MISMATCH", where, repr(q[:100]))
for i, c in enumerate(d.get("candidates", [])):
    for e in c.get("evidence", []): chk(e.get("quote"), f"candidate[{i}]")
for i, p in enumerate(d.get("procedures", [])):
    for e in p.get("evidence", []): chk(e.get("quote"), f"procedure[{i}]")
for i, x in enumerate(d.get("decisions", [])): chk(x.get("quote"), f"decision[{i}]")
for i, x in enumerate(d.get("debates", [])):
    for e in x.get("evidence", []): chk(e.get("quote"), f"debate[{i}]")
print("mismatches:", bad); sys.exit(1 if bad else 0)
