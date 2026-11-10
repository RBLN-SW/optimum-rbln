import json, re, sys, os
S = sys.argv[1]
out = os.path.join(S, "stage_b", "chunks")
TEMPLATE_LINES = [
 "# Pull Request Description", "> **⚠️ Important: Branch Target**",
 "> - **New features, enhancements, and non-critical fixes**: Merge to `dev` branch",
 "> - **Critical hotfixes only**: Merge to `main` branch (must also merge to `dev`)",
 "> Please ensure you've selected the correct base branch before submitting!",
 "<!-- Mark the appropriate option with an [x] -->", "<!-- Provide a brief summary of the changes in this PR -->",
 "<!-- Explain why this change is necessary and what problem it solves -->",
 "<!-- Link any related issues here using the syntax: Closes #123, Fixes #456 -->",
]
def clean(body):
    if not body: return ""
    lines = []
    for ln in body.splitlines():
        t = ln.strip()
        if t in TEMPLATE_LINES or t == ">" : continue
        if re.match(r"^- \[ \] ", t): continue          # unchecked template boxes
        lines.append(ln.rstrip())
    txt = "\n".join(lines)
    txt = re.sub(r"\n{3,}", "\n\n", txt).strip()
    return txt
# PRs
prs = [json.loads(l) for l in open(os.path.join(S, "corpus", "prs.jsonl"))]
kept, dropped = [], 0
for p in prs:
    body = clean(p.get("body"))
    auto = p["author"] in ("renovate", "renovate[bot]", "github-actions", "github-actions[bot]", "app/renovate") \
        or p["title"].startswith(("chore(deps)", "fix(deps)", "Update dependency", "dependency(bot)"))
    if auto or len(body) < 120:
        dropped += 1; continue
    files = [f for f in p["files"] if f.startswith("src/")][:12]
    kept.append(f"=== PR #{p['number']} | {p['title']} | author={p['author']} | base={p['base']} | merged={p['mergedAt']}\nfiles: {', '.join(files)}\n{body}\n")
print(f"PRs kept {len(kept)} dropped {dropped}; chars {sum(map(len,kept))}")
def write_chunks(items, prefix, limit):
    buf, n, size = [], 0, 0
    for it in items:
        if size + len(it) > limit and buf:
            open(os.path.join(out, f"{prefix}_{n:02d}.txt"), "w").write("\n".join(buf)); n += 1; buf, size = [], 0
        buf.append(it); size += len(it)
    if buf: open(os.path.join(out, f"{prefix}_{n:02d}.txt"), "w").write("\n".join(buf)); n += 1
    return n
print("pr chunks:", write_chunks(kept, "pr", 110_000))
# issue / PR conversation comments
ics = [json.loads(l) for l in open(os.path.join(S, "corpus", "issue_comments.jsonl"))]
items = []
for c in ics:
    b = (c["body"] or "").strip()
    if c["user"].endswith("[bot]") or c["user"] in ("github-actions", "renovate", "Copilot") or len(b) < 40: continue
    if b.startswith(("## ", "<!--")) and "Codecov" in b: continue
    items.append(f"=== PR/issue #{c['issue']} | user={c['user']} | {c['created']}\n{b}\n")
print(f"conversation comments kept {len(items)} chars {sum(map(len,items))}")
print("ic chunks:", write_chunks(items, "ic", 110_000))
