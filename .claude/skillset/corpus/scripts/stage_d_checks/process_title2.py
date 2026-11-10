import json, re, sys
from collections import Counter
rows=[json.loads(l) for l in open(sys.argv[1])]
s=[r for r in rows if (r.get('mergedAt') or '')>='2025-01-01']
TYPES={'release','model','performance','refactor','doc','dependency','other'}
# amannn/action-semantic-pull-request v5 default headerPattern
HP=re.compile(r'^(\w*)(?:\(([\w$.\-*/ ]*)\))?: (.*)$')
bad=[]
for r in s:
    m=HP.match(r['title'])
    if not (m and m.group(1) in TYPES and m.group(3).strip()): bad.append(r)
print('merged since 2025-01-01:',len(s),'compliant',len(s)-len(bad),'violations',len(bad))
for r in sorted(bad,key=lambda r:r['mergedAt']): print(r['number'],r['base'],r['mergedAt'][:10],r['title'][:90])
# model bug fixes
fix=[]
for r in s:
    b=r['body'] or ''
    if re.search(r'^\s*- \[[xX]\] Bug fix',b,re.M):
        src=[f for f in r['files'] if f.startswith('src/')]
        if src and all(re.match(r'src/optimum/rbln/(transformers|diffusers)/(models|pipelines)/',f) for f in src):
            m=HP.match(r['title']); fix.append((r['number'],m.group(1) if m else r['title'].split(':')[0]))
c=Counter(t for _,t in fix); print('model-only bugfix PRs',len(fix),c)
print([n for n,t in fix if t!='model'])
