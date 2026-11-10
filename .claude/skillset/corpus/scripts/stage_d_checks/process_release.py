import json, re, sys
rows=[json.loads(l) for l in open(sys.argv[1])]
M={p['number']:p for p in json.load(open(sys.argv[2]))}  # gh pr list --base main --state merged (headRefName)
s=[r for r in rows if (r['mergedAt'] or '')>='2025-01-01']
cand=[r for r in s if r['title'].lower().startswith('release') or (r['base']=='main' and M.get(r['number'],{}).get('headRefName')=='dev')]
TRX=re.compile(r'^release: v\d+\.\d+\.\d+(\.post\d+)?$')
out=[]
for r in sorted(cand,key=lambda r:r['number']):
    head=M.get(r['number'],{}).get('headRefName')
    b=(r['body'] or '')
    rel=bool(re.search(r'^\s*- \[[xX]\] Release',b,re.M))
    d=dict(n=r['number'],date=r['mergedAt'][:10],base=r['base'],head=head,title=r['title'],
           base_ok=r['base']=='main',head_ok=head=='dev',title_ok=bool(TRX.match(r['title'].strip())),release_checked=rel)
    d['ok']=all([d['base_ok'],d['head_ok'],d['title_ok'],d['release_checked']])
    out.append(d); print(d)
json.dump(out,open(sys.argv[3],'w'))
