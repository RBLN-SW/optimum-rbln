import json, subprocess, sys
rows=[json.loads(l) for l in open(sys.argv[1])]
s=[r for r in rows if (r['mergedAt'] or '')>='2025-01-01']
main={p['number']:p for p in json.load(open(sys.argv[2]))}
cand=[r for r in s if r['title'].lower().startswith('release') or (r['base']=='main' and main.get(r['number'],{}).get('headRefName')=='dev')
      or (r['base']=='dev' and any(k in r['title'].lower() for k in ['sync with main','backmerge','back-merge','merge main','main into dev','main to dev','advance main']))]
out=[]
for r in sorted(cand,key=lambda r:r['number']):
    j=json.loads(subprocess.check_output(['gh','pr','view',str(r['number']),'-R','RBLN-SW/optimum-rbln','--json','headRefName,mergeCommit']))
    sha=j['mergeCommit']['oid']
    par=int(subprocess.check_output(['gh','api',f'repos/RBLN-SW/optimum-rbln/commits/{sha}','--jq','.parents|length']).strip())
    out.append(dict(n=r['number'],date=r['mergedAt'][:10],base=r['base'],head=j['headRefName'],title=r['title'],sha=sha[:8],parents=par))
    print(out[-1])
json.dump(out,open(sys.argv[3],'w'))
