import json, re, sys
from collections import Counter
rows=[json.loads(l) for l in open(sys.argv[1])]
s=[r for r in rows if (r.get('mergedAt') or '')>='2025-01-01']
def section(body, name):
    m=re.search(r'^##\s*'+name+r'[^\n]*\n(.*?)(?=^#{1,2}\s|\Z)', body, re.M|re.S)
    return m.group(1) if m else None
def strip(t): return re.sub(r'<!--.*?-->','',t or '',flags=re.S).strip()
res=Counter(); per=[]
for r in s:
    b=r['body'] or ''
    secs={k:section(b,k) for k in ['Type of Change','Changes Overview','Motivation','Related Issues','Verification']}
    tmpl = secs['Type of Change'] is not None
    allsec = all(secs[k] is not None for k in ['Type of Change','Changes Overview','Motivation','Related Issues'])
    checked = re.findall(r'^- \[[xX]\] (.*)$', secs['Type of Change'] or '', re.M)
    other_bad = any(c.strip().startswith('Other') and re.sub(r'^Other(\s*\(please describe\))?:?','',c.strip()).strip()=='' for c in checked)
    motiv = strip(secs['Motivation'])
    bot = r['author'] in ('app/github-actions','github-actions','rebel-bot') or r['title'].startswith('dependency: Update dependency')
    per.append(dict(n=r['number'],base=r['base'],author=r['author'],title=r['title'],tmpl=tmpl,allsec=allsec,nchk=len(checked),other_bad=other_bad,motiv=bool(motiv),verif=secs['Verification'] is not None,bot=bot,date=r['mergedAt'][:10]))
json.dump(per,open(sys.argv[2],'w'))
H=[p for p in per if not p['bot']]
print('total',len(per),'nonbot',len(H))
for k in ['tmpl','allsec','motiv','verif','other_bad']:
    print(k, sum(p[k] for p in H))
print('nchk dist (tmpl only)',Counter(p['nchk'] for p in H if p['tmpl']))
full=[p for p in H if p['allsec'] and p['nchk']==1 and not p['other_bad'] and p['motiv']]
print('full compliance',len(full))
print(Counter(p['author'] for p in per if p['bot']))
# by half-year
for y in ['2025-0','2025-1','2026-0','2026-1']:
    sub=[p for p in H if p['date'][:6]==y]
    if sub: print(y,len(sub),sum(p['tmpl'] for p in sub),sum(p['allsec'] and p['nchk']==1 and not p['other_bad'] and p['motiv'] for p in sub))
