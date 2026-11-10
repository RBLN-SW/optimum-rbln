import json, re, sys
from collections import Counter
rows=[json.loads(l) for l in open(sys.argv[1])]
s=[r for r in rows if (r.get('mergedAt') or '')>='2025-01-01']
def section(body, name):
    m=re.search(r'^##\s*'+name+r'[^\n]*\n(.*?)(?=^#{1,2}\s|^----|\Z)', body, re.M|re.S)
    return m.group(1) if m else None
def strip(t): return re.sub(r'<!--.*?-->','',t or '',flags=re.S).strip()
out=[]
for r in s:
    b=(r['body'] or '').replace('\r\n','\n')
    mirror=bool(re.search(r'\(#\d+\)\s*$', r['title']))
    dep=r['title'].startswith('dependency: Update dependency')
    release=r['title'].lower().startswith('release')
    toc=section(b,'Type of Change')
    checked=[c.strip() for c in re.findall(r'^\s*- \[[xX]\] (.*)$', toc or '', re.M)]
    other_bad=any(c.startswith('Other') and re.sub(r'^Other(\s*\(please describe\))?:?','',c).strip()=='' for c in checked)
    d=dict(n=r['number'],base=r['base'],date=r['mergedAt'][:10],title=r['title'],author=r['author'],
       mirror=mirror,dep=dep,release=release,tmpl=toc is not None,
       secs=all(section(b,k) is not None for k in ['Type of Change','Changes Overview','Motivation','Related Issues']),
       nchk=len(checked),checked=checked,other_bad=other_bad,
       overview=bool(strip(section(b,'Changes Overview'))),motiv=bool(strip(section(b,'Motivation'))),
       verif=re.search(r'^#+\s*(Verification|Test|Testing)',b,re.M|re.I) is not None)
    d['ok']=d['secs'] and d['nchk']==1 and not d['other_bad'] and d['motiv']
    out.append(d)
json.dump(out,open(sys.argv[2],'w'))
H=[d for d in out if not d['mirror'] and not d['dep']]
print('all',len(out),'mirror',sum(d['mirror'] for d in out),'dep',sum(d['dep'] for d in out),'human-authored',len(H))
for k in ['tmpl','secs','overview','motiv','verif','other_bad','ok']: print(k,sum(d[k] for d in H))
print('nchk',Counter(d['nchk'] for d in H if d['tmpl']))
print(Counter(d['base'] for d in H).most_common(4))
for y in ['2025-0','2025-1','2026-0','2026-1']:
    sub=[d for d in H if d['date'][:6]==y]
    print(y,len(sub),'tmpl',sum(d['tmpl'] for d in sub),'1chk',sum(d['nchk']==1 for d in sub),'motiv',sum(d['motiv'] for d in sub),'ok',sum(d['ok'] for d in sub))
