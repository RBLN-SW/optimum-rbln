import json, re, sys
rows=[json.loads(l) for l in open(sys.argv[1])]
s=[r for r in rows if (r['mergedAt'] or '')>='2025-01-01']
DEP=re.compile(r'(depends? on|dependent on|stacked|based on (#|https://github\.com/\S+/pull/)|on top of (#|https://github\.com/\S+/pull/)|need(s)? #\d+|requires? #\d+|blocked by|merged? (after|before|first)|after #\d+ (is |gets )?merge|must land|land(s|ed)? first|retarget)',re.I)
REF=re.compile(r'#\d+|github\.com/[\w-]+/[\w.-]+/pull/\d+')
def clean(b):
    b=(b or '').replace('\r\n','\n'); b=re.sub(r'<!--.*?-->','',b,flags=re.S); return b.split('\n----\n# Conventional commit')[0]
out=[]
for r in s:
    if re.search(r'\(#\d+\)\s*$',r['title']): continue
    b=clean(r['body']); stacked=r['base'] not in ('dev','main')
    m=DEP.search(b)
    if stacked or m:
        out.append(dict(n=r['number'],base=r['base'],stacked=stacked,dep=m.group(0) if m else None,ref=bool(REF.search(b)),
            snip=re.sub(r'\s+',' ',b[max(0,m.start()-60):m.end()+80]) if m else re.sub(r'\s+',' ',b[:0]),title=r['title'][:70]))
json.dump(out,open(sys.argv[2],'w'),ensure_ascii=False,indent=0)
for o in out: print(o['n'],o['base'],'stacked' if o['stacked'] else '-', 'dep' if o['dep'] else 'NODEP','ref' if o['ref'] else 'NOREF','|',o['title'],'|',o['snip'][:160])
