import json, re, sys
rows=[json.loads(l) for l in open(sys.argv[1])]
s=[r for r in rows if (r['mergedAt'] or '')>='2025-01-01' and not re.search(r'\(#\d+\)\s*$',r['title'])]
TPL=re.compile(r'^\s*- \[[ xX]\] [^\n]*\((non-breaking change|fix or feature that would|dev → main)[^\n]*$',re.M)
def clean(b):
    b=(b or '').replace('\r\n','\n'); b=re.sub(r'<!--.*?-->','',b,flags=re.S)
    b=b.split('\n----\n# Conventional commit')[0]
    return TPL.sub('',b)
SIG=re.compile(r'(?<!non-)\bbreaking\b|backward[- ]incompat|backwards[- ]incompat|no longer (accept|support|work)|BC[- ]break|not backward compatible|is removed|are removed|removed? (the )?(deprecated|public)|rename[sd]? .{0,40}(argument|parameter|kwarg|class|key)',re.I)
chk=[]; sig=[]
for r in s:
    b=(r['body'] or '').replace('\r\n','\n')
    checked=bool(re.search(r'^\s*- \[[xX]\] Breaking change',b,re.M))
    m=SIG.search(clean(b)+'\n'+r['title'])
    if checked: chk.append(r['number'])
    if m: sig.append((r['number'],checked,m.group(0),r['title'][:70]))
print('breaking checked',len(chk),sorted(chk))
print('signal PRs',len(sig),'of which checked',sum(c for _,c,_,_ in sig))
for x in sig: print(x)
