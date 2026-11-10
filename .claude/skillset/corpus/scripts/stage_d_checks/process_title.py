import json, re, sys
C=sys.argv[1]
rows=[json.loads(l) for l in open(C)]
s=[r for r in rows if (r.get('mergedAt') or '')>='2025-01-01']
TYPES='release|model|performance|refactor|doc|dependency|other'
rx=re.compile(r'^(?:'+TYPES+r')(?:\([^)]*\))?!?: \S')
bad=[r for r in s if not rx.match(r['title'])]
print(len(s), len(s)-len(bad), len(bad))
from collections import Counter
print(Counter(re.match(r'^([\w-]*)',r['title']).group(1) for r in s).most_common(20))
for r in sorted(bad,key=lambda r:r['mergedAt']): print(r['number'], r['base'], r['mergedAt'][:10], r['title'][:100])
