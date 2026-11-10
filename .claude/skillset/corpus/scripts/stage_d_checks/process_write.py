import json, re
SP='<SCRATCH>'
S=SP+'/stage_d/scripts'; C=SP+'/corpus/prs.jsonl'
R={json.loads(l)['number']:json.loads(l) for l in open(C)}
def t(n): return R[n]['title'][:120]
out=[]

# PRC-008
decl_stacked=[763,756,739,738,725,707,704,663,638,563,367]
decl_notstacked=[(668,'Stacked on #662 라 적었지만 base는 처음부터 dev (662 미머지 상태)'),(360,'need #359 & #356 선언, base는 main→dev; 선행 브랜치 아님')]
undecl=[640,457,401,277,255,251,228,191,189,187,184,167,129,121,119,116,114,111,107,106,73,52,49]
v=[dict(path=f'PR #{n}',line=None,snippet=t(n),note=note) for n,note in decl_notstacked]
v+=[dict(path=f'PR #{n}',line=None,snippet=t(n)+f" (base={R[n]['base']})",note='선행 PR 브랜치에 stack했으나 본문에 선행 PR 번호 없음') for n in undecl]
out.append(dict(id='PRC-008',
 check=dict(kind='corpus',description='2025-01 이후 머지된 PR 중 미머지 선행 PR에 의존하는 PR(다른 PR의 head 브랜치를 base로 둔 PR + 본문에 depends/stacked/based on/need #N 선언이 있는 PR)을 모아, base가 선행 PR 브랜치인지와 본문에 선행 PR 번호가 있는지를 본다; 둘 중 하나라도 빠지면 위반.',
  command=f"python3 {S}/process_stacked.py {C} {S}/process_stacked_out.json && gh pr list -R RBLN-SW/optimum-rbln --state all --limit 1000 --json number,headRefName,baseRefName,state,mergedAt,createdAt > {S}/process_allprs.json  # then map each stacked PR's base to the PR whose headRefName equals it; base-change history: gh api graphql timelineItems(itemTypes:[BASE_REF_CHANGED_EVENT])"),
 compliant=len(decl_stacked), violations=v[:25], violations_total=len(v), verdict='aspirational',
 verdict_reason='의존 PR 36건 중 base=선행 브랜치이면서 본문에 선행 PR을 밝힌 것은 11건(31%)이다. 2026-08 이후 머지분은 10건 중 9건이 준수(668만 base=dev)하지만, 2025~2026-07의 feature-branch 하위 PR 23건은 본문에 선행 PR을 적지 않았다. 머지 순서는 13건 중 3건(638·738의 retarget, 668의 merge after)만, 순서가 어긋났을 때의 대응은 692 한 건만 명시해 이 부분은 근거가 약하다.',
 rule_refinement='미머지 PR 위에 쌓는 PR은 base를 선행 PR의 head 브랜치로 두고 본문 첫 줄에 "Stacked on #N"을 적는다(머지 후 GitHub가 dev로 retarget). 머지 순서와 순서가 어긋났을 때의 대응(닫기 등)은 main/release 브랜치에 out-of-band로 들어가는 PR에만 요구한다.',
 suggested_test=None))

# PRC-003
D=json.load(open(S+'/process_body_out.json'))
H=[d for d in D if not d['mirror'] and not d['dep']]
viol=sorted([d for d in H if not d['ok']],key=lambda d:-d['n'])
def why(d):
    w=[]
    if not d['secs']: w.append('템플릿 절 누락')
    if d['tmpl'] and d['nchk']!=1: w.append(f"유형 {d['nchk']}개 체크")
    if d['other_bad']: w.append('Other 설명 없음')
    if not d['motiv']: w.append('Motivation 비어 있음')
    return ', '.join(w)
out.append(dict(id='PRC-003',
 check=dict(kind='corpus',description='2025-01 이후 머지된 사람 작성 PR(자동 dependency 봇·sync 미러 제외 588건)의 본문에서 템플릿 절(Type of Change/Changes Overview/Motivation/Related Issues) 존재, Type of Change 체크 정확히 1개, Other 체크 시 설명 기재, Motivation 절 내용(주석 제외) 존재를 모두 만족하는지 본다.',
  command=f"python3 {S}/process_body2.py {C} {S}/process_body_out.json"),
 compliant=sum(d['ok'] for d in H),
 violations=[dict(path=f"PR #{d['n']}",line=None,snippet=d['title'][:120],note=why(d)) for d in viol[:25]],
 violations_total=len(viol), verdict='aspirational',
 verdict_reason='템플릿이 자동으로 채워지므로 절 헤더는 497/588건에 있고, 유형 체크도 템플릿을 쓴 509건 중 448건이 정확히 1개다. 그러나 Motivation을 실제로 쓴 PR은 67건(11%)뿐이고, Other를 체크한 145건 중 135건은 설명이 없어 전체 준수는 51건(8.7%)이다. 2026Q3에는 템플릿 없이 Summary/Why/Test plan 형식으로 쓴 PR이 40건으로 템플릿 사용(46건)과 맞먹는다. Verification 절은 템플릿에 없다.',
 rule_refinement='PR 본문에는 무엇을 바꿨는지와 왜 그렇게 했는지(원인이나 기대 효과)를 적는다. 템플릿을 쓸 때는 Type of Change를 하나만 체크하고, Other를 체크하면 내용을 적는다. 템플릿 대신 Summary/Why/Test plan 형식을 허용할지는 팀이 결정해야 한다.',
 suggested_test=None))

# PRC-001
HP=re.compile(r'^(\w*)(?:\(([\w$.\-*/ ]*)\))?: (.*)$'); TY={'release','model','performance','refactor','doc','dependency','other'}
s=[r for r in R.values() if (r['mergedAt'] or '')>='2025-01-01']
bad=sorted([r for r in s if not ((m:=HP.match(r['title'])) and m.group(1) in TY and m.group(3).strip())],key=lambda r:r['mergedAt'])
def note1(r):
    tl=r['title']
    if tl.startswith('Revert'): return 'GitHub 기본 Revert 제목, type 없음'
    if r['base'].startswith('main_test'): return '테스트용 PR, type 없음'
    m=HP.match(tl)
    if m and m.group(1) not in TY: return f"type '{m.group(1)}' 은 허용 목록 밖"
    return 'scope 괄호/형식이 CI headerPattern과 불일치'
out.append(dict(id='PRC-001',
 check=dict(kind='corpus',description='.github/workflows/pr-title-check.yaml(amannn/action-semantic-pull-request@v5, types=release/model/performance/refactor/doc/dependency/other)이 쓰는 기본 headerPattern으로 2025-01 이후 머지된 PR 제목을 검사한다; 매치하지 않거나 type이 목록 밖이면 위반. 보조로 src 변경이 모델 디렉터리뿐인 Bug fix PR의 type이 model인지 센다.',
  command=f"python3 {S}/process_title2.py {C}"),
 compliant=len(s)-len(bad),
 violations=[dict(path=f"PR #{r['number']}",line=None,snippet=r['title'][:120],note=note1(r)) for r in bad[:25]],
 violations_total=len(bad), verdict='verified',
 verdict_reason='머지된 631건 중 610건(96.7%)이 CI와 같은 정규식을 통과한다. 위반 21건은 Revert 자동 제목 4건, main_test 테스트 PR 5건, 그리고 2025~2026-02의 fix:/Dev/hotfix: 등이다. 제목 검사는 pr-title-check.yaml이 CI로 강제하지만, dev 브랜치 보호의 required check(4개 [OB] 체크)에는 들어 있지 않아 실패해도 머지된다. 모델만 고친 Bug fix 80건 중 68건(85%)이 model type이다.',
 rule_refinement=None,
 suggested_test='이미 CI(pr-title-check.yaml)가 검사하므로 pytest보다는 "Check PR title"을 dev 브랜치 보호의 required status check로 추가하는 것을 권한다. Revert 제목은 GitHub 기본값이므로 ignoreLabels 또는 headerPattern에 Revert 예외를 둘지 결정해야 한다.'))

# PRC-005
REL=json.load(open(S+'/process_release_out.json'))
ok=[d for d in REL if d['ok']]
vv=[]
for n,note in [(484,'head=release-0.10.0, 제목 "Release v0.10.0"'),(488,'post 릴리스: head=hotfix, Release 미체크, 제목 형식 다름'),(514,'head=release-v0.10.1, 제목 "release v0.10.1", Release 미체크'),(518,'head=release_v0.10.1, Release 미체크'),(522,'dev→main인데 제목 "Dev", Release 미체크'),(541,'head=release/v0.10.2 (dev 아님)'),(617,'post 릴리스: head=hotfix, Release 미체크, 제목에 부가설명')]:
    vv.append(dict(path=f'PR #{n}',line=None,snippet=t(n),note=note))
vv+= [dict(path='release v0.11.2 (2026-08-28)',line=None,snippet='gh release list: v0.11.2 Latest at the time; no base=main PR merged after 2026-08-26',note='main 삭제 후 dev→main 릴리스 PR 없이 출시'),
      dict(path='release v0.12.0 (2026-10-16)',line=None,snippet='PRs #786 #789 #805 base=release-v0.12.0 (cherry-pick)',note='release-v0.12.0 브랜치 cherry-pick 방식으로 출시')]
out.append(dict(id='PRC-005',
 check=dict(kind='corpus',description='2025-01 이후 제목이 release로 시작하거나 head=dev·base=main인 머지 PR에 대해 base=main, head=dev(gh headRefName), 제목이 ^release: vX.Y.Z(.postN)?$, Type of Change의 Release 체크를 모두 만족하는지 보고, 정식 릴리스(gh release list) 중 해당 PR이 없는 것도 위반으로 센다.',
  command=f"gh pr list -R RBLN-SW/optimum-rbln --state merged --base main --limit 1000 --json number,title,headRefName,mergedAt,mergeCommit > {S}/process_main_prs.json && python3 {S}/process_release.py {C} {S}/process_main_prs.json {S}/process_release_out.json && gh release list -R RBLN-SW/optimum-rbln --limit 40"),
 compliant=len(ok), violations=vv, violations_total=len(vv), verdict='interview',
 verdict_reason='버전 릴리스 PR 13건과 dev→main PR 522 가운데 7건(342·388·554·578·608·611·661)이 네 조건을 모두 만족하고, 2026-04~07 릴리스 5개는 모두 준수했다. 그러나 v0.10.x는 release-* 브랜치를 head로 썼고, post 릴리스(488·617)는 hotfix 브랜치에서 나왔다. v0.11.2와 v0.12.0은 main 없이 출시됐고, v0.12.0은 release-v0.12.0 브랜치에 cherry-pick하는 방식이었다. 팀에 물을 질문: main 삭제 이후 릴리스 절차가 "dev에서 release-vX.Y.Z를 잘라 태그"로 바뀐 것인가? 그렇다면 이 규칙을 폐기하고 새 절차로 다시 써야 하는가?',
 rule_refinement='(2026-07까지 유효) 정식 릴리스 PR은 head=dev, base=main, 제목 `release: vX.Y.Z`, Type of Change=Release로 만든다. post 릴리스는 hotfix 브랜치에서 낸다. 2026-08 이후의 절차는 팀 확인이 필요하다.',
 suggested_test=None))

# PRC-002
drafts=[(822,'model(gemma3): run the standalone Gemma3 text prefill causal, as transformers does'),(820,'model: add multi_token_decoder for speculative decoding'),(795,'model: [mode-2] enable ALBERT DeBERTa-v2 DINOv2 ConvNeXt and Swin classification'),(794,'model: [mode-1] improve existing model export and runtime contracts'),(768,'performance(qwen3.5): Run the GDN decode short conv as the causal_conv1d_update custom op'),(759,'other: test - cover _prepare_prefill_inputs on CPU for every runtime class'),(708,'performance(memory): size the KV cache from the joint allocation, not the sum'),(686,'other(ci): warm the HF cache once, then run pytest fully offline'),(680,'fix: keep torch cos/sin for the compile-time decoderonly rotary constants'),(656,'fix: quote rebel.* union annotations to avoid runtime TypeError'),(643,'performance(weight-free): link deferred weights from Hugging Face checkpoints'),(589,'model(whisper): restore [batch, seq] long-form generate output rank'),(487,'other: fix deprecation version logic'),(480,'other: support non-fp32 dtype for transformers/v0.5.0'),(479,'other: prepare Transformers/v5.0.0rc3'),(446,'test'),(424,'(DO NOT MERGE) batched attention mask'),(417,'model: change condition for dequant'),(398,'test bc ci'),(358,'other: backward compatibility ci'),(323,'Gpt oss sink'),(274,'[DRAFT] Detach Cache Update from Custom Attention OP'),(233,'other: Update Python version'),(210,'other(ci): pyright checking'),(192,'model: Faster compilation of cosmos transformers')]
out.append(dict(id='PRC-002',
 check=dict(kind='corpus',description='GitHub에서 draft 상태인 PR 전부(작업 중 PR의 대리 지표)와 제목에 WIP가 들어간 PR을 대조해, draft PR이 제목에 (WIP)를 붙였는지 본다; draft인데 (WIP)가 없으면 위반.',
  command="gh pr list -R RBLN-SW/optimum-rbln --state all --search 'draft:true' --limit 300 --json number,title,state,createdAt,author; gh pr list -R RBLN-SW/optimum-rbln --state all --search 'WIP in:title' --limit 100 --json number,title,state,isDraft"),
 compliant=4,
 violations=[dict(path=f'PR #{n}',line=None,snippet=s_,note='draft PR인데 제목에 (WIP) 없음') for n,s_ in drafts],
 violations_total=34, verdict='rejected',
 verdict_reason='draft PR 38건 중 (WIP)를 붙인 것은 4건(390·391·490·496)이고 모두 한 작성자(rebel-seinpark)다. 나머지 34건은 GitHub draft 상태만 쓰거나 [DRAFT]·(DO NOT MERGE)를 붙였다. 근거 3건도 같은 작성자이며, "(WIP) model: …" 제목은 pr-title-check.yaml의 headerPattern(^(\\w*)…)에 걸려 실패하므로 PRC-001과 충돌한다.',
 rule_refinement='작업 중인 PR은 제목에 표식을 달지 않고 GitHub Draft로 연다(제목은 conventional commit 형식 유지).',
 suggested_test=None))

# PRC-023
M=json.load(open(S+'/process_merge_out.json'))
extra=[dict(n=619,parents=2),dict(n=516,parents=2)]
allm=[d for d in M]+extra
okm=[d for d in allm if d['parents']==2]
badm=[d for d in M if d['parents']==1]
out.append(dict(id='PRC-023',
 check=dict(kind='script',description='2025-01 이후 머지된 릴리스·동기화·백머지 PR(제목이 release:로 시작, head=dev→base=main, 또는 main↔dev back-merge/sync)의 mergeCommit 부모 수를 gh로 확인한다; 부모가 1개(squash/rebase)이면 위반.',
  command=f"python3 {S}/process_merge.py {C} {S}/process_main_prs.json {S}/process_merge_out.json; for n in 619 516; do sha=$(gh pr view $n -R RBLN-SW/optimum-rbln --json mergeCommit --jq .mergeCommit.oid); gh api repos/RBLN-SW/optimum-rbln/commits/$sha --jq '.parents|length'; done; gh api repos/RBLN-SW/optimum-rbln --jq '{{allow_squash_merge,allow_merge_commit,allow_rebase_merge}}'"),
 compliant=len(okm),
 violations=[dict(path=f"PR #{d['n']}",line=None,snippet=d['title'][:120]+f" (merge {d['sha']}, parents=1)",note='squash로 머지되어 dev↔main 조상 관계가 끊김') for d in badm],
 violations_total=len(badm), verdict='verified',
 verdict_reason='대상 24건 중 21건이 부모 2개인 merge commit이다(619는 618의 merge commit으로 fast-forward). 위반은 2025년 릴리스 342·388과 511(같은 날 513에서 merge commit으로 다시 처리)이고, 2026-01 이후에는 사실상 지켜졌다. 다만 현재 저장소 설정은 allow_merge_commit=false(squash만 허용)이고 원격 main 브랜치는 404로 삭제된 상태라, 이 규칙은 main이 있던 시기에만 실행 가능했다.',
 rule_refinement='브랜치 간 조상 관계를 보존해야 하는 PR(release, sync, back-merge)은 merge commit으로 머지한다. 저장소가 squash만 허용하는 지금은 이런 PR을 머지하기 전에 관리자가 merge commit 허용을 일시적으로 켜야 하는지 확인해야 한다.',
 suggested_test=None))

# STY-015
out.append(dict(id='STY-015',
 check=dict(kind='grep',description='현행 CI(dev 브랜치의 .github/workflows, .buildkite, scripts/ci — 워크트리는 main@2026-07-31이라 gh api로 받아 검사)와 워크트리 .github/workflows에서 평문 IP·내부 호스트·클러스터 마운트 경로·토큰을 찾고, 자격증명이 ${{ secrets.* }}로, 엔드포인트·경로·이미지가 vars/에이전트 변수로 주입되는지와 필수 변수에 ${VAR:?} 가드가 있는지 센다.',
  command=f"D={S}/process_dev_ci; gh api 'repos/RBLN-SW/optimum-rbln/git/trees/dev?recursive=1' --jq '.tree[]|select(.type==\"blob\")|.path' | grep -E '^(\\.buildkite|\\.github/workflows|scripts/ci)/' | while read p; do mkdir -p \"$D/$(dirname $p)\"; gh api \"repos/RBLN-SW/optimum-rbln/contents/$p?ref=dev\" --jq .content | base64 -d > \"$D/$p\"; done; cd $D; grep -rnE '\\b[0-9]{{1,3}}(\\.[0-9]{{1,3}}){{3}}\\b' .; grep -rnoE 'https?://[^ \"]+' .; grep -rhoE '\\$\\{{\\{{ *secrets\\.[A-Za-z_]+' .github | wc -l; grep -rhoE '\"\\\\?\\$\\{{(DEVTOOLS_DOCKER_IMAGE|OPTIMUM_BC_BASE_PATH|UV_CACHE_PATH|UMD_PATH)\\}}\"' .buildkite scripts | wc -l; grep -rnE ':\\?' scripts/ci; cd -; grep -nE 'vars\\.|secrets\\.' .github/workflows/*"),
 compliant=54,
 violations=[dict(path='scripts/ci/slack-report.py (dev@74e97fa4)',line=22,snippet='API = "https://obedients-api.k8s.rebellions.in/v2/organizations"',note='내부 API 엔드포인트 평문 하드코딩 (PR 758이 범위 밖으로 남김)')],
 violations_total=1, verdict='verified',
 verdict_reason='현행 CI에서 자격증명 참조 9건은 모두 ${{ secrets.* }}이고, 이미지·클러스터 경로 34건과 사내 PyPI 호스트·자격증명 11건은 변수로 주입된다. 평문 IP나 클러스터 마운트 경로는 없다. 필수 변수는 bc-steps.sh(BC_BASE_PATH)와 install-rebel-compiler.sh(엔드포인트·계정 3개)에 ${VAR:?not set} 가드가 있다. 위반은 slack-report.py의 Obedients API URL 1건이고, 워크트리의 구 GHA 워크플로우도 vars.DEV_CONTAINER_IMAGE·vars.BC_BASE_PATH·secrets.*만 쓴다(/opt/venv 등은 컨테이너 내부 경로).',
 rule_refinement=None,
 suggested_test='tests/test_conventions.py::test_ci_has_no_plaintext_endpoints — .github/workflows/*.y*ml, .buildkite/**/*.yml, scripts/ci/* 를 읽어 (1) IPv4 리터럴, (2) *.rebellions.in / *.svc / *.cluster.local 호스트, (3) /mnt|/nfs|/data 로 시작하는 절대경로, (4) `token:`/`password:` 키에 ${{ secrets. }}나 ${VAR} 아닌 리터럴 값이 있으면 실패. allowlist: scripts/ci/slack-report.py 의 API 상수(변수화 전까지).'))

# PRC-004
bv=[(645,'refactor: 제거 창이 지난 deprecated API 일괄 제거, 유형 체크 없음'),(271,'"Remove legacy apis"인데 Other로 체크'),(605,'image_prefill_chunk_size 기본 버킷팅 제거(기본 동작 변경), Code refactoring 체크'),(128,'NPU 없을 때 create_runtimes 기본 동작 변경, Bug fix 체크'),(714,'본문에 "Behavior change": from_model 원본 HF 모델 forward 불가, Breaking 미체크')]
out.append(dict(id='PRC-004',
 check=dict(kind='corpus',description='2025-01 이후 머지 PR에서 Breaking change 체크 PR을 세고, 본문(템플릿 체크박스 줄 제외)·제목에 breaking/no longer/removed/rename/default 변경 신호가 있는 PR을 수작업으로 판정해 사용자 관점의 호환성 파괴인데 Breaking을 체크하지 않은 PR을 위반으로 센다.',
  command=f"python3 {S}/process_breaking.py {C}; python3 -c \"import json,re;[print(r['number'],bool(re.search(r'- \\[[xX]\\] Breaking',r['body'] or '')),r['title']) for r in map(json.loads,open('{C}')) if (r['mergedAt'] or '')>='2025-01-01' and re.search(r'\\b(remov|deprecat|drop|rename|no longer|default)',r['title'],re.I)]\""),
 compliant=8,
 violations=[dict(path=f'PR #{n}',line=None,snippet=t(n),note=note) for n,note in bv],
 violations_total=len(bv), verdict='aspirational',
 verdict_reason='Breaking change를 체크한 PR은 8건(50·108·171·183·225·356·559·770)이다. 신호 기반 판정으로 찾은 사용자 관점 호환성 파괴 13건 중 5건(38%)은 체크하지 않았다(645·271·605·128·714). 특히 deprecation 정책에 따른 API 제거(645)도 Breaking으로 표시되지 않았다. 판정은 휴리스틱과 수작업 확인이라 실제 누락은 더 많을 수 있다.',
 rule_refinement='공개 API 제거(deprecation 창 만료 포함), 공개 인자 이름 변경, 기본 동작 변경처럼 기존 사용자 코드가 그대로 동작하지 않는 PR은 Type of Change에서 Breaking change를 체크한다.',
 suggested_test=None))

order=['PRC-008','PRC-003','PRC-001','PRC-005','PRC-002','PRC-023','STY-015','PRC-004']
out.sort(key=lambda o:order.index(o['id']))
for o in out:
    for v_ in o['violations']: v_['snippet']=v_['snippet'][:120]
json.dump(out,open(SP+'/stage_d/out/batch_process.json','w'),ensure_ascii=False,indent=1)
for o in out: print(o['id'],o['verdict'],f"{o['compliant']}/{o['violations_total']}",len(o['violations']))
