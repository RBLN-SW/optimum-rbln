import json, sys
S = sys.argv[1]; SC = f"{S}/stage_d/scripts"; DT = f"{S}/devtree"; CORPUS = f"{S}/corpus/prs.jsonl"
prev = {r["id"]: r for r in json.load(open(f"{S}/stage_d/out/batch_process.json"))}

prc001 = dict(prev["PRC-001"])
prc001["check"] = {
    "kind": "corpus",
    "description": "dev@74e97fa4에서 제목 검사가 어디 있는지(.github/workflows vs .buildkite) 찾고, 그 검사(amannn/action-semantic-pull-request@v5, types=release/model/performance/refactor/doc/dependency/other)의 기본 headerPattern으로 2025-01 이후 머지 PR 제목을 검사한다; 매치하지 않거나 type이 목록 밖이면 위반. dev 브랜치 보호의 required check에 포함되는지도 확인한다.",
    "command": f"cd {DT} && grep -rln 'action-semantic-pull-request' .github .buildkite; grep -rn 'title' .buildkite; python3 {SC}/process_title2.py {CORPUS}; gh api repos/RBLN-SW/optimum-rbln/branches/dev/protection --jq .required_status_checks.contexts; gh run list -R RBLN-SW/optimum-rbln --workflow pr-title-check.yaml --limit 8 --json event,conclusion,createdAt,displayTitle",
}
prc001["verdict_reason"] = (
    "판정 변화 없음(verified, 610/21 그대로). 제목 검사는 Buildkite로 옮겨지지 않았고 dev@74e97fa4에도 GitHub Actions(.github/workflows/pr-title-check.yaml)로 남아 있으며 2026-10-22까지 모든 PR에서 실행되고 있다. "
    "머지된 631건 중 610건(96.7%)이 통과하고, main 삭제 이후(2026-09-11~) 머지된 66건과 2026-08 이후 113건은 위반이 0이다. "
    "다만 dev 브랜치 보호의 required check는 Buildkite의 [OB] 4개(BC·pytest·code quality·docstrings)뿐이라 'Check PR title'이 실패해도 머지는 막히지 않는다(강제가 아니라 권고 수준의 CI 신호)."
)
prc001["rule_refinement"] = "PR 제목은 GHA pr-title-check.yaml이 검사하는 `type(scope): description` 형식을 따른다(type ∈ release/model/performance/refactor/doc/dependency/other). release-vX.Y.Z 브랜치로 가는 cherry-pick PR도 원래 변경의 type(doc:, other: 등)을 쓴다."
prc001["suggested_test"] = "pytest 대신 'Check PR title'(GHA)을 dev와 release-* 브랜치의 required status check로 추가하는 것을 권한다. CI가 Buildkite로 옮겨가는 중이므로, 옮길 경우 pr.yml의 gate 단계에 BUILDKITE_PULL_REQUEST 제목을 같은 정규식으로 검사하는 스텝을 두고 [OB] 컨텍스트로 보고하게 해야 required check에 넣을 수 있다. Revert 기본 제목 예외 여부도 정해야 한다."

sty015 = dict(prev["STY-015"])
sty015["check"] = {
    "kind": "grep",
    "description": "dev@74e97fa4의 .buildkite/**, .github/**(workflows·renovate·템플릿), scripts/ci/*에서 평문 IP·내부 호스트(*.rebellions.in, k8s.)·클러스터 마운트 경로·토큰을 찾고, 자격증명이 ${{ secrets.* }}(GHA) 또는 Buildkite secrets: 매핑으로, 이미지·클러스터 경로·사내 PyPI 엔드포인트가 에이전트 변수로 주입되는지, 필수 변수에 ${VAR:?not set} 가드가 있는지 센다. 평문 내부 호스트·경로가 위반.",
    "command": f"bash {SC}/process_dev_sty015.sh {DT}",
}
sty015["compliant"] = 64
sty015["violations"] = [
    {"path": "scripts/ci/slack-report.py", "line": 22, "snippet": "API = \"https://obedients-api.k8s.rebellions.in/v2/organizations\"",
     "note": "내부 API 엔드포인트 평문; GHA는 같은 값을 secrets.OBEDIENTS_API_URL로 받음"},
    {"path": "scripts/ci/rebel-compiler-override.sh", "line": 12,
     "snippet": "  \"rebel-compiler overridden to \\`${REBEL_COMPILER_VERSION}\\` (pypi.rebellions.in/simple)\" || true",
     "note": "엔드포인트는 변수로 받지만 어노테이션 문구에 내부 호스트 평문"},
]
sty015["violations_total"] = 2
sty015["verdict"] = "verified"
sty015["verdict_reason"] = (
    "판정 유지(verified), 수치 변경 54/1 → 64/2(파일은 이전 gh 스냅숏과 동일하고, 내부 호스트 grep과 Buildkite secrets 매핑 집계를 추가한 결과). "
    "GHA 자격증명 9건은 모두 ${{ secrets.* }}(Obedients URL도 secret), Buildkite 자격증명 18건은 모두 secrets: 매핑, 이미지·클러스터 경로 34건과 사내 PyPI 엔드포인트 3건은 에이전트 변수로 주입되며 평문 IP·마운트 경로는 없다. "
    "필수 변수 가드는 bc-steps.sh·bc-compile-steps.sh(BC_BASE_PATH), install-rebel-compiler.sh(엔드포인트·계정 3개)에 있다. 위반은 Obedients API URL 하드코딩과 override 어노테이션의 사내 PyPI 호스트 문구 2건."
)
sty015["rule_refinement"] = "CI(GHA·Buildkite·scripts/ci) 자격증명은 GHA secrets 컨텍스트 또는 Buildkite secrets: 매핑으로만 받고, 내부 엔드포인트·클러스터 경로·이미지는 에이전트/CI 변수로 주입하며 로그·어노테이션 문구에도 평문으로 적지 않는다. 필수 변수는 ${VAR:?not set}로 가드한다."
sty015["suggested_test"] = "tests/test_conventions.py::test_ci_has_no_plaintext_endpoints — .github/**/*.y*ml, .buildkite/**/*.yml, scripts/ci/*를 읽어 (1) IPv4 리터럴, (2) *.rebellions.in / *.k8s.* / *.svc / *.cluster.local 호스트(주석 제외, 문자열 포함), (3) /mnt|/nfs|/data로 시작하는 절대경로, (4) token/password 키에 ${{ secrets. }}·${VAR}·Buildkite secret 이름이 아닌 리터럴 값이 있으면 실패. allowlist: scripts/ci/slack-report.py:22 API 상수, scripts/ci/rebel-compiler-override.sh:12 어노테이션 문구(변수화 전까지)."

prc005 = dict(prev["PRC-005"])
prc005["check"] = {
    "kind": "script",
    "description": "main 삭제 후 릴리스가 어떻게 나가는지 gh(기본 브랜치·main 존재·release-* 브랜치·ruleset·GitHub release의 target_commitish·base=release-v0.12.0 PR)와 dev@74e97fa4의 deploy-on-tag.yaml·bc-compile-steps.sh·PR 템플릿으로 확인하고, 2026-08-28 이후 정식 릴리스 중 dev→main `release: vX.Y.Z` PR로 나간 것이 있는지 센다.",
    "command": f"bash {SC}/process_dev_release.sh {DT}; python3 -c \"import json,re; [print(p['number'],p['mergedAt'][:10],p['base'],p['title']) for p in map(json.loads,open('{CORPUS}')) if (p.get('mergedAt') or '')>='2026-08-28' and (re.match('(?i)release',p['title']) or p['base'] not in ('dev',))]\"",
}
prc005["compliant"] = 0
prc005["violations"] = [
    {"path": "release v0.11.2 (2026-08-28)", "line": None, "snippet": "v0.11.2 target=release-v0.11.2; release-v0.11.2 == 8861ed75 (dev 조상, cherry-pick 0)",
     "note": "dev에서 release 브랜치를 잘라 태그, dev→main PR 없음"},
    {"path": "release v0.12.0 (2026-10-16)", "line": None, "snippet": "v0.12.0 target=release-v0.12.0; PRs #786 #789 #805 base=release-v0.12.0 (squash cherry-pick)",
     "note": "release 브랜치 + cherry-pick PR, 제목 release: 아님"},
]
prc005["violations_total"] = 2
prc005["verdict"] = "rejected"
prc005["verdict_reason"] = (
    "stale: main branch removed in PR 727. 이전 판정 interview(7/9) → rejected. 원격 main은 404이고 기본 브랜치는 dev이며, PR 727 본문이 \"`dev` is now the default branch and `main` has been removed\"라고 적는다. "
    "main 삭제 이후 정식 릴리스 2개(v0.11.2, v0.12.0)는 모두 dev→main PR 없이 release-vX.Y.Z 브랜치에서 태그로 나갔고, corpus에서 2026-07-31(#661) 이후 `release:` 제목 PR은 0건이다. "
    "PR 템플릿(.github/pull_request_template.md)은 아직 'Release (dev → main merge…)'와 'hotfix는 main으로'를 안내하고 있어 낡았다."
)
prc005["rule_refinement"] = (
    "(저장소가 보여주는 현행 절차, 2026-08 이후) 릴리스는 PR이 아니라 태그로 낸다. 프리릴리스(vX.Y.ZaN, .devN)는 dev에 직접 태그한다(GitHub release target=dev). "
    "정식 릴리스는 dev의 rc 지점(예: v0.12.0rc1=522f1bb8)에서 release-vX.Y.Z 브랜치를 잘라 내고, 필요한 수정은 base=release-vX.Y.Z인 cherry-pick PR(제목은 원래 type 유지, 예: `doc: … (#783)`, squash)로 넣은 뒤 그 브랜치에 vX.Y.ZrcN·vX.Y.Z 태그를 단다. "
    "release-* 브랜치와 release* 태그는 ruleset으로 삭제·강제푸시·업데이트가 막혀 있다(우회 사용자 1명). v* 태그 푸시가 GHA deploy-on-tag.yaml → deploy.yaml(PyPI 업로드)과 vllm-rbln 패키지 갱신 dispatch를 일으키고, Buildkite bc-compile.yml은 정식 버전 태그(bc-compile-steps.sh의 ^[0-9.]+(postN)?$)에서만 BC 아티팩트를 컴파일한다. "
    "PR 템플릿의 Release 항목과 main 안내는 이 절차에 맞게 고쳐야 한다."
)
prc005["suggested_test"] = None

prc023 = dict(prev["PRC-023"])
prc023["check"] = {
    "kind": "script",
    "description": "main 삭제 후에도 브랜치 간 조상 관계를 유지해야 하는 PR(dev↔main sync/back-merge, release→dev back-merge)이 존재하는지 확인한다: 원격 main 존재 여부, 저장소 merge 설정, base=release-v0.12.0 PR의 merge commit 부모 수, release-* → dev PR 유무.",
    "command": f"bash {SC}/process_dev_release.sh {DT}; gh pr list -R RBLN-SW/optimum-rbln --state all --search 'head:release-v0.12.0 OR head:release-v0.11.2' --json number,title,baseRefName",
}
prc023["compliant"] = 0
prc023["violations"] = []
prc023["violations_total"] = 0
prc023["verdict"] = "rejected"
prc023["verdict_reason"] = (
    "stale: main branch removed in PR 727. 이전 판정 verified(21/3) → rejected. 규칙의 대상인 dev↔main 동기화·백머지가 더 이상 존재할 수 없다: 원격 main은 404이고, 저장소는 allow_merge_commit=false(squash만 허용)이다. "
    "현행 release-vX.Y.Z 브랜치로 가는 PR(#786 #789 #805)은 dev 커밋의 cherry-pick이라 모두 squash(부모 1개)로 머지되었고, release 브랜치에서 dev로 되돌아오는 back-merge PR은 0건이다(수정은 dev에 먼저 들어가므로 필요 없음)."
)
prc023["rule_refinement"] = (
    "(현행) 장기 브랜치는 dev 하나이고 릴리스는 dev에서 잘라낸 release-vX.Y.Z 브랜치에 태그로 낸다. 수정은 dev에 먼저 머지한 뒤 base=release-vX.Y.Z cherry-pick PR로 squash 머지하며, 저장소 설정상 merge commit은 쓸 수 없다. "
    "조상 관계를 보존해야 하는 병합(이전 main↔dev 동기화)은 이 절차에서 생기지 않으므로 규칙은 폐기한다. 만약 release 브랜치에만 있는 수정이 생기면 dev로도 별도 PR(cherry-pick)로 넣는다는 규칙으로 대체할지는 팀이 정해야 한다."
)
prc023["suggested_test"] = None

out = [prc001, sty015, prc005, prc023]
json.dump(out, open(f"{S}/stage_d/out_dev/batch_process.json", "w"), ensure_ascii=False, indent=1)
