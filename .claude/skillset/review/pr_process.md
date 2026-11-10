# PR · CI · 브랜치 운영 — 룰 25개

결정 필요 0 · 코드 일치 4 · 코드 불일치 3 · 판단형 14 · 폐기 제안 4. 순서는 결정이 필요한 것부터. 각 룰의 **응답** 줄에서 하나를 `[x]`로 표시하고, 틀리거나 조건부면 **메모**에 한 줄 적어 주세요.

## 코드 불일치(의도는 분명)

### PRC-008 · 코드 불일치(의도는 분명)

**룰** 미머지 PR 위에 쌓는 PR은 base를 선행 PR 브랜치로 두고 본문 첫 줄에 `Stacked on #N`을 적는다.
**조건** 머지 순서가 어긋났을 때의 대응은 main/release 브랜치로 out-of-band 들어가는 PR에만 요구한다.
**근거** [#692](https://github.com/rebellions-sw/optimum-rbln/pull/692) (junamsong): “If a sync containing the re-landed fix reaches main first, **close this PR instead of merging it**”
　　　[#122](https://github.com/rebellions-sw/optimum-rbln/pull/122) (jongho): “This PR should be merged after the update of document”
　　　외 PR 4건
**코드 대조** 코드 불일치(의도는 분명) — 준수 11 / 위반 25. 의존 PR 36건 중 base=선행 브랜치이면서 본문에 선행 PR을 밝힌 것은 11건(31%)이다. 2026-08 이후 머지분은 10건 중 9건이 준수(668만 base=dev)하지만, 2025~2026-07의 feature-branch 하위 PR 23건은 본문에 선행 PR을 적지 않…
**응답** [ ] 맞다 　[ ] 틀리다 　[ ] 조건부로 맞다 　[ ] 필요 없는 룰이다 　[ ] 모른다 　[ ] 기억나지 않는다 　· 메모:

### PRC-003 · 코드 불일치(의도는 분명)

**룰** PR 본문에는 무엇을 바꿨는지와 왜 그렇게 했는지를 적는다.
**조건** 템플릿 사용 시 Type of Change는 하나만 체크하고 Other를 체크하면 내용을 적는다.
**근거** [#567](https://github.com/rebellions-sw/optimum-rbln/pull/567) (junamsong): “- **Memory waste at load**: a BF16 Hub checkpoint (e.g. Qwen2-VL)”
　　　[#564](https://github.com/rebellions-sw/optimum-rbln/pull/564) (jongho): “- [x] Other: dependency range / cross-version compatibility”
　　　외 PR 3건
**코드 대조** 코드 불일치(의도는 분명) — 준수 51 / 위반 537. 템플릿이 자동으로 채워지므로 절 헤더는 497/588건에 있고, 유형 체크도 템플릿을 쓴 509건 중 448건이 정확히 1개다. 그러나 Motivation을 실제로 쓴 PR은 67건(11%)뿐이고, Other를 체크한 145건 중 135건은 설명이 없어 전체 준수는 51건(8.7%)이…
**응답** [ ] 맞다 　[ ] 틀리다 　[ ] 조건부로 맞다 　[ ] 필요 없는 룰이다 　[ ] 모른다 　[ ] 기억나지 않는다 　· 메모:

### PRC-004 · 코드 불일치(의도는 분명)

**룰** 기존 사용자 코드가 그대로 동작하지 않는 PR은 Type of Change에서 Breaking change를 체크한다.
**조건** 공개 API 제거(deprecation 창 만료 포함), 공개 인자 이름 변경, 기본 동작 변경이 해당한다.
**근거** [#559](https://github.com/rebellions-sw/optimum-rbln/pull/559) (thkim): “- [x] Breaking change (fix or feature that would cause existing functionality to not work as expected)”
　　　[#108](https://github.com/rebellions-sw/optimum-rbln/pull/108) (jongho): “- [x] Breaking change (fix or feature that would cause existing functionality to not work as expected)”
**코드 대조** 코드 불일치(의도는 분명) — 준수 8 / 위반 5. Breaking change를 체크한 PR은 8건(50·108·171·183·225·356·559·770)이다. 신호 기반 판정으로 찾은 사용자 관점 호환성 파괴 13건 중 5건(38%)은 체크하지 않았다(645·271·605·128·714).
**응답** [ ] 맞다 　[ ] 틀리다 　[ ] 조건부로 맞다 　[ ] 필요 없는 룰이다 　[ ] 모른다 　[ ] 기억나지 않는다 　· 메모:

## 코드 일치

### PRC-001 · 코드 일치

**룰** PR 제목은 `type(scope): description` 형식을 따른다.
**조건** type은 release/model/performance/refactor/doc/dependency/other이며 release 브랜치 cherry-pick PR도 원래 변경의 type을 쓴다.
**근거** [#446](https://github.com/rebellions-sw/optimum-rbln/pull/446) (yhboo): “title category를 뭘로 넣어야 할지 좀 애매한데, 가이드 주시면 맞춰서 넣겠습니다”
　　　[#735](https://github.com/rebellions-sw/optimum-rbln/pull/735) (dkhong): “type(optional scope): description”
　　　외 PR 2건
**코드 대조** 코드 일치 — 준수 610 / 위반 21
**응답** [ ] 맞다 　[ ] 틀리다 　[ ] 조건부로 맞다 　[ ] 필요 없는 룰이다 　[ ] 모른다 　[ ] 기억나지 않는다 　· 메모:

### STY-015 · 코드 일치

**룰** CI 자격증명은 GHA secrets나 Buildkite secrets 매핑으로만 받고 로그에도 평문으로 적지 않는다.
**조건** 내부 엔드포인트, 클러스터 경로, 이미지는 CI 변수로 주입하고 필수 변수는 `${VAR:?not set}`로 가드한다.
**근거** [#758](https://github.com/rebellions-sw/optimum-rbln/pull/758) (thkim): “They are facts about the cluster, not about this repository.”
　　　[#630](https://github.com/rebellions-sw/optimum-rbln/pull/630) (jongho): “please hide the internal endpoint. no plain text.”
　　　외 PR 1건
**코드 대조** 코드 일치 — 준수 64 / 위반 2
**응답** [ ] 맞다 　[ ] 틀리다 　[ ] 조건부로 맞다 　[ ] 필요 없는 룰이다 　[ ] 모른다 　[ ] 기억나지 않는다 　· 메모:

### PRC-031 · 코드 일치

**룰** BC 컴파일은 안정 릴리스 태그에서만 돌고, 스텝 안에서 태그를 다시 검사한다.
**조건** aN·rcN·.devN 같은 프리릴리스 태그는 BC 아티팩트를 만들지 않는다(scripts/ci/bc-compile-steps.sh:15).
**근거** [#404](https://github.com/rebellions-sw/optimum-rbln/pull/404) (thkim): “Gate deploy-on-tag with release-only regex and trigger BC compile for release tags.”
　　　[#753](https://github.com/rebellions-sw/optimum-rbln/pull/753) (thkim): “the generator holds the regex 'check-tag' held.”
**코드 대조** 코드 일치 — 준수 None / 위반 None
**응답** [ ] 맞다 　[ ] 틀리다 　[ ] 조건부로 맞다 　[ ] 필요 없는 룰이다 　[ ] 모른다 　[ ] 기억나지 않는다 　· 메모:

### PRC-032 · 코드 일치

**룰** 모든 v* 태그는 프리릴리스를 포함해 PyPI 배포와 vllm-rbln 갱신을 트리거한다.
**조건** .github/workflows/deploy-on-tag.yaml 기준. BC 컴파일만 안정 태그로 제한된다(PRC-031).
**근거** [#29](https://github.com/rebellions-sw/optimum-rbln/pull/29) (sunwook): “trigger vllm model compilation of rebel_compiler when tagged optimum-rbln (include pre-release)”
**코드 대조** 코드 일치 — 준수 None / 위반 None
**응답** [ ] 맞다 　[ ] 틀리다 　[ ] 조건부로 맞다 　[ ] 필요 없는 룰이다 　[ ] 모른다 　[ ] 기억나지 않는다 　· 메모:

## 판단형

### PRC-012 · 판단형

**룰** 한 PR은 특정 모델 계열이나 기능 범위로 제한한다.
**조건** 독립적인 기능이나 호스트 측 호환성 변경처럼 얽힌 항목은 별도 PR로 분리한다.
**근거** [#211](https://github.com/rebellions-sw/optimum-rbln/pull/211) (eunji): “I think this PR should be limited to Llama models.”
　　　[#711](https://github.com/rebellions-sw/optimum-rbln/pull/711) (thkim): “Moved to #764 (independent, dev-based) together with the artifact-compatibility handling, since it touches the host side and olde…”
　　　외 PR 4건
**응답** [ ] 맞다 　[ ] 틀리다 　[ ] 조건부로 맞다 　[ ] 필요 없는 룰이다 　[ ] 모른다 　[ ] 기억나지 않는다 　· 메모:

### DEP-018 · 판단형

**룰** 컴파일러 버전, 브랜치, 해시, 컴파일러 수정에 의존하는 PR은 본문에 필요한 의존 대상을 명시한다.
**조건** 의존 PR도 적고 아직 검증되지 않은 범위를 표기한다.
**근거** [#584](https://github.com/rebellions-sw/optimum-rbln/pull/584) (jongho): “Requires the per-chiplet accounting API added in rebel_compiler (SR-239).”
　　　[#480](https://github.com/rebellions-sw/optimum-rbln/pull/480) (seinpark): “You need to below compiier hash”
　　　외 PR 2건
**응답** [ ] 맞다 　[ ] 틀리다 　[ ] 조건부로 맞다 　[ ] 필요 없는 룰이다 　[ ] 모른다 　[ ] 기억나지 않는다 　· 메모:

### PRC-014 · 판단형

**룰** PR 본문에 지원 범위의 한계와 검증하지 못한 모델·경로를 사유와 함께 명시한다.
**조건** 미지원 배치 크기 등이 해당한다.
**근거** [#714](https://github.com/rebellions-sw/optimum-rbln/pull/714) (thkim): “gemma4 received the same mechanical change and was not run (no test in the repo).”
　　　[#711](https://github.com/rebellions-sw/optimum-rbln/pull/711) (thkim): “Not device-verified: GroundingDINO”
　　　외 PR 1건
**응답** [ ] 맞다 　[ ] 틀리다 　[ ] 조건부로 맞다 　[ ] 필요 없는 룰이다 　[ ] 모른다 　[ ] 기억나지 않는다 　· 메모:

### PRC-015 · 판단형

**룰** 동작을 바꾸는 PR은 본문에 동작 변경을 별도 절로 기록하고 호환성 영향 범위를 명시한다.
**조건** 독립적으로 되돌릴 수 있도록 커밋 단위를 나눈다.
**근거** [#606](https://github.com/rebellions-sw/optimum-rbln/pull/606) (jongho): “Note on compatibility: inputs that already fit within 'enc_max_seq_len' are”
　　　[#714](https://github.com/rebellions-sw/optimum-rbln/pull/714) (thkim): “Commits 2 and 3 carry this and can be dropped independently.”
　　　외 PR 1건
**응답** [ ] 맞다 　[ ] 틀리다 　[ ] 조건부로 맞다 　[ ] 필요 없는 룰이다 　[ ] 모른다 　[ ] 기억나지 않는다 　· 메모:

### PRC-019 · 판단형

**룰** 신규 모델·파이프라인 PR 본문에는 참조한 HF 모델 주소와 클래스명, 재현 가능한 예제 실행 명령을 적는다.
**조건** upstream PR이 있으면 클래스명 대신 적을 수 있다.
**근거** [#83](https://github.com/rebellions-sw/optimum-rbln/pull/83) (sunwook): “python3 examples/object-detection/run_detr_resnet50_object_detection.py --from_transformers --batch_size 1 --height_size 224 --wi…”
　　　[#114](https://github.com/rebellions-sw/optimum-rbln/pull/114) (dkhong): “Please refer to the [PR in HuggingFace diffusers](https://github.com/huggingface/diffusers/pulls?q=cosmos)”
　　　외 PR 1건
**응답** [ ] 맞다 　[ ] 틀리다 　[ ] 조건부로 맞다 　[ ] 필요 없는 룰이다 　[ ] 모른다 　[ ] 기억나지 않는다 　· 메모:

### PRC-026 · 판단형

**룰** 릴리스 라인 수정은 해당 릴리스 태그 위에서 만들어 릴리스 라인의 자식 커밋이 되게 한다.
**조건** dev에 반영된 뒤에는 릴리스 브랜치를 dev에 역병합해 조상 관계를 맞춘다.
**근거** [#617](https://github.com/rebellions-sw/optimum-rbln/pull/617) (junamsong): “This PR rebuilds the fix on top of 'v0.11.0' so the post-release becomes a proper child of the release line.”
　　　[#615](https://github.com/rebellions-sw/optimum-rbln/pull/615) (junamsong): “Reconnects 'dev' to 'main' so 'v0.11.0' becomes an ancestor of 'dev' again.”
　　　외 PR 1건
**응답** [ ] 맞다 　[ ] 틀리다 　[ ] 조건부로 맞다 　[ ] 필요 없는 룰이다 　[ ] 모른다 　[ ] 기억나지 않는다 　· 메모:

### PRC-035 · 판단형

**룰** PR 빌드의 테스트 스위트는 경로 기반 조건이나 의존 관계 추론으로 생략하지 않고 모두 실행한다.
**조건** 시간이 길면 커밋 메시지의 영역별·전체 skip 마커로 명시적으로 건너뛴다.
**근거** [#63](https://github.com/rebellions-sw/optimum-rbln/pull/63) (kblee): “[skip-diffusers], [skip-transformers], [skip-llms] 추가합니다.”
　　　[#748](https://github.com/rebellions-sw/optimum-rbln/pull/748) (thkim): “Path gating is removed rather than repaired.”
　　　외 PR 1건
**응답** [ ] 맞다 　[ ] 틀리다 　[ ] 조건부로 맞다 　[ ] 필요 없는 룰이다 　[ ] 모른다 　[ ] 기억나지 않는다 　· 메모:

### PRC-010 · 판단형

**룰** 리뷰 대상 브랜치가 통합 브랜치와 어긋나 있으면 리뷰 전에 통합 브랜치와 동기화한다.
**근거** [#111](https://github.com/rebellions-sw/optimum-rbln/pull/111) (jongho): “reviewing this PR without rbln config refactoring would be like doing the same thing twice.”
　　　[#561](https://github.com/rebellions-sw/optimum-rbln/pull/561) (kblee): “can you rebase your branch with dev when this pr is merged?”
**응답** [ ] 맞다 　[ ] 틀리다 　[ ] 조건부로 맞다 　[ ] 필요 없는 룰이다 　[ ] 모른다 　[ ] 기억나지 않는다 　· 메모:

### PRC-024 · 판단형

**룰** 브랜치 사이로 수정을 옮길 때는 dev에 먼저 머지하고 cherry-pick -x로 출처를 남긴다.
**조건** 현재 옮기는 대상은 release-vX.Y.Z 브랜치다(main은 PR 727에서 삭제).
**근거** [#693](https://github.com/rebellions-sw/optimum-rbln/pull/693) (junamsong): “where it should have been merged in the first place”
　　　[#805](https://github.com/rebellions-sw/optimum-rbln/pull/805) (jswoo): “with '-x' so each commit records its source.”
**응답** [ ] 맞다 　[ ] 틀리다 　[ ] 조건부로 맞다 　[ ] 필요 없는 룰이다 　[ ] 모른다 　[ ] 기억나지 않는다 　· 메모:

### PRC-033 · 판단형

**룰** 같은 CI 테스트 매트릭스나 설치 절차는 실행기·파이프라인마다 따로 두지 않고 스크립트 한 곳에서 관리한다.
**근거** [#738](https://github.com/rebellions-sw/optimum-rbln/pull/738) (thkim): “holds the uv sync recipe that both the pytest pipeline and the generated steps run, so they cannot drift.”
　　　[#753](https://github.com/rebellions-sw/optimum-rbln/pull/753) (thkim): “Two executors mean two matrices to keep in step.”
**응답** [ ] 맞다 　[ ] 틀리다 　[ ] 조건부로 맞다 　[ ] 필요 없는 룰이다 　[ ] 모른다 　[ ] 기억나지 않는다 　· 메모:

### PRC-034 · 판단형

**룰** 테스트 스위트 실행 여부는 NPU를 점유하기 전에 파이프라인 업로드 시점에 결정한다.
**조건** 점유한 job 안에서 건너뛰지 않는다.
**근거** [#737](https://github.com/rebellions-sw/optimum-rbln/pull/737) (thkim): “GHA spawns them at every level and skips them inside the job, which holds an NPU just to skip.”
　　　[#755](https://github.com/rebellions-sw/optimum-rbln/pull/755) (thkim): “'[skip-*]' is decided at upload, not inside the step”
**응답** [ ] 맞다 　[ ] 틀리다 　[ ] 조건부로 맞다 　[ ] 필요 없는 룰이다 　[ ] 모른다 　[ ] 기억나지 않는다 　· 메모:

### PRC-037 · 판단형

**룰** 선택적 리포트·알림 스텝은 `soft_fail`로 두고 커밋 상태를 보고하지 않게 한다.
**조건** 보고 실패가 필수 체크와 야간 빌드의 성공·실패 결과를 바꾸지 않아야 한다.
**근거** [#739](https://github.com/rebellions-sw/optimum-rbln/pull/739) (thkim): “a report that cannot be delivered should not turn a green nightly red.”
　　　[#752](https://github.com/rebellions-sw/optimum-rbln/pull/752) (thkim): “It carries no 'github_commit_status', so it cannot affect a required check.”
**응답** [ ] 맞다 　[ ] 틀리다 　[ ] 조건부로 맞다 　[ ] 필요 없는 룰이다 　[ ] 모른다 　[ ] 기억나지 않는다 　· 메모:

### PRC-040 · 판단형

**룰** CI 테스트 실패를 자동 재시도로 가리지 않고 원인을 확인한다.
**근거** [#755](https://github.com/rebellions-sw/optimum-rbln/pull/755) (thkim): “automatic retry on a sixty-minute NPU suite hides”
　　　[#756](https://github.com/rebellions-sw/optimum-rbln/pull/756) (thkim): “failures this repo has actually seen had causes”
**응답** [ ] 맞다 　[ ] 틀리다 　[ ] 조건부로 맞다 　[ ] 필요 없는 룰이다 　[ ] 모른다 　[ ] 기억나지 않는다 　· 메모:

### PRC-042 · 판단형

**룰** 스케줄 CI 워크플로의 트리거는 main에 얇게 두고 실행 로직은 dev의 재사용 워크플로로 분리한다.
**조건** main에서 스케줄 워크플로를 직접 수정하지 않는다.
**근거** [#440](https://github.com/rebellions-sw/optimum-rbln/pull/440) (thkim): “moving the execution logic to reusable workflows while keeping the schedule triggers minimal”
　　　[#362](https://github.com/rebellions-sw/optimum-rbln/pull/362) (thkim): “Since this is a patch related to PR, merging it into dev is sufficient.”
**응답** [ ] 맞다 　[ ] 틀리다 　[ ] 조건부로 맞다 　[ ] 필요 없는 룰이다 　[ ] 모른다 　[ ] 기억나지 않는다 　· 메모:

## 폐기 제안

### PRC-005 · 폐기 제안

**룰** 릴리스는 PR이 아니라 태그로 낸다.
**조건** 프리릴리스는 dev에 직접 태그하고, 정식 릴리스는 `release-vX.Y.Z` 브랜치를 잘라 cherry-pick PR 후 태그를 단다.
**근거** [#484](https://github.com/rebellions-sw/optimum-rbln/pull/484) (jongho): “Release (dev → main merge for production release)”
　　　[#541](https://github.com/rebellions-sw/optimum-rbln/pull/541) (thkim): “Release (dev → main merge for production release)”
　　　외 PR 2건
**코드 대조** 폐기 제안 — 준수 0 / 위반 2. stale: main branch removed in PR 727. 이전 판정 interview(7/9) → rejected. 원격 main은 404이고 기본 브랜치는 dev이며, PR 727 본문이 "'dev' is now the default branch and 'main' has…
**응답** [ ] 맞다 　[ ] 틀리다 　[ ] 조건부로 맞다 　[ ] 필요 없는 룰이다 　[ ] 모른다 　[ ] 기억나지 않는다 　· 메모:

### PRC-025 · 폐기 제안

**룰** main은 dev에서 넘어오는 동기화 병합으로만 전진시키고 hotfix 직접 반영은 예외로 한다.
**조건** hotfix는 base를 main으로 두고 제목에 `(hotfix)`를 표기하며 dev에 백머지 PR로 반영한다.
**근거** [#692](https://github.com/rebellions-sw/optimum-rbln/pull/692) (junamsong): “main should only advance via dev sync merges, e.g. #661”
　　　[#492](https://github.com/rebellions-sw/optimum-rbln/pull/492) (thkim): “model: hotfix for gpt-oss bug”
　　　외 PR 2건
**응답** [ ] 맞다 　[ ] 틀리다 　[ ] 조건부로 맞다 　[ ] 필요 없는 룰이다 　[ ] 모른다 　[ ] 기억나지 않는다 　· 메모:

### PRC-002 · 폐기 제안

**룰** 작업 중인 PR은 제목에 표식을 달지 않고 GitHub Draft로 연다.
**조건** 제목은 conventional commit 형식을 유지한다.
**근거** [#490](https://github.com/rebellions-sw/optimum-rbln/pull/490) (seinpark): “(WIP) model: support nested submodule for pipeline”
　　　[#496](https://github.com/rebellions-sw/optimum-rbln/pull/496) (seinpark): “(WIP) support nested submodule for pipeline”
　　　외 PR 1건
**코드 대조** 폐기 제안 — 준수 4 / 위반 34. draft PR 38건 중 (WIP)를 붙인 것은 4건(390·391·490·496)이고 모두 한 작성자(rebel-seinpark)다. 나머지 34건은 GitHub draft 상태만 쓰거나 [DRAFT]·(DO NOT MERGE)를 붙였다.
**응답** [ ] 맞다 　[ ] 틀리다 　[ ] 조건부로 맞다 　[ ] 필요 없는 룰이다 　[ ] 모른다 　[ ] 기억나지 않는다 　· 메모:

### PRC-023 · 폐기 제안

**룰** 수정은 dev에 먼저 머지한 뒤 `release-vX.Y.Z` 대상 cherry-pick PR로 squash 머지한다.
**조건** 저장소 설정상 merge commit은 쓸 수 없고, 조상 관계 보존 병합 규칙은 폐기 대상이다.
**근거** [#616](https://github.com/rebellions-sw/optimum-rbln/pull/616) (junamsong): “**Must be merged with "Create a merge commit"** (not Squash, not Rebase).”
　　　[#615](https://github.com/rebellions-sw/optimum-rbln/pull/615) (junamsong): “Squash flattens the merge commit into a single linear commit and destroys the multi-parent topology this PR exists to restore.”
　　　외 PR 1건
**코드 대조** 폐기 제안 — 준수 0 / 위반 0. stale: main branch removed in PR 727. 이전 판정 verified(21/3) → rejected. 규칙의 대상인 dev↔main 동기화·백머지가 더 이상 존재할 수 없다: 원격 main은 404이고, 저장소는 allow_merge_commit=false(s…
**응답** [ ] 맞다 　[ ] 틀리다 　[ ] 조건부로 맞다 　[ ] 필요 없는 룰이다 　[ ] 모른다 　[ ] 기억나지 않는다 　· 메모:
