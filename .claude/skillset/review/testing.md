# 테스트 — 룰 23개

결정 필요 0 · 코드 일치 2 · 코드 불일치 1 · 판단형 20 · 폐기 제안 0. 순서는 결정이 필요한 것부터. 각 룰의 **응답** 줄에서 하나를 `[x]`로 표시하고, 틀리거나 조건부면 **메모**에 한 줄 적어 주세요.

## 코드 불일치(의도는 분명)

### TEST-008 · 코드 불일치(의도는 분명)

**룰** export된 사용자용 모델·파이프라인 클래스는 `RBLN_CLASS`로 등록된 테스트 클래스를 가진다.
**조건** 등록 위치는 `tests/test_{transformers,diffusers,llm}.py` 중 한 곳이고 이미지·오디오 입력 모델은 `IS_MULTIMODAL=True`를 선언한다.
**근거** [#320](https://github.com/rebellions-sw/optimum-rbln/pull/320) (thkim): “The schedule is incorrect because the Multimodal Flag is missing for Gemma3.”
　　　[#15](https://github.com/rebellions-sw/optimum-rbln/pull/15) (dkhong): “test_diffusers에 test case 추가 부탁드립니다~”
　　　외 PR 1건
**코드 대조** 코드 불일치(의도는 분명) — 준수 99 / 위반 31. dev 팁에서 export된 사용자용 모델·파이프라인은 115개다. 이 중 85개가 등록되어 있고(직접 82, Kandinsky combined 경유 3) 30개(26%)가 미등록이다. main에서는 97개 중 67개 등록, 30개 미등록이었다.
**응답** [ ] 맞다 　[ ] 틀리다 　[ ] 조건부로 맞다 　[ ] 필요 없는 룰이다 　[ ] 모른다 　[ ] 기억나지 않는다 　· 메모:

## 코드 일치

### TEST-019 · 코드 일치

**룰** 공유 설정 dict를 in-place로 고치려면 자기 클래스 본문에서 선언하고 `setUpClass`에서만 고친다.
**조건** 테스트 메서드에서는 복사본을 쓰고, 상속받은 dict는 `cls.X = {**cls.X, ...}`로 재바인딩한다.
**근거** [#741](https://github.com/rebellions-sw/optimum-rbln/pull/741) (thkim): “Rebind instead of mutating at both sites, so the class gets its own attribute and the base dict stays empty.”
　　　[#745](https://github.com/rebellions-sw/optimum-rbln/pull/745) (thkim): “build the 'rbln_config' as a new dict instead of updating the class attribute in place”
　　　외 PR 1건
**코드 대조** 코드 일치 — 준수 23 / 위반 3
**응답** [ ] 맞다 　[ ] 틀리다 　[ ] 조건부로 맞다 　[ ] 필요 없는 룰이다 　[ ] 모른다 　[ ] 기억나지 않는다 　· 메모:

### TEST-007 · 코드 일치

**룰** CI의 모든 pytest 호출은 `-vv --durations 0`을 붙이고 정기 실행은 취소 없이 끝까지 돈다.
**조건** `cancel_on_build_failing`과 `-x`/`--maxfail`을 쓰지 않고 보고 단계는 `continue_on_failure: true`인 wait 뒤에 둔다.
**근거** [#241](https://github.com/rebellions-sw/optimum-rbln/pull/241) (jongho): “This PR fixes scheduled pytest, not to fast-fail.”
　　　[#203](https://github.com/rebellions-sw/optimum-rbln/pull/203) (jongho): “Reports the execution time of each test case.”
**코드 대조** 코드 일치 — 준수 7 / 위반 0
**응답** [ ] 맞다 　[ ] 틀리다 　[ ] 조건부로 맞다 　[ ] 필요 없는 룰이다 　[ ] 모른다 　[ ] 기억나지 않는다 　· 메모:

## 판단형

### TEST-004 · 판단형

**룰** 전체 pytest가 필요한 변경은 커밋 메시지에 [pytest-full]을 넣는다.
**조건** CI는 커밋 메시지(BUILDKITE_MESSAGE)에서 이 마커를 읽고 pytest 범위만 넓힌다. 설명하려고 인용해도 전체 빌드가 되므로 인용하지 않는다.
**근거** [#355](https://github.com/rebellions-sw/optimum-rbln/pull/355) (jongho): “'[pytest-full]' implies only the coverage of pytest, not other actions (slack report).”
　　　[#737](https://github.com/rebellions-sw/optimum-rbln/pull/737) (thkim): “Writing the full-test directive into a commit message makes that build a full one”
　　　외 PR 2건
**응답** [ ] 맞다 　[ ] 틀리다 　[ ] 조건부로 맞다 　[ ] 필요 없는 룰이다 　[ ] 모른다 　[ ] 기억나지 않는다 　· 메모:

### TEST-009 · 판단형

**룰** 새 모델을 지원하는 PR에는 모델 전용 pytest를 함께 추가한다.
**조건** 테스트는 upstream 라이브러리의 테스트를 기반으로 작성한다.
**근거** [#735](https://github.com/rebellions-sw/optimum-rbln/pull/735) (seinpark): “Could you add simple pytest for this model?”
　　　[#253](https://github.com/rebellions-sw/optimum-rbln/pull/253) (seinpark): “Feature pytest for PegasusModel”
　　　외 PR 2건
**응답** [ ] 맞다 　[ ] 틀리다 　[ ] 조건부로 맞다 　[ ] 필요 없는 룰이다 　[ ] 모른다 　[ ] 기억나지 않는다 　· 메모:

### TEST-010 · 판단형

**룰** 테스트는 전체 크기 모델 대신 tiny 모델이나 레이어 수를 줄인 모델을 사용한다.
**조건** 파이프라인 테스트는 tiny-random 구성으로 export, generation, reload를 검증한다.
**근거** [#694](https://github.com/rebellions-sw/optimum-rbln/pull/694) (seinpark): “tiny-random pipelines mirroring the real architectures”
　　　[#15](https://github.com/rebellions-sw/optimum-rbln/pull/15) (dkhong): “현재는 full model로 작성하였는데, 추후 tiny model로 변경하겠습니다”
　　　외 PR 2건
**응답** [ ] 맞다 　[ ] 틀리다 　[ ] 조건부로 맞다 　[ ] 필요 없는 룰이다 　[ ] 모른다 　[ ] 기억나지 않는다 　· 메모:

### TEST-011 · 판단형

**룰** 버그 수정에는 수정을 되돌리면 실패하는 회귀 테스트를 함께 넣는다.
**조건** 기존 테스트가 놓친 입력 조건과 같은 계열 모델의 누락된 커버리지를 채운다.
**근거** [#701](https://github.com/rebellions-sw/optimum-rbln/pull/701) (seinpark): “qwen2_5_vl had no output_hidden_states test (qwen3_vl does), so this PR adds one to cover the regression.”
　　　[#659](https://github.com/rebellions-sw/optimum-rbln/pull/659) (thkim): “The only multimodal 'output_hidden_states' test was Gemma3 with a single unpadded prompt”
　　　외 PR 2건
**응답** [ ] 맞다 　[ ] 틀리다 　[ ] 조건부로 맞다 　[ ] 필요 없는 룰이다 　[ ] 모른다 　[ ] 기억나지 않는다 　· 메모:

### TEST-012 · 판단형

**룰** 기능 추가나 동작 변경에는 그 효과를 증명하는 테스트를 함께 추가한다.
**조건** 설정 저장·재로드 왕복, 별칭 매핑, 서브모듈 설정 전파 등이 해당한다.
**근거** [#196](https://github.com/rebellions-sw/optimum-rbln/pull/196) (jongho): “I have added tests that prove my fix is effective or that my feature works (If needed)”
　　　[#27](https://github.com/rebellions-sw/optimum-rbln/pull/27) (thkim): “it adds new unit tests to improve code reliability”
　　　외 PR 2건
**응답** [ ] 맞다 　[ ] 틀리다 　[ ] 조건부로 맞다 　[ ] 필요 없는 룰이다 　[ ] 모른다 　[ ] 기억나지 않는다 　· 메모:

### TEST-013 · 판단형

**룰** 모델, 그래프, 커널 변경은 참조 구현과 같은 입력에서 출력을 비교 검증한다.
**조건** 생성 토큰 일치, logits 상관계수, 비트 일치 등을 쓰고 불일치는 근거와 함께 PR 설명에 기록한다.
**근거** [#817](https://github.com/rebellions-sw/optimum-rbln/pull/817) (seinpark): “The 2 mismatches are near-ties (HF top-1 margin 0.10 and 0.37).”
　　　[#711](https://github.com/rebellions-sw/optimum-rbln/pull/711) (thkim): “compiled with 'dtype=float16', compared against HF fp16 on CPU”
　　　외 PR 2건
**응답** [ ] 맞다 　[ ] 틀리다 　[ ] 조건부로 맞다 　[ ] 필요 없는 룰이다 　[ ] 모른다 　[ ] 기억나지 않는다 　· 메모:

### TEST-017 · 판단형

**룰** CI가 다루지 않는 변경은 병합 전에 실제 NPU에서 컴파일과 추론까지 확인한다.
**조건** 확인하지 못했으면 PR 설명에 미검증 범위를 명시하고 draft로 둔다.
**근거** [#705](https://github.com/rebellions-sw/optimum-rbln/pull/705) (thkim): “Not verified on hardware: only host-side sorting logic is tested (no NPU/compile in the tests)”
　　　[#635](https://github.com/rebellions-sw/optimum-rbln/pull/635) (jongho): “All 33 example scripts were run end-to-end (compile + inference) on RBLN NPUs and pass.”
　　　외 PR 2건
**응답** [ ] 맞다 　[ ] 틀리다 　[ ] 조건부로 맞다 　[ ] 필요 없는 룰이다 　[ ] 모른다 　[ ] 기억나지 않는다 　· 메모:

### TEST-001 · 판단형

**룰** 테스트는 CI 워크플로우가 실제로 실행하는 파일에 두고 기대값은 실제 모델 출력으로 검증한다.
**조건** 실패를 가리는 우회 줄을 남기지 않고 같은 기능의 중복 테스트는 합친다.
**근거** [#475](https://github.com/rebellions-sw/optimum-rbln/pull/475) (jongho): “the tests added to test_transformers and test_llm seem to perform the same function.”
　　　[#722](https://github.com/rebellions-sw/optimum-rbln/pull/722) (jongho): “please remove this line so that we can recognize if this test goes wrong”
　　　외 PR 1건
**응답** [ ] 맞다 　[ ] 틀리다 　[ ] 조건부로 맞다 　[ ] 필요 없는 룰이다 　[ ] 모른다 　[ ] 기억나지 않는다 　· 메모:

### TEST-002 · 판단형

**룰** NPU가 필요 없는 테스트는 NPU를 요청하지 않는 CPU 단위 테스트로 `tests/unit/cpu/`에 둔다.
**조건** 디바이스가 필요한 단위 테스트는 `tests/unit/npu/`에 둔다.
**근거** [#784](https://github.com/rebellions-sw/optimum-rbln/pull/784) (junamsong): “Both decorator tests move to 'tests/unit/cpu/', which 'unit-cpu' collects as a directory, so they run on every PR”
　　　[#759](https://github.com/rebellions-sw/optimum-rbln/pull/759) (thkim): “The method only reads 'self.rbln_config', so the tests bypass '__init__' and hand it a config stub”
　　　외 PR 1건
**응답** [ ] 맞다 　[ ] 틀리다 　[ ] 조건부로 맞다 　[ ] 필요 없는 룰이다 　[ ] 모른다 　[ ] 기억나지 않는다 　· 메모:

### TEST-038 · 판단형

**룰** 테스트 기대값과 경계 입력은 모델의 기본 동작과 지원 범위 상수에 맞춰 작성한다.
**조건** 이들이 바뀌면 기존 테스트를 함께 갱신하거나 정리한다.
**근거** [#325](https://github.com/rebellions-sw/optimum-rbln/pull/325) (seinpark): “As the Qwen2-VL defaults to not using flash attention, this PR modified the related pytest correctly.”
　　　[#571](https://github.com/rebellions-sw/optimum-rbln/pull/571) (jongho): “This made two 'DisallowedTest' cases in 'tests/test_llm.py' stop raising, so they failed.”
　　　외 PR 1건
**응답** [ ] 맞다 　[ ] 틀리다 　[ ] 조건부로 맞다 　[ ] 필요 없는 룰이다 　[ ] 모른다 　[ ] 기억나지 않는다 　· 메모:

### TEST-005 · 판단형

**룰** 하위 호환(BC) 검증과 경계값 설정 조합은 PR마다 실행되는 CI 케이스로 고정한다.
**근거** [#345](https://github.com/rebellions-sw/optimum-rbln/pull/345) (kblee): “I think we need tests to verify bc in each PR.”
　　　[#720](https://github.com/rebellions-sw/optimum-rbln/pull/720) (junamsong): “No CI covers either configuration”
**응답** [ ] 맞다 　[ ] 틀리다 　[ ] 조건부로 맞다 　[ ] 필요 없는 룰이다 　[ ] 모른다 　[ ] 기억나지 않는다 　· 메모:

### TEST-006 · 판단형

**룰** 무거운 추론 테스트와 Hub push 검증은 기본 PR 테스트가 아닌 스케줄 CI에서만 실행한다.
**근거** [#215](https://github.com/rebellions-sw/optimum-rbln/pull/215) (jongho): “Add test for checking whether push-to-hub is working to scheduled CI”
　　　[#203](https://github.com/rebellions-sw/optimum-rbln/pull/203) (jongho): “SDXL inference test to scheduled test only”
**응답** [ ] 맞다 　[ ] 틀리다 　[ ] 조건부로 맞다 　[ ] 필요 없는 룰이다 　[ ] 모른다 　[ ] 기억나지 않는다 　· 메모:

### TEST-015 · 판단형

**룰** 모델·custom op 변경은 대표 구성으로 컴파일 스모크 테스트를 돌려 컴파일러 패턴과 맞는지 확인한다.
**조건** 여러 텐서 병렬 크기, attention 변형과 MoE 모델을 각 하나 이상 포함한다.
**근거** [#566](https://github.com/rebellions-sw/optimum-rbln/pull/566) (thkim): “Compilation smoke test for at least one attention variant and one MoE model to confirm the new signatures still match the RBLN co…”
　　　[#800](https://github.com/rebellions-sw/optimum-rbln/pull/800) (jaehunryu): “Compiles to a single device function. TP4/8/16 scaling was checked.”
**응답** [ ] 맞다 　[ ] 틀리다 　[ ] 조건부로 맞다 　[ ] 필요 없는 룰이다 　[ ] 모른다 　[ ] 기억나지 않는다 　· 메모:

### TEST-016 · 판단형

**룰** PR 설명에는 실행한 테스트 명령, 결과, 실행 환경, 테스트 수를 적는다.
**조건** 기존에 있던 실패는 원인과 함께 표시한다.
**근거** [#495](https://github.com/rebellions-sw/optimum-rbln/pull/495) (jongho): “'pytest tests/test_llm.py' - 316 passed, 8 errors (pre-existing compiler bfloat16 issue in TestQLlamaForCausalLM)”
　　　[#808](https://github.com/rebellions-sw/optimum-rbln/pull/808) (jongho): “On an NPU host, 'tests/unit' (45 tests)”
**응답** [ ] 맞다 　[ ] 틀리다 　[ ] 조건부로 맞다 　[ ] 필요 없는 룰이다 　[ ] 모른다 　[ ] 기억나지 않는다 　· 메모:

### TEST-018 · 판단형

**룰** 메모리 관련 변경과 모델 테스트는 호스트 RSS 증가를 측정한다.
**조건** dense 모델은 체크포인트 크기의 0.5배 미만, MoE 모델은 1.5배 미만이며 최적화 PR은 단계별 RSS를 표로 제시한다.
**근거** [#730](https://github.com/rebellions-sw/optimum-rbln/pull/730) (thkim): “safetensors mmap views, measured ~0.02x); MoE models under 1.5x”
　　　[#714](https://github.com/rebellions-sw/optimum-rbln/pull/714) (thkim): “Measured on one real 'Qwen3MoeSparseMoeBlock'”
**응답** [ ] 맞다 　[ ] 틀리다 　[ ] 조건부로 맞다 　[ ] 필요 없는 룰이다 　[ ] 모른다 　[ ] 기억나지 않는다 　· 메모:

### TEST-020 · 판단형

**룰** 테스트는 같은 프로세스에서 다른 테스트와 함께 돌아도 통과해야 한다.
**조건** 테스트 클래스는 `tearDownClass`에서 만든 모델 객체를 해제한다.
**근거** [#810](https://github.com/rebellions-sw/optimum-rbln/pull/810) (thkim): “Fixed by dropping 'cls.model' and collecting in 'tearDownClass'.”
　　　[#722](https://github.com/rebellions-sw/optimum-rbln/pull/722) (thkim): “the assertion only holds when the test runs alone.”
**응답** [ ] 맞다 　[ ] 틀리다 　[ ] 조건부로 맞다 　[ ] 필요 없는 룰이다 　[ ] 모른다 　[ ] 기억나지 않는다 　· 메모:

### TEST-022 · 판단형

**룰** 테스트 간 실행 순서 의존성은 이름의 알파벳 순서로 보장한다.
**조건** 샤딩은 클래스 단위로 나눠 한 모델이 두 작업에서 동시에 컴파일되지 않게 하고 분할 후에도 수집 순서를 유지한다.
**근거** [#744](https://github.com/rebellions-sw/optimum-rbln/pull/744) (thkim): “so a model never compiles in two jobs at once”
　　　[#462](https://github.com/rebellions-sw/optimum-rbln/pull/462) (jongho): “Since unittest executes tests in alphabetical order, this ensures the push test completes before the pull test”
**응답** [ ] 맞다 　[ ] 틀리다 　[ ] 조건부로 맞다 　[ ] 필요 없는 룰이다 　[ ] 모른다 　[ ] 기억나지 않는다 　· 메모:

### TEST-028 · 판단형

**룰** 런타임이 필요하지 않은 테스트는 런타임을 생성하지 않는다.
**조건** 아티팩트 저장만 하는 실행 등 하드웨어를 점유하는 단계를 뺀다.
**근거** [#120](https://github.com/rebellions-sw/optimum-rbln/pull/120) (jongho): “This PR patches to not create runtime for a test case (save load) that semantically doesn't need runtime to be created”
　　　[#753](https://github.com/rebellions-sw/optimum-rbln/pull/753) (thkim): “the runtimes were what dragged the hardware in.”
**응답** [ ] 맞다 　[ ] 틀리다 　[ ] 조건부로 맞다 　[ ] 필요 없는 룰이다 　[ ] 모른다 　[ ] 기억나지 않는다 　· 메모:

### TEST-029 · 판단형

**룰** 외부 컴파일러 문제로 비활성화한 테스트는 원인이 해결되면 다시 활성화한다.
**조건** 더 이상 건너뛰지 않는 테스트 레벨 게이트는 제거한다.
**근거** [#387](https://github.com/rebellions-sw/optimum-rbln/pull/387) (jongho): “As rebel-compiler is fixed, we can now restore the diffuser-tests that have been disabled.”
　　　[#747](https://github.com/rebellions-sw/optimum-rbln/pull/747) (thkim): “Leaving it in place would read as a claim that is no longer true”
**응답** [ ] 맞다 　[ ] 틀리다 　[ ] 조건부로 맞다 　[ ] 필요 없는 룰이다 　[ ] 모른다 　[ ] 기억나지 않는다 　· 메모:

### TEST-032 · 판단형

**룰** 테스트에서 Hub 접근은 기본 차단하고 명시적으로 표시된 테스트만 온라인을 허용한다.
**조건** push한 값은 main이 아니라 push가 만든 커밋 sha로 읽어 검증한다.
**근거** [#743](https://github.com/rebellions-sw/optimum-rbln/pull/743) (thkim): “Resolve the sha of the commit the push just created and read back at that revision.”
　　　[#688](https://github.com/rebellions-sw/optimum-rbln/pull/688) (junamsong): “테스트가 도는 동안 Hub를 끄고, 다음 두 경우에만 켭니다.”
**응답** [ ] 맞다 　[ ] 틀리다 　[ ] 조건부로 맞다 　[ ] 필요 없는 룰이다 　[ ] 모른다 　[ ] 기억나지 않는다 　· 메모:
