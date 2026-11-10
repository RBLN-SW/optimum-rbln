# 논쟁 v0 (optimum-rbln)

입력 논쟁 43건 → 주제 병합 후 40건 (병합 3건: 최상위 옵션 전파 2건, rbln_config 로드 경계 2건, fp32 캐스트 2건). 해결 13건, 미해결 27건. 영역별로 묶었다. 리뷰 댓글 클러스터(원 D01~D19)는 추출 단계에서 원문 인용이 보존되지 않아 `인용: 없음`으로 표기하고 PR 번호만 남겼다. 해당하는 rules_v0 규칙 id는 이 단계에서 확인하지 못했다.

## rbln_config

### R1. 최상위 컴파일 옵션(batch_size 등)을 서브모듈로 전파할지
- 입장: 작성자(rebel-thkim): 부모 batch_size를 기본값(soft default)으로 전파하고 서브모듈 값을 우선(#681). 이전에는 gemma4 언어 모델에 `force_kwargs`로 강제(#650). 리뷰어: 서브모듈 전파는 의도된 동작이 아니므로 정책을 transformers 쪽에서 논의(#445). 리뷰어: 전역 설정이 connected pipeline으로 넘어가는지 확인 요청(#15)
- 해결: 예 (#681: soft default 전파, `force_kwargs`는 고정이 필요한 서브모듈에만)
- PR: #15, #445, #650, #681
- 인용: "`force_kwargs=True` stays; it still guards `use_inputs_embeds`, which gemma4 requires." (#650) / "Propagate the top-level `batch_size` to the language-model submodule as a **soft default**" (#681)

### R2. 충돌하는 설정 값을 예외로 막을지, 경고 후 사용자 값을 쓸지
- 입장: 작성자: 강제되는 키만 예외, 나머지는 경고 없이 사용자 값 적용(혼합안). 리뷰어(kblee): 예외 처리가 낫다
- 해결: 아니오
- PR: #267
- 인용: 없음

### R3. config 값 검증 범위: 잘못된 값도 필드별로 검사할지
- 입장: 리뷰어(kblee): 0 이하 같은 잘못된 값에 예외 추가. 리뷰어(jongho): 다른 필드도 검사하지 않으므로 이 필드만 추가할 필요 없음
- 해결: 아니오
- PR: #437
- 인용: 없음

### R4. 입력 정규화(dtype 등)를 방어적으로 둘지 최소화할지
- 입장: 작성자(kblee): `_dtype`이 항상 str이라는 보장이 없어 방어적 normalize 유지. 리뷰어(jongho): 과한 설계, 단순 대입으로 충분
- 해결: 아니오
- PR: #389
- 인용: 없음

### R5. output_hidden_states / output_attentions 옵션을 어디서 결정할지
- 입장: #38 작성자: 당분간 `_VisionEncoder`에서 `output_hidden_states=True` 유지, 필요 시 파이프라인별 관리로 전환. #174 작성자: 사용자가 None을 넘기면 컴파일 시점 `rbln_config` 값을 따르는 것이 의도. #631: 미지원 옵션은 config 단계에서 `NotImplementedError`로 거부하고 지원은 후속 작업으로 분리
- 해결: 아니오
- PR: #38, #174, #631
- 인용: 없음

### R6. rbln_config 로드 경계: 객체 게이트 유지 대 부분 객체 축소, dict 입력 처리 범위
- 입장: (초안) 진입점에서 모든 객체를 override로 축소(rebel-seinpark). (최종) 객체 게이트를 유지하고 부분 객체만 호출 지점에서 dict로 변환(rebel-seinpark, #702). dict `rbln_config`는 `rbln_` 접두사로 바꿔 `from_pretrained`에 넘기며, 설정 객체를 `load`로 읽는 경로는 접두사 변환 보장이 안 돼 후속 수정 필요(rebel-seinpark). 리뷰어(jongho): dict 형태 입력 지원 여부 질문
- 해결: 예, 부분(#702 최종안은 객체 게이트 유지. dict 경로 후속 수정은 별도)
- PR: #67, #475, #495, #702
- 인용: "that turned misuse into silence for *all* load paths and was reverted in favor of the call-site conversion." (#702)

### R7. 입력 모양 불일치를 추론 전에 ValueError로 검사할지, 런타임에서만 검사할지
- 입장: 추론 전에 `ValueError`로 조기 검증하고 두 값을 메시지에 명시(rebel-dkhong). 런타임만 모양 검증 지점으로 둠(rebel-seinpark)
- 해결: 아니오
- PR: #770
- 인용: "I think it would be better to explicitly raise a `ValueError` before inference" (#770, rebel-dkhong) / "I'd rather leave the runtime as the single place that validates shapes." (#770, rebel-seinpark)

### R8. 서브모듈 디바이스 수가 줄어들 때 구성 시점에 거부할지, 배치 정보를 조용히 축소할지
- 입장: 구성 시점에 서브모듈을 명시해 거부(대안으로 제시). 배치 정보를 축소해 보수적으로 처리(본 PR, rebel-junamsong)
- 해결: 아니오
- PR: #776
- 인용: "The alternative is to refuse the construction instead, naming the submodule" (#776) / "This takes the conservative route; happy to switch if you would rather it raised." (#776)

### R9. compiler 기능을 쓰는 경로를 옵트인 설정으로 둘지, 옵션 없이 항상 쓸지
- 입장: 기본값 False 옵트인 설정으로 기존 경로 보존(rebel-yskim, #815). 옵션 없이 항상 사용하고 미지원 컴파일러는 검증에서 거부(rebel-seinpark, #803)
- 해결: 아니오
- PR: #803, #815
- 인용: "The option defaults to `False`, preserving the existing compilation path." (#815) / "The conv kernels are always used, as the core kernels already are." (#803)

### R10. 메모리 예산 인자의 표현 형식(비율 실수 대 퍼센트 문자열)
- 입장: 퍼센트나 바이트로 직관적으로 표현하는 초안(rebel-jongho, #629). vLLM 관례에 맞춰 float 비율을 권장 형식으로 하는 리뷰 피드백 반영(rebel-jongho, #637)
- 해결: 예 (float 비율)
- PR: #629, #637
- 인용: "expressed intuitively in bytes or a percentage (`memory_budget`)." (#629) / "Note: the float-vs-`%` representation follows reviewer feedback" (#637)

### R11. 비생성(non-generation) 모델에서 batch_size 속성의 의미
- 입장: 작성자(thkim): 생성 모델에서는 디코더 최대 배치를 뜻하지만 `can_generate`가 false이면 의미가 모호하며 대안은 아직 없음
- 해결: 아니오
- PR: #245
- 인용: 없음

### R12. 미지원 dtype 입력을 예외로 막을지, fp32로 자동 승격할지
- 입장: 리뷰어(jongho): 기존에는 non-fp32에서 실패했는데 새 코드는 조용히 fallback하므로 의도인지 질문. 작성자(dkhong): 미지원 클래스는 예외 대신 fp32로 지원(VAE `force_upcast` 등 혼합 dtype 고려)
- 해결: 아니오
- PR: #673
- 인용: 없음

## submodule

### S1. Qwen3.5 구현을 Qwen3-VL 클래스 상속으로 묶을지, 의도적으로 분리 유지할지
- 입장: 리뷰어(jongho): 메서드 복사가 많아 상속이나 공통 base 제안, 분리가 의도라면 명시 요청. 작성자(seinpark): qwen3_vl 변경 영향을 막기 위해 두 tower를 의도적으로 분리
- 해결: 아니오
- PR: #631
- 인용: 없음

### S2. Cosmos 안전 검사기(safety checker)를 원본 파이프라인 구조대로 둘지, 서브모듈로 옮길지
- 입장: 리뷰어(jongho): 원본 checker 없이 파이프라인을 초기화하고 자체 checker를 붙임. 작성자(seinpark): 가능하나 `rbln_config`를 checker에 전달하는 경로 필요. 작성자(seinpark, #191): 원본과 달리 `_submodules`에 넣어도 되는지 질문
- 해결: 아니오 (#168 결정 기록은 `RBLNCosmosSafetyChecker`가 기존 pipeline 내 `CosmosSafetyChecker`를 대체한다는 것이나, 서브모듈 이동 여부는 #191에서 미결)
- PR: #118, #168, #191
- 인용: 없음

### S3. 서브모듈 아티팩트 저장 위치: 평평한 형제 배치 유지 대 부모 서브폴더 중첩
- 입장: 평평한 저장과 평평한 로드 유지(기존). 부모 서브폴더 안에 중첩 저장하고 부모 기준으로 로드(PR 방식, rebel-seinpark)
- 해결: 예
- PR: #703
- 인용: "Flat save + flat load: a consistent world." (#703)

### S4. 디바이스별 폐지 플래그를 모델마다 추가할지, 범용 create_runtimes 경로로 통일할지
- 입장: 모델별 `_load_visual_runtime` 플래그 추가(#750, rebel-seinpark). 지연 로드로 `create_runtimes=False`를 범용화(#760, rebel-jongho)
- 해결: 예 (#760 범용화, #750은 대체됨)
- PR: #750, #760
- 인용: "#750 proposed adding `_load_visual_runtime` to Qwen2.5-VL and is superseded by this." (#760)

## lifecycle

### L1. generation_config에 transformers 버전 정보를 모든 모델에 저장할지
- 입장: 리뷰어(kblee): 모델 전반에 적용. 작성자(junamsong): Whisper 외 모델은 transformers 4.50.3에서 검증되었고 일반화는 불필요한 오버헤드
- 해결: 아니오
- PR: #97
- 인용: 없음

### L2. deprecation 컷오프에 걸린 테스트를 테스트 쪽에서 조정할지, 컷오프를 옮길지
- 입장: 테스트 실패를 컷오프 도달로 해석해 테스트 변경으로 해결(rebel-junamsong, #784 초기). 0.11.x 아티팩트 호환을 위해 컷오프를 0.13.0으로 이동(rebel-thkim, #785)
- 해결: 예 (컷오프 이동)
- PR: #784, #785
- 인용: "reading the failure as the deprecation having reached its cutoff" (#784) / "0.12.0 is required to stay backward compatible with 0.11.x artifacts" (#785)

## testing

### T1. CI 테스트에 공식 체크포인트를 쓸지, tiny 모델을 쓸지
- 입장: 리뷰어(jongho): tiny 모델이 안 되면 `TEST_LEVEL=FULL`로 PR CI에서 제외하거나 가중치를 업로드. 작성자(seinpark): diffusers 테스트용 tiny 모델을 찾아 추가(PR 52)
- 해결: 예 (tiny 모델 추가)
- PR: #38, #52
- 인용: 없음

### T2. dummy device에서 발생한 CI 실패를 optimum-rbln에서 고칠지 배포 도구에서 고칠지
- 입장: optimum-rbln의 device 처리 변경이 필요하다는 CI 실패 보고. optimum-rbln 변경은 불필요하며 `rbln_deploy`에서 처리(rebel-junamsong)
- 해결: 예 (배포 도구에서 처리)
- PR: #776
- 인용: "No change to optimum-rbln is called for by this failure." (#776)

### T3. 테스트 위치: 기존 모델 테스트 파일에 넣을지 별도 CPU 단위 테스트 디렉터리에 둘지
- 입장: 기존 `test_llm.py`에 포함(검토된 대안). `tests/unit/cpu/`에 별도로 둠(본 PR, rebel-thkim)
- 해결: 아니오
- PR: #746
- 인용: "So porting them into `test_llm.py` — the other option on the table — would be wrong twice over" (#746)

### T4. HF Hub rate limit 대응을 테스트 호출부 helper, CI 2단계 캐시, conftest 훅 중 어디에 둘지
- 입장: 호출부를 cache-first helper로 감쌈(#685, rebel-thkim). CI에서 캐시를 미리 채우고 `HF_HUB_OFFLINE=1`로 실행(#686, rebel-thkim). conftest 훅 한 곳에서 테스트별로 오프라인 여부 결정(#688, rebel-junamsong)
- 해결: 아니오
- PR: #685, #686, #688
- 인용: "Mitigation for the HF rate-limit incident (option 2: CI-level two-phase)." (#686) / "#685는 46개 호출부를 helper로 감쌉니다." (#688)

## pr_process

### P1. 리뷰 지적을 같은 PR에서 고칠지, 후속 PR로 미룰지
- 입장: #114 작성자: 여러 지적을 '다음 PR에서 반영'. #118 리뷰어: 권장되지 않으니 릴리스 전까지 이번 PR에서 수정. #133 작성자: 처음엔 별도 PR로 미루려 했으나 이번 PR에서 반영
- 해결: 예 (암묵적으로 같은 PR 반영으로 수렴, 명문 합의는 없음)
- PR: #114, #118, #133
- 인용: 없음

### P2. 배포 후 수정을 dev 팁에 먼저 넣을지, 릴리스 태그 위에서 만들지
- 입장: dev 팁에 직접 수정한 뒤 릴리스 라인에 같은 수정 반영(rebel-junamsong, #612, #613). 릴리스 태그 위에서 수정을 재구성하고 dev를 역병합(rebel-junamsong, #617)
- 해결: 예 (#617 릴리스 태그 위 재구성)
- PR: #612, #613, #617
- 인용: "Hotfix into `release-v0.11.0`. Same fix as #612 (targets `dev`)." (#613) / "This PR rebuilds the fix on top of `v0.11.0` so the post-release becomes a proper child of the release line." (#617)

### P3. dev→main 백머지 PR의 병합 방식(merge commit 대 squash)
- 입장: merge commit으로만 병합(rebel-junamsong). 저장소 설정은 squash만 허용(`allow_merge_commit: false`)
- 해결: 아니오 (이후 #727에서 main 자체가 제거됨)
- PR: #695
- 인용: "The repository currently allows **squash only** (`allow_merge_commit: false`)" (#695)

## other

### O1. 파이프라인 간 공통 검증 코드를 어디에 둘지(mixin vs utils)
- 입장: 리뷰어(jongho): 비슷한 코드를 합칠 수 있는지 질문. 작성자(dkhong): mixin 또는 pipelines 하위 utils 파일 중 어디에 둘지 의견 요청
- 해결: 아니오
- PR: #54
- 인용: 없음

### O2. 사용하지 않는 인자를 docstring에 적을지 생략할지
- 입장: 작성자(dkhong): UNet처럼 미사용 인자는 생략해 혼란을 줄임. 대안(작성자): 일관성을 위해 kwargs 전체를 항상 기재, 정책 결정 필요
- 해결: 아니오
- PR: #367
- 인용: 없음

### O3. 혼동되는 함수 이름을 바꿔야 하는지
- 입장: 리뷰어(jongho): 혼동될 만한 함수가 없으니 이름 변경 불필요. 작성자(seinpark): 두 이름이 혼동되어 `_update_submodule_config_from_model`로 변경
- 해결: 아니오 (작성자가 변경했으나 명시적 합의 기록 없음)
- PR: #433
- 인용: 없음

### O4. KV 캐시 텐서 이름을 고유하게 할지, 이름을 외부(vLLM)에서 넘길지
- 입장: 리뷰어(eunji): key/value 텐서 이름이 중복이므로 고유 이름. 리뷰어(yskim): `mark_static_address(tensor, name)` 유지하고 vLLM이 이름을 명시적으로 전달. 리뷰어(jongho): kv 튜플 가정이 많아 이름 체계 변경은 다음 PR로
- 해결: 아니오
- PR: #437
- 인용: 없음

### O5. 입력 이름(model_input_names) 순서를 고정할지
- 입장: 리뷰어(jongho): key-value 매핑만으로 충분한지 질문. 작성자(kblee): 컴파일 설정과 forward 인자 불일치를 막기 위해 순서 고정. 리뷰어(jongho): 그 불일치는 사용자 실수이므로 지원 불필요
- 해결: 아니오
- PR: #383
- 인용: 없음

### O6. getattr 기본값으로 fallback할지, 명시적으로 속성을 읽을지
- 입장: 리뷰어(junamsong): 기존 명시적 방식이 더 안전하고 의도가 분명. 작성자(thkim): transformers 4.51 이후 config의 `head_dim`이 None으로 들어와 기존 방식이 실패
- 해결: 아니오
- PR: #162
- 인용: 없음

### O7. RBLN 래퍼 설계 방식: 예제 편의용 컴파일 헬퍼 대 교체 가능한 인스턴스 설계 원칙
- 입장: 예제 코드를 HF 예제와 같게 하면서 구현 방식 검토 중(rebel-seinpark). 교체 가능한 인스턴스로 설계해야 한다는 원칙 적용(rebel-jongho)
- 해결: 예
- PR: #168
- 인용: "I'm still concerning about how to implement this model due to the root philosophy of Optimum-rbln." (#168, rebel-seinpark) / "are designed so that instances of the class can be replaced in existing huggingface code." (#168, rebel-jongho)

### O8. 디코더 전용 모델 래퍼를 Model/CausalLM으로 분리할지, is_causal_lm 플래그를 유지할지
- 입장: Model을 CausalLM의 일부로 두도록 래퍼 분리(리뷰 코멘트). 중복 코드가 늘어나므로 플래그를 유지하고 이후 리팩터링(rebel-kblee)
- 해결: 아니오 (decisions_v0에 2025-07-17 #219 "현재 구조 유지 후 리팩터링" 결정이 있으나 추출 데이터는 미해결 표기)
- PR: #219
- 인용: "so that `Model` can properly be part of `CausalLM`." (#219, rebel-kblee) / "there would have been too much repetition of duplicate code" (#219, rebel-kblee)

### O9. eager 어텐션 상한을 넘는 미지정 attn_impl을 flash_attn으로 자동 승격할지
- 입장: 자동 승격 후 명시적 eager 요청만 거부(#721, rebel-junamsong). 패밀리별 한계 레코드로 검증하고 자동 승격은 후속 PR로 분리(#729, rebel-junamsong)
- 해결: 아니오
- PR: #721, #729
- 인용: "an unset `attn_impl` with `max_seq_len >= 32768` on CR13+" (#721) / "Auto-promoting an unset `attn_impl` to `flash_attn` above the eager cap went away with #721." (#729)

### O10. fp32 캐스트: HF 충실도를 위해 복원할지, 단일 활성 dtype을 유지할지
- 입장: HF 충실도와 fp16 로드 가드를 위해 GatedDeltaNet 커널 fp32 island 복원(rebel-seinpark, #710). 단일 디바이스 dtype 유지를 위해 캐스트를 제거한 기존 결정(#259 계열). 비전 cos/sin 그래프 입력은 HF처럼 fp32로 줄지(HF 원본), 활성 dtype을 유지할지(호스트 및 이전 아티팩트 호환성, rebel-thkim, #711)
- 해결: 아니오 (GatedDeltaNet은 복원으로 해결, 비전 cos/sin은 미결)
- PR: #259, #710, #711
- 인용: "The casts were originally removed to keep the kernel in a single device dtype." (#710) / "stay in the activation dtype here; HF feeds them in fp32." (#711)

### O11. LoRA 경로를 adapter dtype으로 계산할지, 기존처럼 base weight dtype으로 둘지
- 입장: peft 기본 방식대로 adapter 고유 dtype으로 계산. 기존 동작인 base weight dtype 유지(rebel-thkim)
- 해결: 예 (기존 유지)
- PR: #711
- 인용: "so the existing behavior is kept." (#711)

### O12. gate_up_proj 분할 결과를 strided view로 직접 넘길지, contiguous 복사본을 쓸지
- 입장: strided view를 연산에 직접 전달(시도만 함). contiguous 복사본 전달(커밋 1에서 채택)
- 해결: 예 (contiguous 복사본)
- PR: #714
- 인용: "the compiler materializes contiguous copies anyway, so it saves nothing over" (#714)

### O13. decoderonly rotary cos/sin을 numpy로 계산할지 torch(HF 레퍼런스와 동일)로 계산할지
- 입장: numpy 유지, 스레드 구성과 무관한 재현성 확보(#672). 컴파일 상수 경로만 torch로 되돌림, HF 레퍼런스와 1-ulp 차이가 MoE 라우팅을 흔듦(#680)
- 해결: 아니오
- PR: #672, #680
- 인용: "numpy's cos/sin are reproducible across thread counts" (#672) / "In exchange it introduced a **permanent ≤1-ulp rotary difference against the torch-based HF native reference**" (#680)

### O14. 정확도 대비 이득이 일부 구성에서 작은 회귀인 커스텀 op를 모든 구성에 쓸지
- 입장: 모든 구성에 커스텀 op 사용. 이득이 나는 구성에서만 사용(리뷰어 판단 대기, rebel-jaehunryu)
- 해결: 아니오
- PR: #768
- 인용: "Whether to use the op for every configuration is left to the reviewers" (#768)

### O15. KV 캐시 블록 산정에 쓰는 DRAM 버퍼 추정치(tp당 500MB 상향 대 기존 총 1GB 추정 유지)
- 입장: 기존 추정을 유지하면 모델이 커져 실패(rebel-jongho). 버퍼를 tp당 500MB로 키우면 최대 블록 수가 줄어 정밀 추정 필요(rebel-jongho)
- 해결: 아니오
- PR: #84
- 인용: "If we leave the buffer estimation as it is, we get models that blow up." (#84) / "This makes the existing max num blocks smaller, so it seems we need to estimate the buffer size more precisely" (#84)

## 인터뷰 문항 초안

미해결 27건. 보기 (모른다)와 (기억나지 않는다)는 모든 문항에 공통이며 마지막 두 항목이다.

1. [R2] 충돌하는 설정 값(강제 키와 사용자 값)을 어떻게 처리하는가? (a) 충돌 시 항상 예외 (b) 강제되는 키만 예외, 나머지는 경고 없이 사용자 값 적용 (c) 모른다 (d) 기억나지 않는다
2. [R3] 잘못된 config 값(예: 0 이하)을 필드별로 검사해 예외를 내는가? (a) 값 오류마다 필드별로 검사 (b) 다른 필드와 맞춰 별도 검사 없이 둠 (c) 모른다 (d) 기억나지 않는다
3. [R4] config 입력(dtype 등) 정규화의 기준은? (a) 타입을 보장할 수 없으므로 방어적으로 normalize (b) 단순 대입으로 충분, 과한 설계 지양 (c) 모른다 (d) 기억나지 않는다
4. [R5] `output_hidden_states` / `output_attentions` 같은 옵션은 어디서 결정하는가? (a) 모듈 내부 하드코딩 후 필요 시 파이프라인별 관리 (b) 사용자 None이면 컴파일 시점 `rbln_config` 값을 따름 (c) 미지원 옵션은 config 단계에서 `NotImplementedError`로 거부 (d) 모른다 (e) 기억나지 않는다
5. [R7] 입력 모양 불일치는 어디서 검사하는가? (a) 추론 전 `ValueError`로 조기 검증 (b) 런타임만 검증 지점 (c) 모른다 (d) 기억나지 않는다
6. [R8] 서브모듈 디바이스 수가 줄어들 때는? (a) 구성 시점에 서브모듈을 명시해 거부 (b) 배치 정보를 조용히 축소해 보수적으로 처리 (c) 모른다 (d) 기억나지 않는다
7. [R9] compiler 기능 사용 경로는? (a) 기본 False인 옵트인 설정 (b) 옵션 없이 항상 사용, 미지원 컴파일러는 검증에서 거부 (c) 모른다 (d) 기억나지 않는다
8. [R11] `can_generate`가 false인 모델에서 `batch_size`는 무엇을 뜻하는가? (a) 디코더 최대 배치 의미를 그대로 유지(모호함 감수) (b) 별도 의미/이름을 도입 (c) 모른다 (d) 기억나지 않는다
9. [R12] 미지원 dtype 입력은? (a) 예외로 막음 (b) fp32로 자동 승격 (c) 모른다 (d) 기억나지 않는다
10. [S1] Qwen3.5와 Qwen3-VL 구현의 관계는? (a) 상속 또는 공통 base로 묶음 (b) 변경 영향 차단을 위해 의도적으로 분리 유지 (c) 모른다 (d) 기억나지 않는다
11. [S2] Cosmos 안전 검사기는? (a) 원본 pipeline 구조대로 두고 pipeline 안에서 대체 (b) 원본 checker 없이 초기화하고 자체 checker 부착 (c) `_submodules`로 이동 (d) 모른다 (e) 기억나지 않는다
12. [L1] `generation_config`에 transformers 버전 정보를 저장하는 범위는? (a) 모든 모델에 적용 (b) 검증 이슈가 있는 모델(Whisper)에만 적용 (c) 모른다 (d) 기억나지 않는다
13. [T3] CPU 단위 테스트의 위치는? (a) 기존 `test_llm.py`에 포함 (b) `tests/unit/cpu/`에 별도 배치 (c) 모른다 (d) 기억나지 않는다
14. [T4] HF Hub rate limit 대응은 어디에 두는가? (a) 호출부를 cache-first helper로 감쌈 (b) CI에서 캐시 선채움 후 `HF_HUB_OFFLINE=1` (c) conftest 훅 한 곳에서 테스트별 오프라인 결정 (d) 모른다 (e) 기억나지 않는다
15. [P3] dev→main 백머지 PR의 병합 방식은? (a) merge commit만 사용 (b) 저장소 설정대로 squash (c) 모른다 (d) 기억나지 않는다
16. [O1] 파이프라인 간 공통 검증 코드의 위치는? (a) mixin (b) pipelines 하위 utils 파일 (c) 모른다 (d) 기억나지 않는다
17. [O2] 사용하지 않는 인자를 docstring에 적는가? (a) 미사용 인자는 생략 (b) 일관성을 위해 kwargs 전체를 항상 기재 (c) 모른다 (d) 기억나지 않는다
18. [O3] 혼동 가능성이 있는 함수 이름은? (a) 혼동되면 이름을 바꿈 (b) 혼동될 함수가 없으면 변경하지 않음 (c) 모른다 (d) 기억나지 않는다
19. [O4] KV 캐시 텐서 이름은? (a) key/value 각각 고유 이름 (b) `mark_static_address(tensor, name)` 유지, vLLM이 이름을 명시 전달 (c) 이름 체계 변경은 다음 PR로 미룸 (d) 모른다 (e) 기억나지 않는다
20. [O5] `model_input_names` 순서는? (a) 컴파일 설정과 forward 인자 불일치 방지를 위해 순서 고정 (b) key-value 매핑만 사용, 불일치는 사용자 실수로 간주 (c) 모른다 (d) 기억나지 않는다
21. [O6] config 속성 읽기는? (a) 명시적으로 읽음 (b) `getattr` 기본값으로 fallback(transformers 4.51 이후 `head_dim`이 None인 경우 대응) (c) 모른다 (d) 기억나지 않는다
22. [O8] decoder-only 래퍼는? (a) Model/CausalLM으로 분리 (b) `is_causal_lm` 플래그 유지 후 리팩터링 (c) 모른다 (d) 기억나지 않는다
23. [O9] eager 어텐션 상한을 넘는 미지정 `attn_impl`은? (a) `flash_attn`으로 자동 승격하고 명시적 eager만 거부 (b) 패밀리별 한계 레코드로 검증하고 자동 승격은 후속 PR (c) 모른다 (d) 기억나지 않는다
24. [O10] 비전 cos/sin 그래프 입력 dtype은? (a) HF처럼 fp32 (b) 활성 dtype 유지(호스트/이전 아티팩트 호환) (c) 모른다 (d) 기억나지 않는다
25. [O13] decoder-only rotary cos/sin 계산은? (a) numpy 유지(스레드 무관 재현성) (b) 컴파일 상수 경로만 torch로 복귀(HF 레퍼런스와 1-ulp 일치) (c) 모른다 (d) 기억나지 않는다
26. [O14] 일부 구성에서 소폭 회귀가 있는 커스텀 op는? (a) 모든 구성에 사용 (b) 이득이 나는 구성에서만 사용 (c) 모른다 (d) 기억나지 않는다
27. [O15] KV 캐시 블록 산정의 DRAM 버퍼 추정은? (a) 기존 총 1GB 추정 유지 (b) tp당 500MB로 상향하고 정밀 추정 도입 (c) 모른다 (d) 기억나지 않는다
