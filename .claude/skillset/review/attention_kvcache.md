# attention · KV cache — 룰 5개

결정 필요 1 · 코드 일치 0 · 코드 불일치 0 · 판단형 2 · 폐기 제안 2. 순서는 결정이 필요한 것부터. 각 룰의 **응답** 줄에서 하나를 `[x]`로 표시하고, 틀리거나 조건부면 **메모**에 한 줄 적어 주세요.

## 결정 필요

### ATT-003 · 결정 필요

**룰** 컴파일 그래프에 들어가는 rotary cos/sin의 계산 방식(numpy 또는 torch)은 하나로 정해 적용한다.
**조건** 기준은 재현성과 HF 레퍼런스 일치 중 하나여야 한다.
**근거** [#680](https://github.com/rebellions-sw/optimum-rbln/pull/680) (thkim): “In exchange it introduced a **permanent ≤1-ulp rotary difference against the torch-based HF native reference**”
　　　[#672](https://github.com/rebellions-sw/optimum-rbln/pull/672) (thkim): “Host-computed rotary cos/sin fed to compiled graphs as inputs therefore breaks run-to-run reproducibility.”
**질문** cos/sin은 재현성을 위해 numpy로 계산하는가, HF와 같은 torch로 계산하는가?
**응답** [ ] 맞다 　[ ] 틀리다 　[ ] 조건부로 맞다 　[ ] 필요 없는 룰이다 　[ ] 모른다 　[ ] 기억나지 않는다 　· 메모:

## 판단형

### ATT-004 · 판단형

**룰** rope cos/sin 테이블은 로드 시점에 한 번 만들어 재사용한다.
**조건** 정적 inv_freq를 가진 rope 타입에만 쓰고 테이블 범위 밖 위치는 동일 결과의 동적 경로로 폴백한다.
**근거** [#663](https://github.com/rebellions-sw/optimum-rbln/pull/663) (thkim): “Out-of-range positions (e.g. mrope temporal ids) fall back to the bit-identical dynamic numpy path.”
　　　[#653](https://github.com/rebellions-sw/optimum-rbln/pull/653) (dkhong): “To prevent unexpected mismatch while computing rope, calculate them before inference and reuse.”
　　　외 PR 1건
**응답** [ ] 맞다 　[ ] 틀리다 　[ ] 조건부로 맞다 　[ ] 필요 없는 룰이다 　[ ] 모른다 　[ ] 기억나지 않는다 　· 메모:

### ATT-022 · 판단형

**룰** 커스텀 attention 커널이 변경되면 모델 쪽 호출부도 함께 동기화한다.
**조건** 모델은 배치 단위 커널을 호출해 모델 코드를 단순하게 유지한다.
**근거** [#20](https://github.com/rebellions-sw/optimum-rbln/pull/20) (myeongbo): “Update model to call multi-batch attention kernel to simplify model and clear attention kernel writing.”
　　　[#24](https://github.com/rebellions-sw/optimum-rbln/pull/24) (thkim): “This PR is to keep in sync with the custom kernel changes in attention_decode on a batch basis.”
**응답** [ ] 맞다 　[ ] 틀리다 　[ ] 조건부로 맞다 　[ ] 필요 없는 룰이다 　[ ] 모른다 　[ ] 기억나지 않는다 　· 메모:

## 폐기 제안

### ATT-005 · 폐기 제안

**룰** 텐서 병렬 샤딩 대상 축(Q/K head 등)은 텐서 레이아웃의 가장 바깥 축에 둔다.
**조건** 샤딩 절단선이 head 경계에 놓이게 하고, 정적 캐시는 prefill과 decode 그래프에서 같은 방식으로 분할한다.
**근거** [#768](https://github.com/rebellions-sw/optimum-rbln/pull/768) (jaehunryu): “a static cache must be cut the same way in the prefill and decode graphs.”
　　　[#803](https://github.com/rebellions-sw/optimum-rbln/pull/803) (seinpark): “every tensor-parallel cut falls on a head boundary”
　　　외 PR 1건
**응답** [ ] 맞다 　[ ] 틀리다 　[ ] 조건부로 맞다 　[ ] 필요 없는 룰이다 　[ ] 모른다 　[ ] 기억나지 않는다 　· 메모:

### ATT-006 · 폐기 제안

**룰** 직접 설계하는 디바이스 상주 정적 상태 캐시는 64의 배수인 축을 가장 안쪽 축에 둔다.
**조건** 기본값으로 정하는 시퀀스 길이와 청크 크기는 64의 배수로 올림하거나 검증한다.
**근거** [#816](https://github.com/rebellions-sw/optimum-rbln/pull/816) (seinpark): “The text-encoder and vision-encoder sequence lengths are rounded up to multiples of 64.”
　　　[#804](https://github.com/rebellions-sw/optimum-rbln/pull/804) (seinpark): “The device needs the last axis of a static cache to be a multiple of 64.”
**코드 대조** 결정 필요 — 준수 4 / 위반 3. dev에서도 main과 동일하다: 64 단위 정렬은 prefill_chunk_size 검증(modeling_attention_utils.py:95), TST dec_max_seq_len 올림(:237), Qwen3.5 conv_state(innermost=conv_dim, 'a mul…
**응답** [ ] 맞다 　[ ] 틀리다 　[ ] 조건부로 맞다 　[ ] 필요 없는 룰이다 　[ ] 모른다 　[ ] 기억나지 않는다 　· 메모:
