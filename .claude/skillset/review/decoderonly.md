# decoderonly — 룰 3개

결정 필요 0 · 코드 일치 1 · 코드 불일치 0 · 판단형 2 · 폐기 제안 0. 순서는 결정이 필요한 것부터. 각 룰의 **응답** 줄에서 하나를 `[x]`로 표시하고, 틀리거나 조건부면 **메모**에 한 줄 적어 주세요.

## 코드 일치

### DEC-005 · 코드 일치

**룰** 체크포인트 구조가 공통 경로와 다르면 컴파일 전에 가중치를 바꿔 맞추고, 추론 연산이나 전용 클래스를 더하지 않는다.
**조건** 예: 합쳐진 qkv_proj는 q/k/v로 잘라 두고(Phi-3), RoPE 짝 순서가 다르면 q/k 행을 재배열해 둔다(GLM-4).
**근거** [#801](https://github.com/rebellions-sw/optimum-rbln/pull/801) (jaehunryu): “The q/k projection rows are permuted at load time: even rotary dims first, then odd.”
　　　[#798](https://github.com/rebellions-sw/optimum-rbln/pull/798) (jaehunryu): “So the shared q/k/v and gate/up hooks (and their head sharding) apply unchanged.”
**코드 대조** 코드 일치 — 준수 None / 위반 None
**응답** [ ] 맞다 　[ ] 틀리다 　[ ] 조건부로 맞다 　[ ] 필요 없는 룰이다 　[ ] 모른다 　[ ] 기억나지 않는다 　· 메모:

## 판단형

### STY-008 · 판단형

**룰** 추론 경로에서는 매 스텝 텐서를 새로 할당하거나 host와 동기화하거나 불필요하게 복사하지 않는다.
**조건** 마스크는 미리 만들어 재사용하고 broadcast 연산을 쓴다.
**근거** [#563](https://github.com/rebellions-sw/optimum-rbln/pull/563) (thkim): “attn_mask 4d is too slow. You must use 2d attn_mask as input and use brocating ops”
　　　[#604](https://github.com/rebellions-sw/optimum-rbln/pull/604) (eunji): “tensor.sum().item()보다는 이게 저비용이지 않을까요?”
　　　외 PR 1건
**응답** [ ] 맞다 　[ ] 틀리다 　[ ] 조건부로 맞다 　[ ] 필요 없는 룰이다 　[ ] 모른다 　[ ] 기억나지 않는다 　· 메모:

### DEC-009 · 판단형

**룰** 모델 출력(logits, hidden_states 등)은 같은 입력에서 HF 원본 모델과 모양과 뜻이 같아야 한다.
**조건** 예: last_hidden_state는 hidden_states[-1]과 같고, 패딩된 배치의 prefill 출력은 유효 토큰 길이가 아니라 mask 폭으로 돌려준다.
**근거** [#659](https://github.com/rebellions-sw/optimum-rbln/pull/659) (thkim): “Hidden states came back at valid length instead of mask width.”
　　　[#668](https://github.com/rebellions-sw/optimum-rbln/pull/668) (thkim): “violates the base contract that 'last_hidden_state == hidden_states[-1]'”
**응답** [ ] 맞다 　[ ] 틀리다 　[ ] 조건부로 맞다 　[ ] 필요 없는 룰이다 　[ ] 모른다 　[ ] 기억나지 않는다 　· 메모:
