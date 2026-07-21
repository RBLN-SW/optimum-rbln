# Qwen3.5 GatedDeltaNet — RBLN 최적화 구현 정리

이 문서는 Qwen3.5의 `linear_attention`(GatedDeltaNet, 이하 GDN) 레이어를 RBLN에서 컴파일/실행 가능하도록
재작성하면서, **HuggingFace(HF) 원본 구현 대비 무엇을 어떻게 바꿨는지**를 정리한 것이다.

핵심 원칙은 하나다:

> **수학적으로는 HF와 동일한 값을 계산하되, RBLN 컴파일러에서 lowering 되지 않거나 디바이스에서 값이 깨지는 연산만
> 등가(等價)의 다른 연산으로 바꿔 쓴다.**

대상 코드:

- `src/optimum/rbln/transformers/models/qwen3_5/qwen3_5_architecture.py`
  - `rbln_chunk_gated_delta_rule` — PREFILL(청크 병렬) 델타 규칙
  - `rbln_recurrent_gated_delta_rule_step` — DECODE(seq=1) 순차 델타 규칙
  - `Qwen3_5GatedDeltaNet.forward` — short conv + conv_state 캐싱 + 델타 규칙 호출
- 비교 대상 HF: `transformers/models/qwen3_5/modeling_qwen3_5.py`
  - `torch_chunk_gated_delta_rule`, `torch_recurrent_gated_delta_rule`, `torch_causal_conv1d_update`

---

## 0. 전체 구조

Qwen3.5는 하이브리드 디코더다.

- `full_attention` 레이어: gated softmax attention (Qwen3 + output gate + partial RoPE), paged KV cache
- `linear_attention` 레이어: **GatedDeltaNet**, 두 개의 상태를 가짐
  - `conv_state` — short depthwise causal conv의 이전 문맥
  - `recurrent_state` — 델타 규칙의 연상 메모리 `S` (shape `(B, Hv, Dk, Dv)`)

GDN은 두 가지 경로로 동작한다.

| 단계 | 사용 커널 | 이유 |
|------|-----------|------|
| PREFILL | 청크 병렬 델타 규칙 (`rbln_chunk_gated_delta_rule`) | 시퀀스를 한 번에 행렬곱으로 처리 |
| DECODE (seq=1) | 순차 델타 규칙 (`rbln_recurrent_gated_delta_rule_step`) | 토큰 1개씩 상태 갱신 |

두 경로는 동일한 상태 레이아웃을 공유하므로, prefill이 만든 `recurrent_state`를 decode가 그대로 이어받는다.
`conv_state`/`recurrent_state`는 on-device STATIC DRAM 캐시에 올라가며(`mark_static_address`), paged KV
cache처럼 그래프 안에서 `rbln_cache_update`로 읽고 쓴다. 런타임이 0/1 마스크를 주입해 **prefill 윈도우 0에서는
이전 상태를 0으로** 지우고, 이후 윈도우부터는 상태를 이어간다.

또한 optimum-rbln의 prefill 런타임은 프롬프트를 고정 크기 `prefill_chunk_size` 윈도우로 나눠 넣고
윈도우 사이로 `recurrent_state`를 넘긴다. 한 윈도우 **안**은 다시 `gdn_chunk_size`(config, 기본값 =
`prefill_chunk_size`) 크기의 서브청크로 나눠 청크 병렬 규칙을 적용하고 서브청크 사이로도 state를 넘긴다
(`n_chunks = prefill_chunk_size // gdn_chunk_size`; 기본값이면 n_chunks=1 → 한 윈도우 = 한 청크).
이 서브청크 병렬화는 chunk 축을 batch로 두는 3 batch-dim `(B, Hv, n_chunks)` matmul이라 예전 컴파일러에선
OpTiling에 막혔으나(§6 참고), 컴파일러의 5D matmul 지원 이후 컴파일된다.

---

## 1. 배경: Gated Delta Rule 수학

상태 `S ∈ ℝ^{Dk×Dv}` 는 "키를 넣으면 밸류를 뱉는" 연상 메모리다. 토큰 단위 recurrence는 다음과 같다.

```
u_t = β_t (v_t − α_t · Sᵀ_{t-1} k_t)      # 델타: 실제값 − 예측값
S_t = α_t · S_{t-1} + k_t u_tᵀ            # 메모리 갱신 (감쇠 + 새 기록)
o_t = q_tᵀ S_t                            # 출력
```

- `incr_t` : 토큰별 log-decay(<0), `α_t = exp(incr_t)` : 망각 게이트, `β_t` : 학습률
- 누적 `g_t = Σ_{s≤t} incr_s` (cumsum) → `γ_t = exp(g_t)`, 두 위치 사이 감쇠 `γ_t/γ_j = exp(g_t − g_j)`
- `q`는 `1/√Dk`로 스케일, `q`/`k`는 l2norm  (§2 코드의 `incr`/`g` 이름과 동일)

청크 병렬 형태는 이 순차 recurrence를,
- **청크 안(intra-chunk)** 은 삼각행렬 역행렬 + 행렬곱으로 병렬화하고,
- **청크 사이(inter-chunk)** 만 순차적으로 `S`를 넘겨주는

방식으로 재구성한 것이다. 아래 §1.1~§1.4가 그 "삼각행렬 역행렬"이 **왜** 필요한지를 유도한다.

### 1.1 왜 순차적인가 — 병목의 정체

`u_t`는 예측값 `α_t Sᵀ_{t-1} k_t` 때문에 `S_{t-1}`에 의존하고, `S_{t-1}`은 앞선 델타 `u_1..u_{t-1}`를 전부 누적한 것이다.

```
u₁ ─기록→ S₁
           └─예측에 사용→ u₂ ─기록→ S₂
                                    └─예측에 사용→ u₃ ─기록→ S₃ → ...
```

즉 **`u_t`는 앞선 모든 델타에 의존**한다. 토큰을 하나씩 처리하면 O(n) 순차이므로, 청크 안에서 이 사슬을 한 번에
풀어야 병렬화가 된다. "삼각행렬 역행렬"은 바로 이 사슬을 푸는 도구다.

### 1.2 재귀 펼치기 — 델타끼리의 선형 관계

한 청크(토큰 `1..C`, 들어온 상태 `S₀`)에서 상태를 펼치면 (감쇠 `γ_t = exp(g_t)`):

```
S_{t-1} = γ_{t-1} S₀  +  Σ_{j<t} (γ_{t-1}/γ_j) k_j u_jᵀ
```

이를 예측항에 넣고 `α_t γ_{t-1} = γ_t`, `α_t (γ_{t-1}/γ_j) = γ_t/γ_j` 를 쓰면:

```
α_t Sᵀ_{t-1} k_t  =  e^{g_t} S₀ᵀk_t   +   Σ_{j<t} e^{g_t−g_j} (k_t·k_j) u_j
                     └ 이전 청크 기여 ┘    └ 청크 안 앞선 델타들의 간섭 ┘
```

`u_t = β_t (v_t − 위)` 이므로, **입력과 `S₀`만으로 정해지는 부분 `w_t`** 와 **다른 델타에 의존하는 부분**으로 갈린다:

```
u_t  =  w_t  +  Σ_{j<t} A_{tj} · u_j

  w_t    = β_t v_t − β_t e^{g_t} S₀ᵀ k_t          # 다른 델타와 무관 (입력 + 이전 상태)
  A_{tj} = −β_t e^{g_t−g_j} (k_t·k_j)   (j<t)      # 앞선 델타 uⱼ 가 uₜ 를 끌어당기는 간섭
```

### 1.3 삼각 선형시스템 → 왜 `(I − A)⁻¹` 인가

`u_t = w_t + Σ_{j<t} A_{tj} u_j` 를 청크 전체(C개 토큰)에 대해 한꺼번에 쓰면 **하나의 선형 연립방정식**이다:

```
┌ u₁ ┐   ┌ w₁ ┐   ┌  0                 ┐ ┌ u₁ ┐
│ u₂ │ = │ w₂ │ + │ A₂₁   0            │ │ u₂ │
│ u₃ │   │ w₃ │   │ A₃₁  A₃₂   0       │ │ u₃ │
└ u₄ ┘   └ w₄ ┘   └ A₄₁  A₄₂  A₄₃   0  ┘ └ u₄ ┘
 (구하려는 델타)  (입력+S₀)   A = 엄격 하삼각 (델타끼리의 간섭)
```

`A`가 **엄격 하삼각**인 이유: causal이라 토큰 `t`는 자기보다 앞선 `j(<t)`만 본다 → `j ≥ t`는 0이고 대각선도 0
(자기 자신과의 교차 간섭 없음; 자기 항 `β_t v_t`는 이미 `w_t` 안에 있음).

우변의 `u`를 좌변으로 모으면:

```
u = w + A·u   ⟹   (I − A) u = w   ⟹   u = (I − A)⁻¹ w = T·w
```

**이게 `T = (I − A)⁻¹`를 구하는 이유다.** "델타를 하나씩, 앞선 델타를 보정해가며" 계산하던 순차 재귀를,
**`T`라는 행렬 하나를 곱하는 것**으로 대체한다. `T`는 "청크 안의 모든 상호 간섭을 한 방에 풀어주는 연산자"이며,
`A`가 엄격 하삼각(=nilpotent)이라 역행렬이 항상 존재하고 그 결과도 하삼각이다 (§2.2에서 전방대입으로 구성).

### 1.4 델타를 알면 출력은 행렬곱 — 코드 연결

`u`(청크 안 모든 델타)를 얻으면 상태·출력은 순차 없이 끝난다. `S_t = γ_t S₀ + Σ_{j≤t} (γ_t/γ_j) k_j u_jᵀ` 이므로:

```
o_t = q_tᵀ S_t  =  (e^{g_t} q_t)ᵀ S₀   +   Σ_{j≤t} e^{g_t−g_j} (q_t·k_j) u_j
                   └ inter-chunk ┘         └ intra-chunk (하삼각 어텐션 × u) ┘
```

코드(§2.2~§2.3) 대응 — `v_new`가 곧 델타 `u`다:

| 수식 | 코드 |
|---|---|
| `u = T·w` | `v_new = value − v_prime` |
| `value = T·(β v)` | `value = T @ v_beta` |
| `v_prime = T·(β e^g k)·S₀` | `v_prime = (T @ (k_beta·e^g)) @ S₀` |
| `w_t = β_t v_t − β_t e^{g_t} S₀ᵀk_t` | `v_beta − (k_beta·e^g)@S₀` 의 t행 |
| inter-chunk `(e^g q)ᵀ S₀` | `attn_inter = (q·e^g) @ S₀` |
| intra-chunk `Σ_{j≤t} e^{g_t−g_j}(q_t·k_j) u_j` | `attn_intra @ v_new`,  `attn_intra=(q@kᵀ)·decay_mask` |

즉 순차 재귀에서 **유일하게 "풀어야" 하는 것이 델타 `u`** 이고, 하삼각 역행렬 `T`는 그 `u`를 O(n) 순차 없이
한 번에 얻기 위한 도구다. 나머지(출력·상태 갱신)는 전부 행렬곱이다.

---

## 2. PREFILL — `rbln_chunk_gated_delta_rule`

HF `torch_chunk_gated_delta_rule`을 **표현식 단위로(expression-for-expression)** 옮긴 것이다. 아래 표의 모든
차이는 오직 "HF의 그 연산이 RBLN에서 lowering 되게 하기 위한" 것뿐이며 결과값은 동일하다(full window cos≈1.0).

### 2.1 준비 단계

```
q, k = l2norm(q), l2norm(k)
q = q * (1/√Dk)
v_beta = v * β,  k_beta = k * β
incr = g                          # per-token log-decay (cumsum 전)
g    = incr.cumsum(-1)            # 누적 log-decay
decay_mask[i,j] = exp(g_i − g_j)  (i≥j, 하삼각) else 0
```

`decay_mask` (S=4 예시):

```
        j=0        j=1        j=2       j=3
i=0 [ e^0        0          0         0    ]
i=1 [ e^{g1-g0}  e^0        0         0    ]
i=2 [ e^{g2-g0}  e^{g2-g1}  e^0       0    ]
i=3 [ e^{g3-g0}  e^{g3-g1}  e^{g3-g2} e^0  ]
        ↑ 하삼각(과거만) + 감쇠 + causal 마스킹을 한 번에
```

### 2.2 첫 번째 for문 — 청크 안 순차 의존성을 역행렬로 풀기

델타 규칙은 순차적이라, 청크 안 토큰 i의 갱신이 앞선 토큰 j(<i)의 갱신에 의존한다. 이 연쇄 의존을 한 번에 풀기
위해 엄격 하삼각행렬 `A`(코드 변수 `attn`)를 만들고 `T = (I − A)⁻¹`를 구한다.

```python
attn = -((k_beta @ keyᵀ) * decay_mask) * tril_strict      # == A
```

`k_beta @ keyᵀ`의 `[i,j] = β_i (k_i·k_j)`(청크 안 키들끼리의 유사도 ×β)에 `decay_mask`(감쇠)와
`tril_strict`(과거만 남기는 엄격 하삼각)를 곱한 것이다.

```
A[i,j] = −β_i (k_i·k_j) exp(g_i − g_j)   (i>j, 엄격 하삼각) else 0
```

`A`(=`attn`)는 **"청크 안에서 토큰 i가 자기보다 앞선 토큰 j에게 받는 (음의) 간섭"** 을 담는다.

`A` 행렬 (S=4 예시):

```
        j=0    j=1    j=2    j=3
i=0 [  0      0      0      0  ]
i=1 [ a₁₀     0      0      0  ]     a_ij = −β_i (k_i·k_j) e^{g_i−g_j}
i=2 [ a₂₀    a₂₁     0      0  ]     (i > j 인 곳만 값이 있음 · 대각선 0)
i=3 [ a₃₀    a₃₁    a₃₂     0  ]
```

`for` 루프는 **전방 대입(forward substitution)** 으로 `T = (I − A)⁻¹`의 하삼각부를 한 행씩 채운다.

```python
for i in range(1, S):
    row = attn[..., i, :i].clone()   # A의 i행 [0..i-1]  (아직 raw A 값)
    sub = attn[..., :i, :i].clone()  # 이미 완성된 왼쪽 위 i×i 블록 (대각선 = 0)
    attn[..., i, :i] = row + (row.unsqueeze(-1) * sub).sum(-2)
attn = attn + eye                    # 마지막에 대각선 1을 채움 → T = (I − A)⁻¹
```

i번째 행을 채울 때, 위쪽 블록 `sub`는 이미 `T`의 하삼각부가 완성돼 있다 (i=3 예시):

```
row = [a₃₀  a₃₁  a₃₂]        ← 3행, 아직 raw A 값
sub = 완성된 위쪽 블록          새 attn[3,m] = a₃ₘ  +  Σ_{m<k<3} a₃ₖ · sub[k,m]
      [ 0            ] (0행)                  └직접 3→m┘   └중간 토큰 k 경유 모든 경로┘
      [ t₁₀  0       ] (1행)
      [ t₂₀  t₂₁  0  ] (2행)
```

`attn[..., i, :i] = row + (row.unsqueeze(-1) * sub).sum(-2)` 를 단계별로 (i=3):
(m=열은 끝까지 **가로**로 고정, `sum(-2)`는 k=행을 **세로**로 접는 것)

```
① row.unsqueeze(-1)   (3,) → (3,1) 열벡터
   [ a₃₀ ]
   [ a₃₁ ]
   [ a₃₂ ]

② (①) * sub   브로드캐스트 — sub의 행 k를 스칼라 row[k]로 스케일
         m=0       m=1       m=2
   k=0 [ a₃₀·0     a₃₀·0     a₃₀·0  ]   ← ×a₃₀
   k=1 [ a₃₁·t₁₀   a₃₁·0     a₃₁·0  ]   ← ×a₃₁
   k=2 [ a₃₂·t₂₀   a₃₂·t₂₁   a₃₂·0  ]   ← ×a₃₂

③ .sum(-2)   각 열을 k(세로) 방향으로 접음 → (3,) 행벡터  (m은 그대로 가로)
                 ↓세로합            ↓세로합     ↓세로합
              m=0                m=1        m=2
        [ a₃₁·t₁₀+a₃₂·t₂₀    a₃₂·t₂₁     0  ]

④ row + ③   직접경로(row) + 경유경로(③), 둘 다 m 가로 방향
              m=0                m=1        m=2
   row  [ a₃₀               a₃₁        a₃₂ ]
   +③   [ a₃₁·t₁₀+a₃₂·t₂₀    a₃₂·t₂₁     0  ]
   ─────────────────────────────────────────
   =    [ t₃₀               t₃₁        t₃₂ ]   ← T의 3행 완성
```

`sub`의 대각선이 0이라 ②에서 `a₃ₘ·sub[m,m]=0`(각 열 m의 k=m 항이 0) → 직접경로가 중복되지 않고, ④의 `row`가 그 직접항을 딱 한 번 더한다.

**주의: 루프가 도는 동안 `attn`(=`sub`)의 대각선은 0이다.** `+ eye`(대각선 1)는 루프가 끝난 뒤 딱 한 번 더해진다.
루프 중 대각선이 1이면 "직접 경로 `a_im`"이 두 번 세어져(중복 카운팅) 틀린 결과가 나오므로, 대각선이 0이어야 한다.

완성된 변환 행렬 `T = (I − A)⁻¹` (대각선 1, 하삼각):

```
[ 1                 ]
[ t₁₀  1            ]     T = "청크 안의 모든 델타 보정을 한 번에 적용하는 변환 행렬"
[ t₂₀  t₂₁  1       ]     t_im = a_im (직접) + 중간 토큰들을 경유하는 간섭의 총합
[ t₃₀  t₃₁  t₃₂  1  ]
```

이후 UT 변환 (`attn`이 곧 `T`):

```
value      = T @ v_beta                    # 청크 안 델타 보정이 반영된 새 밸류 U (= u_t)
k_cumdecay = T @ (k_beta * exp(g))         # 이전 상태 S가 현재 청크에 준 예측을 빼낼 때(§2.3 v_prime) 사용
```

**RBLN 차이(정확성 관련):** HF는 `(I−A)⁻¹`를 log-depth 제곱 `(I+A)(I+A²)…`로 구할 수도 있으나, 학습된 스케일에서는
중간 거듭제곱이 발산해 수치적으로 불안정하다. 그래서 **HF와 동일한 forward-substitution 루프**를 유지한다.

### 2.3 두 번째 for문 대응부 — intra/inter 출력과 상태 갱신

한 윈도우가 `n_chunks`개 서브청크로 나뉘므로, `for c in range(n_chunks)` 루프로 서브청크마다 아래를 계산하고
`recurrent_state`를 서브청크 사이로 넘긴다 (HF의 inter-chunk 루프와 동일; n_chunks=1이면 1회). 서브청크당:

```
attn_intra = (q @ kᵀ) * decay_mask         # 청크 안 causal 어텐션
v_prime    = k_cumdecay @ S                 # 이전 상태가 예측한 밸류
v_new      = value − v_prime                # 이전 상태 기여 제거 → 최종 델타 u_t
attn_inter = (q * exp(g)) @ S               # 이전 상태로부터의 출력 (inter-chunk)
core       = attn_inter + attn_intra @ v_new
```

### 2.4 상태 갱신 — reverse-cumsum을 matmul로 (핵심 최적화)

HF의 상태 갱신(`torch_chunk_gated_delta_rule` 300–306):

```
S_new = S * exp(g[-1]) + (k * exp(g[-1] − g))ᵀ @ v_new
```

여기서 `g[-1]`은 cumsum의 **마지막 원소 슬라이스**, `g[-1] − g`는 **브로드캐스트 뺄셈**인데, 둘 다 RBLN에서
잘 lowering 되지 않는다(innermost 슬라이스 / `torch.flip` 계열). RBLN은 이를 **reduction + matmul**로 바꾼다.

```python
g_total      = incr.sum(-1, keepdim=True)   # == g[-1]  (마지막 슬라이스 대신 전체 합)
decay_to_end = incr @ tril_strict           # == g[-1] − g  (reverse-cumsum을 matmul로)
S_new = S * exp(g_total) + (k * exp(decay_to_end))ᵀ @ v_new
```

**왜 같은 값인가 (행렬 그림, S=4, `incr = [a0,a1,a2,a3]`):**

- `g_total = a0+a1+a2+a3` = cumsum의 마지막 원소 `g[-1]`. (cumsum의 끝 = 전체 합)
- `decay_to_end[t] = Σ_{s>t} incr[s]` 를 구해야 하는데, `tril_strict[s,t] = 1 iff s>t` 이므로:

```
tril_strict (4×4),  [s,t]=1  iff  s>t
              t=0  t=1  t=2  t=3
  s=0 [ a0 ]   0    0    0    0
  s=1 [ a1 ]   1    0    0    0
  s=2 [ a2 ]   1    1    0    0
  s=3 [ a3 ]   1    1    1    0
   ↑incr        │    │    │    │
                ▼    ▼    ▼    ▼
decay_to_end[t] = Σ_s incr[s]·tril_strict[s,t]

t=0: a1+a2+a3   t=1: a2+a3   t=2: a3   t=3: 0
```

이는 정확히 HF의 `g[-1] − g[t] = Σ_{s>t} incr[s]` 와 같다. 즉,

- HF: "전체합에서 앞부분(cumsum)을 뺀다"
- RBLN: "하삼각 마스크로 뒷부분만 골라 더한다(reverse-cumsum)"

는 같은 "뒤쪽 합"을 구하는 서로 다른 계산 경로일 뿐이다.

### 2.5 PREFILL 우측 패딩 처리

런타임이 마지막 윈도우를 `prefill_chunk_size`로 우측 패딩하는데, 패딩 토큰은 projection bias 때문에 q/k/v/g가
0이 아니다. 이 값들이 청크 합(=recurrent state)과 그 감쇠를 오염시키므로 **패딩 토큰의 g/β를 0으로** 만든다.

```python
g    = g    * valid_mask
beta = beta * valid_mask
```

### 2.6 PREFILL 최적화 요약

| 항목 | HF | RBLN | 이유 |
|------|----|----|------|
| 삼각/단위 행렬 | bool `.tril()` / `masked_fill` / `torch.eye` | float 마스크(`tril_incl`/`tril_strict`), `eye = tril_incl − tril_strict` 로 파생 | `rtosa.where`가 i1 거부, `aten::eye` 미구현 |
| 마스크 생성 | 모듈 상수 | 호출부 `forward` 지역 변수로 inline (rank `(1,1,1,gcs,gcs)`) | named 상수는 weight-sharing 패스가 prefill/decode 간 공유 시도 → gen mode 실패 |
| 상태 갱신 감쇠 | `g[-1]`, `g[-1] − g` (슬라이스/뺄셈) | `incr.sum`, `incr @ tril_strict` (matmul reverse-cumsum) | innermost 슬라이스/`flip` mis-lower |
| 청크 분할 | 내부 `chunk_size` reshape (3 batch dim) | `gdn_chunk_size`로 서브청크 분할 (chunk 축을 `torch.stack`으로, reshape는 rank 오추론) → 3 batch-dim `(B,Hv,n_chunks)` | 예전엔 OpTiling "3 vs 4"로 막혔으나 컴파일러 5D matmul 지원으로 해소 |

---

## 3. DECODE — `rbln_recurrent_gated_delta_rule_step`

HF `torch_recurrent_gated_delta_rule`의 S=1 케이스와 **수학적으로 완전히 동일**하다(cos=1.0). 5단계는 다음과 같다.

```
① decay  : S ← exp(g) · S
② predict: kv_mem = kᵀ S
③ delta  : δ = β (v − kv_mem)
④ update : S ← S + k ⊗ δ
⑤ output : o = qᵀ S      (④에서 갱신된 S 사용)
```

HF와 RBLN 매핑:

| 단계 | HF (broadcast + `sum(-2)`) | RBLN (matmul) |
|------|----------------------------|---------------|
| l2norm | `(x*x).sum(-1)` | `matmul(x, xᵀ)` dot |
| ② predict | `(S * k[…,None]).sum(-2)` | `matmul(k_row, S)` |
| ④ update | `k[…,None] * δ[…,None,:]` | `matmul(k_rowᵀ, δ)` |
| ⑤ output | `(S * q[…,None]).sum(-2)` | `matmul(q_row, S)` |

핵심은 **"broadcast 곱 + `sum(dim=-2)`" 와 "matmul"이 같은 계산**이라는 점이다 (`kᵀS` 예시, Dk=3, Dv=2):

```
matmul:  [k0 k1 k2] · ┌ s00 s01 ┐ = [k0s00+k1s10+k2s20 , k0s01+k1s11+k2s21]
                      │ s10 s11 │
                      └ s20 s21 ┘

broadcast+sum(-2):  S를 k로 원소곱 후 Dk축(세로)으로 합 → 같은 벡터
```

### DECODE 최적화 요약

| 항목 | HF | RBLN | 이유 |
|------|----|----|------|
| 리덕션 | `.sum(dim=-2)` (비-최내축 Dk) | matmul | 비-최내축 리덕션 mis-lower |
| l2norm | `(x*x).sum(-1)` (최내축) | dot-product(matmul) | 작은 decode 텐서에서 최내축 sum이 ≈0 반환 → `rsqrt(eps)` 폭발 |
| seq축 | `query[:,:,i]`, `out[:,:,i]=` | S=1을 `reshape`로 제거 | 동적 StridedSlice / ScatterInfo blocker |

---

## 4. Short Conv 와 conv_state 캐싱

GDN 앞단에는 커널 크기 `K`(=`conv_kernel_size`, 보통 4)의 depthwise **causal** 1D conv가 있다. causal conv는
각 출력마다 "자기 자신 + 직전 `K-1`개 입력"을 필요로 하므로, 이전 문맥을 `conv_state`에 들고 다닌다.

### 4.1 RBLN conv 계산

```python
# conv_state: (B, K-1, conv_dim) → 앞에 붙임(prepend), padding=0 conv
x_cf     = cat([conv_state.T, mixed_qkv.T], dim=time)   # (B, conv_dim, (K-1)+S)
conv_out = F.conv1d(x_cf, weight, bias, padding=0, groups=conv_dim)  # 길이 정확히 S
```

- prepend된 좌문맥 `K-1`개 덕분에 `padding=0`으로도 출력 길이가 정확히 `S`가 되고, 첫 출력 `out[0]`은
  `conv(c0..c_{K-2}, x0)`로 올바른 좌문맥을 본다.
- **첫 prefill 윈도우**는 `conv_state`가 마스크로 0이 되어(§0), HF의 좌측 zero-padding과 동일한 상태가 된다.

### 4.2 HF와의 비교

| 항목 | HF | RBLN | 동일? |
|------|----|----|------|
| conv 출력값(q/k/v 입력) | causal depthwise conv | 동일 | ✅ 수학적으로 동일 |
| 첫 윈도우 좌문맥=0 | `nn.Conv1d`의 내장 `padding=K-1`(zero) | `conv_state × mask`로 0 | ✅ |
| 저장하는 이전 문맥 | 최근 **K개** 입력 | 최근 **K-1개** 유효 입력 | ⚠️ 표현 차이 (최근 K-1개는 동일) |
| 다음 스텝 적용 | prepend/roll 후 conv | prepend 후 conv | ✅ |
| 우측 패딩 처리 | 없음(정확 길이 처리) | `query_position` 기반 동적 take(point-gather) K-1개 + `cat` | RBLN 고정 윈도우 전용 |

### 4.3 왜 HF는 K개, RBLN은 K-1개인가

HF 캐시는 폭이 **K**다(`cache_utils.py`의 `update_conv_state`가 `conv_states[..., -K:]` 유지, mamba
`causal_conv1d_update`의 고정 폭 = 커널 크기 관례). 반면 RBLN은 **K-1**만 저장한다.

**결론: 저장한 K개 중 가장 오래된 1칸은 어떤 (최종) 출력에도 기여하지 않는다.** 즉 정보를 실어 나르는 최소 상태는
`K-1`이며, RBLN은 그만큼만 저장한다.

- **decode(roll 버퍼):** roll로 가장 오래된 열을 흘려보내고 새 토큰을 넣어 K탭 conv → 문맥은 K-1개만 사용.
- **chunk(cat + `padding=K-1` conv):** conv1d가 기계적으로 `c0`부터 다 계산하지만, causal 정렬 슬라이싱
  (`[:K+S]` → `[-S:]`)으로 `c0`이 들어간 앞쪽 출력들은 **버려진다**. 살아남는 첫 출력은 `[c1,c2,c3,n0]`라 `c0`
  미사용.

```
padded conv 입력: [0 0 0  c0 c1 c2 c3 n0 n1 ...]   (padding=K-1=3)
out[0..3] = c0가 들어간 출력  → 슬라이싱으로 폐기 ❌
out[4]    = conv(c1,c2,c3,n0) → 유지 (c0 안 봄) ✅
```

HF의 참조 구현 `torch_causal_conv1d_update`도 RBLN처럼 `cat + padding=0 + [-seq_len:]`을 쓰며, 이 경우
`out[0]=conv(c0..c_{K-1})` 하나만 버리고 나머지를 유지 → 역시 `c0`은 버려지는 출력에만 들어간다.

정리하면 HF가 `K`를 드는 건 정보가 더 필요해서가 아니라 mamba `causal_conv1d_update`의 고정 폭-K roll 버퍼
관례 때문이고, RBLN의 `K-1` 저장은 필요한 최소량만 저장하는 것 — **둘 다 같은 K-1 문맥으로 같은 conv 출력**을 낸다.

### 4.4 우측 패딩에서 "마지막 K-1개 유효" 고르기

prefill 윈도우는 우측 패딩되므로, 원시 `mixed_qkv`의 꼬리 `K-1`개는 패딩(쓰레기값)일 수 있다. 그래서 런타임이 주는
`query_position`(= 마지막 유효 토큰의 0-based 인덱스 = `valid_count − 1`)으로 **마지막 K-1개 유효 입력**을 고른다.
`query_position + i` (i = 1..K-1) 열을 각각 **동적 take(point-gather)** 로 뽑아 `cat`한다.

```python
# query_position 을 (1,) 텐서로 만들어 인덱싱 -> 각 열은 point-gather(take) 로 lower
states         = [x_cf[:, :, query_position.unsqueeze(0) + i] for i in range(1, K)]  # 각 (B, conv_dim, 1)
new_conv_state = torch.cat(states, dim=2).transpose(1, 2)                            # (B, K-1, conv_dim)
```

- `x_cf = [conv_state(0..K-2) | mixed_qkv(K-1..)]`. 유효 토큰은 `mixed_qkv`의 앞 `valid_count`개.
- 뽑는 열 = `[qp+1 .. qp+K-1]` = `[valid_count .. valid_count+K-2]` = 마지막 K-1개 유효 입력.
- `valid_count < K-1`이면 인덱스가 prepend된 `conv_state` 영역까지 파고들어 **이전 윈도우 문맥의 꼬리**를
  자동으로 끌어온다.

```
예) K=4, 윈도우 6, 유효 4개(t0..t3), 패딩 2개(p), query_position=3:
x_cf = [ c0 c1 c2 | t0 t1 t2 t3 p p ]   (index 0..8)
naïve 꼬리 [-3:] = [t3, p, p]           ← ❌ 패딩 섞임
뽑는 열 = 3 + [1,2,3] = [4,5,6] → [t1,t2,t3] ← ✅ 마지막 3개 유효
```

**왜 take K-1개 + cat인가** (다른 방식은 컴파일 실패):
- `index_select`(동적 인덱스 텐서 gather) → RBLN에서 **그래프가 둘로 쪼개짐**.
- `x_cf[:, :, i]`(동적 **스칼라** 인덱스) → 동적 **StridedSlice(SubviewOp) blocker**.
- `query_position`을 `(1,)` 텐서로 만들어 `x_cf[:, :, qp+i]`로 뽑으면 **point-gather(take)** 로 lower되고,
  K-1개를 `cat`하면 단일 그래프로 컴파일된다.

decode 경로는 패딩이 없으므로 HF처럼 그냥 꼬리 `x_cf[..., -(K-1):]`을 쓴다.

---

## 5. 정합성 / 검증 요약

| 경로 | HF 대비 |
|------|---------|
| DECODE (seq=1) | cos = 1.0 (수학적으로 동일) |
| PREFILL | pearsonR ~0.9999 |

부동소수점 재배열(합산 순서: cumsum ↔ sum/matmul, broadcast-sum ↔ matmul)로 인한 비트 단위 차이는 존재할 수
있으나, 위 상관계수 수준에서 무시 가능하다.

검증 범위 (teacher-forced, HF vs 컴파일된 RBLN): text 8·16 레이어, 서브청크 `n_chunks` 1·2, 이미지+텍스트
(vision 인코더 포함, 다중 윈도우 prefill) — 전부 per-step 로짓 pearson ~0.9999. GatedDeltaNet 단독은 CPU eager로도
HF와 cos ≈ 1.0 (단일/다중 윈도우).

---

## 6. RBLN lowering 제약 → 우회 총정리

| 원인이 된 HF 연산 | RBLN 우회 | 발생 위치 |
|---|---|---|
| bool `.tril()` / `masked_fill` / `torch.eye` | float `tril_incl`/`tril_strict`, `eye = tril_incl − tril_strict` | prefill 마스크 |
| 모듈 상수(named) 공유 | `forward` 지역 상수 | weight-sharing gen mode |
| `g[-1]` 슬라이스, `g[-1]−g`, `flip` | `incr.sum`, `incr @ tril_strict` | 상태 갱신 감쇠 |
| 내부 `chunk_size` reshape (3 batch-dim) | `gdn_chunk_size` 서브청크 + `torch.stack` 축 생성; 컴파일러 5D matmul 지원으로 3-batch-dim 통과 | OpTiling (구 "3 vs 4") |
| 비-최내축 `.sum(dim=-2)` | matmul | decode predict/update/output |
| 최내축 `(x*x).sum(-1)` l2norm (작은 텐서 ≈0) | dot-product matmul | decode l2norm |
| `query[:,:,i]` / `out[:,:,i]=` (StridedSlice/Scatter) | S=1 `reshape` | decode seq축 |
| 동적 strided-slice / index_select 로 conv 꼬리 추출 | `query_position`을 `(1,)` 텐서로 만들어 `x_cf[:, :, qp+i]` 동적 take K-1개 + `cat` (index_select는 그래프 분할, 동적 스칼라 subview는 blocker) | conv_state 갱신 |

---

## 7. Runtime 오케스트레이션 — 전체 generate() 흐름

섹션 2~4가 GDN 커널 "한 번의 호출" 내부라면, 여기서는 이 호출들이 vision 인코딩부터 prefill 윈도우들,
decode 스텝들까지 어떻게 엮여 `generate()`를 이루는지 정리한다.
구현: `qwen3_5_runtime_utils.py`(`RBLNQwen3_5RuntimeModel`), `modeling_qwen3_5.py`(`forward` / `_preprocess_prefill`).

### 7.0 두 개의 컴파일 단위

| 단위 | 아티팩트 | 언제 실행 | 상태 캐시 |
|---|---|---|---|
| **visual** (vision 인코더) | 별도 `.rbln` 서브모듈 | prefill 때 **딱 1회** (Python eager 호출) | 없음 (stateless) |
| **language model** | prefill/decode `.rbln` | prefill(청크 루프) + decode(스텝 루프) | full-attn paged KV + linear conv/recurrent 정적 캐시 |

런타임(`RBLNQwen3_5RuntimeModel`)이 GDN을 위해 하는 일은 딱 **3가지 입력 주입**뿐이다. 상태 자체(conv/recurrent/KV)는
KV처럼 device DRAM 정적 캐시에 있고 호출 시 오가지 않는다.

| 주입 입력 | shape (0.8B) | 역할 |
|---|---|---|
| `conv_state_mask` | (1, 3, 6144) | 읽은 conv_state에 곱함 — **0=fresh(폐기), 1=carry** |
| `recurrent_state_mask` | (1, 16, 128, 128) | 읽은 recurrent_state에 곱함 — 0=fresh, 1=carry |
| `valid_mask` | (1, chunk, 1) | 청크 내 토큰별 유효(1)/패딩(0) — chunk-parallel prefill에서 패딩 제외 |

### 7.1 Visual submodule — 입력 처리와 병합

**프로세서**(HF `AutoProcessor.apply_chat_template`)가 이미지를 패치로 잘라 `input_ids`를 만든다:

```
smart_resize:  factor = patch_size · spatial_merge = 16 · 2 = 32
   원본 H×W → h_bar×w_bar (32의 배수, h_bar·w_bar ≤ max_pixels, 종횡비 유지)
grid_thw = (grid_t, grid_h, grid_w) = (1, h_bar/16, w_bar/16)
patch 개수 P      = grid_t·grid_h·grid_w                  # pixel_values = [P, 3·2·16·16 = 1536]
image placeholder = P / spatial_merge² = P / 4            # <image> 토큰(id 248056) 개수
```

`input_ids` = `… <vision_start> <image>×(P/4) <vision_end> "질문" …` 형태.
(예: max_pixels 262144 → P=936, placeholder≈234 · max_pixels 1M → P=3888, placeholder=972.)

**vision forward**(`RBLNQwen3_5VisionModel.forward`): patch_embed → 위치임베딩 → transformer 블록 →
**2×2 patch merger** → `image_embeds [P/4, H_lm]`. 패치수를 `visual.max_seq_len` 버킷으로 pad→실행→valid만 trim
(따라서 `visual: {max_seq_len: N}`은 P 이상이어야 한다 — 작으면 vision forward에서 IndexError).

**병합**(`_preprocess_prefill`):
```python
inputs_embeds = embed_tokens(input_ids)                        # 텍스트 임베딩 (placeholder 포함)
image_embeds  = self.visual(pixel_values, grid_thw)            # [P/4, H_lm]
mask = (input_ids == config.image_token_id)                    # placeholder 위치
inputs_embeds = inputs_embeds.masked_scatter(mask, image_embeds)   # 그 자리 → vision 임베딩 (순서대로 1:1)
```
placeholder 개수 == `image_embeds` 행수(P/4)라 1:1로 맞아떨어진다.

### 7.2 mRoPE 위치 임베딩

같은 `_preprocess_prefill`에서 `_get_rope_index_func(input_ids, mm_token_type_ids, image_grid_thw)`가 3D mRoPE
좌표를 만든다 — 텍스트 토큰은 순차, **이미지 토큰은 grid 기반 2D(height/width)** 좌표. → `position_embed`(부분 RoPE
cos/sin)로 `prefill_decoder`에 전달. decode는 `cache_position + rope_deltas`로 이어간다. (position_ids는 프로세서가
아니라 모델이 생성한다.)

### 7.3 PREFILL 런타임 — `prefill_forward`

프롬프트를 `prefill_chunk_size`(=128) 윈도우로 나눠 넣고 GDN 상태를 윈도우 사이로 넘긴다. `prompt=255` 예시
(→ 256으로 pad, 2 윈도우):

```
                conv/recurrent state_mask   valid_mask (1,128,1)
window0 step=0     ZEROS (fresh 시작)         [1]×128            ← stale DRAM 폐기, 새 상태 write
window1 step=128   ONES  (carry)              [1]×127 + [0]×1    ← window0 상태 이어받음, 우측 패딩 1개 제외
                                              (valid = min(128, 255−128) = 127)
```

- **`conv/recurrent_state_mask`** — window0에서만 0. 런타임이 device DRAM을 직접 못 지우므로 GDN이 읽은 상태에
  ×0을 곱해 "논리적 리셋"을 한다(§0). 이후 윈도우는 ×1로 이전 윈도우가 쓴 상태를 그대로 이어받는다.
- **`valid_mask`** — GDN이 `g`/`beta`에 곱해, 마지막 부분 윈도우의 우측 패딩을 recurrent-state 합·decay·conv_state
  추출에서 제외한다(§2.5). full 윈도우는 전부 1.
- **`query_position`** — 이 윈도우의 마지막 유효 토큰 인덱스(= `valid_count − 1`). ① `logits_to_keep=1`이라 매 윈도우가
  단일 logits 행을 덮어써 최종값 = **마지막 윈도우의 next-token logits**, ② GDN conv 꼬리 추출(§4.4)이 이 값으로
  마지막 K-1 유효 토큰을 고른다.

### 7.4 DECODE 런타임 — `decode_forward`

토큰 1개(seq=1), `cache_position` 한 칸씩 전진, 순차 델타 규칙(§3). `state_mask`=ones(carry), `valid_mask`=ones —
셋 다 decode 그래프에선 prefill-phase 게이팅으로 pruned되어 실질 무시된다(안전용으로만 전달).

### 7.5 `_run` — pruned 입력 이름 매핑

```python
order = self.runtime._index_to_input_name         # rebel이 dead input 제거 후 남긴 입력 순서
args  = [named_inputs[order[k]] for k in range(len(order))]
out   = super().forward(*args); return out[0]      # 정적 캐시는 order에 없음(in-graph); logits만 keep
```
이름 기반 매핑이라 런타임이 실제로 keep한 입력만 정확한 순서로 전달된다 — decode에서 pruned된 mask를 넘겨도 무시된다.

### 7.6 전체 타임라인

```
이미지 ─[processor]→ pixel_values, grid_thw ─→ self.visual (별도 .rbln) ─→ image_embeds [P/4, H]
텍스트+<image> input_ids ─[embed_tokens]→ inputs_embeds ──masked_scatter(input_ids==image_token_id)──┘
                                                    │  + mRoPE position_embed
                                                    ▼
                              prefill_decoder (LM prefill)
                                window0(mask=0, fresh) → window1(mask=1, carry) → …   (conv/recurrent 캐리)
                                                    ▼  마지막 윈도우의 next-token logits
                              decode step0 → step1 → …   (seq=1, 이미지 없음, 순차 규칙)
```

full-attn paged-KV와 linear conv/recurrent 정적 캐시가 prefill 윈도우들 → decode 스텝들에 걸쳐 device에서 연속으로
이어지고, 런타임은 그 사이 (1) 3개 마스크 주입, (2) 청크 윈도잉, (3) 이름 기반 입력 매핑만 담당한다.

---

### 참고

- 자세한 배경은 프로젝트 메모 `rbln-qwen35-deltanet-compile` 참고.
- 코드 주석은 `qwen3_5_architecture.py`에 HF와 라인 단위로 대응(`# HF:`)되어 있으므로 side-by-side로 읽으면 좋다.
