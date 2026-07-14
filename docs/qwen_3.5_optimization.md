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
윈도우 사이로 `recurrent_state`를 넘긴다. 따라서 **한 번의 호출 == 하나의 델타 청크**이며,
HF `torch_chunk_gated_delta_rule` 내부의 `chunk_size` 재분할(reshape)은 존재하지 않는다.

---

## 1. 배경: Gated Delta Rule 수학

상태 `S ∈ ℝ^{Dk×Dv}` 는 "키를 넣으면 밸류를 뱉는" 연상 메모리다. 토큰 단위 recurrence는 다음과 같다.

```
u_t = β_t (v_t − α_t · Sᵀ_{t-1} k_t)      # 델타: 실제값 − 예측값
S_t = α_t · S_{t-1} + k_t u_tᵀ            # 메모리 갱신 (감쇠 + 새 기록)
o_t = q_tᵀ S_t                            # 출력
```

- `α_t = exp(g_t)` : 토큰별 게이트(망각 계수), `β_t` : 학습률
- 청크 내 누적 게이트 `γ_t = exp(cumsum(g)_t)` → 두 위치 사이 감쇠 `exp(g_i − g_j)`
- `q`는 `1/√Dk`로 스케일, `q`/`k`는 l2norm

청크 병렬 형태는 이 순차 recurrence를,
- **청크 안(intra-chunk)** 은 삼각행렬 역행렬 + 행렬곱으로 병렬화하고,
- **청크 사이(inter-chunk)** 만 순차적으로 `S`를 넘겨주는

방식으로 재구성한 것이다.

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
위해 삼각행렬 `A`를 만들고 `(I − A)⁻¹`를 구한다.

```
A[i,j] = −β_i (k_i·k_j) exp(g_i − g_j)   (i>j, 엄격 하삼각) else 0
```

`for` 루프는 **전방 대입(forward substitution)** 으로 `(I − A)⁻¹`의 하삼각부를 만든다.

```python
for i in range(1, S):
    row = attn[..., i, :i].clone()   # A의 i행 (raw)
    sub = attn[..., :i, :i].clone()  # 이미 완성된 위쪽 블록 (대각선 = 0)
    attn[..., i, :i] = row + (row.unsqueeze(-1) * sub).sum(-2)
attn = attn + eye                    # 마지막에 대각선 1을 채움 → T = (I − A)⁻¹
```

**주의: 루프가 도는 동안 `attn`의 대각선은 0이다.** `+ eye`(대각선 1)는 루프가 끝난 뒤 딱 한 번 더해진다.
루프 중 대각선이 1이면 "직접 경로 `a_im`"이 두 번 세어져(중복 카운팅) 틀린 결과가 나오므로, 대각선이 0이어야 한다.

```
새 attn[i,m] = a_im  +  Σ_{m<k<i} a_ik · sub[k,m]
              └직접 i→m┘   └중간 토큰 k 경유 경로┘
```

이후 UT 변환:

```
value      = T @ v_beta                    # 청크 안 델타 보정이 반영된 새 밸류 U (= u_t)
k_cumdecay = T @ (k_beta * exp(g))         # 이전 상태 S가 현재 청크에 준 예측을 빼낼 때 사용
```

**RBLN 차이(정확성 관련):** HF는 `(I−A)⁻¹`를 log-depth 제곱 `(I+A)(I+A²)…`로 구할 수도 있으나, 학습된 스케일에서는
중간 거듭제곱이 발산해 수치적으로 불안정하다. 그래서 **HF와 동일한 forward-substitution 루프**를 유지한다.

### 2.3 두 번째 for문 대응부 — intra/inter 출력과 상태 갱신

RBLN은 "한 호출 = 한 청크"라 실제 for문은 없고, 한 청크에 대한 다음 계산만 수행한다.

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
| 삼각/단위 행렬 | bool `.tril()` / `masked_fill` / `torch.eye` | float 마스크(`tril_incl`/`tril_strict`) + 전달된 `eye` | `rtosa.where`가 i1 거부, `aten::eye` 미구현 |
| 마스크/단위 행렬 생성 | 모듈 상수 | `forward` 지역 변수(arange 비교)로 inline | named 상수는 weight-sharing 패스가 prefill/decode 간 공유 시도 → gen mode 실패 |
| 상태 갱신 감쇠 | `g[-1]`, `g[-1] − g` (슬라이스/뺄셈) | `incr.sum`, `incr @ tril_strict` (matmul reverse-cumsum) | innermost 슬라이스/`flip` mis-lower |
| 청크 분할 | 내부 `chunk_size` reshape (3 batch dim) | 없음. 한 호출 = 한 청크 (2 batch dim `(B,Hv)`) | 3 batch-dim matmul이 OpTiling "memory size mismatch (3 vs 4)" 유발 |

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
| 우측 패딩 처리 | 없음(정확 길이 처리) | `valid_count` 기반 `index_select` | RBLN 고정 윈도우 전용 |

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

prefill 윈도우는 우측 패딩되므로, 원시 `mixed_qkv`의 꼬리 `K-1`개는 패딩(쓰레기값)일 수 있다. 그래서 유효 토큰
개수 `valid_count`를 이용해 **마지막 K-1개 유효 입력**을 고른다.

```python
valid_count    = valid_mask.sum()                       # 이 윈도우의 유효 토큰 수 (batch=1이라 스칼라)
idx            = valid_count + arange(K-1)               # x_cf에서 마지막 K-1 유효 열
new_conv_state = index_select(x_cf, dim=time, idx)       # (B, K-1, conv_dim)
```

- `x_cf = [conv_state(0..K-2) | mixed_qkv(K-1..)]`. 유효 토큰은 `mixed_qkv`의 앞 `valid_count`개.
- 마지막 K-1개 유효 열 = `x_cf` 인덱스 `[valid_count .. valid_count+K-2]`.
- `valid_count < K-1`이면 인덱스가 prepend된 `conv_state` 영역까지 파고들어 **이전 윈도우 문맥의 꼬리**를
  자동으로 끌어온다.

```
예) K=4, 윈도우 6, 유효 4개(t0..t3), 패딩 2개(p):
x_cf = [ c0 c1 c2 | t0 t1 t2 t3 p p ]   (index 0..8)
naïve 꼬리 [-3:] = [t3, p, p]           ← ❌ 패딩 섞임
idx = 4 + [0,1,2] = [4,5,6] → [t1,t2,t3] ← ✅ 마지막 3개 유효
```

동적 `index_select`(point-gather)는 RBLN에서 lowering 되지만, 동적 strided-slice는 SubviewOp/ScatterInfo
blocker다. decode 경로는 패딩이 없으므로 HF처럼 그냥 꼬리 `x_cf[..., -k_1:]`을 쓴다.

---

## 5. 정합성 / 검증 요약

| 경로 | HF 대비 |
|------|---------|
| DECODE (seq=1) | cos = 1.0 (수학적으로 동일) |
| PREFILL | pearsonR ~0.9999 |

부동소수점 재배열(합산 순서: cumsum ↔ sum/matmul, broadcast-sum ↔ matmul)로 인한 비트 단위 차이는 존재할 수
있으나, 위 상관계수 수준에서 무시 가능하다.

---

## 6. RBLN lowering 제약 → 우회 총정리

| 원인이 된 HF 연산 | RBLN 우회 | 발생 위치 |
|---|---|---|
| bool `.tril()` / `masked_fill` / `torch.eye` | float 마스크 + inline arange 비교 | prefill 마스크 |
| 모듈 상수(named) 공유 | `forward` 지역 상수 | weight-sharing gen mode |
| `g[-1]` 슬라이스, `g[-1]−g`, `flip` | `incr.sum`, `incr @ tril_strict` | 상태 갱신 감쇠 |
| 3 batch-dim matmul (내부 청크 분할) | 2 batch-dim, 한 호출 = 한 청크 | OpTiling |
| 비-최내축 `.sum(dim=-2)` | matmul | decode predict/update/output |
| 최내축 `(x*x).sum(-1)` l2norm (작은 텐서 ≈0) | dot-product matmul | decode l2norm |
| `query[:,:,i]` / `out[:,:,i]=` (StridedSlice/Scatter) | S=1 `reshape` | decode seq축 |
| 동적 strided-slice로 conv 꼬리 추출 | 동적 `index_select`(point-gather) | conv_state 갱신 |

---

### 참고

- 자세한 배경은 프로젝트 메모 `rbln-qwen35-deltanet-compile` 참고.
- 코드 주석은 `qwen3_5_architecture.py`에 HF와 라인 단위로 대응(`# HF:`)되어 있으므로 side-by-side로 읽으면 좋다.
