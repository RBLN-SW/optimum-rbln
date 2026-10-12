# CogVideoX enable — progress

- 시작 시각: **2026-10-12 23:14:06 KST** (`date` 출력: `Mon Oct 12 11:14:06 PM KST 2026`)
- 종료 기한: 2026-10-13 11:14 KST (12시간)
- 브랜치: `seinpark/cogvideox` (from `origin/dev` @ e2fa89a6)
- 환경: `~/venv_optimum`, optimum-rbln editable (`~/optimum-rbln/src/...`), rebel 0.12.1.dev99+g18b188f8 (`~/rebel_compiler`), diffusers 0.38.0, torch 2.11.0+cpu, transformers 5.15.1
- HF_HOME=/mnt/shared_data/groups/sw_dev/.cache/huggingface

## 로그

### 환경 변경 (venv_optimum)
- `pytest-xdist==3.8.0` 설치 (repo dev 의존성, uv.lock 버전). `-n 1` 실행에 필요.
- `cosmos-guardrail` 0.1.0 → **0.3.2** (`--no-deps`). repo 코드가 `cosmos_guardrail.cosmos_guardrail.Qwen3Guard`를 import하는데 0.1.0에는 없어서
  `tests/test_diffusers.py` 수집 자체가 ImportError로 실패했다 (내 변경과 무관).

### 2026-10-12 23:14 — 시작, 환경 확인
- `import rebel / diffusers / optimum.rbln` 모두 OK. `optimum.rbln.__file__` = `~/optimum-rbln/src/optimum/rbln/__init__.py` (editable 설치 확인).
- NPU 32개 모두 idle (rbln-stat). VAE는 `RBLN_DEVICES=8`, Transformer는 `RBLN_DEVICES=12` 사용.

### 설계 메모 (VAE)
- diffusers `AutoencoderKLCogVideoX._decode`는 latent를 **chunk 단위**로 decode한다: 첫 chunk = `min(n, 2 + n % 2)` latent frame,
  이후 2 frame씩 (49 frames → latent 13 → 3 + 2×5). GroupNorm 통계와 시간축 upsample(`CogVideoXUpsample3D`)이 chunk에 의존하므로,
  CPU 원본과 같은 출력을 내려면 RBLN도 같은 chunking을 써야 한다. → `decoder_0` = 첫 chunk (49 frames면 latent 3), `decoder_n` = latent 2.
- conv cache: kernel-3 causal conv 38개 (conv_in, mid 2×2, up_blocks 4×4×2, conv_out), 모두 depth 2. kernel-1 conv(`conv_y`/`conv_b`)는 cache를 안 쓰므로 제외.
  Wan처럼 상수 테이블 대신 **모듈 트리에서 도출**(`get_cache_specs`) → 축소 config(P1 테스트)에도 그대로 동작.
- Wan 패턴 그대로: idx 0(conv_in cache, z의 slice)은 runtime 출력/입력, idx 1..37은 static buffer + `rbln_cache_update`, `CompileContext(use_weight_sharing=True)`, 프레임 루프는 `vae.py`의 새 클래스 `RBLNRuntimeCogVideoXVAEDecoder`.
- CPU 검증: D0/DN 래퍼 + host 루프를 CPU에서 (`rbln_cache_update`를 in-place 에뮬레이션) 돌리면 `vae.decode`와 **max|diff| = 0.0** (latent 7, 8x12 / 6x9 해상도). 로직은 정확.

### 컴파일러 이슈 1: `rbln_cache_update` shape 제약 (해결 — workaround)
- 증상: 480x720 첫 컴파일에서 60x90 해상도 cache 12개 (`(1,512,2,5400)`)가
  `'rblnTensor.nn.contrib_custom_cache_update' op wrong shape for custom op for running on device` (Partition.cpp `verifyCustomOpsDevRunnable`) → `RBLNCompileError [DEVICE_GRAPH_CONVERSION]`.
- repro: `examples/cogvideo/repro_cache_update_shape.py` (shape마다 별도 프로세스). 결과: flat `h*w` 축이 **64의 배수**여야 한다.
  5400 FAIL(partition), 21600 FAIL(SIGABRT: `a memory.store still writes a DRAM-view concatenate; FoldMemoryTransfer must fold every such store into chunks`),
  5408/5440/5632/21632/86400/345600 PASS.
- workaround: cache의 flat 축을 64 배수로 zero-pad (`_CACHE_HW_ALIGN`, `_to_cache` pad / `_from_cache` slice). 정렬된 shape은 변경 없음. 컴파일러 수정 불필요.

### P0-1a VAE 단계별 결과
| 단계 | 컴파일 그래프 | 컴파일 | 시간 | host peak RSS | device peak | r (overall) | 비고 |
|---|---|---|---|---|---|---|---|
| latent 1 (1 frame) | decoder_0 (lat 1) | OK | 35.4s | 11.9 GB (time -v) | 4.38 GiB (rbln8) | 0.999992 | PSNR 68.2dB, a=1.00115 b=-6.5e-5, allclose False (max\|diff\| 0.236) |
| latent 3 (9 frames) | decoder_0 (lat 3), num_frames=9 | OK | 70.1s | 31.2 GB | 8.66 GiB | 0.999988 | 9 frame 모두 r≈1.0000, PSNR 66–69dB, a=1.00114 b=-8.5e-5 |
| latent 3, 49-frame graphs (d0+dn, 캐시 end-write, fp32) | d0 lat3 + dn lat2 | 컴파일 OK | ~2분 | 12.6 GB | — | — | **runtime 생성 OOM**: dn `Intermediate` 7.78 GB 추가 할당 실패 |
| 같음 + `num_devices=2` (RSD) | | 컴파일 OK | | | — | — | 동일 OOM (RSD가 conv 그래프를 나누지 않음, node당 intermediate 동일) |
| 49-frame graphs, per-conv cache write (hook), fp32 cache | d0+dn | 컴파일 OK | 116s | | — | — | OOM: d0 10.32 GiB(Intermediate 8.73) + dn 6.79 GiB(DramTensor 5.27 = fp32 static cache) |
| 49-frame graphs, per-conv write, **bf16 cache** | d0+dn | OK, 로드 OK | 120.7s | 42.8 GB | 14.91 GiB | **frame 0–8 r≈1.0, PSNR 67–68dB / frame 9–16 (decoder_n) 깨짐** | dn 출력 ≈ 4e9 상수, PSNR 5dB → dn 경로 버그 (아래 이슈 3) |
| 49-frame graphs, end write, bf16 cache | d0+dn | 컴파일 OK | 127.3s | | — | — | OOM: d0 8.66 GiB (Intermediate 4.40) + dn 9.14 GiB (**별도** Intermediate 7.25) |
| **49-frame graphs, end write, bf16 cache, decoder_n 먼저 컴파일** | d0+dn | **OK** | 131.5s | 11.1 GB | **13.41 GiB** | latent 5 (17 frames): **0.999992**, 17 frame 모두 r=1.0000, PSNR 66.9–68.9dB | d0가 dn의 intermediate를 재사용 (d0 alloc 1.36 GiB, dn 12.04 GiB) |
| **latent 13 (49 frames)** 같은 아티팩트 | d0 + 5×dn | **OK** | (재사용) | 11.1 GB | 13.41 GiB | **0.999994** | **P0-1b 달성.** a=1.00145 b=-5.4e-5, rel_l2 0.0036, mean\|diff\| 5.5e-4, max\|diff\| 0.656, allclose False. frame별 r min 0.999974 (frame 0), 마지막 dn chunk 평균 0.999996 → conv_cache 누적 오차 없음. PSNR 66.5–70.0dB. decode 18.2s |

추가 발견 (CPU, `chunking_cpu.py`): chunk 분할을 HF와 다르게 하면 결과가 달라진다. `[1,2×6]` → r=0.993, max|diff|=45.7 (깨짐), `[1×13]` → T=1 chunk는 시간축 upsample이 안 돼 13 frame만 나옴.
→ decoder_0 = latent 3, decoder_n = latent 2 (HF와 동일)는 필수. 메모리 문제를 chunk 축소로 우회할 수 없다.

### 컴파일러 이슈 3: static cache의 write-after-read hazard (workaround: end write 유지)
- 메모리를 줄이려고 각 conv 직후에 cache를 쓰도록(forward hook) 바꾸자 decoder_n 출력이 깨졌다. 256x384 비교:
  per-conv write → dn frame r 0.003–0.99 (bf16/fp32 모두), end write → 17 frame 모두 r=1.0000 (bf16/fp32 모두).
- repro: `examples/cogvideo/repro_cache_update_war.py`. graph W: cache←x0, graph RW: out=conv(cat(cache,x1)); cache←x1.
  새 cache 값이 old cache에 의존하지 않으면 write가 read보다 먼저 실행된다 (`end`/`mid`/`dep_mul0`/`dep_slicecat` 모두 out이 x1 기준: r(out, cache=x1)=0.999996).
  write 값이 **read를 소비한 conv의 출력**에 의존하면 순서가 지켜진다 (`dep_h`, `dep_hslice`: r(out, expected)=0.999996).
- 그러나 VAE의 per-conv write에 `0 * conv_out[..., :1]` 의존성을 넣어도 256x384에서 여전히 깨졌다 (dn frame r 0.35–0.90). 원인 미확정
  (per-conv 빌드에서는 dn이 intermediate를 따로 갖지 않고 d0 것을 공유하는 등 메모리 배치가 달라짐).
- 결론: Wan과 같은 **end write**를 유지. VAE 전체 그래프에서는 end write가 정확하다 (위 표). 단 toy repro에서는 end write도 hazard가 나므로,
  정확성은 실제 그래프 순서에 기대고 있다 → 컴파일러 쪽 수정 필요 (static buffer의 WAR 의존성 보장).

### 메모리 이슈: 그래프 간 intermediate 공유는 "먼저 컴파일된 그래프의 버퍼가 충분히 클 때만"
- `rebel/src/compile/codegen/codegen.cc` `AllocIntermediateMemory`: 같은 CompileSession의 이전 compile_id가 만든 intermediate 버퍼로 pool을 seed하고,
  크기가 맞으면 재사용, 아니면 새로 만든다. d0(4.40 GiB)를 먼저 컴파일하면 dn(7.25 GiB)이 새 버퍼를 만들어 합계 17.8 GiB → OOM.
- workaround: `get_compiled_model`에서 **decoder_n을 먼저 컴파일** → d0가 dn의 버퍼를 재사용 → 13.41 GiB로 한 device에 들어감. 컴파일러 수정 불필요.
- (참고) `num_devices=2` RSD는 conv 그래프를 나누지 않아 도움이 안 됨. chunk 축소는 HF와 결과가 달라져 불가 (위 CPU 실험).

### 컴파일러 이슈 2: 240x360 VAE 컴파일 assert
- 240x360 (latent 30x45)로 디버그용 축소 컴파일 시 `[@MLIR] assert:ScatterInfo size mismatch with shape-based size` (SIGABRT). 4가지 변형(write 위치 × cache dtype) 모두 동일. 480x720에선 발생 안 함.
  → 디버그 해상도를 256x384 (모든 cache h*w가 64 배수, padding 불필요)로 변경.

### P0-2 Transformer 단계별 결과 (랜덤 초기화, 실제 config: heads 30, head_dim 64, text_embed_dim 4096, max_text_seq 226, patch 2; 입력 49 frames → latent 13×60×90, seq 17550+226)
- 설계: `time_proj`(sinusoid, weight 없음)는 host에서 계산하고 그래프 입력 `timestep_proj`로 넣는다 (device sin/cos 정밀도 회피). 나머지 HF forward는 그대로.

| num_layers | 컴파일 | 컴파일 시간 | host peak RSS | device peak | RBLN fwd | r | 비고 |
|---|---|---|---|---|---|---|---|
| 1 | OK | 66.0s | 6.1 GB | 0.72 GiB | 0.35s | 0.999996 | max\|diff\| 0.0128, a=1.00158, allclose True |
| 2 | OK | 74.4s | | | 0.64s | 0.999996 | allclose True |
| 8 | OK | 157.4s | | 1.91 GiB | 2.54s | 0.999995 | max\|diff\| 0.0116, a=1.00142, allclose True |
| 16 | OK | 294.2s | | 2.98 GiB | 5.05s | 0.999993 | allclose True. alloc: CmdStream 0.117 GiB, Kernel 1.76 GiB |
| 18 | OK | 340.3s | | 2.79 GiB (alloc) | 5.67s | 0.999992 | allclose True. CmdStream 0.180 GiB |
| 19 | OK | 350.5s | | | 6.03s | 0.999993 | allclose True |
| **20** | 컴파일 OK | 366.2s | | 3.01 GiB (alloc) | — | — | **실행 hang: `SYS_KERNEL_TIMEOUT`**. CmdStream 0.199 GiB |
| 24 | 컴파일 OK | 435.3s | | 3.49 GiB (alloc) | — | — | hang (`SYS_KERNEL_TIMEOUT`). CmdStream 0.238 GiB |
| 30 | 컴파일 OK | 553.0s / 613.5s | | 4.20 GiB (alloc) | — | — | hang. **timeout 600s로도 hang** → 느린 게 아니라 멈춤. CmdStream 0.299 GiB |

- **깨지기 시작하는 레이어 수: 20** (1/2/8/16/18/19 OK, 20/24/30 hang). 메모리는 충분(≤4.2 GiB)하고, 시간도 레이어당 ~0.31s로 선형이라 컴파일러/런타임 쪽 스케일 문제(그래프 크기, CmdStream 크기 등)로 보인다.
- **workaround: 그래프 분할.** `RBLNCogVideoXTransformer3DModelConfig.layers_per_graph`(기본 16) 단위로 블록을 나눠 여러 그래프로 컴파일하고 host에서 연결
  (`CogVideoXTransformer3DModelSegment`: 첫 segment = time/patch embedding + 블록, 마지막 = 블록 + norm/proj/unpatchify. 중간 텐서 hidden_states (B,17550,1920) 등은 host 경유).
  CPU에서 HF forward와 max|diff| 0.0 (5 layers, 2-layer segment) 확인.

| 구성 | 컴파일 | 시간 | host peak RSS | device | RBLN fwd | r | 비고 |
|---|---|---|---|---|---|---|---|
| L30 random, 2 graph × 15 | OK | 499.1s | 24.9 GB | 5.89 GiB | 9.44s | 0.999990 | max\|diff\| 0.0159, a=1.00151 b=2.5e-5, allclose True |
| **L30 실제 가중치 (THUDM/CogVideoX-2b), 2 graph × 15** | **OK** | 420.3s | 24.8 GB | 5.89 GiB | 9.44s | **0.999970** | **P0-2c 달성.** max\|diff\| 0.0317, mean\|diff\| 0.0041, rel_l2 0.0085, a=0.99951 b=-0.00202, allclose False (max diff가 atol 2e-2 초과). latent frame별 r 0.99996–0.99997. 입력: seed 0, hidden (1,13,16,60,90), text (1,226,4096), t=500 |

#### transformer hang 조건 정리 (CmdStream 크기와 상관)
| 구성 | CmdStream (GiB) | 결과 |
|---|---|---|
| batch 1, 15 layers (segment) | 0.111 | OK |
| batch 1, 16 | 0.117 | OK |
| batch 1, 18 | 0.180 | OK |
| batch 1, 19 | 0.189 | OK |
| batch 1, 20 | 0.199 | hang |
| batch 1, 24 | 0.238 | hang |
| batch 1, 30 | 0.299 | hang |
| batch 2 (CFG), 15 (segment) | 0.328 / 0.324 | hang (P1 E2E 첫 시도) |
| batch 2 (CFG), 6 (segment ×5) | 0.098–0.133 | OK |
- 한 그래프의 command stream이 약 0.19 GiB를 넘으면 hang한다 (원인은 컴파일러/런타임 쪽, 미확정). batch 2는 layer당 CmdStream이 약 2배.
- `layers_per_graph` 기본값 = `max(1, 15 // batch_size)` (batch 1 → 2×15, batch 2 → 5×6). batch는 파이프라인이 CFG로 2배로 바꿀 수 있어 컴파일 시점에 결정.

- batch 2 (CFG) 실제 가중치 L30, 기본값 5 graph × 6: 컴파일 1339.4s, forward 19.06s, device peak 11.47 GiB,
  **r=0.999969**, max|diff| 0.0336, mean|diff| 0.0041, rel_l2 0.0086, a=0.99898 b=-0.00203, allclose False. latent frame별 r 0.99996–0.99997.
  산출물: `outputs/20261013_0138_transformer_L30_real_b2/`

### P0-3 회귀 테스트 (2026-10-13 00:26)
- `RBLN_DEVICES=16,17,18,19 python -m pytest -n 1 tests/test_diffusers.py -k "Cosmos or SVD" -vv`
  → **35 passed, 5 skipped, 15 subtests passed in 245.27s**. skip 5개는 각 클래스의 `test_automap` (RBLN_AUTO_CLASS 없음).
  Cosmos2.5 Predict / Cosmos2.5 Transfer / Cosmos2 Text2Image / Cosmos2 Video2World / SVD 모두 test_generate, save_load 등 통과.

### P1-7 테스트 `TestCogVideoXModel`
- tiny 파이프라인을 `setUpClass`에서 로컬 생성 → 임시 디렉터리에 `save_pretrained` → `HF_MODEL_ID`로 사용 (hub 업로드 없음).
  T5 1 layer d_model 32 (토크나이저는 CogVideoX-2b 것), transformer 1 layer 2 heads×16, VAE block_out_channels (8,8,8,8) layers_per_block 1.
  17 frames 128x128 (latent 5 → decoder_0 + decoder_n 모두 사용, latent 16x16이라 cache h*w 모두 64 배수).
- 첫 실행: `test_generate` 실패 — `CogVideoXPipeline`이 `self.transformer.cache_context("cond_uncond")`(diffusers CacheMixin)를 호출.
  → `RBLNCogVideoXTransformer3DModel.cache_context` no-op context manager 추가.
- `RBLN_DEVICES=24,25,26,27 python -m pytest -n 1 tests/test_diffusers.py -k CogVideoX -vv` → **7 passed, 1 skipped (test_automap), 3 subtests passed in 31.84s**.

### P1-5 Pipeline E2E 진행 기록
- CPU golden: `examples/cogvideo/run_e2e.py cpu` (diffusers CogVideoXPipeline fp32, prompt=panda, seed 42, 49 frames, 50 steps, guidance 6) — CPU에서 약 63s/step.
- RBLN 1차 (`20261013_0034_e2e`, 실패): export=True 컴파일+저장 1392s OK, export=False 로드 OK, 생성 실패 — `cache_context` 없음 → 추가.
- RBLN 2차 (`20261013_0107_e2e`, 실패): T5가 fp16으로 컴파일돼(fp32 T5-XXL은 한 device에 안 들어감) 파이프라인 latent가 fp16 → transformer 입력 dtype 오류.
  → `RBLNCogVideoXPipeline.encode_prompt`가 transformer dtype(fp32)으로 embeds를 반환 (latent/scheduler fp32 유지), transformer forward에서 입력 cast.
- RBLN 3차 (`20261013_0110_e2e`, 실패): batch 2 transformer segment(15 layers, CmdStream 0.328 GiB)가 hang → `layers_per_graph` 기본값 변경 (위).
- **RBLN 4차 (`20261013_0112_e2e`, 성공)**: `RBLNCogVideoXPipeline.from_pretrained("THUDM/CogVideoX-2b", export=True)` 컴파일+`save_pretrained` 1587s
  → `export=False` 재로드 58.1s → 49 frames 생성 954.9s (50 steps, CFG batch 2, transformer 5 graph × 6 layers). devices: text_encoder/transformer/vae 각 1개.
  - CPU golden 대비 (같은 prompt, seed 42): **video r=0.955, PSNR 23.6 dB**, frame별 r 0.925–0.972, PSNR 21.3–25.6 dB. 최종 latent r=0.985.
    a=0.962 b=-0.017, mean|diff| 0.054, max|diff| 2.05.
  - 눈으로 확인: `compare_cpu_vs_rbln.png` (frame 0/12/24/36/48, 위 CPU / 아래 RBLN) — 구도, 판다/기타, 움직임이 같다. 배경 판다 자세 등 세부만 다르다. 깨짐 없음.
  - 차이의 원인 분리: CPU golden의 latent를 RBLN VAE로 decode → golden video와 r=0.999991, PSNR 57.3 dB. 즉 VAE는 정확하고,
    차이는 50 step 동안 누적되는 denoising 궤적 차이다 (transformer 1 step r=0.99997, T5는 device에서 fp16).

### P1-6 예제
- `RBLN_DEVICES=20,21,22 python examples/cogvideo/run_cogvideo.py --model-dir .../compiled/pipeline_cogvideox_2b_v2 --out-dir .../outputs/20261013_0158_run_cogvideo_example`
  → 49 frames + video_rbln.mp4 저장, 17:46 (로드 + 50 steps + decode), Exit 0. 프레임 확인: 판다/기타/대나무 숲, 깨짐 없음.

### 최종 재확인 (2026-10-13 02:0x, 최종 코드 c00255bd 기준)
- `pytest -n 1 tests/test_diffusers.py -k CogVideoX -vv` → 7 passed, 1 skipped, 3 subtests passed
- `pytest -n 1 tests/test_diffusers.py -k "Cosmos or SVD" -vv` → 35 passed, 5 skipped, 15 subtests passed
- `git diff origin/dev..HEAD`: 22 files, +2274 / -0. 공유 코드(`modeling.py`, `modeling_base.py`, `configuration_utils.py`, `modeling_diffusers.py`) 변경 없음,
  `vae.py`는 새 클래스 `RBLNRuntimeCogVideoXVAEDecoder` 추가만.

## 요약 / 남은 블로커 / 다음 단계
**P0 전부 달성, P1 전부 달성.**
- P0-1 VAE decoder: latent 1 → 3 → 5 → 13 단계별 컴파일·실행. 49 frames r=0.999994 (frame별 r ≥ 0.999974, 누적 오차 없음).
- P0-2 Transformer: L1/2/8/16/18/19 단일 그래프 OK, L20 이상 단일 그래프는 device hang → 그래프 분할로 L30 실제 가중치 r=0.999970 (batch 2: 0.999969).
- P0-3 회귀: Cosmos/SVD 35 passed. P0-4: ruff 깨끗 (브랜치 기준), 로컬 commit.
- P1: 파이프라인 E2E (golden 대비 PSNR 23.6dB, 시각적 동일, VAE 단독 57.3dB), 예제, 테스트.

컴파일러 쪽에 남은 문제 (workaround로 우회, 컴파일러 수정은 하지 않음):
1. `rbln_cache_update`: flat 축이 64 배수가 아니면 컴파일 실패 / assert (`repro_cache_update_shape.py`). workaround: zero-pad.
2. static buffer의 write-after-read 순서 미보장 (`repro_cache_update_war.py`). VAE는 Wan과 같이 end write로 맞게 동작하지만, toy에서는 end write도 깨짐 → 근본 수정 필요.
3. 그래프 간 intermediate 공유가 "먼저 컴파일된 그래프가 클 때만" 동작 → decoder_n을 먼저 컴파일해서 우회.
4. 한 그래프의 command stream이 약 0.19 GiB를 넘으면 실행 hang (`SYS_KERNEL_TIMEOUT`). → transformer를 `layers_per_graph` 단위로 분할. 최소 repro 미작성
   (재현: `verify_transformer.py --num-layers 20 --random-init --layers-per-graph 64`).
5. 240x360 VAE에서 `ScatterInfo size mismatch` assert (축소 해상도 디버그용이라 우회만 함).

다음 단계 후보:
- 위 1–4를 컴파일러 팀에 repro와 함께 보고 (특히 2: 정확성 문제).
- transformer segment 사이 hidden_states를 host 대신 device tensor로 넘기기 (현재 step당 ~0.26 GB 왕복).
- VAE encoder (image-to-video, `uses_encoder=True`), CogVideoX-5b (rotary embedding) 지원.
- T5를 fp32로 (2 device) 돌리면 golden과의 E2E 차이가 줄어드는지 확인.

## 산출물 (영상/프레임)
- `/home/seinpark/nas_data/cogvideox/outputs/20261012_2326_vae_stageA_lat1/` — VAE latent 1 → 1 frame, CPU vs RBLN (frames_cpu/, frames_rbln/ PNG), r=0.999992
- `/home/seinpark/nas_data/cogvideox/outputs/20261012_2334_vae_stageB_lat3_f9_d0only/` — VAE latent 3 → 9 frames (decoder_0만), r=0.999988, video_cpu.mp4 / video_rbln.mp4
- `/home/seinpark/nas_data/cogvideox/outputs/20261012_2352_vae_stageC_lat5_f17/` — **깨진 결과 (보관)**: latent 5 → 17 frames, frame 0–8 정상 / 9–16 (decoder_n) ≈4e9 상수. dn 버그 근거
- `/home/seinpark/nas_data/cogvideox/outputs/20261013_0005_dbg_dn_256x384_*` (4개) — dn 디버그 256x384: `conv_*` = **깨진 결과 (보관, per-conv write)**, `end_*` = 정상
- `/home/seinpark/nas_data/cogvideox/outputs/20261013_0009_dbg_dn_256x384_convdep_bf16/` — **깨진 결과 (보관)**: per-conv write + conv 출력 의존성, dn frame r 0.35–0.90
- `/home/seinpark/nas_data/cogvideox/outputs/20261013_0015_vae_stageC_lat5_f17/` — latent 5 → 17 frames (d0 + 1×dn), r=0.999992, 정상
- `/home/seinpark/nas_data/cogvideox/outputs/20261013_0015_vae_stageD_lat13_f49/` — **P0-1b**: latent 13 → 49 frames (d0 + 5×dn), r=0.999994, 전 frame PNG + video_cpu.mp4/video_rbln.mp4 + latents.pt
- `/home/seinpark/nas_data/cogvideox/outputs/20261013_0033_e2e_cpu_golden/` — **P1-5 CPU golden** (fp32, panda prompt, seed 42, 50 steps): video_cpu.mp4, 49 frames, latents_cpu.pt
- `/home/seinpark/nas_data/cogvideox/outputs/20261013_0034_e2e/`, `20261013_0107_e2e/`, `20261013_0110_e2e/` — **실패한 E2E 시도 (영상 없음, meta.json에 원인)**
- `/home/seinpark/nas_data/cogvideox/outputs/20261013_0112_e2e/` — **P1-5 RBLN E2E** video_rbln.mp4, 49 frames, latents_rbln.pt, compare_cpu_vs_rbln.png. r=0.955, PSNR 23.6 dB vs golden
- `/home/seinpark/nas_data/cogvideox/outputs/20261013_0158_run_cogvideo_example/` — **P1-6** 예제 실행 결과 (짧은 panda prompt, seed 42): video_rbln.mp4, 49 frames
- `/home/seinpark/nas_data/cogvideox/outputs/20261013_0138_transformer_L30_real_b2/` — transformer batch 2 실제 가중치, r=0.999969
- `/home/seinpark/nas_data/cogvideox/outputs/20261012_2328_transformer_L1_rand/`, `..._L2_rand/`, `..._L8_rand/`, `20261013_0009_transformer_L16_rand/` — transformer 입력/출력 텐서 (inputs.pt, outputs.pt, meta.json)
- `/home/seinpark/nas_data/cogvideox/outputs/20261013_0032_transformer_L30_rand_seg/` — L30 random, 2 graph, r=0.999990
- `/home/seinpark/nas_data/cogvideox/outputs/20261013_0031_transformer_L30_real/` — **P0-2c**: L30 실제 가중치, r=0.999970
