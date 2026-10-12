# CogVideoX enable — progress

- 시작 시각: **2026-10-12 23:14:06 KST** (`date` 출력: `Mon Oct 12 11:14:06 PM KST 2026`)
- 종료 기한: 2026-10-13 11:14 KST (12시간)
- 브랜치: `seinpark/cogvideox` (from `origin/dev` @ e2fa89a6)
- 환경: `~/venv_optimum`, optimum-rbln editable (`~/optimum-rbln/src/...`), rebel 0.12.1.dev99+g18b188f8 (`~/rebel_compiler`), diffusers 0.38.0, torch 2.11.0+cpu, transformers 5.15.1
- HF_HOME=/mnt/shared_data/groups/sw_dev/.cache/huggingface

## 로그

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
| 30 | 컴파일 OK | 553.0s | | | — | — | **실행 시 `SYS_KERNEL_TIMEOUT`** (60s; L8이 2.5s라 느린 게 아니라 hang). 조사 중 (L16 bisect) |

## 산출물 (영상/프레임)
- `/home/seinpark/nas_data/cogvideox/outputs/20261012_2326_vae_stageA_lat1/` — VAE latent 1 → 1 frame, CPU vs RBLN (frames_cpu/, frames_rbln/ PNG), r=0.999992
- `/home/seinpark/nas_data/cogvideox/outputs/20261012_2334_vae_stageB_lat3_f9_d0only/` — VAE latent 3 → 9 frames (decoder_0만), r=0.999988, video_cpu.mp4 / video_rbln.mp4
- `/home/seinpark/nas_data/cogvideox/outputs/20261012_2352_vae_stageC_lat5_f17/` — **깨진 결과 (보관)**: latent 5 → 17 frames, frame 0–8 정상 / 9–16 (decoder_n) ≈4e9 상수. dn 버그 근거
- `/home/seinpark/nas_data/cogvideox/outputs/20261013_0005_dbg_dn_256x384_*` (4개) — dn 디버그 256x384: `conv_*` = **깨진 결과 (보관, per-conv write)**, `end_*` = 정상
- `/home/seinpark/nas_data/cogvideox/outputs/20261013_0009_dbg_dn_256x384_convdep_bf16/` — **깨진 결과 (보관)**: per-conv write + conv 출력 의존성, dn frame r 0.35–0.90
- `/home/seinpark/nas_data/cogvideox/outputs/20261013_0015_vae_stageC_lat5_f17/` — latent 5 → 17 frames (d0 + 1×dn), r=0.999992, 정상
- `/home/seinpark/nas_data/cogvideox/outputs/20261013_0015_vae_stageD_lat13_f49/` — **P0-1b**: latent 13 → 49 frames (d0 + 5×dn), r=0.999994, 전 frame PNG + video_cpu.mp4/video_rbln.mp4 + latents.pt
- `/home/seinpark/nas_data/cogvideox/outputs/20261012_2328_transformer_L1_rand/`, `..._L2_rand/`, `..._L8_rand/` — transformer 입력/출력 텐서 (inputs.pt, outputs.pt, meta.json)
