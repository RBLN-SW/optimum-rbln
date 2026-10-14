# CogVideoX-5b-I2V enable — progress

- 시작 시각: **2026-10-13 21:51:39 KST** (`date`: `Tue Oct 13 09:51:39 PM KST 2026`)
- 종료 기한: 2026-10-14 09:51 KST (12시간)
- 브랜치: `seinpark/cogvideox` (HEAD e56eddd7, origin과 같음). 로컬 commit만.
- 모델: `zai-org/CogVideoX-5b-I2V` (공유 HF 캐시, snapshot a6f0f485, 파일 크기 hub와 일치 확인)
- 환경: `~/venv_optimum`, optimum-rbln editable, rebel 0.12.1.dev99+g18b188f8, diffusers 0.38.0

## 로그

### P0-1 VAE encoder (2026-10-13 21:58) — 달성
- 구현: `_VAECogVideoXEncoder` (graph 하나, `vae.encoder(x, conv_cache=None)`), `RBLNRuntimeCogVideoXVAEEncoder` (1 frame만 허용), config의 `uses_encoder` 허용.
  compile config 순서 `decoder_0, decoder_n, encoder`, 컴파일 순서 `decoder_n → decoder_0 → encoder` (같은 CompileContext, intermediate 재사용).
  `__post_init__`은 runtime을 이름으로 찾는다 (`create_runtimes=False`일 때 `self.model`이 UnavailableRuntime이라 index로 꺼냄 — zip(strict) 실패를 고침).
- 검증 `examples/cogvideo/verify_vae_encoder.py` (RBLN_DEVICES=8): 이미지 = t2v CPU golden 첫 frame (480x720 PNG).
  - latent mean: **r=0.999981**, max|diff| 0.103, mean|diff| 0.0060, rel_l2 0.0075, a=1.00415 b=-0.00026, allclose False
  - moments (mean+logvar): r=0.999997
  - 컴파일 128.6s. device alloc: decoder_0 1.36 / decoder_n 11.62 (Intermediate 7.25) / **encoder 0.25 GiB (intermediate 공유)**, device peak 13.23 GiB (rbln8)
  - 산출물: `/home/seinpark/nas_data/cogvideox/outputs/20261013_2158_vae5bi2v_encoder/` (input_image.png, encode.pt, meta.json)

### P0-2 Transformer — 진행 중
- 구현: graph wrapper가 `image_rotary_emb_cos`, `image_rotary_emb_sin` (`[F*(H/p)*(W/p), head_dim]`, float32)을 추가 입력으로 받아 HF forward에
  `image_rotary_emb=(cos, sin)`으로 넘긴다 (`use_rotary_positional_embeddings`일 때만 input_info에 추가). 1.5 전용(`ofs_embedding`, `patch_size_t`)은 NotImplementedError.
  RBLN forward는 rotary 모델이면 `image_rotary_emb`를 필수로 받는다. RoPE 표는 파이프라인 `_prepare_rotary_positional_embeddings` (1.0 경로)와 같은 방식.
- 1단계 (2026-10-13 22:02): 5b-I2V config `num_layers=1` 랜덤, 실제 shape (2,13,32,60,90) / (2,226,4096) / RoPE (17550,64), TP8, batch 2
  → 컴파일 84.4s, **r=0.999990**, max|diff| 0.0144, a=1.00196 b=-0.00025, allclose True. device당 CmdStream 0.004 GiB
  (42 layer면 단순 계산으로 약 0.17 GiB → hang 기준 ~0.19 GiB에 근접). 산출물 `outputs/20261013_2202_i2v5b_transformer_L1_rand_tp8_b2/`
- 2단계: 실제 가중치 42 layer, TP8, batch 2 — 실행 중

### P1-6 tiny I2V 테스트 — 완료 (2026-10-13 22:07)
- `tests/test_diffusers.py`: `_save_tiny_cogvideox_pipeline(save_dir, image_to_video=True)`가 `CogVideoXImageToVideoPipeline`을
  로컬에서 만든다 (hub 업로드 없음). transformer `in_channels=8`(영상 latent 4 + 이미지 latent 4), `out_channels=4`,
  `use_rotary_positional_embeddings=True`, `use_learned_positional_embeddings=True`. VAE는 t2v 테스트와 같은 64채널.
- `TestCogVideoXImageToVideoModel`: 480x720 단색 이미지, 17 frame, 2 step. `test_encode_image`는 VAE encoder 출력 shape
  `(1, 4, 1, 60, 90)` 확인.
- `RBLN_DEVICES=8..15 python -m pytest -n 1 tests/test_diffusers.py -k CogVideoXImageToVideo -vv`
  → **8 passed, 1 skipped (test_automap), 3 subtests passed in 49.19s**

### P0-4 코드 품질 — 통과 (2026-10-13 22:07, 커밋 전)
- `ruff format --check` / `ruff check`: 추적 파일 + 신규 src 파일 모두 통과. (저장소 전체 `ruff check .`의 11건은 모두
  커밋하지 않는 로컬 스크래치 파일: `examples/image-to-text/*`, `qwen_test.py`)
- `mypy`: Success: no issues found in 331 source files
- `python scripts/check_inits.py`: exit 0
- `validate_docstrings.py`: 변경된 src 8개 파일 각각 exit 0

### P0-2 2단계 (42 layer 실제 가중치, TP8, batch 2) — SYS_KERNEL_TIMEOUT (2026-10-13 22:22)
- 컴파일 887.0s, 저장 `compiled/i2v5b_transformer_L42_real_tp8_b2`. device alloc 합계 18.488 GiB.
  device별 CmdStream [0.121, 0.082, 0.082, 0.082, 0.098, 0.082, 0.082, 0.082] GiB,
  Kernel [2.725, 1.25 x7], Intermediate [0.863, 0.408 ...] GiB.
- 첫 forward에서 `SYS_KERNEL_TIMEOUT ... (code: 2 Wait job kernel timeout) ([cmd_dispatcher] Failed to WaitForCompletion, seq=0.)`.
- 재확인: 같은 아티팩트를 `rbln_config={"timeout": 600}`으로 로드해도 실행 시작 ~20s 후 같은 에러
  → 사용자 runtime timeout(기본 60s)이 아니라 device 쪽 job watchdog.
- CmdStream이 0.121 GiB로 2b에서 본 hang 기준(~0.19 GiB)보다 작다 → "CmdStream 크기" 가설은 이 경우를 설명 못 함.
  대안 가설: 한 job(그래프 1회 실행)의 device 실행 시간이 watchdog(약 10s?)을 넘는다. 근거: 2b TP8 30 layer = 4.1s/step,
  2b 1 device 19 layer(추정 ~10.4s) OK / 20 layer(~10.9s) hang. 5b는 layer당 연산 ~2.5배, 42 layer → TP8에서 ~14s 추정.
- 다음 시도: (a) TP16 42 layer 단일 그래프 (NPU 16-31), (b) 가설 확인용 21 layer TP8 (NPU 0-7) — step 시간 측정.

### P0-3 회귀 테스트 — 통과 (2026-10-13 22:24, NPU 8-15)
- `python -m pytest -n 1 tests/test_diffusers.py -k CogVideoX -vv` → **15 passed, 2 skipped, 6 subtests passed in 101.80s**
  (t2v `TestCogVideoXModel` + I2V `TestCogVideoXImageToVideoModel`, skip은 `test_automap` 2개)
- `python -m pytest -n 1 tests/test_diffusers.py -k "Cosmos or SVD" -vv` → **35 passed, 5 skipped, 15 subtests passed in 137.56s**

### P0-2 가설 확인: 21 layer 실제 가중치, TP8, batch 2 (2026-10-13 22:37)
- 컴파일 466.7s, CmdStream [0.061, 0.043 ...] GiB. **r=0.999992**, max|diff| 0.189, mean|diff| 0.0170, rel_l2 0.00449,
  a=0.99841 b=0.00446, allclose False. 산출물 `outputs/20261013_2237_i2v5b_transformer_L21_real_tp8_b2/`
- warm 실행 시간: 5.16s, 5.03s → 42 layer 단일 그래프는 ~10s/step. 2b에서 본 경계(1 device 19 layer ~10.4s OK, 20 layer hang)와
  같은 크기 → device job watchdog ~10s 가설과 맞다. CmdStream 크기 가설은 아님 (42 layer TP8 = 0.121 GiB에서도 hang).

### P0-2 TP16 시도 — SYS_KERNEL_TIMEOUT (2026-10-13 22:58)
- 42 layer 실제 가중치, TP16 단일 그래프 (NPU 16-31): 컴파일 1511.1s, alloc 합계 24.562 GiB, device당 CmdStream 최대 0.09 GiB.
  첫 forward에서 같은 `SYS_KERNEL_TIMEOUT (code: 2 Wait job kernel timeout)`. → device 수를 늘려도 한 job의 실행 시간이 충분히 줄지 않는다.

### P0-2 layer split 재도입 (2026-10-13 23:00)
- `RBLNCogVideoXTransformer3DModelConfig.layers_per_graph` (기본 None → 자동). 블록을 ceil(num_layers / layers_per_graph)개
  그래프로 균등 분할, host에서 연결. 자동 값 = `MAX_GRAPH_WORK // (batch x video_tokens x inner_dim)`,
  `MAX_GRAPH_WORK = 21 x 2 x 17550 x 3072` (5b 21 layer 측정치). → 2b(30 x 1920, batch 2): 33 → 1 그래프 (t2v 아티팩트 이름/동작 동일),
  5b-I2V(42 x 3072): 21 → 2 그래프 (`segment_0_layers_0_21`, `segment_1_layers_21_42`).
- `CogVideoXTransformer3DModelSegment`: 첫 segment = time_proj(그래프 안) + time_embedding + patch_embed(learned pos emb 포함) + split,
  나머지 = (hidden_states, encoder_hidden_states, emb), 마지막 = norm_final/norm_out/proj_out/unpatchify. 모든 segment가 RoPE cos/sin 입력.
- CPU 확인: 5 layer tiny 모델을 [(0,2),(2,3),(3,5)]로 연결 → HF forward 대비 max|diff| 0 (rope 있음/없음 둘 다).
- tiny I2V 테스트: transformer 2 layer + `layers_per_graph=1` → 2 그래프. `test_transformer_segments` 추가.
  `RBLN_DEVICES=8..15 python -m pytest -n 1 tests/test_diffusers.py -k CogVideoX -vv` → **16 passed, 2 skipped, 6 subtests passed in 126.81s**
- 42 layer 실제 가중치 TP8 batch 2, 기본값(2 그래프) 검증 실행 중 (NPU 0-7).

### P0-2 Transformer — 완료 (2026-10-13 23:21)
- 명령: `RBLN_DEVICES=0..7 python verify_transformer.py --model-id zai-org/CogVideoX-5b-I2V --batch-size 2 --tensor-parallel-size 8 --compiled-dir .../i2v5b_transformer_L42_real_tp8_b2_seg2`
- 기본 `layers_per_graph` → 2 그래프 (`segment_0_layers_0_21`, `segment_1_layers_21_42`), 컴파일 818.2s.
  alloc: segment_0 12.756 GiB / segment_1 12.838 GiB (8 device 합), CmdStream device당 최대 0.061 GiB.
- RBLN forward 10.12s (2 그래프 합). **r=0.999992**, max|diff| 0.0324, mean|diff| 0.00268, rel_l2 0.00526,
  a=1.00343 b=0.000025, allclose False (atol 2e-2 / rtol 1e-2 — 출력 크기 대비 소수 원소).
- 산출물 `outputs/20261013_2321_i2v5b_transformer_L42_real_tp8_b2_seg2/`

### P0-3 재실행 (layer split 이후, 2026-10-13 23:24)
- `-k CogVideoX` → 16 passed, 2 skipped (23:02). `-k "Cosmos or SVD"` → **35 passed, 5 skipped, 15 subtests passed in 135.25s**

### P0-4 — 완료 (2026-10-13 23:24)
- ruff format/check (추적 + 신규 src), mypy (331 files OK), check_inits exit 0, docstrings 9개 파일 exit 0.
- 로컬 커밋 `7a93cd68 model: CogVideoX-5b-I2V (image-to-video) pipeline` on `seinpark/cogvideox` (origin보다 1 ahead, push 안 함).

### P1-5 I2V E2E — 진행 중
- CPU golden: `python run_i2v_e2e.py cpu` (22:07 시작, 50 step ~230s/step → ~01:15 종료 예상). 이미지 = 2b golden 첫 frame, panda prompt, seed 42, guidance 6.
- RBLN: `RBLN_DEVICES=16..25 python run_i2v_e2e.py rbln --compiled-dir .../pipeline_cogvideox_5b_i2v` (23:24 시작):
  export=True → save_pretrained → export=False 로드 → 49 frame 생성. 이후 `compare` 모드로 golden과 비교.

### P1-5 I2V E2E — 완료 (2026-10-14 01:17)
- CPU golden: `python run_i2v_e2e.py cpu` → 11134s, `outputs/20261013_2207_i2v_e2e_cpu_golden/`
- RBLN: `RBLN_DEVICES=16..25 python run_i2v_e2e.py rbln --compiled-dir .../pipeline_cogvideox_5b_i2v`
  → `from_pretrained("zai-org/CogVideoX-5b-I2V", export=True)` + `save_pretrained` 1285s, `export=False` 로드 329.4s,
  49 frame 생성 507.1s (text_encoder 1 + transformer TP8 2 그래프 + VAE 1 device). `outputs/20261013_2324_i2v_e2e/`
- 비교 (`run_i2v_e2e.py compare`): video **PSNR 24.89 dB, r=0.970683**, max|diff| 2.054, mean|diff| 0.0299, rel_l2 0.157,
  a=0.97584 b=-0.0161, allclose False. frame 0: 50.0 dB / r=0.9999, frame 23까지 ≥26.6 dB / r≥0.98, 이후 ~22 dB / r~0.94.
  final latents r=0.992072.
- 원인 분리: RBLN VAE로 CPU golden latents를 decode → **r=0.999993, 56.51 dB** (frame별 54.14~58.39 dB).
  transformer 1 step r=0.999992 (P0-2). → E2E 차이는 50 step denoising 누적 drift (t2v 2b에서도 24.4 dB / r=0.962, VAE 57.3 dB로 같은 양상).
- 육안: 같은 장면/구도/움직임, artifact 없음. frame 0 = 입력 이미지.

## 상태 요약 (2026-10-14 01:18)
| 조건 | 상태 | 근거 |
|---|---|---|
| P0-1 VAE encoder | 완료 | latent mean r=0.999981, decoder_0/decoder_n/encoder 한 device (peak 13.23 GiB) |
| P0-2 Transformer | 완료 | L1 random r=0.999990; 42 layer 실제 가중치 TP8 batch 2 RoPE r=0.999992 (2 그래프 x 21 layer; 단일 그래프는 TP8/TP16 모두 SYS_KERNEL_TIMEOUT) |
| P0-3 회귀 | 완료 | -k CogVideoX 16 passed 2 skipped; -k "Cosmos or SVD" 35 passed 5 skipped |
| P0-4 코드 품질 | 완료 | ruff/mypy/check_inits/docstrings 통과, 로컬 커밋 7a93cd68 (push 안 함) |
| P1-5 I2V E2E | 완료 | export→save→export=False, 49 frame, PSNR 24.89 dB / r=0.971 (VAE 단독 56.51 dB) |
| P1-6 tiny I2V 테스트 | 완료 | TestCogVideoXImageToVideoModel (setUpClass 로컬 생성, transformer 2 그래프) |

### P0-4 ruff 정정 (2026-10-14 01:25)
- 작업 트리에서 그대로 실행: `ruff format --check .` → 422 files already formatted, exit 0.
  `ruff check .` → **Found 11 errors, exit 1**. 11개 모두 이 작업과 무관한 미추적(`??`) 스크래치 파일:
  `examples/image-to-text/{dump_batch_state,iso_fwdsub_drift,run_qwen3_5,test_host_dtype_divergence,test_qwen3_5_regression}.py`, `qwen_test.py`.
  사용자 파일이라 수정/삭제하지 않음.
- 커밋 `7a93cd68`을 `git archive`로 깨끗한 디렉터리에 풀어 그대로 실행: `ruff format --check .` → 380 files already formatted, exit 0;
  `ruff check .` → All checks passed!, exit 0.
- 이전 상태 요약의 "ruff 통과"는 추적 파일 기준이었음 (작업 트리 전체 기준 아님).
- 사용자 결정 (2026-10-14): P0-4 ruff는 커밋 `7a93cd68` 기준 결과(깨끗한 export에서 `ruff format --check .` / `ruff check .` 모두 exit 0)로 인정.
  → **완료 조건 (5b-I2V) 전부 충족.**

## layers_per_graph 삭제 (2026-10-14 11:50, 사용자 지시)
- "Wait job kernel timeout"(code 2)의 출처: 런타임 `CommandDispatcher::WaitForCompletion` → `rblnWaitJob` 반환값 2 = 커널 드라이버
  lockup 판정. 기준은 드라이버 모듈 파라미터 `/sys/module/rebellions/parameters/timeout_s = 6`
  ("Device lockup timeout in sec, default 6s", `tolerance_cnt = 4`). `rbln_config`의 `timeout`은 code 1(BUSY)에만 해당.
  → 앞의 "약 10s watchdog" / "CmdStream ~0.19 GiB" 설명은 추정이었고 정확하지 않음. TP16은 실행 시간을 재지 않음.
- 사용자가 드라이버 timeout을 직접 해결하기로 함. TP8 모델을 21 layer 그래프 2개로 나누는 것은 받아들이지 않음
  → `layers_per_graph`, `CogVideoXTransformer3DModelSegment`, `MAX_GRAPH_WORK` 삭제. transformer는 다시 한 그래프 (RoPE 입력 유지).
- 삭제 후 `RBLN_DEVICES=8..15 python -m pytest -n 1 tests/test_diffusers.py -k CogVideoX -vv` → 15 passed, 2 skipped, 6 subtests passed in 86.28s.
- 주의: 위 P0-2 (r=0.999992)와 P1-5 E2E (PSNR 24.89 dB) 결과는 2 그래프 버전으로 측정한 값. 드라이버 timeout 해결 후
  42 layer TP8 한 그래프로 다시 측정해야 현재 코드 기준 결과가 된다.
