# CogVideoX on RBLN — 인수인계 (handoff)

이 문서는 CogVideoX 작업을 이어받는 에이전트를 위한 것이다. 이 문서만 읽고 바로 작업을 이어갈 수 있게 썼다.
날짜별 상세 기록은 [cogvideox_progress.md](cogvideox_progress.md) (2b text-to-video)와
[cogvideox_i2v_progress.md](cogvideox_i2v_progress.md) (5b image-to-video)에 있다. 두 문서의 앞부분 설명 중 일부는
나중에 바뀌었다. **이 문서와 다르면 이 문서가 맞다.**

마지막 갱신: 2026-10-14 15시, 브랜치 `seinpark/cogvideox` @ `64163788` (+ 이 문서 갱신 커밋)

---

## 1. 현재 상태 요약

| 항목 | 상태 |
|---|---|
| 브랜치 / PR | `seinpark/cogvideox` → `dev`, PR [#804](https://github.com/RBLN-SW/optimum-rbln/pull/804). origin과 같음 (push 완료) |
| `THUDM/CogVideoX-2b` text-to-video | 동작. `RBLNCogVideoXPipeline`. 현재 코드로 다시 컴파일하고 E2E 확인함: 모델 카드 dtype (VAE 포함 전체 fp16), fp32 transformer 둘 다 (2026-10-14, §6) |
| `zai-org/CogVideoX-5b-I2V` image-to-video | 동작. `RBLNCogVideoXImageToVideoPipeline`. 42 layer 한 그래프, TP8, VAE 포함 전체 bf16으로 E2E 확인함 (2026-10-14, §6). 드라이버 lockup timeout이 꺼진 뒤부터 실행된다 (아래 블로커) |
| dtype | 모든 submodule (VAE 포함)이 `torch_dtype`대로 컴파일된다 (upstream과 같게). 예제는 모델 카드 dtype: 2b `float16`, 5B 모델 `bfloat16` (§3 "dtype"). 커밋 `1c522b42` (VAE), `64163788` (예제) |
| `zai-org/CogVideoX1.5-5B-I2V` | 시작하지 않음. **사용자가 지시할 때만** 한다 |
| 테스트 | `pytest tests/test_diffusers.py -k CogVideoX -vv` → 15 passed, 2 skipped (`test_automap`). `-k "Cosmos or SVD"` → 35 passed, 5 skipped (2026-10-14, `63acc798`) |
| PR 본문 | 2b + 5b-I2V, dtype, 한 그래프 5b-I2V 결과로 갱신함 (2026-10-14). 제목 `model: add CogVideoX text-to-video and image-to-video pipelines`. 원본 `~/nas_data/cogvideox/pr804_body_draft_20261014_v2.md` |

### (해결됨, 서버 설정) 블로커: 5b-I2V transformer가 device에서 "Wait job kernel timeout"
- **2026-10-14 14:55 확인: 이 서버의 `/sys/module/rebellions/parameters/timeout_s`가 `0` (lockup 검사 꺼짐)이다.**
  누가 언제 바꿨는지는 확인하지 않았다 (사용자가 해결하기로 한 항목). 이 상태에서 5b-I2V 42 layer 한 그래프 (TP8, bf16)가
  step당 약 11.5s로 timeout 없이 50 step을 돌았다. `timeout_s`가 다시 6이 되면 아래 증상이 다시 난다.
- 증상: 42 layer transformer를 한 그래프로 컴파일하면 컴파일·로드는 되지만 첫 실행에서
  `SYS_KERNEL_TIMEOUT ... (code: 2 Wait job kernel timeout) ([cmd_dispatcher] Failed to WaitForCompletion, seq=0.)`.
  TP8, TP16 모두 같다.
- 원인: 커널 드라이버의 device lockup timeout.
  - 런타임 `CommandDispatcher::WaitForCompletion` (`~/rebel_compiler/rebel/src/runtime/base/command_dispatcher.cc:77`)이
    `rblnWaitJob`을 부르고, 반환값 2가 "Wait job kernel timeout"이다 (`rebel/src/common/status.cc:272`).
  - 기준은 드라이버 모듈 파라미터 `/sys/module/rebellions/parameters/timeout_s = 6`
    (`modinfo`: "Device lockup timeout in sec (0 = disabled, default 6s)"), `tolerance_cnt = 4`.
  - `rbln_config={"timeout": N}`은 code 1 (BUSY, 사용자 timeout)에만 해당한다. 올려도 소용없다.
- 측정값이 "한 번 실행이 약 6초를 넘으면 실패"와 맞는다 (증명은 아님. 드라이버 소스는 보지 않았다):

  | 구성 | 1회 실행 | 결과 |
  |---|---|---|
  | 2b, TP1, batch 1, 19 layer | 6.03s (host 측정) | OK |
  | 2b, TP1, batch 1, 20 layer | (~6.3s 추정) | timeout |
  | 2b, TP1, batch 2, 15 layer | (~9s 추정) | timeout |
  | 2b, TP8, batch 2, 30 layer | 4.12s | OK |
  | 5b-I2V, TP8, batch 2, 21 layer | 5.03–5.16s | OK |
  | 5b-I2V, TP8, batch 2, 21 layer, 10-14 런타임 (fp32, bf16 둘 다) | 5.7–6.1s | OK |
  | 5b-I2V, TP8, batch 2, 42 layer | (~10s 추정) | timeout |
  | 5b-I2V, TP16, batch 2, 42 layer | 측정 안 함 | timeout |
  | 5b-I2V, TP8, batch 2, 42 layer, bf16, `timeout_s=0` | E2E step당 약 11.5s | OK |

- **사용자 결정 (2026-10-14): 이 timeout은 사용자가 직접 해결한다. transformer를 여러 그래프로 나누는 우회
  (`layers_per_graph`)는 받아들이지 않는다 — 코드에서 삭제했다 (`612ea376`). 다시 넣지 않는다.**
- 이전 설명 "device당 CmdStream이 ~0.19 GiB를 넘으면 hang"은 틀렸다 (42 layer TP8은 CmdStream 0.121 GiB에서도 실패).

---

## 2. 환경

| 항목 | 값 |
|---|---|
| 서버 | NPU 32개 (`rbln-stat`). `RBLN_DEVICES=...`로 물리 device를 고른다 (rbln_config의 device 번호는 이 목록의 index) |
| venv | `~/venv_optimum` (uv/.venv 아님). `source ~/venv_optimum/bin/activate` |
| optimum-rbln | `~/optimum-rbln` editable 설치 |
| rebel-compiler | `~/rebel_compiler` (`python/rebel`, 런타임 C++ 소스 포함). 2026-10-14 13시 기준 `0.12.1.dev102+g29213901`. 10-13 측정은 대부분 `dev99+g18b188f8` |
| 버전 | transformers 5.15.1, diffusers 0.38.0, torch 2.11.0+cpu, peft 0.19.1, accelerate 1.13.0 |
| HF 캐시 | `export HF_HOME=/mnt/shared_data/groups/sw_dev/.cache/huggingface` (팀 공유. 지우거나 정리하지 않는다. 2b, 5b-I2V 가중치 있음) |
| NAS | `~/nas_data` = `/mnt/shared_data/users/seinpark/nas_data` |

- 다른 서버에서 `peft<=0.17.x` + transformers 5.x이면 `cannot import name 'HybridCache' from 'transformers'`로
  `optimum.rbln` import가 실패한다 (peft가 모듈 최상단에서 `HybridCache`를 import). `peft>=0.18.0`으로 올리면 된다.
- `tests/test_diffusers.py` 수집에 `cosmos-guardrail`이 필요하다. 2026-10-14 확인: 이 venv에는 `cosmos-guardrail 0.3.1`이 있고
  `pytest-xdist`는 없다. 그래서 `-n 1`을 빼고 실행한다 (`-n 1`을 주면 `unrecognized arguments: -n`). venv는 고치지 않았다.

---

## 3. 코드 구조

새 파일/클래스만 추가했다. 공유 코드 (`modeling.py`, `modeling_base.py`, `configuration_utils.py`, `modeling_diffusers.py`,
`vae.py`의 기존 클래스)는 바꾸지 않았다.

| 파일 | 내용 |
|---|---|
| `src/optimum/rbln/diffusers/models/autoencoders/autoencoder_kl_cogvideox.py` | `RBLNAutoencoderKLCogVideoX`. 그래프 `decoder_0`, `decoder_n`, (I2V) `encoder`. conv cache 표, cache 레이아웃 |
| `src/optimum/rbln/diffusers/models/autoencoders/vae.py` | 새 런타임 클래스 `RBLNRuntimeCogVideoXVAEDecoder` (chunk 루프), `RBLNRuntimeCogVideoXVAEEncoder` (1 frame) |
| `src/optimum/rbln/diffusers/configurations/models/configuration_autoencoder_kl_cogvideox.py` | VAE config. `num_devices > 1`이면 ValueError. `cache_dtype` 기본값 = VAE dtype |
| `src/optimum/rbln/diffusers/models/transformers/transformer_cogvideox.py` | `RBLNCogVideoXTransformer3DModel`. 한 그래프, timestep·RoPE를 그래프 입력으로 |
| `src/optimum/rbln/diffusers/configurations/models/configuration_transformer_cogvideox.py` | transformer config. `num_devices` 기본 8, 8 미만이면 ValueError (`MIN_NUM_DEVICES`) |
| `src/optimum/rbln/diffusers/pipelines/cogvideo/pipeline_cogvideox.py` | `RBLNCogVideoXPipeline` |
| `src/optimum/rbln/diffusers/pipelines/cogvideo/pipeline_cogvideox_image2video.py` | `RBLNCogVideoXImageToVideoPipeline` |
| `src/optimum/rbln/diffusers/configurations/pipelines/configuration_cogvideox.py` | `RBLNCogVideoXPipelineConfig` (`_vae_uses_encoder=False`), `RBLNCogVideoXImageToVideoPipelineConfig` (`True`) |
| `tests/test_diffusers.py` | `_save_tiny_cogvideox_pipeline(save_dir, image_to_video)`, `TestCogVideoXModel`, `TestCogVideoXImageToVideoModel` |

export는 `define_import_structure`가 `__all__`에서 자동으로 만든다. `from optimum.rbln import RBLNCogVideoXImageToVideoPipeline` 동작.

### VAE (`RBLNAutoencoderKLCogVideoX`)
- Wan VAE 패턴. 프레임 루프는 그래프 밖 (런타임 클래스).
- **chunk 분할은 diffusers `AutoencoderKLCogVideoX._decode`와 같아야 한다**: 첫 chunk = `min(n, 2 + n % 2)` latent frame,
  이후 2 frame씩 (49 frames → latent 13 → 3 + 2×5). GroupNorm과 시간축 upsample이 chunk에 의존해서, 다르게 나누면 결과가 깨진다.
  - `decoder_0`: 첫 chunk, cache 없이 시작. `decoder_n`: 이후 chunk, 이전 cache를 받는다.
- conv cache 표는 모듈 트리에서 만든다 (`get_cache_convs`). 축소 config에도 동작.
- cache는 static buffer + `torch.ops.rbln_custom_ops.rbln_cache_update`. **쓰기는 그래프 끝에서 한 번에** (아래 이슈 3).
- cache dtype (`cache_dtype`) 기본값은 VAE dtype. device는 16-bit로만 저장하므로 fp32로 선언해도 DRAM은 bf16과 같다
  (2b VAE fp32 그래프에서 fp32 cache / bf16 cache: peak 둘 다 12.982 GiB, 출력 frame md5 동일, §6).
- `get_compiled_model`은 **`decoder_n`을 먼저 컴파일**한다 (`decoder_n`, `decoder_0`, `encoder` 순, 한
  `CompileContext(use_weight_sharing=True)`). 그래야 나머지가 `decoder_n`의 intermediate 버퍼를 재사용해 한 device에 들어간다.
- cache 레이아웃 (`_cache_layout(c, h, w)`):
  - `flat` `(n, c, d, h*w)` — `h*w`가 64의 배수일 때.
  - `chlast` `(n, h*w, d, c)` — 아니고 `c`가 64의 배수일 때. 쓰기 `x.permute(0,3,4,2,1).reshape(n,h*w,d,c)`,
    읽기 `cache.reshape(n,h,w,d,c).permute(0,4,3,1,2)` (그래프 안의 실제 transpose).
  - 둘 다 아니면 `NotImplementedError`.
  - 480x720에서 60x90, 120x180 cache (h*w = 5400, 21600)가 `chlast`.
- encoder (I2V, `uses_encoder=True`): `_VAECogVideoXEncoder`가 `encoder(x, conv_cache=None)` + `quant_conv`. 입력 1 frame
  `[B, 3, 1, H, W]`. `encode()`는 `AutoencoderKLOutput(latent_dist=DiagonalGaussianDistribution(h))`. sampling은 host
  (파이프라인의 `retrieve_latents`가 generator로 sample).
- `__post_init__`은 `self.model[i]`를 index로 고른다 (`create_runtimes=False`면 `self.model`이 `UnavailableRuntime`이라 길이 0).

### Transformer (`RBLNCogVideoXTransformer3DModel`)
- HF forward 전체를 한 그래프로 (`CogVideoXTransformer3DModelWrapper`). SD3처럼 timestep sinusoid도 그래프 안.
- 그래프 입력: `hidden_states [B, F_lat, C_in, h, w]`, `encoder_hidden_states [B, 226, 4096]`, `timestep [B]` (float32),
  rotary 모델이면 추가로 `image_rotary_emb_cos`, `image_rotary_emb_sin` `[F_lat*(h/2)*(w/2), head_dim]` (float32).
  RoPE 표는 파이프라인이 host에서 만든다 (`_prepare_rotary_positional_embeddings`).
- CFG 때문에 batch는 2 (파이프라인 config가 transformer `batch_size`를 2배로).
- `cache_context`는 no-op (diffusers 파이프라인이 `self.transformer.cache_context("cond_uncond")`를 부른다).
- CogVideoX 1.5 (`ofs_embedding`, `patch_size_t`)는 `NotImplementedError`.
- `num_devices` (TP) 기본 8, 8 미만 금지.

### 파이프라인
- `encode_prompt`는 upstream 그대로다. 예전의 `RBLNCogVideoXPromptDtypeMixin` (embeds를 transformer dtype으로 바꿔
  latent/scheduler를 transformer dtype으로 유지)은 upstream과 맞추려고 지웠다 (사용자 결정, 10-14).
  `torch_dtype` 없이 로드하면 T5는 checkpoint dtype (2b fp16, 5b-I2V bf16), transformer·VAE는 fp32라서 latent/scheduler는
  upstream처럼 T5 dtype으로 돈다. 에러는 없다: RBLN transformer `forward`와 VAE runtime이 입력을 컴파일된 dtype으로 바꾼다
  (10-14 확인: 2b·5b-I2V `torch_dtype` 없는 산출물로 2 step, latent dtype = T5 dtype).
- (예전 설명 "fp32 T5-XXL은 한 device에 안 들어간다"는 틀렸다: 10-14 측정에서 2b T5 fp32 컴파일·실행 OK,
  device peak fp32 8.68 / fp16 8.676 GiB. device가 16-bit로 저장하기 때문.)
- 파이프라인 config가 VAE에 `num_devices=1`을 넘긴다.
- device 배치: text_encoder 1 + transformer 8 (또는 16) + VAE 1.

### dtype
- 각 submodule은 **로드된 dtype으로 컴파일된다** (`modeling_base.py`의 `rbln_config.dtype = model.dtype`).
  `RBLN...Pipeline.from_pretrained(..., export=True, torch_dtype=X)`의 `torch_dtype`은 diffusers
  `DiffusionPipeline.from_pretrained`로 그대로 가서 모든 submodule이 X로 로드된다. 컴파일 뒤에는 dtype이 고정이다.
- `torch_dtype`이 없으면: diffusers 모듈 (transformer, VAE)은 fp32, transformers의 T5는 checkpoint dtype.
  그래서 10-13까지의 산출물은 transformer가 fp32다 (`pipeline_cogvideox_2b_tp8_63acc798`, `pipeline_cogvideox_5b_i2v_tp8/16`).
- 모델 카드 권장 (학습 precision): CogVideoX-2b **FP16**, CogVideoX-5b / 5b-I2V / 1.5-5B-I2V **BF16**.
  checkpoint: 2b T5·transformer F16, 5b-I2V T5·transformer BF16, VAE는 둘 다 F32.
  예제 (`examples/cogvideo/`, `~/nas_data/cogvideox/generate_i2v.py`)는 `--torch-dtype auto` = 이 dtype으로 컴파일한다
  (`cog_utils.MODEL_CARD_DTYPE`). CPU golden은 정확도 기준이라 항상 fp32.
- VAE도 로드된 dtype으로 컴파일된다 (upstream diffusers가 `torch_dtype`대로 VAE를 돌리는 것과 같다). 예전 코드의
  fp32 강제 변환 (`_reconstruct_model_if_needed`)은 지웠다 (사용자 결정, 10-14). fp32 VAE가 필요하면 upstream처럼
  dict로 준다: `torch_dtype={"default": torch.bfloat16, "vae": torch.float32}`. VAE runtime이 입력을 VAE dtype으로 바꾼다
  (`z.to(self.dtype)`, `x.to(self.dtype)`).
- device는 16-bit로만 저장한다 (사용자 설명). 그래서 fp32로 선언한 cache도 DRAM이 bf16과 같고, fp32와 bf16 transformer의
  실행 시간도 같다 (아래). fp32 그래프와 16-bit 그래프의 차이는 host 쪽 입출력 dtype과 가중치를 16-bit로 바꾸는 경로다.
- `torch_dtype` 값 (diffusers 0.38.0): 안 주면 diffusers 모듈 fp32 + T5 checkpoint dtype (섞임). `"auto"`는 파이프라인에서
  지원하지 않는다: `"Passed torch_dtype auto is not a torch.dtype. Defaulting to torch.float32"` 경고 후 전부 fp32
  (`pipeline_utils.py:792`). 전부 fp32여도 RBLN에서 컴파일·실행된다 (T5 fp32도 한 device). dtype 또는 dict만 의미가 있다.
- VAE config의 `force_upcast=True`는 CogVideoX에서 아무 효과가 없다 (diffusers 0.38.0): `AutoencoderKLCogVideoX`는 config에
  저장만 하고, `pipelines/cogvideo/`는 읽지 않는다. 읽는 파이프라인 (SDXL 등)도 fp16일 때만 upcast한다.
- 5b-I2V transformer 21 layer: bf16과 fp32의 device 실행 시간이 같다 (같은 런타임, 같은 device에서 번갈아 측정, 약 6.0s).
  bf16으로 바꿔도 빨라지지 않는다.

---

## 4. 바꾸지 말 것 (사용자 결정과 그 이유)

1. **Transformer는 TP ≥ 8, 한 그래프.** 레이어 분할 (`layers_per_graph`) 금지. timeout은 사용자가 드라이버 쪽에서 해결.
2. **VAE는 device 1개. RSD (`num_devices > 1`) 금지.** config가 막는다. (RSD는 conv 그래프를 나누지도 않아 메모리 이득도 없었다.)
3. **VAE cache 레이아웃은 flat/chlast (transpose)만.** zero-pad 방식은 버렸다 (작은 VAE에서 실행 시
   `The input(feat_cache_1) is not specified.`). 기록만 docs에 있다. 코드 주석에 pad 방식을 남기지 않는다.
4. **timestep은 그래프 안에서** (SD3 방식). host에서 `time_proj`를 하던 이전 방식은 버렸다.
5. **공유 코드를 바꾸지 않는다.** 새 클래스를 추가한다. t2v 동작을 바꾸지 않는다.
6. **PR 범위: 코드, 테스트, 사용자가 요청한 docs와 `examples/cogvideo/` 검증 스크립트.** 그 밖의 것을 넣기 전에 묻는다.
7. **dtype은 upstream을 따른다 (2026-10-14).** 모든 submodule (VAE 포함)은 로드된 `torch_dtype`대로 컴파일한다. VAE를
   fp32로 강제하지 않는다. `cache_dtype` 기본값은 VAE dtype (fp32 예외 없음: device는 16-bit로 저장해서 DRAM 차이가 없다).
   예제는 모델 카드 dtype을 준다. `encode_prompt` 등 파이프라인 동작도 upstream 그대로 둔다 (dtype을 바꾸는 override 금지).

---

## 5. 컴파일러/런타임 이슈 (우회만 함, 컴파일러는 수정하지 않음)

1. `rbln_cache_update`: cache의 마지막 축이 64의 배수가 아니면 컴파일 실패
   (`wrong shape for custom op`, `Partition.cpp verifyCustomOpsDevRunnable`). DRAM 텐서 마지막 축은 128B (bf16 64원소) 단위로
   패딩돼 저장되기 때문. → `chlast` 레이아웃으로 우회.
2. `(n, c, d, h, w)` 활성값을 permute 없이 `(n, h*w, d, c)`로 reshape해서 저장하면 `FoldMemoryTransfer` assert
   ("a memory.store still writes a DRAM-view concatenate"). tile 조각이 버퍼 축의 상자(box)가 되지 않아서. permute하면 된다.
3. static cache의 write-after-read 순서가 보장되지 않는다. conv마다 바로 cache를 쓰면 `decoder_n` 출력이 깨진다.
   → 모든 cache 쓰기를 그래프 끝에. (toy repro에서는 끝 쓰기도 깨진다. 정확성은 실제 그래프 순서에 기대고 있다. 컴파일러 수정 필요.)
4. 그래프 간 intermediate 공유는 먼저 컴파일된 그래프의 버퍼가 충분히 클 때만 된다 (`codegen.cc AllocIntermediateMemory`).
   → `decoder_n` 먼저 컴파일.
5. device lockup timeout (위 블로커).
6. 작은 해상도 VAE (120x120, 240x360)에서 `ScatterInfo size mismatch with shape-based size` assert. 테스트는 480x720을 쓴다.

repro 스크립트: `examples/cogvideo/repro_cache_update_shape.py`, `repro_cache_update_war.py`, `repro_cache_layout.py`.

---

## 6. 검증 결과

지표 규칙 (사용자 요구): Pearson r ≥ 0.99를 주 지표로, max/mean |diff|, relative L2, 기울기 a·절편 b (`y ≈ a*x + b`),
`allclose(atol=2e-2, rtol=1e-2)`를 함께 보고한다. 기준은 CPU fp32 diffusers.

### 2b text-to-video (480x720, 49 frames)
| 대상 | 결과 |
|---|---|
| VAE decode (같은 latent) | r=0.999994, frame별 r ≥ 0.999974, 18.9s (transpose만 남긴 최종 코드), device 12.98 GiB |
| Transformer 30 layer 실제 가중치, TP8, batch 2, 한 그래프 | r=0.999974, 4.12s/step. **timestep sinusoid를 host에서 계산하던 옛 코드로 측정.** 그래프 안으로 옮긴 뒤에는 1 layer만 확인 (t=999/500/1에서 r=0.999976/0.999991/0.999996, `e56eddd7`) |
| E2E 50 step, guidance 6, seed 42, **전체 fp16** (VAE 포함, 모델 카드 dtype, 2026-10-14) | 영상 PSNR 24.06 dB, r=0.959. 최종 latent r=0.9879, channel 평균 제거 r=0.953. 생성 235.5s. 이 산출물의 fp16 VAE로 CPU latent를 decode하면 r=0.999989, 55.63 dB. `20261014_1503_e2e/` |
| E2E 같은 조건, transformer fp16 + VAE fp32 (VAE fp32 강제가 있던 코드) | latent이 위와 비트까지 같다 (VAE만 다름). 영상 PSNR 24.06 dB, r=0.959. 육안으로 같은 장면·움직임, artifact 없음. `20261014_1417_e2e/` |
| E2E 같은 조건, transformer fp32 (`torch_dtype` 없음, 2026-10-14) | 영상 PSNR 24.65 dB, r=0.964. 최종 latent r=0.9885, channel 평균 제거 r=0.955. 생성 237.6s. 육안으로 같은 장면·움직임, artifact 없음. 이 산출물의 RBLN VAE로 CPU latent를 decode하면 r=0.999991, 57.29 dB → 차이는 denoising 누적. `20261014_1330_e2e/` |
| E2E (옛 코드, fp32, 2026-10-13, 참고) | PSNR 24.36 dB, r=0.962, 최종 latent r=0.9881, 생성 206.4s |
| 기준값: seed 43 (다른 영상) vs seed 42 golden | PSNR 12.31 dB, 영상 r=0.364, 최종 latent r=0.831, channel 평균 제거 r=0.329. `20261014_1411_e2e/` (정확도 결과 아님) |

- "최종 latent r" = `pipe(..., output_type="latent")`가 돌려준 50 step 후 latent (decode 전, `(1, 13, 16, 60, 90)`)를
  CPU golden의 같은 텐서와 1차원으로 펴서 잰 Pearson r (`cog_utils.compare`).
- **latent의 raw r은 부풀려진다**: latent channel 16개의 평균이 서로 크게 달라서, channel 평균만 채운 텐서 (공간 정보 없음)도
  CPU latent와 r=0.863이다. latent는 channel별 평균 (frame·h·w 축)을 빼고 잰 r ("channel 평균 제거 r")과 rel_l2로 본다.
  참고값 (channel 평균 제거 r): 같은 CPU 영상의 다음 latent frame (영상 4 frame 뒤) 0.56–0.62, 1 latent pixel (8 px) 이동 0.55,
  다른 seed 0.33. RBLN vs CPU 0.95는 이것들보다 훨씬 가깝다. 영상도 같은 이유로 PSNR을 함께 본다.
- 50 step 누적 차이 (channel 평균 제거 rel_l2 약 0.30)가 sampling 자체의 민감도인지는 증명하지 않았다. 확인 방법 (안 함):
  CPU fp32 golden을 다시 돌리면서 매 step transformer 출력에 RBLN 1 step 오차만큼 (rel_l2 약 0.5%) 교란을 넣고 같은 지표를 본다.
- VAE cache dtype과 DRAM (2b VAE fp32 그래프, 480x720, 49 frames, rebel dev102, 같은 코드로 동시에 측정):

  | | fp32 cache | bf16 cache |
  |---|---|---|
  | `decoder_0` 할당 | 1.362 GiB | 1.362 GiB |
  | `decoder_n` 할당 | 11.618 GiB (DramTensor 2.666, Intermediate 7.248, Input 0.824, Init 0.494, Kernel 0.234, Output 0.119) | 같음 |
  | device peak (`rbln-stat`) | 12.982 GiB | 12.982 GiB |
  | decode / CPU 대비 | 22.89s, r=0.999994 | 22.80s, r=0.999994 |
  | 출력 frame (uint8 PNG 49장) | md5 동일 | md5 동일 |

  `outputs/20261014_1518_vae_f49_fp32_cachefp32/`, `outputs/20261014_1518_vae_f49_fp32_cachebf16/`. 예전 기록의 "fp32 cache OOM
  (DramTensor 5.27 GiB)"은 conv마다 cache를 쓰던 옛 구성의 값이다. 참고 (의미 미확인): 두 산출물 모두 `compiled_model_stats`에
  `feat_cache_21` 이후가 `host_op_reason: WrongShape`로 나온다.
- 생성 시간이 206.4s → 237.6s로 늘었다. 같은 fp32 5b-I2V 21 layer 산출물 (10-13 dev99 컴파일)도 10-13에 5.05s,
  10-14 런타임 (dev102)에서 약 6.0s라서, 런타임 업데이트 쪽으로 보인다. device 번호도 달라서 확정은 아니다.

### 5b-I2V (480x720, 49 frames)
| 대상 | 결과 | 비고 |
|---|---|---|
| VAE encoder (1 frame 이미지) | latent mean r=0.999981, max\|diff\| 0.103, rel_l2 0.0075 | decoder_0 1.36 + decoder_n 11.62 + encoder 0.25 GiB, peak 13.23 GiB, 한 device |
| Transformer 1 layer 랜덤, TP8, batch 2, RoPE | r=0.999990, allclose True | |
| Transformer 21 layer 실제 가중치, TP8, batch 2, **bf16** | r=0.999989, rel_l2 0.0050 (fp32: r=0.999992, rel_l2 0.0045) | `20261014_1434_transformer_L21_real_bfloat16/` |
| Transformer 42 layer 실제 가중치, TP8, batch 2, **bf16**, 한 그래프 | r=0.999990, rel_l2 0.0058, max\|diff\| 0.042. 1회 실행 12.01s | `20261014_1525_transformer_L42_real_bfloat16/`. `timeout_s=0`에서 측정 |
| E2E 42 layer 한 그래프, TP8, **전체 bf16** (VAE 포함, 모델 카드 dtype, 2026-10-14) | PSNR 24.32 dB, 영상 r=0.967, frame 0 48.3 dB. 최종 latent r=0.9911, channel 평균 제거 r=0.968. 생성 559.1s | `20261014_1503_i2v_e2e/`. 육안으로 같은 장면·구도·움직임, artifact 없음 |
| VAE decoder 단독: CPU golden latent decode | bf16 VAE r=0.999984, 55.36 dB / fp32 VAE r=0.999991, 56.27 dB | 각 E2E 폴더의 `vae_on_golden_latents.json` (fp32는 scratch) |
| E2E 같은 조건, transformer bf16 + VAE fp32 (VAE fp32 강제가 있던 코드) | PSNR 23.60 dB, 영상 r=0.961, frame 0 48.5 dB. channel 평균 제거 latent r=0.964. 생성 578.3s | `20261014_1417_i2v_e2e/`. VAE encoder dtype이 달라 image latent부터 다르다 → 위 E2E와 PSNR 차이를 VAE 정확도로 읽지 않는다 |
| Transformer 42 layer 실제 가중치, TP8, batch 2, fp32 | r=0.999992, rel_l2 0.0053 | **삭제된 2 그래프 버전으로 측정 (참고)** |
| E2E fp32, 2 그래프 (2026-10-13, 참고) | PSNR 24.89 dB, r=0.971. frame 0 50.0 dB. channel 평균 제거 latent r=0.971. 생성 507.1s | **2 그래프 버전.** CPU latent를 RBLN VAE로 decode하면 56.51 dB |
| tiny I2V 테스트 | 통과 | |

- 아직 안 한 것: 42 layer 한 그래프 TP16, fp32 한 그래프 E2E (`pipeline_cogvideox_5b_i2v_tp8/16`은 컴파일만 되어 있다).

---

## 7. 산출물 위치

### 컴파일된 모델 `~/nas_data/cogvideox/compiled/`
| 경로 | 현재 코드로 로드 | 내용 |
|---|---|---|
| `pipeline_cogvideox_5b_i2v_bf16_vaebf16_tp8/` | 가능 (확인함, E2E 실행함) | 5b-I2V 전체 파이프라인, **전체 bf16** (VAE·cache 포함, 현재 코드). transformer 42 layer TP8 한 그래프. device 10개. `generate_i2v.py --tp 8` 기본값 |
| `pipeline_cogvideox_5b_i2v_bf16_tp8/` | 가능 (확인함, E2E 실행함) | 같음, 단 VAE fp32 + cache bf16 (VAE fp32 강제가 있던 코드) |
| `pipeline_cogvideox_5b_i2v_tp8/` | 가능 (확인함) | 5b-I2V 전체 파이프라인, T5 bf16 + transformer **fp32**. transformer 42 layer TP8 한 그래프. device 10개 |
| `pipeline_cogvideox_5b_i2v_tp16/` | 가능 (확인함) | 같음, TP16. device 18개. bf16 TP16 폴더는 아직 없다 |
| `pipeline_cogvideox_5b_i2v/` | 불가 | transformer 2 그래프 (삭제된 `layers_per_graph`). text_encoder/VAE는 위 두 폴더와 하드링크 |
| `pipeline_cogvideox_2b_fp16_vaefp16_tp8/` | 가능 (확인함, E2E 실행함) | 2b 전체 파이프라인, **전체 fp16** (VAE·cache 포함, 현재 코드). transformer TP8 한 그래프. device 10개 |
| `pipeline_cogvideox_2b_fp16_tp8/` | 가능 (확인함, E2E 실행함) | 같음, 단 VAE fp32 + cache bf16 (VAE fp32 강제가 있던 코드) |
| `vae_f49_480x720_fp32_cachefp32/`, `vae_f49_480x720_fp32_cachebf16_dev102/` | 가능 | 2b VAE 단독 (fp32 그래프), cache fp32 / bf16 DRAM 비교용 |
| `t5_2b_fp32/`, `t5_2b_fp16/` | 가능 | 2b T5 단독 fp32 / fp16 (device 메모리 비교용, 둘 다 약 8.68 GiB) |
| `pipeline_cogvideox_2b_tp8_63acc798/` | 가능 (확인함) | 2b 전체 파이프라인, `63acc798` 코드, rebel dev102. T5 fp16 + transformer **fp32**. device 10개 |
| `i2v5b_transformer_L21_real_tp8_b2_bf16/`, `i2v5b_transformer_L42_real_tp8_b2_bf16/` | 가능 | 5b-I2V transformer 단독 bf16 (21 / 42 layer) |
| `pipeline_cogvideox_2b/`, `pipeline_cogvideox_2b_v2/`, `pipeline_cogvideox_2b_tp8/`, `transformer_L*` | 불가 | 옛 코드 (그래프 입력 `timestep_proj`) |
| `vae5bi2v_f49_480x720_enc/` | 가능 | 5b-I2V VAE (encoder + decoder) 단독 |

컴파일 조건 (5b-I2V): 480x720, 49 frames, batch 2 (CFG, guidance_scale > 1), 텍스트 226 토큰. 다른 조건이면 다시 컴파일.
컴파일 시간: 파이프라인 전체 약 21분 (TP8 transformer 약 15분), TP16 transformer 약 25분. `export=False` 로드 약 2–5분.
2b 컴파일 조건: 480x720, 49 frames, batch 2 (guidance 6), 226 토큰. 컴파일+저장 769s (TP8 transformer 약 8분), 로드 255s.
10-14 bf16/fp16 컴파일 (다른 컴파일과 동시에 실행): 2b fp16 1100s, 5b-I2V bf16 1506s.

### 영상 생성 스크립트
- `~/nas_data/cogvideox/generate_i2v.py --tp 8|16 --image <png> --prompt "..."` → `~/nas_data/cogvideox/outputs/<ts>_i2v_gen_tp8/`
  (video.mp4, frames/, input_image.png, meta.json). 기본 폴더는 `compiled/pipeline_cogvideox_5b_i2v_bf16_vaebf16_tp<tp>` (10-14 변경,
  이전 판은 `generate_i2v.py.bak_20261014`). TP8은 같은 산출물로 `run_i2v_e2e.py`가 E2E를 돌렸다. 이 스크립트 자체로는 아직 안 돌렸다.

### 검증 스크립트 `examples/cogvideo/` (repo에 있음)
모두 같은 폴더의 `cog_utils.py`를 import한다 (`compare`, `psnr`, `new_out_dir`, `write_meta`). 결과 폴더 위치는
`COGVIDEOX_OUT_ROOT` (기본 `~/nas_data/cogvideox/outputs`). 같은 파일의 복사본이 `~/nas_data/cogvideox/scripts/`에도 있다.
| 스크립트 | 용도 |
|---|---|
| `verify_vae.py` | VAE decoder vs CPU |
| `verify_vae_encoder.py` | VAE encoder (I2V) vs CPU `latent_dist.mean` |
| `verify_transformer.py --model-id <id> --num-layers N --tensor-parallel-size 8 --batch-size 2 --compiled-dir <dir>` | transformer vs CPU fp32. 컴파일 → 저장 → alloc 출력 → 로드 → 비교. 컴파일 dtype `--torch-dtype` (기본 모델 카드 dtype) |
| `run_e2e.py cpu|rbln|compare` | 2b E2E (CPU golden fp32, RBLN, 비교). RBLN 컴파일 dtype `--torch-dtype` (기본 fp16) |
| `run_i2v_e2e.py cpu|rbln|compare` | 5b-I2V E2E. RBLN 컴파일 dtype `--torch-dtype` (기본 bf16) |
| `run_cogvideo.py` | 2b 생성 예제. `--torch-dtype` (기본 float16) |
| `repro_*.py` | 컴파일러 이슈 repro |

### 결과 `~/nas_data/cogvideox/outputs/<YYYYMMDD_HHMM>_<desc>/`
실행마다 새 폴더, `meta.json` 포함, 깨진 결과도 보관. 주요 폴더:
- `20261013_0033_e2e_cpu_golden/` — 2b CPU golden (panda prompt, seed 42)
- `20261013_1207_e2e/` — 2b RBLN E2E (TP8, 옛 코드)
- `20261014_1330_e2e/` — 2b RBLN E2E (TP8, 현재 코드, transformer fp32). `compare_cpu_vs_rbln.png`, `vae_on_golden_latents.json` 포함
- `20261014_1411_e2e/` — 2b seed 43 기준값 (다른 영상)
- `20261014_1417_e2e/` — 2b RBLN E2E, transformer fp16 + VAE fp32
- `20261014_1503_e2e/` — 2b RBLN E2E, 전체 fp16 (모델 카드 dtype, 현재 코드)
- `20261014_1417_i2v_e2e/` — 5b-I2V RBLN E2E, transformer bf16 + VAE fp32, 42 layer 한 그래프 TP8
- `20261014_1503_i2v_e2e/` — 5b-I2V RBLN E2E, 전체 bf16 (모델 카드 dtype, 현재 코드), 42 layer 한 그래프 TP8
- `20261014_1434_transformer_L21_real_bfloat16/`, `20261014_1525_transformer_L42_real_bfloat16/` — 5b-I2V transformer bf16 21 / 42 layer
- `20261014_1518_vae_f49_fp32_cachefp32/`, `20261014_1518_vae_f49_fp32_cachebf16/` — VAE cache dtype DRAM 비교
- `20261013_2158_vae5bi2v_encoder/` — 5b-I2V VAE encoder
- `20261013_2237_i2v5b_transformer_L21_real_tp8_b2/` — 21 layer, 실행 시간 근거
- `20261013_2321_i2v5b_transformer_L42_real_tp8_b2_seg2/` — 42 layer (2 그래프 버전)
- `20261013_2207_i2v_e2e_cpu_golden/`, `20261013_2324_i2v_e2e/` — 5b-I2V E2E

---

## 8. 자주 쓰는 명령

```bash
source ~/venv_optimum/bin/activate
export HF_HOME=/mnt/shared_data/groups/sw_dev/.cache/huggingface
cd ~/optimum-rbln

# 테스트 (device 8개 필요: transformer TP8). 이 venv에는 pytest-xdist가 없어서 -n 1을 쓰지 않는다
RBLN_DEVICES=8,9,10,11,12,13,14,15 python -m pytest tests/test_diffusers.py -k CogVideoX -vv
RBLN_DEVICES=8,9,10,11,12,13,14,15 python -m pytest tests/test_diffusers.py -k "Cosmos or SVD" -vv   # 회귀

# 2b E2E (컴파일된 폴더가 있으면 컴파일을 건너뛰고 로드만 한다)
RBLN_DEVICES=16,17,18,19,20,21,22,23,24,25 python examples/cogvideo/run_e2e.py rbln \
  --golden ~/nas_data/cogvideox/outputs/20261013_0033_e2e_cpu_golden \
  --compiled-dir ~/nas_data/cogvideox/compiled/pipeline_cogvideox_2b_fp16_vaefp16_tp8

# 5b-I2V E2E (bf16). golden은 CPU fp32 (약 3시간, 재사용)
RBLN_DEVICES=16,17,18,19,20,21,22,23,24,25 python examples/cogvideo/run_i2v_e2e.py rbln \
  --golden ~/nas_data/cogvideox/outputs/20261013_2207_i2v_e2e_cpu_golden \
  --compiled-dir ~/nas_data/cogvideox/compiled/pipeline_cogvideox_5b_i2v_bf16_vaebf16_tp8

# 코드 품질
ruff format --check . && ruff check . && mypy && python scripts/check_inits.py
python .github/scripts/validate_docstrings.py <file>   # 한 번에 파일 하나
```
- 작업 트리의 `ruff check .`는 이 작업과 무관한 untracked 스크래치 파일 (`examples/image-to-text/*`, `qwen_test.py`) 때문에 실패한다.
  사용자 파일이므로 건드리지 않는다. 커밋 기준으로 확인: `git archive HEAD | tar -x -C <tmp>` 후 그 안에서 ruff 실행.
- 긴 컴파일·실행은 백그라운드로 돌리고 로그를 `~/nas_data/cogvideox/logs/`에 남겼다.

---

## 9. 다음 할 일

1. 5b-I2V 한 그래프 측정 (`timeout_s=0`이 된 뒤 TP8 bf16은 E2E와 21/42 layer 검증 완료). 남은 것: TP16 bf16
   (컴파일 필요), 필요하면 fp32 한 그래프 E2E (`pipeline_cogvideox_5b_i2v_tp8`). PR #804 본문의 5b-I2V 결과 표와
   "two 21-block graphs" 문단을 bf16 한 그래프 값으로 바꾼다 (본문 수정은 사용자에게 먼저 묻는다).
2. (선택) 2b transformer 30 layer를 현재 코드·fp16으로 다시 측정 (`verify_transformer.py --model-id THUDM/CogVideoX-2b
   --num-layers 30 --tensor-parallel-size 8 --batch-size 2`). 지금 표의 r=0.999974는 옛 코드 fp32 값이다.
3. (선택) 생성 시간 증가 (2b 206.4s → 237.6s, 21 layer 5.05s → 약 6.0s) 원인 확인. 런타임 dev99 → dev102가 후보.
4. (선택) 50 step 누적 차이가 sampling 민감도인지 확인 (§6 2b의 CPU 교란 실험).
5. 컴파일러 팀 보고 후보: 이슈 1–4 (특히 3, 정확성 문제).
6. CogVideoX1.5-5B-I2V는 사용자가 지시할 때만. 위험: video token 약 2.6배 (실행 시간 증가), 768x1360x81 frames에서
   VAE decoder 메모리 약 3배 (추정 약 30 GiB, device 15.7 GiB), 가중치는 공유 캐시에 없다.

---

## 10. 사용자와 일하는 규칙

- 답은 한국어로. repo 문서/커밋은 간결하게 (CLAUDE.md: ASD-STE 100).
- **commit/push 전에 diff를 보여주고 명시적인 OK를 받는다.** 남의 rebase를 해결하거나 줄을 옮기는 등 요청 밖의 변경을 하지 않는다.
- **GitHub에 사용자 이름으로 댓글/리뷰를 달지 않는다.** 문구 초안만 준다.
- 원격에 새 커밋이 있으면 그 위에 쌓는다. 원격 커밋을 되돌리거나 덮어쓰지 않는다.
- 줄임말·변형 이름을 쓸 때는 처음에 정확히 정의한다 (코드와 shape까지). 예: "chlast = `(n, h*w, d, c)`, 쓰기 시 permute".
- 추정과 측정을 구분해서 말한다. 확인하지 않은 원인을 사실처럼 말하지 않는다.
- 사용자 결정 (TP ≥ 8, VAE 단일 device, 그래프 분할 금지 등)을 바꾸는 우회가 필요하면 먼저 묻는다.
- 공유 HF 캐시는 읽기만. 가중치를 다시 받거나 지우지 않는다.
- 결과 폴더는 덮어쓰지 않는다. 깨진 결과도 보관한다.
