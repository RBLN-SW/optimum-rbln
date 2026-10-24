# Frozen native Chronos-2 B512 CA25 validation

This scheduler-only diagnostic loads six CPU-compiled stages from the staged
CommandStreamGen build. It preserves context 8192, horizon 1024, all 512 series,
all twelve encoder blocks and the full 577x1x512x512 joint group mask. Embedding,
encoders and both forecast projections execute on the NPU. Host work is the
original native preprocessing, explicit intermediate transfers and the original
FP32 projection/residual sum and inverse normalization.

The driver inherits the original `TestChronos2ModelBatch512` methods unchanged:
`test_automap`, `test_generate`, `test_get_rbln_config_class`,
`test_infer_framework`, `test_model_save_dir`, `test_model_save_dir_load`,
`test_numerical_parity`, `test_save_artifacts`, and `test_save_load`. It requires
nine methods run and exactly the original automap skip. The original five
parity scenarios, NRMSE <= 0.05/PCC >= 0.99, exact repeats and return-to-first remain
unchanged. An additional invocation of that same original numerical-parity
method runs after save/fresh reload with genuinely new compiled-model objects.
The original goldens are unchanged.

The sole runtime adaptation is diagnostic `_create_runtimes`: each stage's
loader uses the compiled-model object passed by that load invocation. It never
reopens the original artifact source path. Every request cold-loads and releases
one stage before loading the next. Runtime-owned outputs are cloned to CPU
before release. The existing product wrapper still performs its original FP32
forecast tuple sum. This uses sequential loading; an all-resident configuration
is not claimed to fit.

Export/compile are blocked. Before any runtime, the harness validates the pinned
manifest SHA, every stage checksum/size, reviewed source and single-library
compile provenance, full native stage interfaces, the reported device-function
and fallback summary, and original frozen input identity. The separate `executor-audit/audit.md` traces every actual serialized
node and host constant: only padding and nonlearned mask preparation stay on
the host; all learned stages own device Kernel data. It then checks
actual SDK inspection against the frozen metadata. Runtime execution requires
a scheduler `BUILDKITE_BUILD_ID`, unset SAVE/REUSE artifact modes, published
SDK329 and an explicit expected SHA256 for its sole mapped librbln.so.
The source artifacts use source4b/primitiveffb and the reviewed logging-only
staged DSO; SDK329 is a runtime ABI control, not a claim that these artifacts
came from that wheel.

Use one scheduler-owned CA25 resource, CPU 16, memory 64 GiB. The parent owns NAS
publication, immutable manifest/runtime-image pins and the scheduler launch.
Neither command below authorizes a local NPU run.

CPU-only artifact validation, after all six stages finish:

```sh
PYTHONPATH=src python -m scripts.diagnostics.chronos2_frozen_validation \
  --artifacts "$CHRONOS_FROZEN_ROOT" --manifest-name manifest-recovered.json \
  --manifest-sha256 "$CHRONOS_MANIFEST_SHA256" \
  --out verify-frozen --verify-only
```

Scheduler execution with the platform-pinned SDK329 library SHA:

```sh
PYTHONPATH=src python -m scripts.diagnostics.chronos2_frozen_validation \
  --artifacts "$CHRONOS_FROZEN_ROOT" --manifest-name manifest-recovered.json \
  --manifest-sha256 "$CHRONOS_MANIFEST_SHA256" \
  --runtime-library-sha256 "$SDK329_LIBRBLN_SHA256" --out frozen-ca25-results
```

Do not set SAVE_ARTIFACTS_PATH or REUSE_ARTIFACTS_PATH: those modes skip native
parity. Events record the nine-method result/skip count, fresh-reload object
identities and parity, and per-stage cold load/inference+host-copy/unload host
wall times with static allocation estimates. It refuses SAVE/REUSE variables
even when empty. These are host timings, not NPU query-profiler measurements or a matched performance comparison. Any original
suite, fresh-reload parity, checksum or geometry failure makes the job fail.
No successful scheduler/NPU acceptance is asserted by local mock tests.

The source run preserves its post-save UTF-8 log-counter failure in
`manifest-original-failed.json`. `manifest-recovered.json` records successful
checksummed forecast artifact recovery without recompilation, and leaves its
lost exact compile timer null. The recovered manifest SHA is
`0eeb40e41cae94010206d8426646e8347bf76d07873663eee5df7cd66bae93c5`.
Static all-resident allocation is 42,836,426,752 bytes; the sequential-stage peak
is 9,225,371,648 bytes, plus runtime overhead and host intermediate buffers.
