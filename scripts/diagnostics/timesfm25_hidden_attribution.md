# Final hidden/head attribution probe

This B1 probe retains the original point and spread heads and adds the final transformer hidden `[1,1,1280]` as a third output. All 231,289,280 learned parameters remain in the NPU core. It does not use the split-head treatment, replace native forecasts, or clear the failed original CI794/split-head CI797 acceptance gates.

Four compressed fixtures preserve the exact four input tensors from SHA-verified CI794 traces: positive/negative falling step0, negative h1024 step0, and teacher-forced negative h1024 step7. Each `.pt.gz` is below 500 KiB; the manifest records its SHA256, original trace SHA256, source build/job/path and each tensor's raw-byte hash. They also retain the original NPU raw outputs and CPU raw references as comparison evidence. These are diagnostic input snapshots, not new native goldens or runtime substitute outputs. Existing six native fixtures remain unchanged.

For each input the probe records actual NPU point/spread/hidden, an original compiled two-output baseline, CPU point/spread/hidden on identical inputs, and CPU head results evaluated on the identical NPU hidden. The CPU learned evaluations are oracle controls only. Actual point/spread outputs always come directly from the full NPU core. Tensor evidence is serialized before metrics.

For either head, let `P` be the actual NPU head output, `H` its NPU hidden, and `F` the FP32 CPU head. With CPU reference hidden `H0`, the output-error vector decomposes as:

`P - F(H0) = [P - F(H)] + [F(H) - F(H0)]`.

The first component measures head evaluation on an identical hidden input, including weight representation and arithmetic differences; the second measures propagation of trunk-state differences through the CPU reference head. This does not identify a particular compiler pass, primitive, normalization or attention defect. Component RMS magnitudes do not add; signed dot-product fractions describe projection onto the total error and are not independent variance percentages. Per-field and whole-head comparisons are retained for both heads.

Exposing hidden can change compilation/fusion. Therefore the probe **requires a same-device original compiled baseline**, stores both original raw outputs, and checks bitwise equality with the instrumented first two outputs on all four inputs. Any change is preserved and makes the final diagnostic assertion fail as an instrumentation confound, even if metrics improve. It also checks input immutability and error-vector recomposition. Raw CPU comparison metrics are descriptive and have no acceptance flag; `product_accepted` and `native_acceptance_exercised` remain false.

CPU proof from this worktree:

```sh
/home/jaehunryu/model-enable/20261023-chronos2/compiler-venv/bin/python -m scripts.diagnostics.timesfm25_hidden_attribution --cpu-only --output /home/jaehunryu/model-enable/20261024-timesfm25/hidden-attribution-cpu
```

It strictly exports the full original-max graph with all parameters, tests the same exported graph on all four saved inputs, requires both original raw outputs to remain bitwise equal, and requires same-hidden CPU heads and eager/exported outputs to match exactly. Saved CI794 CPU references must remain within the existing tight CPU tolerance. This mode does not compile or create an NPU runtime.

Pending owned-CI entrypoints, **not executed locally**:

```sh
uv run --no-sync python -m scripts.diagnostics.timesfm25_hidden_attribution --npu RBLN-CR13 --device 0 --baseline-artifact /path/to/original-core-b1.rbln
uv run --no-sync python -m scripts.diagnostics.timesfm25_hidden_attribution --npu RBLN-CA25 --device 0 --compile-baseline
```

The reused artifact must match the target NPU and original two-output B1 shape. `--compile-baseline` explicitly creates an original same-device control instead; it is mutually exclusive with artifact reuse. SDK329 remains pinned. The fixture inputs and actual model execution are fixed B1; this probe adds no B2 coverage. Artifacts go to `debug-timesfm25-hidden/**/*`. No pipeline modification or CI dispatch is included; publication waits for the owning agent's review of actual CI797.

The implementation reuses the original attributed native learned-core modules and inspects the unchanged Transformers 5.15.1 point/quantile residual blocks. It adds a returned view of the final hidden and oracle measurements only. The root TimesFM worktree/CI797 head is untouched; this worktree is isolated from commit `0aa95674`.
