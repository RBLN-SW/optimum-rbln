# Split-head precision control

The learned core returns four FP32-interface tensors: point output/residual `[B,128,10]`, then quantile output/residual `[B,1024,10]`. `SplitHeadAdapter` performs only the two elementwise FP32 additions before calling the unchanged native normalization, autoregressive feedback, flip invariance, quantile spread adjustment and per-series positivity path. All 231,289,280 learned parameters, including both branches of each head and all 20 transformer layers, remain in the NPU core. Input tensors retain the original maximum 512-patch signature. There is no learned CPU offload in the actual adapter.

This tests whether exposing the residual branches and moving their final sum to FP32 improves the existing numerical failures. It does not establish that final-add cancellation caused CI794. Changing graph outputs can also change compiler fusion or tiling. Same-input raw branch/merged comparisons and an original compiled-core control must be examined before making a causal claim. No acceptance threshold, golden, constant case or h1024 case is removed or changed; the original diagnostics remain intact and CI794 remains failed.

CPU proof (run from this repository):

```sh
/home/jaehunryu/model-enable/20261023-chronos2/compiler-venv/bin/python -m scripts.diagnostics.timesfm25_split_heads --cpu-only --output /home/jaehunryu/model-enable/20261024-timesfm25/split-heads-cpu
```

The CPU mode preserves each of all six saved fixtures' original batch sizes, including B2 mixed/falling fixtures. It requires full forecast bitwise equality between original merged heads and split heads plus FP32 addition, along with the unchanged tight native CPU control. A strict original-max B1 Torch export checks all four raw outputs, verifies exact eager/export equality and merged/original equality, and retains every learned parameter. A real-weight CPU stand-in exercises the same raw recording/adapter code, including the original two-output baseline comparison. Its four raw branches and merged/baseline outputs must agree exactly. CPU results are proof of algebra/export/orchestration only.

Pending owned CI examples:

```sh
uv run --no-sync python -m scripts.diagnostics.timesfm25_split_heads --batches 1 --npu RBLN-CR13 --device 0 --baseline-artifact /path/to/original-CI794-core-b1.rbln
uv run --no-sync python -m scripts.diagnostics.timesfm25_split_heads --batches 1 --npu RBLN-CA25 --device 0 --compile-baseline
```

The original artifact must match the selected NPU. A CR13 artifact cannot control a CA25 result; `--compile-baseline` explicitly compiles the original two-output core on the same hardware instead. The options are mutually exclusive. The compiler remains pinned to SDK329. The new script defaults to B1 and accepts explicit larger static batches; no larger-batch support is inferred from CPU results. Pipeline artifact capture must include `debug-timesfm25-split-heads/**/*`; pipeline publication/dispatch is left to the owning agent.

For every first-run forecast call and sign/AR step, raw artifacts preserve exact inputs, four actual NPU branches, four CPU oracle branches on the same inputs, FP32 merged outputs, and original compiled raw outputs when available. Metrics are computed only after serialization. CPU oracle evaluation is diagnostic only. `call00..call07` are unflipped h1024 steps; `call08..call15` are flipped steps. h128 uses one call per sign. The report includes branch metrics, merged per-field metrics, original-baseline raw and native comparisons, cold/warm invocation times, repeated forecasts, return-to-first and saved-model reload equality. Repeats/reload use the NPU adapter without extra CPU oracle tracing. Artifacts retain the original baseline's failures as controls while the treatment uses exactly the same native numerical gates. This diagnostic always reports `product_accepted: false`; actual NPU evidence is pending.

The branch expression and dtype cast were checked against `TimesFm2_5ResidualBlock.forward` in Transformers 5.15.1 (Apache-2.0): `output_layer(activation(input_layer(x)))` and `residual_layer(x)`, with input cast to the learned input-layer weight dtype. Running-statistics and forecasting utilities are reused directly from the existing attributed native diagnostic.
