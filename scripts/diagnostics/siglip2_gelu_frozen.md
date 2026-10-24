# Frozen native SigLIP2 GELU A/B

This isolated diagnostic executes the full native B1 checkpoint at text length
64 and patch budget 256. The original four image geometries and texts come from
`TestSiglip2Model.test_numerical_parity`. A native HF CPU model supplies validation
outputs; learned product inference remains in the frozen compiled model.

The artifact manifest pins source compiler `4b1449d1`, source Optimum `e861627a`,
the frontend file and mapped C++ library SHA256, and all six files per variant.
The runner additionally pins both compiled binary checksums and sizes. BEFORE
uses the historical erf conversion, AFTER honors requested tanh GELU. Both have
three device functions. Their host constant-table gathers, masks, rsqrt and
transfers are the existing full-model partition; `host_op_reason` labels alone
cannot establish learned CPU offload. The source build's embedded compiler
version is stale, so source identity comes from the verified manifest.

The scheduler pipeline `.buildkite/siglip2-gelu-frozen.yml` runs the existing
quality and docstring gates, then allocates one CA25 on the k8s queue with 16 CPUs
and 64 GiB memory. It installs the product-pinned SDK329 runtime and **does not
compile**. Successful results establish only frozen source4b B1 A/B behavior,
not SDK330/package acceptance or other batch sizes.

CPU-only checksum/interface inspection (SDK329 environment, no runtime created):

```sh
python -m scripts.diagnostics.siglip2_gelu_frozen \
  --artifacts /mnt/shared_data/users/jaehunryu/nas_data/model-enable-20261024/siglip2-gelu-ab-source4b \
  --verify-only
```

The scheduler-owned invocation supplies `--device 0`. Before evaluation, every
case saves inputs, CPU expected outputs, actual outputs and exact repeats for all
eight native fields. Both initial and fresh runtime loads execute all four cases,
then return to the first input. Reloaded outputs must exactly equal initial
outputs. Timing records contain cold/warm host wall times and load times; they
are diagnostic observations, not a performance claim.

All non-scalar outputs require PCC >= 0.99; every output requires NRMSE <= 0.05
and finite values. Both embedding outputs additionally require minimum cosine
>= 0.99. Baseline numerical failures remain in results. Baseline completion,
shape, finite, exact repeat, return and reload checks still apply. AFTER must
pass every criterion. Final failure occurs only after both variants complete,
with raw tensors and JSON results preserved even when numerical criteria fail.

Local CPU/mock checks:

```sh
python -m pytest -q tests/unit/cpu/test_siglip2_gelu_frozen.py
```
