# Native512 MPNet shared-position-bias scheduler proof

This isolated diagnostic starts at product commit
`e5acc2960ec48eb3bcf122ad53428ff878a775f2` and copies only the reviewed
11-line architecture change and 25-line CPU regression from
`shared-position-bias`. The active full B2048 compile and CI804 baseline remain
unchanged. No additional product implementation or learned CPU route is added.

The diagnostic pipeline preserves the original team-member, quality and docstring
gates, then requests one UDC CR13, CPU 16 and 64 GiB memory. It overrides the
compiler to published `0.12.1.dev163+g0c6d4a5e.prod`; the driver verifies that
exact version, wheel Python import and one mapped library with SHA256
`ff6af17e0ff975ac22100e92da58f74bb61ba06999c94aceb4c829c6654f96ed`
(verified against the installed wheel RECORD).

The driver collects only these original classes:

- `TestMPNetModelNativeMaxBatch1`
- `TestMPNetModelNativeMaxBatch2`
- `TestMPNetModelNativeMaxBatch4`

Each retains all nine original methods, for 27 methods and no skipped methods.
The original checkpoint/revision, tokenization, row padding masks, native512
input shapes, pooled-cosine/PCC thresholds, exact repeats, return-to-first and
save/load tests are unchanged. The original text rotation gives B2/B4 distinct
row masks, exercising the singleton learned-position-bias broadcast against
actual row-dependent masks. B1 is the control.

The only diagnostic lifecycle hook extends each class teardown: save the
candidate, release the original runtimes, load genuinely new compiled-model
objects and invoke that same original `test_embedding_parity` method again.
These three additional oracle invocations are explicitly separate from the 27
original collected methods. No numerical assertion is reimplemented or relaxed.
The original teardown still removes each class's artifacts and runtimes.

Runtime execution requires `BUILDKITE_BUILD_ID`, rejects dummy devices, and
rejects SAVE_ARTIFACTS_PATH/REUSE_ARTIFACTS_PATH even when empty. Source hashes
pin the reviewed architecture and both original acceptance-test files. The
pipeline clears source-library overrides and dummy mode, but does not clear
artifact-mode misconfiguration. Per-class/reload events and original parity
metrics go to the diagnostic artifacts.

CPU-only collection/source verification, without model compilation or runtimes:

```sh
RBLN_DUMMY_DEVICE=1 PYTHONPATH=src python \
  -m scripts.diagnostics.mpnet_shared_bias_native512 \
  --out verify-mpnet-native512 --verify-only
```

The scheduler command is defined in
`.buildkite/pr.yml` on this diagnostic branch.
A successful B1/B2/B4 matrix establishes this candidate's small native512
correctness. It does not establish B2048 capacity, allocation or performance.
