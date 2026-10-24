#!/usr/bin/env bash
# Run the unchanged failing classifier in a fresh scheduler container.
set -euo pipefail
: "${BUILDKITE_BUILD_ID:?This diagnostic requires the scheduler}"
if [ -n "${SAVE_ARTIFACTS_PATH:-}${REUSE_ARTIFACTS_PATH:-}" ]; then
  echo "Real numerical execution is required." >&2
  exit 2
fi
mkdir -p debug-deberta-classifier
snapshot() {
  for name in memory.current memory.peak memory.max memory.events; do
    if [ -r "/sys/fs/cgroup/$name" ]; then
      cat "/sys/fs/cgroup/$name" > "debug-deberta-classifier/$1-$name"
    fi
  done
}
snapshot before
trap 'snapshot after' EXIT
uv run --no-sync python -c "import importlib.metadata as m; assert m.version('rebel-compiler') == '0.12.1.dev249+gdef44cb7.prod'"
uv run --no-sync pytest -n 1 \
  tests/test_deberta_capacity.py::TestDebertaV3ForSequenceClassificationBatch64CR13::test_numerical_parity \
  --junitxml junit-deberta-classifier-b64.xml \
  -o junit_logging=all --capture=tee-sys -rP -vv --durations 0
