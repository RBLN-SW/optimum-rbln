#!/usr/bin/env bash
# run-suite.sh <suite> [shard]
# Runs one suite. The [skip-*] directives are guards on the steps that call this.
set -euo pipefail

suite="${1:?usage: run-suite.sh <suite> [shard]}"
shard="${2:-}"   # index/count, e.g. 3/6

# A backward-compatibility run is one release's artifacts, compiled at its tag
# (SAVE) or reloaded later (REUSE): one test carries it, and no shard.
bc=()
if [ -n "${REUSE_ARTIFACTS_PATH:-}" ]; then
  bc=(-k test_generate)
  shard=""
elif [ -n "${SAVE_ARTIFACTS_PATH:-}" ]; then
  bc=(-k test_save_artifacts)
  shard=""
fi

# Named after the step, so the report can tell two BC releases apart after it
# downloads every junit file into one directory, and labelled inside so it can
# group what failed by the step that ran it.
label="${BUILDKITE_LABEL:-$suite}"
report=(--junitxml "junit-$(printf '%s' "$label" | tr -cs '[:alnum:]' '-').xml" -o junit_suite_name="$label")

echo "--- :pytest: ${suite}${shard:+ (shard ${shard})}"
case "$suite" in
  unit-cpu)
    uv run --no-sync pytest tests/unit/cpu "${report[@]}" -vv --durations 0 ;;
  config)
    uv run --no-sync pytest -n 1 tests/test_config.py "${report[@]}" -vv --durations 0 "${bc[@]}" ;;
  transformers)
    uv run --no-sync pytest -n 1 tests/test_transformers.py "${report[@]}" ${shard:+--shard "$shard"} -vv --durations 0 "${bc[@]}" ;;
  electra-capacity-cr13)
    if [ -n "${SAVE_ARTIFACTS_PATH:-}${REUSE_ARTIFACTS_PATH:-}" ]; then
      echo "electra-capacity-cr13 requires real execution, not artifact compatibility mode." >&2
      exit 2
    fi
    batch="${CAPACITY_BATCH_SIZE:-32}"
    case "$batch" in 32|64|128|256|512|1024|2048) ;; *) echo "Unsupported capacity batch: $batch" >&2; exit 2 ;; esac
    export RBLN_TEST_NPU=RBLN-CR13
    status=0
    # Separate processes release each head's runtimes before the next head. Each
    # original test still retains its first model during its exact reload check.
    for head in TestElectraModel TestElectraForPreTraining TestElectraForSequenceClassification; do
      test_class="${head}Batch${batch}CR13"
      uv run --no-sync pytest -n 1 tests/test_electra_capacity.py -k "$test_class and test_numerical_parity" \
        --junitxml "junit-${suite}-${test_class}.xml" -o junit_suite_name="${label}-${test_class}" \
        -o junit_logging=all --capture=tee-sys -rP -vv --durations 0 || status=1
    done
    exit "$status" ;;
  diffusers)
    uv run --no-sync pytest -n 1 tests/test_diffusers.py "${report[@]}" ${shard:+--shard "$shard"} -vv --durations 0 "${bc[@]}" ;;
  llm)
    uv run --no-sync pytest -n 1 tests/test_llm.py "${report[@]}" ${shard:+--shard "$shard"} -vv --durations 0 "${bc[@]}" ;;
  cli-basic)
    uv run --no-sync scripts/test_cli.py basic ;;
  cli-argument-parsing)
    uv run --no-sync scripts/test_cli.py argument-parsing ;;
  cli-error-handling)
    uv run --no-sync scripts/test_cli.py error-handling ;;
  *)
    echo "unknown suite: $suite" >&2; exit 2 ;;
esac
