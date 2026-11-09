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
