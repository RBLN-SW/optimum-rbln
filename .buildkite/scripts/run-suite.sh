#!/usr/bin/env bash
# run-suite.sh <suite> [shard]
# Runs one suite of the GHA matrix, honoring the same [skip-*] commit-message
# directives. BUILDKITE_MESSAGE is the PR head commit message, as in GHA.
set -euo pipefail

suite="${1:?usage: run-suite.sh <suite> [shard]}"
shard="${2:-}"   # index/count, e.g. 3/6

case "$suite" in
  transformers) tag="[skip-transformers]" ;;
  diffusers)    tag="[skip-diffusers]" ;;
  llm)          tag="[skip-llms]" ;;
  cli-*)        tag="[skip-cli]" ;;
  *)            tag="" ;;
esac
if [ -n "$tag" ] && [[ "${BUILDKITE_MESSAGE:-}" == *"$tag"* ]]; then
  echo "Found $tag in commit message, skipping $suite"
  exit 0
fi

# A backward-compatibility run loads an older release's artifacts on a dummy
# device, so only test_generate applies and the suite is small enough not to split.
bc=()
if [ -n "${REUSE_ARTIFACTS_PATH:-}" ]; then
  bc=(-k test_generate)
  shard=""
fi

echo "--- :pytest: ${suite}${shard:+ (shard ${shard})}"
case "$suite" in
  config)
    uv run --no-sync pytest -n 1 tests/test_config.py -vv --durations 0 "${bc[@]}" ;;
  transformers)
    uv run --no-sync pytest -n 1 tests/test_transformers.py ${shard:+--shard "$shard"} -vv --durations 0 "${bc[@]}" ;;
  diffusers)
    uv run --no-sync pytest -n 1 tests/test_diffusers.py ${shard:+--shard "$shard"} -vv --durations 0 "${bc[@]}" ;;
  llm)
    uv run --no-sync pytest -n 1 tests/test_llm.py ${shard:+--shard "$shard"} -vv --durations 0 "${bc[@]}" ;;
  cli-basic)
    uv run --no-sync .github/scripts/test_cli.py basic ;;
  cli-argument-parsing)
    uv run --no-sync .github/scripts/test_cli.py argument-parsing ;;
  cli-error-handling)
    uv run --no-sync .github/scripts/test_cli.py error-handling ;;
  *)
    echo "unknown suite: $suite" >&2; exit 2 ;;
esac
