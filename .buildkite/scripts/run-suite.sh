#!/usr/bin/env bash
# run-suite.sh <suite> [group]
# Runs one suite of the GHA matrix, honoring the same [skip-*] commit-message
# directives. BUILDKITE_MESSAGE is the PR head commit message, as in GHA.
set -euo pipefail

suite="${1:?usage: run-suite.sh <suite> [group]}"
group="${2:-}"

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
  group=""
fi

echo "--- :pytest: ${suite}${group:+ (group ${group}/4)}"
case "$suite" in
  config)
    uv run --no-sync pytest -n 1 tests/test_config.py -vv --durations 0 "${bc[@]}" ;;
  transformers)
    uv run --no-sync pytest -n 1 tests/test_transformers.py -vv --durations 0 "${bc[@]}" ;;
  diffusers)
    uv run --no-sync pytest -n 1 tests/test_diffusers.py -vv --durations 0 "${bc[@]}" ;;
  llm)
    if [ -n "$group" ]; then
      uv run --no-sync pytest -n 1 tests/test_llm.py --splits 4 --group "$group" -vv --durations 0
    else
      [ ${#bc[@]} -gt 0 ] || { echo "llm needs a group (1-4)" >&2; exit 2; }
      uv run --no-sync pytest -n 1 tests/test_llm.py -vv --durations 0 "${bc[@]}"
    fi ;;
  cli-basic)
    uv run --no-sync .github/scripts/test_cli.py basic ;;
  cli-argument-parsing)
    uv run --no-sync .github/scripts/test_cli.py argument-parsing ;;
  cli-error-handling)
    uv run --no-sync .github/scripts/test_cli.py error-handling ;;
  *)
    echo "unknown suite: $suite" >&2; exit 2 ;;
esac
