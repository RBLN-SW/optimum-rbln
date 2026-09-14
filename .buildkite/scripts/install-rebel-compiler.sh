#!/usr/bin/env bash
# Install the rebel-compiler pinned in .github/version.yaml into the synced venv.
# Same file the GHA workflows read, so the two CIs test the same compiler.
#
# REBEL_COMPILER_VERSION overrides the pin. The compiler's dev-branch CI sets it
# when it triggers this pipeline, so its downstream gate tests the wheel that
# build just produced instead of silently re-testing the pinned one.
set -euo pipefail

: "${REBEL_PYPI_INTERNAL_ENDPOINT:?not set}"
: "${UV_INDEX_REBELLIONS_USERNAME:?not set}"
: "${UV_INDEX_REBELLIONS_PASSWORD:?not set}"

version="${REBEL_COMPILER_VERSION:-}"
if [ -z "$version" ]; then
  version=$(grep '^rebel_compiler_version:' .github/version.yaml | cut -d ':' -f2 | tr -d ' ')
  [ -n "$version" ] || { echo "rebel_compiler_version not found in .github/version.yaml" >&2; exit 1; }
fi

host="${REBEL_PYPI_INTERNAL_ENDPOINT%/}"
index="https://${UV_INDEX_REBELLIONS_USERNAME}:${UV_INDEX_REBELLIONS_PASSWORD}@${host#https://}/simple"

echo "--- :package: rebel-compiler==${version}"
uv pip install --extra-index-url "$index" "rebel-compiler==${version}"

# `uv pip` resolves $VIRTUAL_ENV before the project's .venv and the devtools
# image sets one, so check the environment the suites actually run in instead of
# trusting the install. Silently landing elsewhere would test the pinned
# compiler and still report green -- how rebel_compiler #13336 went unnoticed.
uv run --no-sync python - "${version}" <<'PY'
import importlib.metadata as md
import sys

from packaging.version import Version

want = sys.argv[1]
got = md.version("rebel-compiler")
if Version(got) != Version(want):
    sys.exit(f"rebel-compiler is {got} in the test environment, expected {want}")
print(f"verified rebel-compiler=={got}")
PY

if [ -n "${REBEL_COMPILER_VERSION:-}" ]; then
  buildkite-agent annotate --style success --context rebel-compiler \
    "rebel-compiler overridden to \`${version}\`" || true
fi
