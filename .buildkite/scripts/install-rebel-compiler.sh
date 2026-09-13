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

if [ -n "${REBEL_COMPILER_VERSION:-}" ]; then
  buildkite-agent annotate --style success --context rebel-compiler \
    "rebel-compiler overridden to \`${version}\`" || true
fi
