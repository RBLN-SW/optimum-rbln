#!/usr/bin/env bash
# Install the rebel-compiler pinned in .github/version.yaml into the synced venv.
# Same file the GHA workflows read, so the two CIs test the same compiler.
set -euo pipefail

: "${REBEL_PYPI_INTERNAL_ENDPOINT:?not set}"
: "${UV_INDEX_REBELLIONS_USERNAME:?not set}"
: "${UV_INDEX_REBELLIONS_PASSWORD:?not set}"

version=$(grep '^rebel_compiler_version:' .github/version.yaml | cut -d ':' -f2 | tr -d ' ')
[ -n "$version" ] || { echo "rebel_compiler_version not found in .github/version.yaml" >&2; exit 1; }

host="${REBEL_PYPI_INTERNAL_ENDPOINT%/}"
index="https://${UV_INDEX_REBELLIONS_USERNAME}:${UV_INDEX_REBELLIONS_PASSWORD}@${host#https://}/simple"

echo "--- :package: rebel-compiler==${version}"
uv pip install --extra-index-url "$index" "rebel-compiler==${version}"
