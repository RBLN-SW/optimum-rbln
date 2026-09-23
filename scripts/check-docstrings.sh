#!/usr/bin/env bash
# Mirrors .github/workflows/test-docstrings.yml: check that mkdocstrings can parse
# the src/*.py files this PR changes. DOCSTRINGS_ALL_FILES=1 checks every file.
set -euo pipefail

base="${BUILDKITE_PULL_REQUEST_BASE_BRANCH:-dev}"
if [ "${DOCSTRINGS_ALL_FILES:-}" = "1" ]; then
  files=$(find src -name '*.py' -type f | sort)
else
  git fetch -q origin "$base"
  files=$(git diff --name-only "origin/${base}...HEAD" | grep '\.py$' | grep '^src/' || true)
fi

# modeling_t5.py links to ../../../index.md, which the throwaway mkdocs project
# cannot resolve; drop this once the link is absolute.
exclude=(
  src/optimum/rbln/transformers/models/t5/modeling_t5.py
)
selected=()
for f in $files; do
  for e in "${exclude[@]}"; do
    if [ "$f" = "$e" ]; then echo "skip excluded: $f"; continue 2; fi
  done
  [ -f "$f" ] && selected+=("$f")
done
if [ ${#selected[@]} -eq 0 ]; then
  echo "No Python files to test"
  exit 0
fi

echo "--- :package: mkdocs env"
uv venv -q --python 3.12 /tmp/docstrings-venv
uv pip install -q --python /tmp/docstrings-venv mkdocs mkdocs-material mkdocstrings mkdocstrings-python
export PATH="/tmp/docstrings-venv/bin:$PATH"

echo "--- :page_facing_up: ${#selected[@]} files"
printf '%s\n' "${selected[@]}" | xargs -P 16 -n 1 sh -c \
  'python scripts/validate_docstrings.py "$0" && echo "ok $0" || { echo "FAILED $0"; exit 1; }'
echo "All docstring tests passed"
