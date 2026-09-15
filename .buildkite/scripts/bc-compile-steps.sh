#!/usr/bin/env bash
# bc-compile-steps.sh [tag]
#
# Emits the steps that compile one release's artifacts into BC_BASE_PATH, for
# `buildkite-agent pipeline upload`. bc-steps.sh reloads those artifacts with the
# then-current code on every later PR and nightly.
#
# The UI filter lets every v* tag through, so the release policy lives here: a
# pre-release emits nothing and the build passes empty. Same regex the GHA
# check-tag job held, kept where it can be read and run (bash bc-compile-steps.sh
# v0.11.3rc1).
#
# No compiler version to pass: the checkout is the tag, so sync.sh installs the
# compiler that release pinned in .github/version.yaml.
set -euo pipefail

tag="${1:-${BUILDKITE_TAG:-}}"
[ -n "$tag" ] || { echo "no tag: pass one or set BUILDKITE_TAG" >&2; exit 1; }
: "${BC_BASE_PATH:?not set}"

# Accept X.Y.Z[.more][.postN|postN]; reject rc, dev and anything else.
version="${tag#v}"
if ! [[ "$version" =~ ^[0-9]+(\.[0-9]+)*(\.post[0-9]+|post[0-9]+)?$ ]]; then
  echo "$tag is not a stable release; no artifacts to compile" >&2
  exit 0
fi

# Directory names encode the tag with underscores, as bc-steps.sh decodes them.
encoded="${tag//./_}"

cat <<'EOF'
notify:
  - github_commit_status:
      context: "[OB] Optimum-RBLN BC compile"
EOF
echo "steps:"
for suite in transformers diffusers llm; do
  # The attached NPU fixes the artifacts' target SoC. CA22, as the GHA ca22-1
  # runner produced -- every release already under BC_BASE_PATH is CA22.
  cat <<EOF
  - label: ":floppy_disk: compile $tag $suite"
    key: "bc-compile-${suite}"
    image: "\${DEVTOOLS_DOCKER_IMAGE}"
    resources:
      npu:
        count: 1
        product: "RBLN-CA22"
    secrets:
      UV_INDEX_REBELLIONS_USERNAME: REBEL_SW_DEV_USERNAME
      UV_INDEX_REBELLIONS_PASSWORD: REBEL_SW_DEV_PASSWORD
      HF_TOKEN: HF_TOKEN
      HF_HOME: HF_HOME
    env:
      OPTIMUM_RBLN_TEST_LEVEL: "full"
      SAVE_ARTIFACTS_PATH: "$BC_BASE_PATH/$encoded"
    timeout_in_minutes: 180
    command:
      - "bash .buildkite/scripts/sync.sh"
      - "bash .buildkite/scripts/run-suite.sh $suite"
EOF
done
