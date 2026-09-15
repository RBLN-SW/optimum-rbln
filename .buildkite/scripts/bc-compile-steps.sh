#!/usr/bin/env bash
# bc-compile-steps.sh [tag]
#
# Emits the steps that compile one release's artifacts into BC_BASE_PATH, which
# bc-steps.sh reloads on every later PR and nightly.
#
# The tag comes from the build. BC_TAG names it on a build that has none, which
# is how a release gets backfilled -- the workflow_dispatch bc_compile.yaml took.
# Either way the steps take the release's own tree from the tag, so the artifacts
# are what that release compiled, with the compiler it pinned.
#
# The UI filter passes every v*, so the release policy lives here: a pre-release
# emits nothing and the build passes empty.
#
# REBEL_COMPILER_VERSION on the build overrides the compiler the tag pinned, and
# then names it in every label -- artifacts compiled with anything but the pin are
# not what that release shipped.
set -euo pipefail

tag="${1:-${BUILDKITE_TAG:-${BC_TAG:-}}}"
[ -n "$tag" ] || { echo "no tag: set BC_TAG on a build that is not a tag build" >&2; exit 1; }
: "${BC_BASE_PATH:?not set}"

version="${tag#v}"
if ! [[ "$version" =~ ^[0-9]+(\.[0-9]+)*(\.post[0-9]+|post[0-9]+)?$ ]]; then
  echo "$tag is not a stable release; no artifacts to compile" >&2
  exit 0
fi

# bc-steps.sh decodes these back.
encoded="${tag//./_}"
override="${REBEL_COMPILER_VERSION:-}"

cat <<'EOF'
notify:
  - github_commit_status:
      context: "[OB] Optimum-RBLN BC compile"
EOF
echo "steps:"
for suite in transformers diffusers llm; do
  # RBLN_FORCE_NPU_NAME fixes the target SoC without hardware, so the compile
  # holds no NPU. CA22, as every release already under BC_BASE_PATH was built on.
  # setUpClass still creates a dummy runtime, which dlopens librbln-thunk.so, so
  # the pod needs the shared UMD the BC steps already borrow.
  #
  # The build's commit is the tag only when a tag triggered it, so check the tag
  # out explicitly and put .buildkite back: releases older than the migration
  # carry none, and the scripts running the job have to come from somewhere that
  # does. Same split the GHA job had -- workflow from dev, ref: <tag> for code.
  cat <<EOF
  - label: ":floppy_disk: compile $tag $suite${override:+ @ $override}"
    key: "bc-compile-${suite}"
    image: "\${DEVTOOLS_DOCKER_IMAGE}"
    secrets:
      UV_INDEX_REBELLIONS_USERNAME: REBEL_SW_DEV_USERNAME
      UV_INDEX_REBELLIONS_PASSWORD: REBEL_SW_DEV_PASSWORD
      HF_TOKEN: HF_TOKEN
      HF_HOME: HF_HOME
    env:
      LD_LIBRARY_PATH: "/mnt/shared_data/cross-volume/umd:/mnt/shared_data/umd:/mnt/cross_data/umd"
      RBLN_FORCE_NPU_NAME: "RBLN-CA22"
      OPTIMUM_RBLN_TEST_LEVEL: "full"
      SAVE_ARTIFACTS_PATH: "$BC_BASE_PATH/$encoded"
    timeout_in_minutes: 180
    command:
      # Fail in seconds, not after a three-hour compile, if this pod's
      # /mnt/shared_data is not the volume holding the other releases.
      - "test -d $BC_BASE_PATH && mkdir -p $BC_BASE_PATH/$encoded"
      - "git fetch -q origin refs/tags/$tag && git checkout -q --detach FETCH_HEAD"
      - "git checkout -q \$\$BUILDKITE_COMMIT -- .buildkite"
      - "bash .buildkite/scripts/sync.sh"
      - "bash .buildkite/scripts/run-suite.sh $suite"
EOF
done
