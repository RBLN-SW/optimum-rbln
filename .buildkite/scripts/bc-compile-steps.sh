#!/usr/bin/env bash
# bc-compile-steps.sh [tag]
#
# Emits the steps that compile one release's artifacts into BC_BASE_PATH, which
# bc-steps.sh reloads on every later PR and nightly. BC_TAG names the tag on a
# build that has none, which is how a release gets backfilled.
#
# The UI filter passes every v*, so the release policy lives here: a pre-release
# emits nothing and the build passes empty.
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
  # RBLN_FORCE_NPU_NAME fixes the target SoC without hardware: CA22, as every
  # release under BC_BASE_PATH was built on. UMD as in bc-steps.sh -- the dummy
  # runtime setUpClass creates still dlopens the thunk.
  #
  # The build's commit is the tag only when a tag triggered it. .buildkite is the
  # one path not taken from the tag: older releases carry none.
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
      # mkdir -p alone would happily create it on the wrong volume.
      - "test -d $BC_BASE_PATH && mkdir -p $BC_BASE_PATH/$encoded"
      - "git fetch -q origin refs/tags/$tag && git checkout -q --detach FETCH_HEAD"
      - "git checkout -q \$\$BUILDKITE_COMMIT -- .buildkite"
      - "bash .buildkite/scripts/sync.sh"
      - "bash .buildkite/scripts/run-suite.sh $suite"
EOF
done
