#!/usr/bin/env bash
# bc-steps.sh {--latest|--all}
#
# Emits the backward-compatibility steps for `buildkite-agent pipeline upload`.
# Each release under BC_BASE_PATH holds the artifacts that release compiled
# (written by the tag build, .buildkite/bc-compile.yml); a step reloads them with
# the current code.
#
# The load uses a dummy device, so no NPU is involved -- but creating the runtime
# still dlopens librbln-thunk.so, and a CPU pod has no UMD of its own (build #29
# failed every model with "Failed to load the RBLN Thunk library"), hence UMD_PATH.
# The llm step asks for far more memory than the others, matching the 128GB runner
# it replaces.
#
# --latest is what a PR runs (the newest release only), --all is the nightly.
set -euo pipefail

mode="${1:?usage: bc-steps.sh --latest|--all}"
: "${BC_BASE_PATH:?not set}"
[ -d "$BC_BASE_PATH" ] || { echo "BC_BASE_PATH not found: $BC_BASE_PATH" >&2; exit 1; }

# Directory names encode the tag with underscores (v0_11_2); sort -V on the
# decoded form picks the newest release the same way the GHA workflow does.
dirs=$(find "$BC_BASE_PATH" -maxdepth 1 -mindepth 1 -type d -not -name '.*' -printf '%f\n' | tr '_' '.' | sort -V)
[ -n "$dirs" ] || { echo "no release directories under $BC_BASE_PATH" >&2; exit 1; }
case "$mode" in
  --latest) tags=$(printf '%s\n' "$dirs" | tail -1) ;;
  --all)    tags="$dirs" ;;
  *)        echo "unknown mode: $mode" >&2; exit 2 ;;
esac

cat <<'EOF'
notify:
  - github_commit_status:
      context: "[OB] Optimum-RBLN BC"
EOF
echo "steps:"
for tag in $tags; do
  encoded="${tag//./_}"
  for suite in transformers diffusers llm; do
    # llm answers to [skip-llms], the others to their own name.
    case "$suite" in llm) skip="skip-llms" ;; *) skip="skip-$suite" ;; esac
    if [ "$suite" = llm ]; then memory="128Gi"; else memory="32Gi"; fi
    cat <<EOF
  - label: ":rewind: BC $tag $suite"
    key: "bc-${encoded}-${suite}"
    if: build.message !~ /\[${skip}\]/
    image: "\${DEVTOOLS_DOCKER_IMAGE}"
    resources:
      cpu:
        requests: "4"
        limits: "4"
      memory:
        requests: "$memory"
        limits: "$memory"
    secrets:
      UV_INDEX_REBELLIONS_USERNAME: REBEL_SW_DEV_USERNAME
      UV_INDEX_REBELLIONS_PASSWORD: REBEL_SW_DEV_PASSWORD
      HF_TOKEN: HF_TOKEN
      HF_HOME: HF_HOME
    env:
      LD_LIBRARY_PATH: "\${UMD_PATH}"
      OPTIMUM_RBLN_TEST_LEVEL: "full"
      REUSE_ARTIFACTS_PATH: "$BC_BASE_PATH/$encoded"
    timeout_in_minutes: 60
    artifact_paths: "junit-*.xml"
    command:
      - "bash .buildkite/scripts/sync.sh"
      - "bash .buildkite/scripts/run-suite.sh $suite"
EOF
  done
done
