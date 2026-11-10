#!/usr/bin/env bash
# usage: process_dev_sty015.sh <repo-root>   (read-only scan of CI files for STY-015)
cd "$1"; F=".buildkite .github scripts/ci"
echo "--IPv4";            grep -rnE '\b[0-9]{1,3}(\.[0-9]{1,3}){3}\b' $F
echo "--URLs";            grep -rnoE 'https?://[^ "'"'"')]+' $F
echo "--internal hosts";  grep -rnE 'rebellions\.in|\.svc\b|cluster\.local|k8s\.' $F
echo "--abs mount paths"; grep -rnE '(^|[ "=:])/(mnt|nfs|data|shared|home)/' $F
echo "--GHA secrets refs";   grep -rhoE '\$\{\{ *secrets\.[A-Za-z_]+' .github | wc -l
echo "--BK secret mappings"; { grep -rhE '^\s+(- GIT_PAT|[A-Z_]+: (BUILDKITE_API_TOKEN|SLACK_OAUTH_TOKEN|REBEL_SW_DEV_[A-Z]+|HF_TOKEN|HF_HOME|MODEL_HUB_ACCESS))$' .buildkite scripts/ci; } | wc -l
echo "--env-injected image/cluster vars"; grep -rhoE '\$\{(DEVTOOLS_DOCKER_IMAGE|OPTIMUM_BC_BASE_PATH|UV_CACHE_PATH|UMD_PATH)\}' .buildkite scripts/ci | wc -l
echo "--PyPI endpoint via var"; grep -rho 'REBEL_PYPI_INTERNAL_ENDPOINT' $F | wc -l
echo "--required-var guards"; grep -rnE ': "\$\{[A-Z_]+:\?not set\}"' scripts/ci
