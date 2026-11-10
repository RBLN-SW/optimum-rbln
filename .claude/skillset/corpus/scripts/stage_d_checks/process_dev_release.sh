#!/usr/bin/env bash
# usage: process_dev_release.sh <repo-root>   (read-only: how releases happen once main is gone)
R=RBLN-SW/optimum-rbln; cd "$1"
gh api repos/$R --jq '{default_branch,allow_squash_merge,allow_merge_commit,allow_rebase_merge}'
gh api repos/$R/branches/main --jq .name 2>&1 | tail -1
git ls-remote --heads https://github.com/$R | grep -cE . ; git ls-remote --heads https://github.com/$R | grep -E 'refs/heads/(main|release-v0\.1[12].*)$'
for id in $(gh api repos/$R/rulesets --jq '.[].id'); do gh api repos/$R/rulesets/$id --jq '{name,target,include:.conditions.ref_name.include,rules:[.rules[].type]}'; done
for t in $(gh release list -R $R --limit 12 --json tagName --jq '.[].tagName'); do gh api repos/$R/releases/tags/$t --jq '"\(.tag_name) target=\(.target_commitish) prerelease=\(.prerelease) \(.published_at[:10])"'; done
gh pr list -R $R --state merged --search "base:release-v0.12.0" --json number,title,headRefName,mergedAt
for n in 786 789 805; do sha=$(gh pr view $n -R $R --json mergeCommit --jq .mergeCommit.oid); echo "$n parents=$(gh api repos/$R/commits/$sha --jq '.parents|length')"; done
grep -nE 'tags:|v\*|ref_name' .github/workflows/deploy-on-tag.yaml; grep -nE 'BUILDKITE_TAG|stable release' scripts/ci/bc-compile-steps.sh
grep -nE 'main' .github/pull_request_template.md
