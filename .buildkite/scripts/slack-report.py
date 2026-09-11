#!/usr/bin/env python3
"""Post this build's suite results to Slack.

Replaces .github/workflows/slack_report.yaml: the job states come from the
Obedients REST API rather than the GitHub Actions one, and the versions are read
from the checkout instead of being passed in as workflow inputs.
"""

from __future__ import annotations

import json
import os
import re
import sys
import urllib.request
from pathlib import Path


API = "https://obedients-api.k8s.rebellions.in/v2/organizations"


def api_get(path: str, token: str) -> dict:
    req = urllib.request.Request(f"{API}/{path}", headers={"Authorization": f"Bearer {token}"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read().decode())


def pinned_version(path: str, pattern: str) -> str:
    try:
        match = re.search(pattern, Path(path).read_text())
    except OSError:
        return "unknown"
    return match.group(1) if match else "unknown"


def summarize(jobs: list[dict], prefix: str) -> tuple[str, int, int]:
    """A one-line verdict for the jobs whose label starts with prefix."""
    picked = [j for j in jobs if (j.get("label") or "").startswith(prefix)]
    if not picked:
        return "⏭️ Not run", 0, 0
    failed = [j for j in picked if j.get("state") in ("failed", "broken", "canceled")]
    ran = [j for j in picked if j.get("state") != "skipped"]
    if not ran:
        return "⏭️ Skipped", 0, 0
    if failed:
        names = ", ".join(sorted((j["label"].split(": ", 1)[-1]) for j in failed))
        return f"❌ {len(failed)}/{len(ran)} failed - `{names}`", len(failed), len(ran)
    return f"✅ {len(ran)}/{len(ran)} passed", 0, len(ran)


def main() -> int:
    token = os.environ.get("OBEDIENTS_API_TOKEN")
    slack_token = os.environ.get("SLACK_BOT_TOKEN")
    # A scheduled run is the real nightly; anything else is someone trying it out.
    scheduled = os.environ.get("BUILDKITE_SOURCE") == "schedule"
    channel = os.environ.get("SLACK_CI_REPORTER_CHANNEL" if scheduled else "SLACK_CI_PLAYGROUND_CHANNEL")
    if not (token and slack_token and channel):
        print("OBEDIENTS_API_TOKEN, SLACK_BOT_TOKEN or the channel is unset", file=sys.stderr)
        return 1

    org = os.environ["BUILDKITE_ORGANIZATION_SLUG"]
    pipeline = os.environ["BUILDKITE_PIPELINE_SLUG"]
    number = os.environ["BUILDKITE_BUILD_NUMBER"]
    build = api_get(f"{org}/pipelines/{pipeline}/builds/{number}", token)

    # This step is still running, and its own state would always read as such.
    me = os.environ.get("BUILDKITE_JOB_ID")
    jobs = [j for j in build.get("jobs", []) if j.get("id") != me]

    pytest_status, pytest_failed, _ = summarize(jobs, ":pytest:")
    bc_status, bc_failed, _ = summarize(jobs, ":rewind:")
    gates = [j for j in jobs if (j.get("label") or "").startswith((":mag:", ":page_facing_up:"))]
    gates_failed = [j for j in gates if j.get("state") in ("failed", "broken")]

    title_base = os.environ.get("SLACK_TITLE", "Optimum-RBLN Nightly")
    if gates_failed:
        title = f"❌ {title_base} - code quality"
    elif pytest_failed:
        title = f"❌ {title_base}"
    elif bc_failed:
        title = f"❌ {title_base} - BC"
    else:
        title = f"✅ {title_base}"

    commit = build.get("commit", "")[:8]
    repo = os.environ.get("BUILDKITE_REPO", "").removesuffix(".git").split("github.com")[-1].lstrip(":/")
    fields = [
        ("*Commit*", f"<https://github.com/{repo}/commit/{commit}|{commit}>"),
        ("*Build*", f"<{build.get('web_url', '')}|View Details>"),
        ("*Compiler*", "`" + pinned_version(".github/version.yaml", r"rebel_compiler_version:\s*(\S+)") + "`"),
        ("*Transformers*", "`" + pinned_version("pyproject.toml", r'"transformers[<>=]{2}([^",]+)') + "`"),
        ("*Diffusers*", "`" + pinned_version("pyproject.toml", r'"diffusers[<>=]{2}([^",]+)') + "`"),
    ]
    blocks = [{"type": "header", "text": {"type": "plain_text", "text": title}}]
    blocks += [
        {"type": "section", "fields": [{"type": "mrkdwn", "text": k}, {"type": "mrkdwn", "text": v}]}
        for k, v in fields
    ]
    blocks += [
        {"type": "divider"},
        {
            "type": "section",
            "fields": [{"type": "mrkdwn", "text": "*Pytest*"}, {"type": "mrkdwn", "text": pytest_status}],
        },
        {"type": "section", "fields": [{"type": "mrkdwn", "text": "*BC*"}, {"type": "mrkdwn", "text": bc_status}]},
    ]

    print(f"{title}\n  pytest: {pytest_status}\n  BC: {bc_status}")
    payload = json.dumps({"channel": channel, "text": title, "blocks": blocks}).encode()
    req = urllib.request.Request(
        "https://slack.com/api/chat.postMessage",
        data=payload,
        headers={
            "Authorization": f"Bearer {slack_token}",
            "Content-Type": "application/json; charset=utf-8",
        },
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        body = json.loads(resp.read().decode())
    # Slack answers 200 even when it refuses the message.
    if not body.get("ok"):
        print(f"Slack refused the message: {body.get('error')}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
