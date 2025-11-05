# Copyright 2025 Rebellions Inc. All rights reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import re
import sys

from github import Auth, Github


def read_checklist_from_template():
    template_path = ".github/pull_request_template.md"
    checklist_items = []

    with open(template_path, "r") as file:
        content = file.read()
        # Find the checklist section
        checklist_section = re.search(r"## Checklist\n(.*?)\n\n", content, re.DOTALL)
        if checklist_section:
            checklist = checklist_section.group(1)
            # Extract individual checklist items
            checklist_items = re.findall(r"- \[ \] (.*)", checklist)

    return checklist_items


def is_release_pr(pr):
    """Check if this is a release PR (dev -> main)"""
    return pr.base.ref == "main" and pr.head.ref == "dev"


def validate_checklist(body, expected_items):
    for item in expected_items:
        if f"- [x] {item}" not in body:
            print(f"item : {item}")
            return False
    return True


def main():
    github_token = os.getenv("GITHUB_TOKEN")
    pr_number = os.getenv("PR_NUMBER")
    repo_name = os.getenv("GITHUB_REPOSITORY")

    if not all([github_token, pr_number, repo_name]):
        print("Missing required environment variables")
        sys.exit(1)

    g = Github(auth=Auth.Token(github_token))
    repo = g.get_repo(repo_name)
    pr = repo.get_pull(int(pr_number))

    # Skip checklist validation for release PRs (dev -> main)
    if is_release_pr(pr):
        print("This is a release PR (dev -> main). Skipping checklist validation.")
        print("Release PRs follow a different approval process.")
        sys.exit(0)

    expected_items = read_checklist_from_template()

    if not expected_items:
        print("No checklist items found in the PR template.")
        sys.exit(1)

    if validate_checklist(pr.body, expected_items):
        print("All checklist items are marked. PR is valid.")
        sys.exit(0)
    else:
        print(f"expected items : {expected_items}")
        print("Not all checklist items are marked. PR is invalid.")
        sys.exit(1)


if __name__ == "__main__":
    main()
