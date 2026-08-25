"""Keep the suite off the account-level Hugging Face Hub rate limit.

`from_pretrained` revalidates even a warm cache with one HEAD request per file,
and this suite loads ~90 models against an account shared with the other CI
pipelines. Tests therefore run with the Hub switched off. A test keeps it on
only when it is marked `@pytest.mark.hub` or when a repo it declares is not in
the cache yet, so adding a model needs no separate warm-up step.

Setting `HF_HUB_OFFLINE` explicitly disables this and leaves the mode alone.
"""

import os
import warnings

from huggingface_hub import constants, snapshot_download
from huggingface_hub.errors import LocalEntryNotFoundError


HF_HUB_OFFLINE_IS_PINNED = "HF_HUB_OFFLINE" in os.environ
CACHED_REPOS = set()


def pytest_configure(config):
    config.addinivalue_line("markers", "hub: test needs the Hub online")


def pytest_runtest_setup(item):
    # Runs before setUpClass, which is where most models are loaded.
    if HF_HUB_OFFLINE_IS_PINNED:
        return
    constants.HF_HUB_OFFLINE = item.get_closest_marker("hub") is None and all(
        is_cached(repo_id, revision) for repo_id, revision in declared_repos(item)
    )


def declared_repos(item):
    """Hub repos the test declares, through class attributes or a `HUB_REPOS`
    tuple on the class or module for repos loaded from inside a test body."""
    repos = set()
    cls = getattr(item, "cls", None)
    if cls is not None:
        revision = (getattr(cls, "HF_CONFIG_KWARGS", None) or {}).get("revision")
        for attr, rev in (("HF_MODEL_ID", revision), ("CONTROLNET_ID", None)):
            repo_id = getattr(cls, attr, None)
            if isinstance(repo_id, str) and "/" in repo_id:
                repos.add((repo_id, rev))
    for owner in (cls, getattr(item, "module", None)):
        for repo_id in getattr(owner, "HUB_REPOS", ()):
            repos.add((repo_id, None))
    return repos


def is_cached(repo_id, revision):
    # Only hits are remembered: a repo downloaded by the first test that needs it
    # serves the rest of the session from the cache.
    if (repo_id, revision) in CACHED_REPOS:
        return True
    try:
        snapshot_download(repo_id, revision=revision, local_files_only=True)
    except LocalEntryNotFoundError:
        warnings.warn(f"{repo_id} is not in the HF cache; tests using it will download it from the Hub", stacklevel=2)
        return False
    CACHED_REPOS.add((repo_id, revision))
    return True
