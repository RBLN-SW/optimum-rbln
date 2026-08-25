"""Serve Hub loads from the local cache instead of revalidating them.

`from_pretrained` sends one HEAD request per file even on a warm cache, against
an account shared with the other CI pipelines. Tests therefore run with the Hub
switched off; a test keeps it on when it is marked `@pytest.mark.hub` or when a
repo it declares is not cached yet.
"""

import warnings

from huggingface_hub import constants, snapshot_download
from huggingface_hub.errors import LocalEntryNotFoundError


def pytest_configure(config):
    config.addinivalue_line("markers", "hub: test needs the Hub online")


def pytest_runtest_setup(item):
    # A hook, not a fixture: this has to run before setUpClass, where models are loaded.
    constants.HF_HUB_OFFLINE = item.get_closest_marker("hub") is None and all(
        is_cached(repo_id, revision) for repo_id, revision in declared_repos(item)
    )


def declared_repos(item):
    cls = getattr(item, "cls", None)
    revision = (getattr(cls, "HF_CONFIG_KWARGS", None) or {}).get("revision")
    repos = [(getattr(cls, "HF_MODEL_ID", None), revision), (getattr(cls, "CONTROLNET_ID", None), None)]
    return [(repo_id, rev) for repo_id, rev in repos if repo_id]


def is_cached(repo_id, revision):
    try:
        snapshot_download(repo_id, revision=revision, local_files_only=True)
    except LocalEntryNotFoundError:
        warnings.warn(f"{repo_id} is not in the HF cache; it will be downloaded from the Hub", stacklevel=2)
        return False
    return True
