"""Hooks shared by the whole suite.

Sharding deals whole classes round-robin: a class compiles its model in setUpClass, and
the shard has to stay in collection order.

Hub loads are served from the local cache instead of revalidated: `from_pretrained` sends
one HEAD request per file even on a warm cache, against an account shared with the other CI
pipelines. Tests therefore run with the Hub switched off; a test keeps it on when it is
marked `@pytest.mark.hub` or when a repo it declares is not cached yet.
"""

import os
import warnings

import pytest
from huggingface_hub import constants, snapshot_download
from huggingface_hub.errors import LocalEntryNotFoundError


def pytest_addoption(parser):
    parser.addoption("--shard", default="", metavar="INDEX/COUNT", help="run one of COUNT shards of the test classes")


def pytest_configure(config):
    config.addinivalue_line("markers", "hub: test needs the Hub online")


def pytest_collection_modifyitems(config, items):
    if not config.getoption("--shard"):
        return
    index, count = (int(n) for n in config.getoption("--shard").split("/"))
    classes = list(dict.fromkeys(item.cls for item in items))
    keep = set(classes[index - 1 :: count])
    items[:] = [item for item in items if item.cls in keep]


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


@pytest.fixture(scope="session", autouse=True)
def _artifacts_only():
    """A save run compiles and copies; a runtime would only need a device.

    Held for the session rather than per compile: setUpClass is overridden all over
    tests/, and several of those compile without going through the base class.
    """
    if not os.environ.get("SAVE_ARTIFACTS_PATH"):
        yield
        return

    from optimum.rbln.configuration_utils import ContextRblnConfig

    with ContextRblnConfig(create_runtimes=False):
        yield
