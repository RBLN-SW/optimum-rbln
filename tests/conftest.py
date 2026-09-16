"""Whole classes, dealt round-robin: a class compiles its model in setUpClass, and
the shard has to stay in collection order."""

import os

import pytest


def pytest_addoption(parser):
    parser.addoption("--shard", default="", metavar="INDEX/COUNT", help="run one of COUNT shards of the test classes")


def pytest_collection_modifyitems(config, items):
    if not config.getoption("--shard"):
        return
    index, count = (int(n) for n in config.getoption("--shard").split("/"))
    classes = list(dict.fromkeys(item.cls for item in items))
    keep = set(classes[index - 1 :: count])
    items[:] = [item for item in items if item.cls in keep]


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
