"""Whole classes, dealt round-robin: a class compiles its model in setUpClass, and
the shard has to stay in collection order."""

import os
from unittest.mock import patch

import pytest


def pytest_addoption(parser):
    parser.addoption("--shard", default="", metavar="INDEX/COUNT", help="run one of COUNT shards of the test classes")


def pytest_configure(config):
    # BC artifacts must be real: a fake run would save (or fail to read) JSON stubs.
    bc = [name for name in ("SAVE_ARTIFACTS_PATH", "REUSE_ARTIFACTS_PATH") if os.environ.get(name)]
    if bc and os.environ.get("OPTIMUM_RBLN_REAL_COMPILE") != "1":
        raise pytest.UsageError(f"{', '.join(bc)} needs OPTIMUM_RBLN_REAL_COMPILE=1.")


@pytest.hookimpl(trylast=True)
def pytest_collection_modifyitems(config, items):
    # Apply -m/-k selection first so compiled-models shards contain only the
    # selected model classes, rather than gaps left by the fake-only classes.
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


@pytest.fixture(scope="class", autouse=True)
def _fake_compile(request):
    """Default to API-only execution; marked classes always compile, including setUpClass."""
    from .fake_rbln import fake_rbln

    if request.node.get_closest_marker("requires_compile") is not None:
        with patch.dict(os.environ, {"OPTIMUM_RBLN_REAL_COMPILE": "1"}):
            yield
    else:
        with fake_rbln():
            yield
