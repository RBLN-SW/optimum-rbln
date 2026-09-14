"""Whole classes, dealt round-robin: a class compiles its model in setUpClass, and
the shard has to stay in collection order."""


def pytest_addoption(parser):
    parser.addoption(
        "--shard", default="", metavar="GROUP/SPLITS", help="run one of SPLITS shards of the test classes"
    )


def pytest_collection_modifyitems(config, items):
    if not config.getoption("--shard"):
        return
    group, splits = (int(n) for n in config.getoption("--shard").split("/"))
    classes = list(dict.fromkeys(item.cls for item in items))
    keep = set(classes[group - 1 :: splits])
    items[:] = [item for item in items if item.cls in keep]
