"""`--shard GROUP/SPLITS` runs one slice of a suite.

It selects whole classes: a class compiles its model in setUpClass, so one split
across two shards pays for that twice. Dealing them out round-robin spreads the
heavy model families, which are defined next to each other, and leaves the shard
in collection order, which the suites still rely on.
"""


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
