"""Whole classes, dealt round-robin: a class compiles its model in setUpClass, and
the shard has to stay in collection order."""


def pytest_addoption(parser):
    parser.addoption("--shard", default="", metavar="INDEX/COUNT", help="run one of COUNT shards of the test classes")


def pytest_collection_modifyitems(config, items):
    if not config.getoption("--shard"):
        return
    index, count = (int(n) for n in config.getoption("--shard").split("/"))
    classes = list(dict.fromkeys(item.cls for item in items))
    keep = set(classes[index - 1 :: count])
    items[:] = [item for item in items if item.cls in keep]
