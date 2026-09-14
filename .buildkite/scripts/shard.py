#!/usr/bin/env python3
"""shard.py <group> <splits>

Pick the test classes for one shard: reads `pytest --collect-only -q` on stdin,
writes the class node ids this shard owns on stdout.

By class, because a class compiles its model in setUpClass and splitting one
across two shards pays for that twice. Round-robin, because the expensive model
families sit next to each other in the file, so taking every nth class spreads
them -- and leaves each shard in collection order, which the suite still relies
on. Nothing to keep up to date when a test is added or removed.
"""

import sys


group, splits = int(sys.argv[1]), int(sys.argv[2])

units = []
for line in sys.stdin:
    node = line.strip()
    if not node.startswith("tests/") or "::" not in node:
        continue
    unit = node.rsplit("::", 1)[0] if node.count("::") > 1 else node
    if unit not in units:
        units.append(unit)

if not units:
    sys.exit("shard.py: collected no tests -- did `pytest --collect-only` fail?")

shard = units[group - 1 :: splits]
if not shard:
    sys.exit(f"shard.py: group {group} of {splits} is empty -- more shards than classes?")

print("\n".join(shard))
