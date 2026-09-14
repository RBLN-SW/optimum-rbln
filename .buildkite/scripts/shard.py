#!/usr/bin/env python3
"""shard.py <weights-file> <group> <splits>

Pick the test classes that belong to one shard. Reads `pytest --collect-only -q`
output on stdin and writes the selected class node ids on stdout.

Splitting by class rather than by test keeps a model's compile in one shard --
pytest-split cuts wherever the test count lands, and a class split across two
shards pays for setUpClass twice. Classes are packed longest-first using the
measured seconds in the weights file; one the file does not list gets the median
of its test file, so a newly added test always runs, just not perfectly balanced.
"""

import statistics
import sys


weights_path, group, splits = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])

weights = {}
for line in open(weights_path):
    line = line.split("#", 1)[0].strip()
    if line:
        seconds, unit = line.split(None, 1)
        weights[unit] = float(seconds)

units = []
for line in sys.stdin:
    node = line.strip()
    if not node.startswith("tests/") or "::" not in node:
        continue
    # A class holds the compile, so shard on it; a module-level test is its own unit.
    unit = node.rsplit("::", 1)[0] if node.count("::") > 1 else node
    if unit not in units:
        units.append(unit)

if not units:
    sys.exit("shard.py: collected no tests -- did `pytest --collect-only` fail?")

median = {}
for path in {u.split("::", 1)[0] for u in units}:
    known = [w for u, w in weights.items() if u.startswith(path + "::")]
    median[path] = statistics.median(known) if known else 1.0


def weight(unit):
    return weights.get(unit, median[unit.split("::", 1)[0]])


shards = [[] for _ in range(splits)]
totals = [0.0] * splits
for unit in sorted(units, key=lambda u: (-weight(u), u)):
    i = totals.index(min(totals))
    shards[i].append(unit)
    totals[i] += weight(unit)

if not shards[group - 1]:
    sys.exit(f"shard.py: group {group} of {splits} is empty -- more shards than classes?")

print("\n".join(shards[group - 1]))
