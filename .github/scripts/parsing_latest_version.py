import sys

from packaging.version import parse


versions = []
for line in sys.stdin:
    version_str = line.strip()
    if version_str:
        try:
            versions.append(parse(version_str))
        except Exception:
            pass

if versions:
    latest_version = sorted(versions)[-1]
    print(str(latest_version))
