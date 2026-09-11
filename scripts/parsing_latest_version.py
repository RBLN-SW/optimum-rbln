import argparse
import sys

from packaging.version import parse


def main():
    parser = argparse.ArgumentParser(description="Parse and find latest version from stdin")
    parser.add_argument(
        "--exclude-prerelease",
        action="store_true",
        help="Exclude prerelease versions (alpha, beta, rc, etc.)",
    )
    args = parser.parse_args()

    versions = []
    for line in sys.stdin:
        version_str = line.strip()
        if version_str:
            try:
                version = parse(version_str)
                if args.exclude_prerelease and version.is_prerelease:
                    continue
                versions.append(version)
            except Exception:
                pass

    if versions:
        latest_version = sorted(versions)[-1]
        print(str(latest_version))


if __name__ == "__main__":
    main()
