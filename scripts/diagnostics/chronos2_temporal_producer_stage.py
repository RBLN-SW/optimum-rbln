"""Checksum-verified NAS producer to common-storage staging; no NPU imports."""

import argparse
import json
import os
import shutil
from pathlib import Path

from scripts.diagnostics.chronos2_temporal_producer_run import verify_bundle


def stage(source, destination, manifest_sha256):
    verify_bundle(source, manifest_sha256)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        verify_bundle(destination, manifest_sha256)
        return "verified_existing"
    temporary = destination.with_name(destination.name + f".partial-{os.getpid()}")
    if temporary.exists():
        raise ValueError("Retaining existing partial staging directory; refusing to overwrite")
    shutil.copytree(source, temporary)
    verify_bundle(temporary, manifest_sha256)
    temporary.rename(destination)
    return "staged_verified"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    args = parser.parse_args()
    print(json.dumps({"status": stage(args.source, args.destination, args.manifest_sha256)}), flush=True)


if __name__ == "__main__":
    main()
