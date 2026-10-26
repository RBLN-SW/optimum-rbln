"""Relay verified user-owned activation fixtures into the current CI build."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path


def digest(path):
    checksum = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            checksum.update(block)
    return checksum.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    assert args.source.is_dir(), "Fixture source mount unavailable; no NPU computation attempted"
    manifest = json.loads(args.manifest.read_text())
    # Validate the complete source before copying or publishing any fixture.
    for row in manifest["files"]:
        assert Path(row["filename"]).name == row["filename"]
        source = args.source / row["filename"]
        assert source.stat().st_size == row["bytes"]
        assert digest(source) == row["sha256"]
    args.destination.mkdir(parents=True, exist_ok=False)
    rows = []
    for row in manifest["files"]:
        target = args.destination / row["filename"]
        shutil.copyfile(args.source / row["filename"], target)
        assert target.stat().st_size == row["bytes"]
        assert digest(target) == row["sha256"]
        rows.append({"filename": row["filename"], "bytes": row["bytes"], "sha256": row["sha256"]})
    receipt = {
        "scope": "Verified original activation input relay; no model or NPU execution",
        "source_build": manifest["build"],
        "files": rows,
        "all_source_and_copy_hashes_verified": True,
    }
    (args.destination / "producer-receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print("GELU_FIXTURE_SOURCE_VERIFIED " + json.dumps({"files": len(rows), "bytes": sum(r["bytes"] for r in rows)}))


if __name__ == "__main__":
    main()
