"""Atomically stage the frozen CI806 baseline from k8s NAS onto the UDC route."""

import argparse
import hashlib
import json
import shutil
import tempfile
from pathlib import Path


SOURCE = Path("/mnt/shared_data/users/jaehunryu/nas_data/model-enable-20261024/modernbert-ci806/offset-2.pt")
DESTINATION = Path("/mnt/cross_data/model-enable/jaehunryu/modernbert-ci806/offset-2.pt")
SHA256 = "bed9721c662ca682645dee9469a2dbe17c3028559ec4a388c21a3e87f2b267e4"
BYTES = 50465191
FILENAME = "offset-2.pt"


def verify(path):
    if path.is_symlink() or not path.is_file() or path.stat().st_size != BYTES:
        raise ValueError("Baseline must be an ordinary file with the exact reviewed size")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    if digest.hexdigest() != SHA256:
        raise ValueError("Baseline SHA256 mismatch; refusing to publish or overwrite")


def stage(source=SOURCE, destination=DESTINATION):
    source, destination = Path(source), Path(destination)
    if source.name != FILENAME or destination.name != FILENAME:
        raise ValueError("Expected the frozen offset-2.pt filename")
    if source.resolve() == destination.resolve() or source.resolve().parent in destination.resolve().parents:
        raise ValueError("Destination must be independent of immutable source")
    verify(source)
    directory = destination.parent

    def reuse():
        if directory.is_symlink() or not directory.is_dir() or {p.name for p in directory.iterdir()} != {FILENAME}:
            raise ValueError("Existing staging directory contains unknown or conflicting content")
        verify(destination)
        return {"state": "reused", "sha256": SHA256, "bytes": BYTES, "destination": str(destination)}

    if directory.exists() or directory.is_symlink():
        return reuse()
    directory.parent.mkdir(parents=True, exist_ok=True)
    lock = directory.with_name(f".{directory.name}.stage-lock")
    try:
        lock.mkdir()
    except FileExistsError as error:
        if directory.exists():
            return reuse()
        raise FileExistsError("Another baseline staging writer owns the lock") from error
    try:
        if directory.exists():
            return reuse()
        with tempfile.TemporaryDirectory(prefix=f".{directory.name}.staging-", dir=directory.parent) as name:
            temporary = Path(name)
            shutil.copy2(source, temporary / FILENAME)
            verify(temporary / FILENAME)
            if directory.exists():
                return reuse()
            temporary.rename(directory)
    finally:
        lock.rmdir()
    return {"state": "staged", "sha256": SHA256, "bytes": BYTES, "destination": str(destination)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--destination", type=Path, default=DESTINATION)
    args = parser.parse_args(argv)
    print("MODERNBERT_BASELINE_STAGED " + json.dumps(stage(args.source, args.destination)), flush=True)


if __name__ == "__main__":
    main()
