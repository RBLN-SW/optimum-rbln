"""Stage the exact frozen MPNet bundle from Kubernetes NAS onto the shared UDC route.

Uses the passed DINO route's sibling-directory rename and a cooperative mkdir lock.
"""

import argparse
import hashlib
import json
import shutil
import tempfile
from pathlib import Path

from scripts.diagnostics import mpnet_b2048_frozen as harness


SOURCE = Path("/mnt/shared_data/users/jaehunryu/nas_data/model-enable-20261024/mpnet-native-b2048-hybrid-hcomm-sdk329")
DESTINATION = Path("/mnt/cross_data/model-enable/jaehunryu/mpnet-native-b2048-hybrid-hcomm-sdk329")
MANIFEST_SHA256 = "c15b26e1a3bd1f5af0bdf1482aed7e585c50e4244403b12b05505cb02729a585"


def verify_bundle(directory):
    expected = harness.FILES | {"manifest.json"}
    if directory.is_symlink() or not directory.is_dir() or {p.name for p in directory.iterdir()} != expected:
        raise ValueError("Staging requires exactly five model files and the manifest in an ordinary directory")
    if any(p.is_symlink() or not p.is_file() for p in directory.iterdir()):
        raise ValueError("Every staged entry must be an ordinary file")
    manifest = harness.verify_files(directory)
    if hashlib.sha256((directory / "manifest.json").read_bytes()).hexdigest() != MANIFEST_SHA256:
        raise ValueError("Frozen manifest differs from the reviewed SHA256; refusing to overwrite")
    required = {
        "runtime_sdk": harness.SDK,
        "compiler_kind": "source-hybrid root4b + AA + f66 HComm",
        "compiler_dso_sha256": harness.HYBRID_DSO,
        "checkpoint": harness.CHECKPOINT,
        "revision": harness.REVISION,
        "reviewed_product_head": harness.PRODUCT,
        **harness.SOURCE_CONTRACT,
        "npu": "RBLN-CR13",
        "batch_size": 2048,
        "max_seq_len": 512,
    }
    if any(manifest.get(key) != value for key, value in required.items()):
        raise ValueError("Staged bundle differs from the reviewed frozen MPNet contract")
    if manifest["files"]["compiled_model.rbln"]["sha256"] != harness.COMPILED_SHA256:
        raise ValueError("Staged compiled binary differs from the reviewed frozen SHA256")


def stage_lock(destination):
    return destination.with_name(f".{destination.name}.stage-lock")


def stage(source=SOURCE, destination=DESTINATION):
    source, destination = Path(source), Path(destination)
    if source.resolve() == destination.resolve() or source.resolve() in destination.resolve().parents:
        raise ValueError("Staged destination must be independent of the immutable source")
    verify_bundle(source)
    manifest_bytes = (source / "manifest.json").read_bytes()

    def receipt(state):
        return {
            "state": state,
            "source": str(source),
            "destination": str(destination),
            "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
            "compiled_sha256": harness.COMPILED_SHA256,
            "files": sorted(harness.FILES | {"manifest.json"}),
        }

    def reuse():
        verify_bundle(destination)
        if (destination / "manifest.json").read_bytes() != manifest_bytes:
            raise ValueError("Existing destination has a conflicting manifest; refusing to overwrite")
        return receipt("reused")

    if destination.exists() or destination.is_symlink():
        return reuse()
    destination.parent.mkdir(parents=True, exist_ok=True)
    lock = stage_lock(destination)
    try:
        lock.mkdir()
    except FileExistsError as error:
        if destination.exists():
            return reuse()
        raise FileExistsError(f"Refusing to bypass another staging writer: {lock}") from error
    try:
        # All writers of this immutable target use the same mkdir lock. Do not
        # remove another writer's lock; an interrupted unpublished owner requires review.
        if destination.exists():
            return reuse()
        with tempfile.TemporaryDirectory(prefix=f".{destination.name}.staging-", dir=destination.parent) as directory:
            temporary = Path(directory)
            for name in sorted(harness.FILES | {"manifest.json"}):
                shutil.copy2(source / name, temporary / name)
            verify_bundle(temporary)
            if (temporary / "manifest.json").read_bytes() != manifest_bytes:
                raise ValueError("Manifest changed during staging")
            if destination.exists():
                return reuse()
            temporary.rename(destination)
    finally:
        lock.rmdir()
    return receipt("staged")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--destination", type=Path, default=DESTINATION)
    args = parser.parse_args(argv)
    print("MPNET_STAGED " + json.dumps(stage(args.source, args.destination)), flush=True)


if __name__ == "__main__":
    main()
