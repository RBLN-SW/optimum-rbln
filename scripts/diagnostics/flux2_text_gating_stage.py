"""Checksum-verified k8s-to-UDC staging for the native FLUX2 block-intermediate experiment."""

import json
import os
import shutil
from pathlib import Path

from scripts.diagnostics.flux2_text_gating import verify_bundle


SOURCE = Path(
    "/mnt/shared_data/users/jaehunryu/nas_data/model-enable-20261025/flux2-klein/text-gating-probes-bundle-sdk333"
)
DESTINATION = Path("/mnt/cross_data/model-enable/jaehunryu/flux2-klein-text-gating-probes-sdk333")


def stage(source=SOURCE, destination=DESTINATION):
    verify_bundle(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        verify_bundle(destination)
        return "verified_existing"
    temporary = destination.with_name(destination.name + f".partial-{os.getpid()}")
    if temporary.exists():
        raise ValueError("Existing staging prefix retained; refusing to overwrite it")
    shutil.copytree(source, temporary)
    verify_bundle(temporary)
    temporary.rename(destination)
    return "staged_verified"


if __name__ == "__main__":
    print(json.dumps({"status": stage()}), flush=True)
