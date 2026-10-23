import argparse
import hashlib
import importlib.metadata
import json
import shutil
from pathlib import Path

from tests.test_transformers import (
    TestDinov2ForImageClassificationNativeSizeBatch1024,
    TestDinov2ModelNativeSizeBatch1024,
)


def main():
    assert importlib.metadata.version("rebel-compiler") == "0.12.1.dev327+g00334c8e"
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", action="store_true")
    args = parser.parse_args()
    root = Path("/mnt/cross_data/model-enable/jaehunryu/dinov2-source970-b1024")
    if args.stage:
        source_root = Path("/mnt/shared_data/users/jaehunryu/nas_data/model-enable-20261024/dinov2-source970-b1024")
        assert not root.exists(), root
        root.parent.mkdir(parents=True, exist_ok=True)
        temporary = root.with_name(root.name + ".staging")
        assert not temporary.exists(), temporary
        shutil.copytree(source_root, temporary)
        manifest = json.loads((temporary / "provenance.json").read_text())
        for name, artifact in manifest["artifacts"].items():
            with (temporary / name / artifact["file"]).open("rb") as stream:
                assert hashlib.file_digest(stream, "sha256").hexdigest() == artifact["sha256"]
        temporary.rename(root)
        print("DINOV2_STAGED " + str(root), flush=True)
        return
    provenance = json.loads((root / "provenance.json").read_text())
    source = provenance["source_results"]["environment"]["source_provenance"]
    assert source["head"] == "970308c6eec6e0b794d9a20313d1dc77bc050837"
    assert source["primitive"] == "ffb04afbbf16627a93805257542b201b9fa3afb8"
    assert source["full_build_exit"] == 0
    report = {"source": source, "runtime": importlib.metadata.version("rebel-compiler"), "tests": []}
    out = Path("debug-dinov2-source1024")
    out.mkdir(exist_ok=True)
    for cls, name in [
        (TestDinov2ModelNativeSizeBatch1024, "encoder-native-b1024"),
        (TestDinov2ForImageClassificationNativeSizeBatch1024, "classification-native-b1024"),
    ]:
        artifact = root / name
        manifest = json.loads((artifact / "enablement_compile.json").read_text())
        assert manifest["rbln_config"]["batch_size"] == 1024
        assert manifest["rbln_config"]["npu"] == "RBLN-CR13"
        assert manifest["physical"] == [518, 518]
        binary = artifact / provenance["artifacts"][name]["file"]
        with binary.open("rb") as stream:
            assert hashlib.file_digest(stream, "sha256").hexdigest() == provenance["artifacts"][name]["sha256"]
        local = Path(cls.get_rbln_local_dir())
        assert not local.exists(), local
        shutil.copytree(artifact, local)
        row = {"test": cls.__name__, "state": "not run", "artifact_sha256": provenance["artifacts"][name]["sha256"]}
        report["tests"].append(row)
        try:
            cls.model = cls.RBLN_CLASS.from_pretrained(local, rbln_device=cls.DEVICE)
            cls("test_numerical_parity").test_numerical_parity()
            row["state"] = "passed"
        except Exception as error:
            row["state"] = "failed"
            row["error"] = repr(error)
            raise
        finally:
            (out / "results.json").write_text(json.dumps(report, indent=2))
            cls.tearDownClass()
    print("DINOV2_SOURCE1024 " + json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
