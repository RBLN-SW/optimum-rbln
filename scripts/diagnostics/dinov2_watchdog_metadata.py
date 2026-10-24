import json
import platform
from pathlib import Path


def main():
    report = {"kernel_release": platform.release(), "drivers": {}, "errors": []}
    for path in Path("/sys/module").glob("*/parameters/timeout_s"):
        module = path.parent.parent
        if not any(name in module.name.lower() for name in ["rebel", "rbln"]):
            continue
        try:
            driver = {"timeout_s": path.read_text().strip(), "parameter_path": str(path)}
            for name in ["version", "srcversion"]:
                if (module / name).exists():
                    driver[name] = (module / name).read_text().strip()
            report["drivers"][module.name] = driver
        except OSError as error:
            report["errors"].append(str(error))
    report["watchdog_visible"] = bool(report["drivers"])
    out = Path("debug-dinov2-watchdog")
    out.mkdir(exist_ok=True)
    (out / "metadata.json").write_text(json.dumps(report, indent=2))
    print("DINOV2_WATCHDOG_METADATA " + json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
