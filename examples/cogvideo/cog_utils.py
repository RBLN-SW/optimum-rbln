"""Shared helpers for the CogVideoX verification scripts (metrics, output folders, device memory)."""

import datetime
import json
import os
import resource
import subprocess
import threading
from pathlib import Path

import numpy as np
import torch


# Each run writes a new folder here. Override with COGVIDEOX_OUT_ROOT.
OUT_ROOT = Path(os.environ.get("COGVIDEOX_OUT_ROOT", "/home/seinpark/nas_data/cogvideox/outputs"))

# Inference dtype from the model cards ("use the precision the model was trained with"): FP16 for CogVideoX-2b,
# BF16 for the 5B models. Keyed by the repo name, so THUDM/ and zai-org/ ids both match.
MODEL_CARD_DTYPE = {
    "CogVideoX-2b": torch.float16,
    "CogVideoX-5b": torch.bfloat16,
    "CogVideoX-5b-I2V": torch.bfloat16,
    "CogVideoX1.5-5B-I2V": torch.bfloat16,
}
TORCH_DTYPE_CHOICES = ["auto", "float16", "bfloat16", "float32"]


def resolve_torch_dtype(model_id: str, name: str) -> torch.dtype:
    """`name` is "auto" (the model card dtype) or a torch dtype name."""
    if name != "auto":
        return getattr(torch, name)
    return MODEL_CARD_DTYPE[model_id.rstrip("/").split("/")[-1]]


def dtype_name(dtype) -> str:
    return str(dtype).removeprefix("torch.")


def compiled_dtypes(pipe) -> dict[str, str]:
    """The dtype each compiled submodule of an RBLN CogVideoX pipeline runs at (the VAE conv caches use the VAE's)."""
    return {name: dtype_name(getattr(pipe, name).rbln_config.dtype) for name in ("text_encoder", "transformer", "vae")}


def compare(name: str, out: torch.Tensor, ref: torch.Tensor) -> dict:
    """Pearson r plus the stats that r alone hides (offset, scale, absolute error)."""
    from scipy.stats import pearsonr

    x = ref.detach().flatten().to(torch.float64).numpy()
    y = out.detach().flatten().to(torch.float64).numpy()
    r = float(pearsonr(x, y).statistic)
    diff = np.abs(y - x)
    a, b = np.polyfit(x, y, 1)
    stats = {
        "pearson_r": r,
        "max_abs_diff": float(diff.max()),
        "mean_abs_diff": float(diff.mean()),
        "rel_l2": float(np.linalg.norm(y - x) / np.linalg.norm(x)),
        "slope_a": float(a),
        "intercept_b": float(b),
        "allclose_atol2e-2_rtol1e-2": bool(torch.allclose(out.float(), ref.float(), atol=2e-2, rtol=1e-2)),
    }
    print(
        f"[{name}] r={r:.6f}  max|diff|={stats['max_abs_diff']:.5f}  mean|diff|={stats['mean_abs_diff']:.6f}  "
        f"rel_l2={stats['rel_l2']:.5f}  y~{a:.5f}*x+{b:+.6f}  allclose={stats['allclose_atol2e-2_rtol1e-2']}"
    )
    return stats


def psnr(out: torch.Tensor, ref: torch.Tensor, data_range: float = 2.0) -> float:
    mse = torch.mean((out.double() - ref.double()) ** 2).item()
    return float("inf") if mse == 0 else 10 * np.log10(data_range**2 / mse)


def new_out_dir(desc: str) -> Path:
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    path = OUT_ROOT / f"{stamp}_{desc}"
    suffix = 1
    while path.exists():
        path = OUT_ROOT / f"{stamp}_{desc}_{suffix}"
        suffix += 1
    path.mkdir(parents=True)
    return path


def video_to_uint8(video: torch.Tensor) -> np.ndarray:
    """(1, 3, F, H, W) in [-1, 1] -> (F, H, W, 3) uint8."""
    v = ((video[0].float().clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8)
    return v.permute(1, 2, 3, 0).numpy()


def save_video_and_frames(video: torch.Tensor, out_dir: Path, tag: str, fps: int = 8) -> None:
    import imageio
    from PIL import Image

    frames = video_to_uint8(video)
    frame_dir = out_dir / f"frames_{tag}"
    frame_dir.mkdir(exist_ok=True)
    for i, frame in enumerate(frames):
        Image.fromarray(frame).save(frame_dir / f"frame_{i:04d}.png")
    if len(frames) > 1:
        imageio.mimsave(out_dir / f"video_{tag}.mp4", list(frames), fps=fps)


def git_commit(repo: str = str(Path(__file__).resolve().parents[2])) -> str:
    out = subprocess.run(["git", "-C", repo, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "-C", repo, "status", "--porcelain", "src"], capture_output=True, text=True).stdout
    return out + ("+dirty" if dirty.strip() else "")


def write_meta(out_dir: Path, meta: dict) -> None:
    import rebel

    meta = {
        "optimum_rbln_commit": git_commit(),
        "rebel_compiler_version": rebel.__version__,
        "RBLN_DEVICES": os.environ.get("RBLN_DEVICES"),
        **meta,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2, default=str))


def host_peak_rss_gb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2


class DeviceMemSampler:
    """Polls `rbln-stat -j` in the background and keeps the peak used memory per device."""

    def __init__(self, interval: float = 0.5):
        self.interval = interval
        self.peak: dict[str, int] = {}
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _sample(self):
        try:
            data = json.loads(subprocess.run(["rbln-stat", "-j"], capture_output=True, text=True).stdout)
        except Exception:
            return
        for dev in data.get("devices", []):
            used = int(dev["memory"]["used"])
            self.peak[dev["device"]] = max(self.peak.get(dev["device"], 0), used)

    def _run(self):
        while not self._stop.is_set():
            self._sample()
            self._stop.wait(self.interval)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join()
        self._sample()

    def summary(self) -> dict[str, float]:
        return {dev: round(used / 1024**3, 3) for dev, used in self.peak.items() if used > 0}
