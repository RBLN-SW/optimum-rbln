# /// script
# dependencies = ["huggingface_hub"]
# ///
"""Warm the local HF cache for every Hub repo the test suite references.

Run this online before pytest; the suite itself runs with HF_HUB_OFFLINE=1 so a
warm cache produces zero Hub API calls. Only repos missing from the cache are
downloaded, so in steady state this script is also network-silent.

Usage: python warm_hf_cache.py [--dry-run]
"""

import ast
import re
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SCAN_PATHS = [
    REPO_ROOT / "tests",
    REPO_ROOT / ".github" / "scripts" / "test_cli.py",
]
REPO_ID_RE = re.compile(r"^[A-Za-z0-9][\w.-]*/[A-Za-z0-9][\w.-]*$")
# Repo-shaped strings that are not Hub repos (e.g. placeholder ids used with an
# in-memory config).
DENYLIST = {"dummy/mistral"}
COMMIT_HASH_RE = re.compile(r"^[0-9a-f]{40}$")
NON_REPO_SUFFIX_RE = re.compile(
    r"\.(png|jpe?g|gif|bmp|webp|json|jsonl|txt|md|py|csv|tsv|wav|mp3|mp4|safetensors|bin|pt|pth|rbln|ya?ml|lock|toml|log|so)$",
    re.IGNORECASE,
)
# Files a test-time from_pretrained never reads; skipping them keeps a fresh
# warm-up from downloading duplicate or alternative-format checkpoints.
IGNORE_PATTERNS = [
    "*.gguf",
    "*.onnx",
    "*.onnx_data",
    "*.msgpack",
    "*.h5",
    "*.tflite",
    "*.ot",
    "*consolidated*",
    "original/*",
]


def looks_like_repo_id(value: str) -> bool:
    return bool(REPO_ID_RE.match(value)) and not NON_REPO_SUFFIX_RE.search(value) and value not in DENYLIST


def string_constants(node: ast.AST):
    for sub in ast.walk(node):
        if isinstance(sub, ast.Constant) and isinstance(sub.value, str):
            yield sub.value


def collect_candidates(paths):
    """Yield (repo_id, revision or None) referenced by the scanned sources.

    A pinned revision (a 40-hex "revision" literal) is associated only with the
    HF_MODEL_ID assigned in the same class body; every other repo-shaped string
    is warmed at its default revision.
    """
    candidates = {}
    files = []
    for path in paths:
        files.extend(sorted(path.rglob("*.py")) if path.is_dir() else [path])

    for file in files:
        tree = ast.parse(file.read_text(), filename=str(file))
        class_nodes = [n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]

        for cls in class_nodes:
            model_id = None
            revision = None
            for stmt in cls.body:
                if isinstance(stmt, ast.Assign) and any(
                    isinstance(t, ast.Name) and t.id == "HF_MODEL_ID" for t in stmt.targets
                ):
                    if isinstance(stmt.value, ast.Constant) and isinstance(stmt.value.value, str):
                        model_id = stmt.value.value
            for value in string_constants(cls):
                if COMMIT_HASH_RE.match(value):
                    revision = value
            if model_id and looks_like_repo_id(model_id):
                candidates.setdefault((model_id, revision), file.name)

        for value in string_constants(tree):
            if looks_like_repo_id(value):
                candidates.setdefault((value, None), file.name)

    return dict(sorted(candidates.items(), key=lambda item: (item[0][0], item[0][1] or "")))


def is_cached(repo_id: str, revision: str | None) -> bool:
    from huggingface_hub.constants import HF_HUB_CACHE

    repo_dir = Path(HF_HUB_CACHE) / f"models--{repo_id.replace('/', '--')}"
    if revision:
        return (repo_dir / "snapshots" / revision).is_dir()
    # An unpinned load resolves the "main" ref, so the ref and its snapshot must both exist.
    ref = repo_dir / "refs" / "main"
    if not ref.is_file():
        return False
    return (repo_dir / "snapshots" / ref.read_text().strip()).is_dir()


def main() -> int:
    dry_run = "--dry-run" in sys.argv
    candidates = collect_candidates(SCAN_PATHS)

    missing = [(repo, rev) for (repo, rev) in candidates if not is_cached(repo, rev)]
    print(f"{len(candidates)} repo candidates, {len(missing)} not in cache")
    if dry_run:
        for repo, rev in candidates:
            state = "cached" if (repo, rev) not in missing else "MISSING"
            print(f"  [{state}] {repo}" + (f" @ {rev}" if rev else ""))
        return 0

    from huggingface_hub import snapshot_download

    failures = []
    for repo, rev in missing:
        try:
            snapshot_download(repo, revision=rev, ignore_patterns=IGNORE_PATTERNS)
            print(f"warmed {repo}" + (f" @ {rev}" if rev else ""))
        except Exception as err:  # noqa: BLE001 - a false-positive candidate must not fail the build
            failures.append((repo, rev, err))
            print(f"warning: could not warm {repo}: {err}")

    if failures:
        print(f"{len(failures)} candidate(s) skipped; if a test needs one of them it will fail offline")
    return 0


if __name__ == "__main__":
    sys.exit(main())
