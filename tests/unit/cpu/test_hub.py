import httpx
from huggingface_hub.errors import RemoteEntryNotFoundError

from optimum.rbln.utils import hub


def test_cached_file_removed_from_the_repo_is_skipped(tmp_path, monkeypatch):
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    (snapshot / "rbln_config.json").write_text("{}")
    (snapshot / "model.rbln").write_bytes(b"")
    (snapshot / "removed.rbln").write_bytes(b"")
    requested = []

    def fake_hf_hub_download(repo_id, filename, **kwargs):
        requested.append(filename)
        if filename == "removed.rbln":
            response = httpx.Response(404, request=httpx.Request("GET", f"https://huggingface.co/{repo_id}"))
            raise RemoteEntryNotFoundError(f"{filename} is not in {repo_id}", response=response)
        return str(snapshot / filename)

    monkeypatch.setattr(hub, "try_to_load_from_cache", lambda **kwargs: str(snapshot / "rbln_config.json"))
    monkeypatch.setattr(hub, "hf_hub_download", fake_hf_hub_download)

    model_dir = hub.pull_compiled_model_from_hub(
        "org/model",
        subfolder="",
        token=None,
        revision=None,
        cache_dir=None,
        force_download=False,
        local_files_only=False,
    )

    assert model_dir == snapshot
    assert sorted(requested) == ["model.rbln", "rbln_config.json", "removed.rbln"]
