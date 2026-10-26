import inspect
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from tests import test_base


@pytest.mark.parametrize("returns_commit", [True, False])
def test_push_checks_its_own_commit_when_remote_head_changes(monkeypatch, returns_commit):
    monkeypatch.setenv("HF_AUTH_TOKEN", "test-token")
    monkeypatch.setenv("HF_USER_ID", "test-user")
    monkeypatch.setattr(test_base.random, "getrandbits", lambda bits: 123)
    api = Mock()
    api.repo_info.return_value.sha = "later-concurrent-commit"
    monkeypatch.setattr(test_base, "HfApi", Mock(return_value=api))
    loader = Mock(return_value=SimpleNamespace(from_local=123))
    monkeypatch.setattr(test_base.AutoConfig, "from_pretrained", loader)
    commit = SimpleNamespace(oid="own-commit") if returns_commit else None
    model = SimpleNamespace(config=SimpleNamespace(), save_pretrained=Mock(return_value=commit))
    case = SimpleNamespace(model=model, is_diffuser=lambda: False, get_hf_remote_dir=lambda: "test-model")

    inspect.unwrap(test_base.BaseHubTest.TestHub.test_push_to_hub)(case)

    assert loader.call_args.kwargs["revision"] == ("own-commit" if returns_commit else "later-concurrent-commit")
    if returns_commit:
        api.repo_info.assert_not_called()
    else:
        api.repo_info.assert_called_once()
