"""`_is_compiled` asks whether a repo already holds a compiled model, to decide `export`.

The answer needs a repo listing, which no cache can serve, so an unreachable Hub has to read as
"nothing compiled here" rather than propagate out of `from_pretrained`.
"""

from huggingface_hub import HfApi
from huggingface_hub.errors import OfflineModeIsEnabled

from optimum.rbln import RBLNBertModel


def test_is_compiled_is_false_when_the_hub_is_unreachable(monkeypatch):
    def offline(*args, **kwargs):
        raise OfflineModeIsEnabled("offline mode is enabled")

    monkeypatch.setattr(HfApi, "list_repo_files", offline)

    assert RBLNBertModel._is_compiled("hf-tiny-model-private/tiny-random-BertModel") is False
