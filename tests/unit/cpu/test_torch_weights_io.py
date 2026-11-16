import pickle

import pytest
import torch

from optimum.rbln.utils.save_utils import load_torch_weights, save_torch_weights


class _Payload:
    def __reduce__(self):
        return (print, ("unpickled",))


def test_artifact_roundtrip(tmp_path):
    # the shapes save_torch_artifacts writes: nested state dicts and bare parameters
    artifacts = {
        "embed_tokens": torch.nn.Embedding(4, 2).state_dict(),
        "image_newline": torch.nn.Parameter(torch.randn(2)),
    }
    path = tmp_path / "torch_artifacts.pth"
    save_torch_weights(artifacts, path)

    loaded = load_torch_weights(path)
    assert torch.equal(loaded["embed_tokens"]["weight"], artifacts["embed_tokens"]["weight"])
    assert isinstance(loaded["image_newline"], torch.nn.Parameter)
    assert torch.equal(loaded["image_newline"], artifacts["image_newline"])


def test_rejects_arbitrary_pickle(tmp_path):
    path = tmp_path / "adapter_model.bin"
    # nosemgrep: trailofbits.python.pickles-in-pytorch.pickles-in-pytorch
    torch.save({"weight": _Payload()}, path)

    with pytest.raises(pickle.UnpicklingError):
        load_torch_weights(path)
