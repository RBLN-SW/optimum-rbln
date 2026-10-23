import pytest
from transformers import BertConfig

from optimum.rbln import RBLNAutoModel, RBLNBertModel


@pytest.mark.parametrize("architectures", [None, []])
def test_auto_model_accepts_config_without_architecture_hint(architectures):
    config = BertConfig(architectures=architectures)
    assert RBLNAutoModel.get_rbln_cls("unused", export=True, config=config) is RBLNBertModel


def test_auto_model_keeps_warning_for_explicit_different_architecture():
    config = BertConfig(architectures=["BertForMaskedLM"])
    with pytest.warns(UserWarning, match="does not match the expected architecture"):
        assert RBLNAutoModel.get_rbln_cls("unused", export=True, config=config) is RBLNBertModel
