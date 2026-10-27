import gc
import os

import pytest
from transformers import AutoModel, AutoTokenizer

from optimum.rbln import RBLNMPNetModel

from . import test_transformers


pytestmark = pytest.mark.skipif(
    os.environ.get("OPTIMUM_RBLN_TEST_LEVEL", "default").lower() != "full"
    or bool(os.environ.get("SAVE_ARTIFACTS_PATH") or os.environ.get("REUSE_ARTIFACTS_PATH")),
    reason="Native batch parity requires full coverage on a real scheduler NPU.",
)


@pytest.mark.parametrize("batch_size", [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048])
def test_mpnet_native_batch_parity_and_reload(batch_size, tmp_path):
    npu = os.environ.get("RBLN_TEST_NPU")
    if batch_size >= 512 and npu != "RBLN-CR13":
        pytest.skip("Native batches >= 512 require a CR13 lane.")
    tokenizer = AutoTokenizer.from_pretrained(
        test_transformers.TestMPNetModel.HF_MODEL_ID, **test_transformers.TestMPNetModel.HF_CONFIG_KWARGS
    )
    reference = (
        AutoModel.from_pretrained(
            test_transformers.TestMPNetModel.HF_MODEL_ID, **test_transformers.TestMPNetModel.HF_CONFIG_KWARGS
        )
        .float()
        .eval()
    )
    save_dir = tmp_path / "reloaded"
    model = RBLNMPNetModel.from_pretrained(
        test_transformers.TestMPNetModel.HF_MODEL_ID,
        **test_transformers.TestMPNetModel.HF_CONFIG_KWARGS,
        rbln_max_seq_len=512,
        rbln_batch_size=batch_size,
        rbln_npu=npu,
        rbln_device=0,
        model_save_dir=tmp_path / "compiled",
    )
    try:
        test_transformers.TestMPNetModel.assert_embedding_parity(model, reference, tokenizer, batch_size)
        model.save_pretrained(save_dir)
        model = None
        gc.collect()
        model = RBLNMPNetModel.from_pretrained(save_dir, rbln_device=0)
        test_transformers.TestMPNetModel.assert_embedding_parity(model, reference, tokenizer, batch_size)
    finally:
        del model
        gc.collect()
