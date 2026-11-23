import hashlib
import os

import pytest
import rebel
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from optimum.rbln import RBLNLlamaForCausalLM
from optimum.rbln.cli import _handle_kvcache_num_blocks


@pytest.mark.skipif(
    not os.getenv("RBLN_DEVICE_MAP") or os.getenv("RBLN_DUMMY_DEVICE") == "1",
    reason="Set RBLN_DEVICE_MAP to a physical CA25 test device.",
)
def test_decoder_with_caller_owned_cache(tmp_path):
    from torch_rbln._internal.rsd_utils import get_physical_device_ids

    devices = get_physical_device_ids(0)
    if not devices or len(devices) > 2 or rebel.get_npu_name(devices[0]) != "RBLN-CA25":
        pytest.skip("This flash attention configuration targets one or two CA25 devices.")

    torch.manual_seed(42)
    hf_model = (
        LlamaForCausalLM(
            LlamaConfig(
                hidden_size=128,
                intermediate_size=256,
                num_hidden_layers=1,
                num_attention_heads=2,
                num_key_value_heads=2,
                vocab_size=128,
                max_position_embeddings=4096,
            )
        )
        .half()
        .eval()
    )
    opts = {
        "npu": "RBLN-CA25",
        "num_devices": len(devices),
        "create_runtimes": False,
        "max_seq_len": 4096,
        "prefill_chunk_size": 64,
        "batch_size": 2,
        "decoder_batch_sizes": [1, 2],
        "attn_impl": "flash_attn",
        "kvcache_partition_len": 1024,
    }
    ids = torch.tensor([[6, 3, 9]])

    def generate(model):
        return model.generate(ids, attention_mask=torch.ones_like(ids), max_new_tokens=5, do_sample=False)

    directory = tmp_path / "device"
    RBLNLlamaForCausalLM.from_model(hf_model, model_save_dir=directory, rbln_config=opts)
    expected = generate(RBLNLlamaForCausalLM.from_pretrained(directory, rbln_device=devices))
    binaries = {p: hashlib.sha256(p.read_bytes()).digest() for p in directory.glob("*.rbln")}
    _handle_kvcache_num_blocks(str(directory), get=False, set_value=6)
    assert all(hashlib.sha256(p.read_bytes()).digest() == digest for p, digest in binaries.items())

    model = RBLNLlamaForCausalLM.from_pretrained(directory, rbln_device=devices)
    initial = model.get_kv_cache()
    assert initial["past_key_values_0"].shape[0] == 6
    replacement = {name: torch.zeros_like(t) for name, t in initial.items()}
    model.set_kv_cache(replacement)
    assert torch.equal(generate(model), expected)
    assert all(t.cpu().count_nonzero() > 0 for t in replacement.values())
    assert all(t.cpu().count_nonzero() == 0 for t in initial.values())
