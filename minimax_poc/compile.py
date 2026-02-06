import os

from optimum.rbln import RBLNMiniMaxM2ForCausalLM


model_id = "MiniMaxAI/MiniMax-M2.1"
model = RBLNMiniMaxM2ForCausalLM.from_pretrained(
    model_id=model_id,
    export=True,
    rbln_config={
        "batch_size": 1,
        "max_seq_len": 1024,
        "tensor_parallel_size": 4,
        "quantization": {"weights": "fp8", "kv_caches": "fp16"},
        "create_runtimes": False,
        "npu": "RBLN-CR03",
    },
    num_hidden_layers=3,
    trust_remote_code=True,
)
model.save_pretrained(os.path.basename(model_id))
