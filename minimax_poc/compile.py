import os

from configuration_minimax_m2 import RBLNMiniMaxM2ForCausalLMConfig

# Support both:
# - `python -m minimax.compile` (package execution; relative imports work)
# - `python minimax/compile.py`  (script execution; relative imports fail)
from modeling_minimax_m2 import RBLNMiniMaxM2ForCausalLM

from optimum.rbln import RBLNAutoConfig, RBLNAutoModel


RBLNAutoModel.register(RBLNMiniMaxM2ForCausalLM, exist_ok=True)
RBLNAutoConfig.register(RBLNMiniMaxM2ForCausalLMConfig, exist_ok=True)


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
