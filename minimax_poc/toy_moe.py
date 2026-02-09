import fire
import rebel
import torch
from scipy import stats
from transformers import AutoModelForCausalLM

from optimum.rbln.minimax_poc.minimax_m2_architecture import MiniMaxM2SparseMoeBlock


DATA_BASE_PATH = "/mnt/shared_data/groups/sw_dev/thkim/minimax/"


def _w8a16_block_fp8_matmul(self, input: torch.Tensor) -> torch.Tensor:
    bs0, bs1 = self.block_size
    out_features, in_features = self.weight.shape
    out_blocks = out_features // bs0
    in_blocks = in_features // bs1

    w = self.weight.view(out_blocks, bs0, in_blocks, bs1).to(input.dtype)  # (OB, bs0, IB, bs1)
    s = self.weight_scale_inv.view(out_blocks, 1, in_blocks, 1).to(input.dtype)  # (OB, 1,  IB, 1)

    weight = (w * s).view(out_features, in_features)
    output = torch.nn.functional.linear(input, weight, self.bias)

    return output


def _load_moe_native_model():
    from transformers.integrations.finegrained_fp8 import FP8Linear
    from transformers.quantizers.quantizer_finegrained_fp8 import FineGrainedFP8HfQuantizer

    FineGrainedFP8HfQuantizer.validate_environment = lambda self, *a, **k: None
    FP8Linear.forward = _w8a16_block_fp8_matmul

    model = AutoModelForCausalLM.from_pretrained("MiniMaxAI/MiniMax-M2.1", trust_remote_code=True, num_hidden_layers=1)
    moe_sparse_block = model.model.layers[0].block_sparse_moe
    moe_sd = torch.load(DATA_BASE_PATH + "moe_sd.pt")
    moe_sparse_block.load_state_dict(moe_sd)

    return moe_sparse_block


def main(batch_size: int = 1, prefill_chunk_size: int = 128, compile: bool = False):
    model = _load_moe_native_model()
    wrapped_model = MiniMaxM2SparseMoeBlock(model).eval()

    b4_inputs = torch.load(DATA_BASE_PATH + "moe_inputs.pt")  # torch.Size([4, 143, 3072]
    inputs = b4_inputs[:batch_size, :prefill_chunk_size, :].to(torch.float32)

    if compile:
        compiled_model = rebel.compile_from_torch(
            wrapped_model, input_info=[("input", (batch_size, prefill_chunk_size, 3072), torch.float32)]
        )
        compiled_model.save("toy_moe.rbln")

    rbln_model = rebel.Runtime("toy_moe.rbln", tensor_type="pt")
    rbln_output = rbln_model.run(inputs)
    golden_output, _ = model(inputs)

    print("rbln_output")
    print(rbln_output)
    print("golden_output")
    print(golden_output)
    print("Max l1 difference:", torch.max(torch.abs(rbln_output.reshape(-1) - golden_output.reshape(-1))))
    print(
        "Pearson correlation coefficient:",
        stats.pearsonr(rbln_output.detach().numpy().reshape(-1), golden_output.detach().numpy().reshape(-1)),
    )


if __name__ == "__main__":
    fire.Fire(main)
