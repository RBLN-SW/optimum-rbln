import rebel
import torch
from scipy import stats
from transformers.integrations.finegrained_fp8 import FP8Linear


def _w8a16_block_fp8_matmul(self, input: torch.Tensor) -> torch.Tensor:
    _input_dtype = input.dtype
    out_features, in_features = self.weight.shape

    weight_scale = self.weight_scale_inv.repeat_interleave(self.block_size[0], 0).unsqueeze(-1)
    weight = self.weight.view(out_features, in_features // self.block_size[1], self.block_size[1])
    weight = (weight.to(_input_dtype) * weight_scale).view(out_features, in_features)
    output = torch.nn.functional.linear(input, weight, self.bias)

    return output


FP8Linear.forward = _w8a16_block_fp8_matmul

inputs = torch.randn(1, 1024, 3072)

sd = torch.load("sd.pt")


class toy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fp8_linear = FP8Linear(3072, 6144, block_size=(128, 128))
        self.fp8_linear.load_state_dict(sd)

    def forward(self, input):
        output = self.fp8_linear(input)
        return output


model = toy().eval()
compiled_model = rebel.compile_from_torch(model, input_info=[("input", (1, 1024, 3072), torch.float)])
compiled_model.save("toy.rbln")


rbln_model = rebel.Runtime("toy.rbln", tensor_type="pt")
golden_model = model


rbln_output = rbln_model.run(inputs)
golden_output = golden_model(inputs)

print("Max l1 difference:", torch.max(torch.abs(rbln_output.reshape(-1) - golden_output.reshape(-1))))
print(
    "Pearson correlation coefficient:",
    stats.pearsonr(rbln_output.detach().numpy().reshape(-1), golden_output.detach().numpy().resahpe(-1)),
)

breakpoint()
