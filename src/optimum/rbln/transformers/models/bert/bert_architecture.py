import torch


class BertModelWrapper(torch.nn.Module):
    def __init__(self, model, rbln_config):
        super().__init__()
        self.model = model
        self.rbln_config = rbln_config

    def forward(self, *args, **kwargs):
        output = self.model(*args, **kwargs)
        if isinstance(output, torch.Tensor):
            return output
        elif isinstance(output, tuple):
            return tuple(x for x in output if x is not None)
        return output
