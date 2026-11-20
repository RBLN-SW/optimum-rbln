import torch


class BertModelWrapper(torch.nn.Module):
    def __init__(self, model, rbln_config):
        super().__init__()
        self.model = model
        self.rbln_config = rbln_config

    def forward(self, *args, **kwargs):
        # Bind graph inputs by name: a dropped input (e.g. `token_type_ids`) must not shift the rest.
        input_names = getattr(self.rbln_config, "model_input_names", None)
        if input_names and len(args) == len(input_names):
            kwargs.update(zip(input_names, args, strict=True))
            args = ()
        output = self.model(*args, **kwargs)
        if isinstance(output, torch.Tensor):
            return output
        elif isinstance(output, tuple):
            return tuple(x for x in output if x is not None)
        return output
