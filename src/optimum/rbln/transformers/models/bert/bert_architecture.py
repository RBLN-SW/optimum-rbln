import torch
from transformers.modeling_attn_mask_utils import _prepare_4d_attention_mask


class BertModelWrapper(torch.nn.Module):
    def __init__(self, model, rbln_config):
        super().__init__()
        self.model = model
        self.rbln_config = rbln_config

    def forward(self, *args, **kwargs):
        # TODO: make this to use `create_bidirectional_mask` in transformers v5
        args = list(args)
        input_names = getattr(self.rbln_config, "model_input_names", None) or [
            "input_ids",
            "attention_mask",
            "token_type_ids",
        ]
        if "attention_mask" in input_names:
            idx = input_names.index("attention_mask")
            if idx < len(args) and args[idx] is not None and args[idx].dim() == 2:
                args[idx] = _prepare_4d_attention_mask(args[idx], torch.float32)
        if "attention_mask" in kwargs and kwargs["attention_mask"] is not None and kwargs["attention_mask"].dim() == 2:
            kwargs["attention_mask"] = _prepare_4d_attention_mask(kwargs["attention_mask"], torch.float32)

        output = self.model(*args, **kwargs)
        if isinstance(output, torch.Tensor):
            return output
        elif isinstance(output, tuple):
            return tuple(x for x in output if x is not None)
        return output
