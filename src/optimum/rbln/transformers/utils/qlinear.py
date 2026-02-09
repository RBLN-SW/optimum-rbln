from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class QLinear(nn.Module):
    def __init__(
        self,
        weight: Optional[torch.Tensor] = None,
        bias: Optional[torch.Tensor] = None,
        weight_scale: Optional[torch.Tensor] = None,
        input_scale: Optional[torch.Tensor] = None,
        dynamic: bool = False,
    ):
        super().__init__()

        self.weight = weight
        self.bias = bias
        self.weight_scale = weight_scale
        self.input_scale = input_scale
        self.dynamic = dynamic

        if weight_scale is None:
            raise ValueError("weight_scale is required")

    def dtype(self) -> torch.dtype:
        return self.weight.dtype

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class QIntLinear(QLinear):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        iinfo = torch.iinfo(self.dtype())
        finfo = torch.finfo(x.dtype)
        if self.dynamic:
            if self.input_scale is not None:
                raise NotImplementedError("Dynamic quantization with input_scale is not supported.")
            x_max = x.abs().max(dim=-1, keepdim=True).values
            x_scale = x_max / iinfo.max
            x_scale = torch.clamp(x_scale, min=finfo.eps)

            x = (x / x_scale).clamp(min=iinfo.min, max=iinfo.max)
        else:
            if self.input_scale is not None:
                x = (x / self.input_scale).clamp(min=iinfo.min, max=iinfo.max)

        weight = self.weight * self.weight_scale
        qact = F.linear(x, weight, self.bias)

        if self.dynamic:  # Dequantize
            qact = qact * x_scale

        return qact


class QFloatLinear(QLinear):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.input_scale is not None:
            finfo = torch.finfo(self.dtype())
            x = (x / self.input_scale).clamp(min=finfo.min, max=finfo.max)

        weight = self.weight.to(self.weight_scale.dtype) * self.weight_scale

        qact = F.linear(x, weight, self.bias)
        # if self.input_scale is not None:
        #     qact = qact * self.input_scale

        return qact
