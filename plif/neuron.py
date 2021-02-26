from typing import Any, Optional, Tuple
import math

import torch
import torch.jit as jit
import torch.nn as nn


class SpikeFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, x: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(x)
        return x.gt(0.0).float()

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> torch.Tensor:
        (x,) = ctx.saved_tensors
        grad_input = grad_output.clone()
        # Derivative of 1/pi * arctan(pi * x) + 0.5
        spike_pseudo_grad = 1.0 / (math.pi * math.pi * x * x + 1.0)
        return grad_input * spike_pseudo_grad


@jit.ignore
def spike(x: torch.Tensor) -> torch.Tensor:
    return SpikeFunction.apply(x)


class PLIF(nn.Module):
    def __init__(self, a_0: float):
        super().__init__()

        # Shared between neurons in a layer
        self.a = nn.Parameter(torch.tensor(a_0))
        # Just for convenience
        self.register_buffer("thresh", torch.tensor(1.0))

    def forward(
        self, x: torch.Tensor, v: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Get state
        if v is None:
            v = torch.zeros_like(x)

        # Hidden state update
        # Sigmoid to prevent numerical instability
        h = v + torch.sigmoid(self.a) * (-v + x)
        z = spike(h - self.thresh)
        # Hard reset; detach as suggested by Zenke 2021
        v = h * (1.0 - z.detach())
        return z, v
