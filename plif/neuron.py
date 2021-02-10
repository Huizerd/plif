from typing import Any, List, Tuple, Union
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
    def __init__(self, shape: List[int], a_0: float):
        super().__init__()

        # Shared between neurons in a layer
        self.a = nn.Parameter(torch.tensor(a_0))
        # Just for convenience
        self.register_buffer("thresh", torch.tensor(1.0))
        # Opted for saving state here instead of in network
        self.register_buffer("v", torch.zeros(*shape))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Hidden state update
        # Sigmoid to prevent numerical instability
        h = self.v + 1.0 / torch.sigmoid(self.a) * (-self.v + x)
        s = spike(h - self.thresh)
        # Hard reset; detach as suggested by Zenke 2021
        self.v = h * (1.0 - s.detach())
        return s

    def reset(self):
        # XXX: is this needed?
        with torch.no_grad():
            self.v.fill_(0)


class StatelessPLIF(nn.Module):
    def __init__(self, a_0: float):
        super().__init__()

        # Shared between neurons in a layer
        self.a = nn.Parameter(torch.tensor(a_0))
        # Just for convenience
        self.register_buffer("thresh", torch.tensor(1.0))

    def forward(
        self, x: torch.Tensor, v: Union[torch.Tensor, None]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Get state
        if v is None:
            v = torch.zeros_like(x)

        # Hidden state update
        # Sigmoid to prevent numerical instability
        h = v + 1.0 / torch.sigmoid(self.a) * (-v + x)
        z = spike(h - self.thresh)
        # Hard reset; detach as suggested by Zenke 2021
        v = h * (1.0 - z.detach())
        return z, v


class StatelessPLIFScripted(nn.Module):
    def __init__(self, a_0: float):
        super().__init__()

        # Shared between neurons in a layer
        self.a = nn.Parameter(torch.tensor(a_0))
        # Just for convenience
        self.register_buffer("thresh", torch.tensor(1.0))

    def forward(
        self, x: torch.Tensor, v: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Get state
        if v.dim() == 0:
            v = torch.zeros_like(x)

        # Hidden state update
        # Sigmoid to prevent numerical instability
        h = v + 1.0 / torch.sigmoid(self.a) * (-v + x)
        z = spike(h - self.thresh)
        # Hard reset; detach as suggested by Zenke 2021
        v = h * (1.0 - z.detach())
        return z, v


class Dummy(nn.Module):
    def __init__(self, shape: List[int]):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def reset(self):
        pass
