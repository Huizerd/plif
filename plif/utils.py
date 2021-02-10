import inspect
from typing import List, Optional, OrderedDict, Tuple, Union

import torch
import torch.nn as nn


class SequentialState(nn.Sequential):
    """
    Adapted from https://github.com/norse/norse/blob/master/norse/torch/module/sequential.py
    """

    def __init__(self, modules: OrderedDict[str, nn.Module]):
        super().__init__()

        self.stateful_layers = []
        for name, module in modules.items():
            self.add_module(name, module)
            # Identify all the stateful layers
            signature = inspect.signature(module.forward)
            self.stateful_layers.append("v" in signature.parameters)

    def forward(
        self, x: torch.Tensor, states: Union[List[torch.Tensor], None]
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        states = [None] * len(self) if states is None else states

        for i, module in enumerate(self):
            if self.stateful_layers[i]:
                x, state = module(x, states[i])
                states[i] = state
            else:
                x = module(x)
        return x, states


class SequentialStateScripted(nn.Sequential):
    """
    Adapted from https://github.com/norse/norse/blob/master/norse/torch/module/sequential.py
    """

    def __init__(self, modules: OrderedDict[str, nn.Module]):
        super().__init__()

        self.stateful_layers = []
        self.len = len(modules)
        for name, module in modules.items():
            self.add_module(name, module)
            # Identify all the stateful layers
            signature = inspect.signature(module.forward)
            self.stateful_layers.append("v" in signature.parameters)

    def forward(
        self, x: torch.Tensor, states: Optional[List[torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        if states is None:
            states = [torch.tensor(0, dtype=x.dtype, device=x.device)] * self.len

        print(self.stateful_layers)
        for i, module in enumerate(self):
            if self.stateful_layers[i]:
                x, state = module(x, states[i])
                states[i] = state
            else:
                x = module(x)
        return x, states
