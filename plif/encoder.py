import torch
import torch.nn as nn


class CurrentEncoding(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x
