import torch
import torch.nn as nn


class AvgPoolDecoding(nn.Module):
    def __init__(self, kernel: int, stride: int):
        super().__init__()

        # Average pooling over last dimension
        self.pool = nn.AvgPool1d(kernel, stride)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Assumes (batch, seq, neurons), pools over neurons
        return self.pool(x)
