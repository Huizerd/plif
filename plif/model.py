from typing import List, Optional

import torch
import torch.jit as jit
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from hydra.utils import instantiate
from omegaconf import DictConfig


class RegularModelMNIST(nn.Module):
    """
    Regular ANN architecture used for *MNIST datasets.
    """

    def __init__(
        self,
        seq_len: int,
        encoder: DictConfig,
        neuron: DictConfig,
        decoder: DictConfig,
    ):
        super().__init__()

        # Number of time steps
        self.seq_len = seq_len

        # Encoder
        self.encoder = instantiate(encoder)

        # Conv block 1
        # No biases anywhere
        self.conv1 = nn.Conv2d(1, 128, 3, 1, bias=False)
        # Batch norm per-channel, so identical for all timesteps
        self.bn1 = nn.BatchNorm2d(128)
        self.pool1 = nn.MaxPool2d(2, 2)

        # Conv block 2
        self.conv2 = nn.Conv2d(128, 128, 3, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.do1 = nn.Dropout2d()

        # FC block
        self.fc1 = nn.Linear(128 * 5 * 5, 2048, bias=False)
        self.do2 = nn.Dropout()
        self.fc2 = nn.Linear(2048, 100, bias=False)

        # Decoder
        # Pools spatially
        self.decoder = instantiate(decoder)

    def forward(self, x_in: torch.Tensor) -> torch.Tensor:
        # (batch, channel, height, width)
        batch, _, _, _ = x_in.shape

        out: List[torch.Tensor] = []
        for _ in range(self.seq_len):
            # Encoder
            x = self.encoder(x_in)

            # Conv block 1
            x = self.conv1(x)
            x = self.bn1(x)
            x = F.relu(x)
            x = self.pool1(x)

            # Conv block 2
            x = self.conv2(x)
            x = self.bn2(x)
            x = F.relu(x)
            x = self.pool2(x)
            x = self.do1(x)

            # FC block
            x = x.view(batch, -1)
            x = self.fc1(x)
            x = F.relu(x)
            x = self.do2(x)
            x = self.fc2(x)
            x = F.relu(x)

            # Decoder
            x = x.view(batch, 1, -1)
            out += [self.decoder(x)]

        # Sum over time (temporal pooling)
        return torch.stack(out).sum(0).view(batch, -1)


class SpikingModelMNIST(nn.Module):
    """
    SNN architecture used for *MNIST datasets.
    """

    def __init__(
        self,
        seq_len: int,
        encoder: DictConfig,
        neuron: DictConfig,
        decoder: DictConfig,
    ):
        super().__init__()

        # Number of time steps
        self.seq_len = seq_len

        # Encoder
        self.encoder = instantiate(encoder)

        # Conv block 1
        # No biases anywhere
        self.conv1 = nn.Conv2d(1, 128, 3, 1, bias=False)
        # Batch norm per-channel, so identical for all timesteps
        self.bn1 = nn.BatchNorm2d(128)
        self.plif1 = instantiate(neuron)
        self.pool1 = nn.MaxPool2d(2, 2)

        # Conv block 2
        self.conv2 = nn.Conv2d(128, 128, 3, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(128)
        self.plif2 = instantiate(neuron)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.do1 = nn.Dropout2d()

        # FC block
        self.fc1 = nn.Linear(128 * 5 * 5, 2048, bias=False)
        self.plif3 = instantiate(neuron)
        self.do2 = nn.Dropout()
        self.fc2 = nn.Linear(2048, 100, bias=False)
        self.plif4 = instantiate(neuron)

        # Decoder
        # Pools spatially
        self.decoder = instantiate(decoder)

    def forward(self, x_in: torch.Tensor) -> torch.Tensor:
        # (batch, channel, height, width)
        batch, _, _, _ = x_in.shape

        s1: Optional[torch.Tensor] = None
        s2: Optional[torch.Tensor] = None
        s3: Optional[torch.Tensor] = None
        s4: Optional[torch.Tensor] = None
        out: List[torch.Tensor] = []
        for _ in range(self.seq_len):
            # Encoder
            x = self.encoder(x_in)

            # Conv block 1
            x = self.conv1(x)
            x = self.bn1(x)
            z, s1 = self.plif1(x, s1)
            x = self.pool1(z)

            # Conv block 2
            x = self.conv2(x)
            x = self.bn2(x)
            z, s2 = self.plif2(x, s2)
            x = self.pool2(z)
            x = self.do1(x)

            # FC block
            x = x.view(batch, -1)
            x = self.fc1(x)
            z, s3 = self.plif3(x, s3)
            x = self.do2(z)
            x = self.fc2(x)
            z, s4 = self.plif4(x, s4)

            # Decoder
            z = z.view(batch, 1, -1)
            out += [self.decoder(z)]

        # Sum over time (temporal pooling)
        return torch.stack(out).sum(0).view(batch, -1)


class Classifier(pl.LightningModule):
    def __init__(
        self,
        model: DictConfig,
        script: bool,
        batch_size: int,
        lr: float,
        optimizer: DictConfig,
        scheduler: DictConfig,
    ):
        super().__init__()

        self.accuracy = pl.metrics.Accuracy()

        self.model = instantiate(model)
        if script:
            self.model = jit.script(self.model)

        # These can be set by auto_scale_batch and auto_lr_find
        self.batch_size = batch_size
        self.lr = lr

        self.optim = optimizer
        self.sched = scheduler

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, batch_idx, "train")

    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, batch_idx, "val")

    def test_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, batch_idx, "test")

    def _shared_step(
        self, batch: torch.Tensor, batch_idx: int, prefix: str
    ) -> torch.Tensor:
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)

        for name, param in self.model.named_parameters():
            if "plif" in name:
                self.log(f"{prefix} {name}", param)

        # for name, param in self.model.named_parameters():
        #     self.log(f"{prefix} {name} grad", param.grad.mean())

        self.log(f"{prefix} loss", loss)
        self.log(f"{prefix} acc", self.accuracy(F.softmax(logits, 1), y))
        return loss

    def configure_optimizers(self):
        optimizer = instantiate(self.optim, self.parameters(), self.lr)
        scheduler = instantiate(self.sched, optimizer)

        if scheduler is not None:
            return [optimizer], [scheduler]
        else:
            return optimizer
