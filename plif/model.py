from collections import OrderedDict

import torch
import torch.jit as jit
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from hydra.utils import instantiate
from omegaconf import DictConfig

from plif.utils import SequentialState


class SNN(nn.Module):
    """
    Architecture used for *MNIST datasets.
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
        self.conv1 = nn.Sequential(
            OrderedDict(
                [
                    # No biases anywhere
                    ("conv", nn.Conv2d(1, 128, 3, 1, bias=False)),
                    # Batch norm per-channel, so identical for all timesteps
                    ("bn", nn.BatchNorm2d(128)),
                    ("plif", instantiate(neuron, [128, 26, 26])),
                    ("pool", nn.MaxPool2d(2, 2)),
                ]
            )
        )

        # Conv block 2
        self.conv2 = nn.Sequential(
            OrderedDict(
                [
                    ("conv", nn.Conv2d(128, 128, 3, 1, bias=False)),
                    ("bn", nn.BatchNorm2d(128)),
                    ("plif", instantiate(neuron, [128, 11, 11])),
                    ("pool", nn.MaxPool2d(2, 2)),
                    ("do", nn.Dropout2d()),
                ]
            )
        )

        # FC block
        self.fc = nn.Sequential(
            OrderedDict(
                [
                    ("fc1", nn.Linear(128 * 5 * 5, 2048, bias=False)),
                    ("plif1", instantiate(neuron, [2048])),
                    ("do", nn.Dropout()),
                    ("fc2", nn.Linear(2048, 100, bias=False)),
                    ("plif2", instantiate(neuron, [100])),
                ]
            )
        )

        # Decoder
        # XXX: pool only spatially?
        self.decoder = instantiate(decoder)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (batch, channel, height, width)
        batch, _, _, _ = x.shape

        # Reset neurons
        # TODO: make nicer
        self.conv1.plif.reset()
        self.conv2.plif.reset()
        self.fc.plif1.reset()
        self.fc.plif2.reset()

        # out = []

        for _ in range(self.seq_len):
            s = self.encoder(x)
            s = self.conv1(s)
            s = self.conv2(s)
            s = s.view(batch, -1)
            s = self.fc(s)
            s = s.view(batch, 1, -1)
            # XXX: or also pool temporally?
            # out.append(self.decoder(s))
            s = self.decoder(s)

        return s.view(batch, -1)
        # return torch.stack(out).mean(0).view(batch, -1)


class StatelessSNN(nn.Module):
    """
    Architecture used for *MNIST datasets, with stateless neurons.
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
        self.conv1 = SequentialState(
            OrderedDict(
                [
                    # No biases anywhere
                    ("conv", nn.Conv2d(1, 128, 3, 1, bias=False)),
                    # Batch norm per-channel, so identical for all timesteps
                    ("bn", nn.BatchNorm2d(128)),
                    ("plif", instantiate(neuron)),
                    ("pool", nn.MaxPool2d(2, 2)),
                ]
            )
        )

        # Conv block 2
        self.conv2 = SequentialState(
            OrderedDict(
                [
                    ("conv", nn.Conv2d(128, 128, 3, 1, bias=False)),
                    ("bn", nn.BatchNorm2d(128)),
                    ("plif", instantiate(neuron)),
                    ("pool", nn.MaxPool2d(2, 2)),
                    ("do", nn.Dropout2d()),
                ]
            )
        )

        # FC block
        self.fc = SequentialState(
            OrderedDict(
                [
                    ("fc1", nn.Linear(128 * 5 * 5, 2048, bias=False)),
                    ("plif1", instantiate(neuron)),
                    ("do", nn.Dropout()),
                    ("fc2", nn.Linear(2048, 100, bias=False)),
                    ("plif2", instantiate(neuron)),
                ]
            )
        )

        # Decoder
        # XXX: pool only spatially?
        self.decoder = instantiate(decoder)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (batch, channel, height, width)
        batch, _, _, _ = x.shape

        s_conv1, s_conv2, s_fc = None, None, None
        for _ in range(self.seq_len):
            z = self.encoder(x)
            z, s_conv1 = self.conv1(z, s_conv1)
            z, s_conv2 = self.conv2(z, s_conv2)
            z = z.view(batch, -1)
            z, s_fc = self.fc(z, s_fc)
            z = z.view(batch, 1, -1)
            # XXX: or also pool temporally?
            z = self.decoder(z)

        return z.view(batch, -1)


class SpikingClassifier(pl.LightningModule):
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

        self.log(f"{prefix} loss", loss)
        self.log(f"{prefix} acc", self.accuracy(logits, y))
        return loss

    def configure_optimizers(self):
        optimizer = instantiate(self.optim, self.parameters(), self.lr)
        scheduler = instantiate(self.sched, optimizer)

        if scheduler is not None:
            return [optimizer], [scheduler]
        else:
            return optimizer
