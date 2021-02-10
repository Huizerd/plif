import hydra
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from hydra.utils import instantiate
from omegaconf import OmegaConf

from plif.model import SpikingClassifier


@hydra.main(config_path="conf", config_name="config")
def main(cfg):

    # Seed
    pl.seed_everything(cfg.training.seed)

    # Could decrease training speed
    # Also see https://github.com/pytorch/pytorch/issues/38342#issuecomment-644324034
    torch._C._jit_set_profiling_executor(False)
    torch._C._jit_set_profiling_mode(False)

    # Dataset
    dm = instantiate(cfg.dataset)
    # Model
    model = SpikingClassifier(
        cfg.model,
        cfg.training.script,
        cfg.training.batch_size,
        cfg.training.lr,
        cfg.training.optimizer,
        cfg.training.scheduler,
    )

    # Logging
    logger = WandbLogger(
        **cfg.logging, config=OmegaConf.to_container(cfg, resolve=True)
    )
    # Watch model
    # logger.watch(model)

    # Training
    trainer = pl.Trainer(**cfg.trainer, logger=logger)
    trainer.tune(model, datamodule=dm)
    trainer.fit(model, datamodule=dm)

    # Testing
    # trainer.test(datamodule=dm)


if __name__ == "__main__":
    main()
