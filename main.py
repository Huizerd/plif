import hydra
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from hydra.utils import instantiate
from omegaconf import OmegaConf

from plif.model import Classifier
from plif.utils import zeromean_unitvar_transform


@hydra.main(config_path="conf", config_name="config")
def main(cfg):

    # Seed
    pl.seed_everything(cfg.training.seed)

    # Increases speed a bit
    # Also see https://github.com/pytorch/pytorch/issues/38342#issuecomment-644324034
    torch._C._jit_set_profiling_executor(False)
    torch._C._jit_set_profiling_mode(False)

    # Dataset
    # Call transform here until Hydra supports recursive calls
    tf = zeromean_unitvar_transform()
    dm = instantiate(
        cfg.dataset, train_transforms=tf, val_transforms=tf, test_transforms=tf
    )
    # Model
    model = Classifier(
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
    # Watch model if no TorchScript
    if not cfg.training.script:
        logger.watch(model, log="all")

    # Training
    trainer = pl.Trainer(**cfg.trainer, logger=logger)
    trainer.tune(model, datamodule=dm)  # tunes training if desired
    trainer.fit(model, datamodule=dm)

    # Testing: only do once!
    # trainer.test(datamodule=dm)


if __name__ == "__main__":
    main()
