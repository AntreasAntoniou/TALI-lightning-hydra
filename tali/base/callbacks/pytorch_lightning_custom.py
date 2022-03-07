from typing import List, Dict

from pytorch_lightning import Callback, Trainer, LightningModule


class RunValidationOnTrainStart(Callback):
    def __init__(self):
        super().__init__()

    def on_train_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        trainer.validate(model=pl_module, val_dataloaders=pl_module.val_dataloader())
