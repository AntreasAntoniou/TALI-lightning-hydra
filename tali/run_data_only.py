import logging

import hydra
import torch
import tqdm
from omegaconf import DictConfig
from pytorch_lightning import seed_everything

from tali.utils.storage import pretty_print_dict

log = logging.getLogger(__name__)


@hydra.main(config_path="configs", config_name="default-a100-2g")
def sample_datamodule(config: DictConfig):

    seed_everything(config.seed, workers=True)

    log.info(f"{pretty_print_dict(dict(config))}")

    datamodule = hydra.utils.instantiate(config.data, _recursive_=False)
    datamodule.setup(stage="fit")
    with tqdm.tqdm(total=len(datamodule.train_dataloader())) as pbar:
        for item_batch in datamodule.train_dataloader():
            pbar.update(1)


if __name__ == "__main__":
    sample_datamodule()
