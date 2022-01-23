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
    total_valid = 0
    total_invalid = 0
    with tqdm.tqdm(total=len(datamodule.train_dataloader())) as pbar:
        for item_batch, none_count in datamodule.train_dataloader():
            total_valid += len(item_batch)
            total_invalid += none_count
            pbar.update(1)
            pbar.set_description(f'valid count: {total_valid}, '
                                 f'invalid cound {total_invalid},'
                                 f'percentage of invalids '
                                 f'{total_invalid / (total_valid + total_invalid)}')


if __name__ == "__main__":
    sample_datamodule()
