import logging
from time import sleep

import hydra
import torch
import tqdm
from omegaconf import DictConfig
import matplotlib.pyplot as plt
from pytorch_lightning import seed_everything, LightningDataModule
from torchvision.utils import make_grid
import numpy as np

from tali.utils.storage import pretty_print_dict

log = logging.getLogger(__name__)


def plot_with_spectrum(x, rate=48000):
    """Plot the given waveform (timeSeries), both as time-domain and as its
    frequency-domain spectrum. Returns a matplotlib.figure.Figure object."""
    fig, axs = plt.subplots(2)
    n = len(x)
    # (1) Plot time-domain data:
    timesMsec = np.arange(n) * 1000.0 / rate
    axs[0].plot(timesMsec, x)
    # Limit the X axis to our input samples:
    axs[0].set_xlabel("Time (ms)")
    axs[0].grid(True)
    # (2) Compute and plot frequency spectrum:
    return fig


def sample_datamodule(config: DictConfig):
    seed_everything(config.seed, workers=True)

    log.info(f"{pretty_print_dict(dict(config))}")

    datamodule: LightningDataModule = hydra.utils.instantiate(
        config.datamodule, _recursive_=False
    )
    datamodule.setup(stage="fit")
    datamodule.setup(stage='test')
    total_valid = 0
    total_invalid = 0
    for _ in range(config.trainer.max_epochs):
        total_samples = len(datamodule.val_dataloader()) * config.batch_size
        with tqdm.tqdm(total=len(datamodule.val_dataloader()), smoothing=0.0) as pbar:
            for idx, item_batch in enumerate(datamodule.val_dataloader()):
                sleep(0.5)
                total_valid += len(item_batch['image'])
                total = (idx + 1) * config.batch_size
                total_invalid = total - total_valid
                pbar.update(1)
                pbar.set_description(f'valid count: {total_valid}, '
                                     f'invalid count: {total_invalid}, '
                                     f'percentage of invalid: '
                                     f'{total_invalid / (total_valid + total_invalid)}')

        total_samples = len(datamodule.train_dataloader()) * config.batch_size
        with tqdm.tqdm(total=len(datamodule.train_dataloader()), smoothing=0.0) as pbar:
            for idx, item_batch in enumerate(datamodule.train_dataloader()):
                sleep(0.5)
                total_valid += len(item_batch['image'])
                total = (idx + 1) * config.batch_size
                total_invalid = total - total_valid
                pbar.update(1)
                pbar.set_description(f'valid count: {total_valid}, '
                                     f'invalid count: {total_invalid}, '
                                     f'percentage of invalid: '
                                     f'{total_invalid / (total_valid + total_invalid)}')

        total_samples = len(datamodule.test_dataloader()) * config.batch_size
        with tqdm.tqdm(total=len(datamodule.test_dataloader()), smoothing=0.0) as pbar:
            for idx, item_batch in enumerate(datamodule.test_dataloader()):
                sleep(0.5)
                total_valid += len(item_batch['image'])
                total = (idx + 1) * config.batch_size
                total_invalid = total - total_valid
                pbar.update(1)
                pbar.set_description(f'valid count: {total_valid}, '
                                     f'invalid count: {total_invalid}, '
                                     f'percentage of invalid: '
                                     f'{total_invalid / (total_valid + total_invalid)}')


            # log.info(item_batch)

            # grid = make_grid(
            #     torch.cat(
            #         [
            #             item_batch["image"],
            #             item_batch["video"].view(
            #                 -1,
            #                 *item_batch["video"].shape[2:],
            #             ),
            #         ],
            #         dim=0,
            #     ),
            #     normalize=True,
            #     value_range=None,
            #     scale_each=True,
            #     pad_value=0,
            #     nrow=30,
            # )
            #
            # for item_idx, audio_item in enumerate(item_batch["audio"]):
            #     logging.info(f'{torch.mean(audio_item), torch.std(audio_item)}, ')
            #     # torchaudio.save(
            #     #     f"save_example_default_{item_idx}.wav",
            #     #     audio_item.permute([1, 0]),
            #     #     44100,
            #     # )
            #     for channel in range(2):
            #         fig = plotWithSpectrum(audio_item[channel], rate=44100)
            #         fig.savefig(f"{config.exp_dir}/{item_idx}_{channel}.png")
            #         plt.close()
            #
            # plt.imshow(grid.permute([1, 2, 0]).cpu().numpy())
            # plt.show()
            # input("Press Enter to continue...")
