import logging as log
import multiprocessing as mp
from typing import Any, Optional

import torchvision.transforms as transforms
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

import tali
from tali.config_repository import TALIDatasetConfig
from tali.datasets.datasets import TALIMultiModalDataset
from tali.datasets.tokenizers import HuggingFaceBPETokenizer
from tali.datasets.utils.helpers import SubSampleAudioFrames, SubSampleVideoFrames


class BaseDataModule(LightningDataModule):
    def __init__(self, **kwargs):
        super(BaseDataModule, self).__init__()

    def prepare_data(self, **kwargs):
        raise NotImplementedError

    def configure_dataloaders(self, **kwargs):
        raise NotImplementedError

    def setup(self, stage: Optional[str] = None):
        raise NotImplementedError

    @staticmethod
    def add_dataset_specific_args(self):
        raise NotImplementedError

    def dummy_dataloader(self):
        raise NotImplementedError

    def train_dataloader(self):
        raise NotImplementedError

    def val_dataloader(self):
        raise NotImplementedError

    def test_dataloader(self):
        raise NotImplementedError


class TALIDataModule(BaseDataModule):
    def __init__(
        self,
        config: Any = TALIDatasetConfig,
        persistent_workers: bool = False,
        pin_memory: bool = True,
        prefetch_factor: int = 2,
        batch_size: int = 32,
        num_workers: int = mp.cpu_count(),
        shuffle_train: bool = True,
        shuffle_eval: bool = False,
        train_start_index: int = 0,
        val_start_index: int = 0,
        test_start_index: int = 0,
    ):
        super(TALIDataModule, self).__init__()
        self.save_hyperparameters(logger=False)
        self.config = config
        self.batch_size = batch_size
        self.persistent_workers = persistent_workers
        self.pin_memory = pin_memory
        self.prefetch_factor = prefetch_factor
        self.num_workers = num_workers
        self.shuffle_train = shuffle_train
        self.shuffle_eval = shuffle_eval
        self.train_start_index = train_start_index
        self.val_start_index = val_start_index
        self.test_start_index = test_start_index
        self.tokenizer = HuggingFaceBPETokenizer(
            context_length=config.text_context_length
        )

        self.transform_train = {
            "image": [],
            "audio": transforms.Compose(
                [
                    SubSampleAudioFrames(
                        num_frames=config.num_audio_frames_per_datapoint
                    ),
                ]
            ),
            "video": [
                SubSampleVideoFrames(num_frames=config.num_video_frames_per_datapoint)
            ],
            "text": transforms.Compose(
                [
                    lambda x: self.tokenizer.forward(x),
                ]
            ),
        }

        self.transform_eval = {
            "image": [],
            "audio": transforms.Compose(
                [
                    SubSampleAudioFrames(
                        num_frames=config.num_audio_frames_per_datapoint
                    ),
                ]
            ),
            "video": [
                SubSampleVideoFrames(num_frames=config.num_video_frames_per_datapoint)
            ],
            "text": transforms.Compose(
                [
                    lambda x: self.tokenizer.forward(x),
                ]
            ),
        }

    def prepare_data(self, **kwargs):
        # download
        pass

    def setup(self, stage: Optional[str] = None):

        if stage == "fit" or stage is None:
            self.val_set = TALIMultiModalDataset(
                config=self.config,
                set_name="val",
                transforms=self.transform_eval,
                start_index=self.val_start_index,
            )

            self.train_set = TALIMultiModalDataset(
                config=self.config,
                set_name="train",
                transforms=self.transform_train,
                start_index=self.train_start_index,
            )

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.test_set = TALIMultiModalDataset(
                config=self.config,
                set_name="test",
                transforms=self.transform_eval,
                start_index=self.test_start_index,
            )

    def dummy_dataloader(self):
        return DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            prefetch_factor=self.prefetch_factor,
            collate_fn=tali.datasets.utils.helpers.collate_resample_none,
            persistent_workers=self.persistent_workers,
            drop_last=True,
        )

    def train_dataloader(self):

        return DataLoader(
            dataset=self.train_set,
            batch_size=self.batch_size,
            shuffle=self.shuffle_train,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            prefetch_factor=self.prefetch_factor,
            collate_fn=tali.datasets.utils.helpers.collate_resample_none,
            persistent_workers=self.persistent_workers,
            drop_last=True,
        )

    def val_dataloader(self):

        return DataLoader(
            dataset=self.val_set,
            batch_size=self.batch_size,
            shuffle=self.shuffle_eval,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            prefetch_factor=self.prefetch_factor,
            collate_fn=tali.datasets.utils.helpers.collate_resample_none,
            persistent_workers=self.persistent_workers,
            drop_last=True,
        )

    def test_dataloader(self):

        return DataLoader(
            dataset=self.test_set,
            batch_size=self.batch_size,
            shuffle=self.shuffle_eval,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            prefetch_factor=self.prefetch_factor,
            collate_fn=tali.datasets.utils.helpers.collate_resample_none,
            persistent_workers=self.persistent_workers,
            drop_last=True,
        )
