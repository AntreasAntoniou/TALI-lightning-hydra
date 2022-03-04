import functools
import logging as log
import multiprocessing as mp
from dataclasses import dataclass
from typing import Any, Optional

import torchvision.transforms as transforms
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

import tali
from tali.config_repository import (
    DatasetConfig,
    DataLoaderConfig,
    DatasetDirectoryConfig,
)
from tali.datasets.datasets import TALIMultiModalDataset, DummyMultiModalDataset
from tali.datasets.tokenizers import HuggingFaceBPETokenizer
from tali.datasets.utils.helpers import (
    SubSampleAudioFrames,
    SubSampleVideoFrames,
    collate_fn_replace_corrupted,
)


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
        dataset_config: DatasetConfig,
        dataloader_config: DataLoaderConfig,
    ):
        super().__init__()
        self.dataset_config = dataset_config
        self.datamodule_config = dataloader_config

        self.dataset_class = (
            DummyMultiModalDataset
            if self.datamodule_config.use_dummy_dataloader
            else TALIMultiModalDataset
        )
        self.tokenizer = HuggingFaceBPETokenizer(
            context_length=dataset_config.text_context_length
        )

        self.transform_train = {
            "image": [],
            "audio": transforms.Compose(
                [
                    SubSampleAudioFrames(
                        num_frames=dataset_config.num_audio_frames_per_datapoint
                    ),
                ]
            ),
            "video": [
                SubSampleVideoFrames(
                    num_frames=dataset_config.num_video_frames_per_datapoint
                )
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
                        num_frames=dataset_config.num_audio_frames_per_datapoint
                    ),
                ]
            ),
            "video": [
                SubSampleVideoFrames(
                    num_frames=dataset_config.num_video_frames_per_datapoint
                )
            ],
            "text": transforms.Compose(
                [
                    lambda x: self.tokenizer.forward(x),
                ]
            ),
        }
        self.save_hyperparameters(logger=True)

    def prepare_data(self, **kwargs):
        # download
        pass

    def setup(self, stage: Optional[str] = None):

        if stage == "fit" or stage is None:
            self.val_set = self.dataset_class(
                config=self.dataset_config,
                set_name="val",
                transforms=self.transform_eval,
                start_index=self.datamodule_config.val_start_index,
                num_samples=self.datamodule_config.val_num_samples,
            )

            self.train_set = self.dataset_class(
                config=self.dataset_config,
                set_name="train",
                transforms=self.transform_train,
                start_index=self.datamodule_config.train_start_index,
                num_samples=self.datamodule_config.train_num_samples,
            )

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.test_set = self.dataset_class(
                config=self.dataset_config,
                set_name="test",
                transforms=self.transform_eval,
                start_index=self.datamodule_config.test_start_index,
                num_samples=self.datamodule_config.test_num_samples,
            )

    def train_dataloader(self):

        collate_fn = functools.partial(
            collate_fn_replace_corrupted, dataset=self.train_set
        )

        return DataLoader(
            dataset=self.train_set,
            batch_size=self.datamodule_config.batch_size,
            shuffle=self.datamodule_config.shuffle_train,
            num_workers=self.datamodule_config.num_workers,
            pin_memory=self.datamodule_config.pin_memory,
            prefetch_factor=self.datamodule_config.prefetch_factor,
            collate_fn=collate_fn,
            persistent_workers=self.datamodule_config.persistent_workers,
            drop_last=True,
        )

    def val_dataloader(self):

        collate_fn = functools.partial(
            collate_fn_replace_corrupted, dataset=self.val_set
        )

        return DataLoader(
            dataset=self.val_set,
            batch_size=self.datamodule_config.batch_size,
            shuffle=self.datamodule_config.shuffle_eval,
            num_workers=self.datamodule_config.num_workers,
            pin_memory=self.datamodule_config.pin_memory,
            prefetch_factor=self.datamodule_config.prefetch_factor,
            collate_fn=collate_fn,
            persistent_workers=self.datamodule_config.persistent_workers,
            drop_last=True,
        )

    def test_dataloader(self):

        collate_fn = functools.partial(
            collate_fn_replace_corrupted, dataset=self.test_set
        )

        return DataLoader(
            dataset=self.test_set,
            batch_size=self.datamodule_config.batch_size,
            shuffle=self.datamodule_config.shuffle_eval,
            num_workers=self.datamodule_config.num_workers,
            pin_memory=self.datamodule_config.pin_memory,
            prefetch_factor=self.datamodule_config.prefetch_factor,
            collate_fn=collate_fn,
            persistent_workers=self.datamodule_config.persistent_workers,
            drop_last=True,
        )
