import os

import pytest
import torch
import tqdm
from torch.utils.data import default_collate

from tali.config_repository import TALIDatasetConfig, ImageShape, ModalityConfig
from tali.datasets.dataloaders import SampleParallelismDataLoader
from tali.datasets.datasets import DummyMultiModalDataset


@pytest.mark.parametrize("batch_size", [32])
@pytest.mark.parametrize("shuffle", [False, True])
@pytest.mark.parametrize("num_workers", [8])
@pytest.mark.parametrize("prefetch_factor", [1, 8])
@pytest.mark.parametrize("num_samples", [1000])
def test_data_loader(batch_size, shuffle, num_workers, prefetch_factor, num_samples):
    dataset = DummyMultiModalDataset(
        config=TALIDatasetConfig(
            modality_config=ModalityConfig(),
            num_video_frames_per_datapoint=10,
            num_audio_frames_per_datapoint=88200,
            num_audio_sample_rate=44100,
            image_shape=ImageShape(channels=3, width=224, height=224),
            text_context_length=77,
        ),
    )
    collate_fn = default_collate
    dataloader = SampleParallelismDataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        collate_fn=collate_fn,
    )
    print(f"Start sampling {len(dataloader)} batches")
    with tqdm.tqdm(total=len(dataloader)) as pbar:
        for i, batch in enumerate(dataloader):
            assert len(batch) == 4
            print(len(batch))
            pbar.update(1)
