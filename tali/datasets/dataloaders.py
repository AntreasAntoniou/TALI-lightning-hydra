from typing import Any

from torch.utils.data import Dataset

DataLoader(
    dataset=self.train_set,
    batch_size=self.batch_size,
    shuffle=self.shuffle_train,
    num_workers=self.num_workers,
    pin_memory=self.pin_memory,
    prefetch_factor=self.prefetch_factor,
    collate_fn=collate_fn,
    persistent_workers=self.persistent_workers,
    drop_last=True,
)


class SampleParallelismDataLoader(object):
    def __init__(
        self,
        dataset: Dataset,
        batch_size: int,
        shuffle: bool,
        num_workers: int,
        pin_memory: bool,
        prefetch_factor: int,
        collate_fn: Any,
        persistent_workers: bool,
        drop_last: bool,
    ):
        self.dataset = dataset  # type: Dataset
