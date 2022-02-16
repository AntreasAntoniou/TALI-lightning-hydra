import os
import threading
from concurrent.futures import ProcessPoolExecutor, as_completed, ThreadPoolExecutor
from random import shuffle
from time import sleep
from typing import Any
import numpy as np
from torch.utils.data import Dataset, DataLoader


class SampleParallelismDataLoader(object):
    def __init__(
        self,
        dataset: Dataset,
        batch_size: int,
        shuffle: bool,
        num_workers: int,
        prefetch_factor: int,
        collate_fn: Any,
        **kwargs,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor
        self.collate_fn = collate_fn
        self.index_cache = []
        self.batch_cache = []
        self.data_sampler_thread = None
        self.data_sampler_stop_event = threading.Event()
        self.data_sample_is_running = False
        self.data_loader_done = False
        self.precaching_done = False

    def start_data_sampler(self):
        if not self.data_sample_is_running:
            self.data_sampler_thread = threading.Thread(
                target=self.run_data_sampler, args=(self.data_sampler_stop_event,)
            )
            self.data_sampler_thread.daemon = True
            self.data_sampler_thread.start()
            self.data_sample_is_running = True

    def stop_data_sampler(self):
        self.data_sampler_thread.stop()
        self.data_sample_is_running = False

    def resample_index_cache(self):
        index_cache = set(list(range(len(self.dataset))))

        if self.shuffle:
            index_cache = list(index_cache)
            shuffle(index_cache)
            index_cache = set(index_cache)

        return index_cache

    def run_data_sampler(self, stop_event):

        cur_batch = []
        self.index_cache = self.resample_index_cache()
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            while stop_event.is_set() is False:
                if len(self.index_cache) < self.batch_size:
                    self.data_loader_done = True
                    continue
                num_samples_collected = len(self.batch_cache) * self.batch_size + len(
                    cur_batch
                )
                num_samples_needed = self.batch_size * self.prefetch_factor
                num_samples_missing = num_samples_needed - num_samples_collected
                num_samples_to_sample = min(num_samples_missing, len(self.index_cache))
                chosen_indexes = np.random.choice(
                    a=list(self.index_cache), size=num_samples_to_sample, replace=False
                )
                for index in chosen_indexes:
                    self.index_cache.remove(index)
                futures = [
                    executor.submit(self.dataset.__getitem__, i) for i in chosen_indexes
                ]
                for sample in as_completed(futures):
                    result = sample.result()
                    if result is not None:
                        cur_batch.append(result)
                        if len(cur_batch) == self.batch_size:
                            self.batch_cache.append(self.collate_fn(cur_batch))
                            cur_batch = []

    def __next__(self):
        if self.data_loader_done:
            self.stop_data_sampler()
            raise StopIteration

        self.start_data_sampler()
        while len(self.batch_cache) == 0:
            pass

        return self.batch_cache.pop(0)

    def __len__(self):
        return len(self.dataset) // self.batch_size

    def __iter__(self):
        return self


# class SampleParallelismDataLoader(object):
#     def __init__(
#         self,
#         dataset: Dataset,
#         batch_size: int,
#         shuffle: bool,
#         num_workers: int,
#         pin_memory: bool,
#         prefetch_factor: int,
#         collate_fn: Any,
#         persistent_workers: bool,
#         drop_last: bool,
#     ):
#         self.batch_size = batch_size
#         self.num_workers = num_workers
#         self.prefetch_factor = prefetch_factor
#         self.dataset = dataset
#         self.collate_fn = collate_fn
#         self.dataloader = DataLoader(
#             dataset=dataset,
#             batch_size=1,
#             shuffle=shuffle,
#             num_workers=num_workers,
#             pin_memory=pin_memory,
#             prefetch_factor=prefetch_factor * batch_size,
#             collate_fn=collate_fn,
#             persistent_workers=persistent_workers,
#             drop_last=drop_last,
#         )
#
#     def get_batches(self):
#         batch_cache = []
#         for batch in self.dataloader:
#             if batch is not None:
#                 batch_cache.append(batch)
#             if len(batch_cache) == self.batch_size:
#                 batch_ready = self.collate_fn(batch_cache)
#                 del batch_cache[:]
#                 yield batch_ready
#
#         raise StopIteration
#
#     def __len__(self):
#         return len(self.dataset) // self.batch_size
#
#     def __iter__(self):
#         return self
