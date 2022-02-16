import os
import threading
from concurrent.futures import ProcessPoolExecutor, as_completed
from random import shuffle
from time import sleep
from typing import Any
import numpy as np
from torch.utils.data import Dataset


class SampleParallelismDataLoader(object):
    def __init__(
        self,
        dataset: Dataset,
        batch_size: int,
        shuffle: bool,
        num_workers: int,
        prefetch_factor: int,
        collate_fn: Any,
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
            shuffle(index_cache)

        return index_cache

    def run_data_sampler(self, stop_event):

        cur_batch = []
        self.index_cache = self.resample_index_cache()
        with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
            while True:
                if len(self.index_cache) < self.batch_size:
                    self.data_loader_done = True
                    continue
                chosen_indexes = np.random.choice(
                    a=list(self.index_cache), size=self.batch_size, replace=False
                )
                for index in chosen_indexes:
                    self.index_cache.remove(index)
                print("calling futures")
                futures = [
                    executor.submit(self.dataset.__getitem__, i) for i in chosen_indexes
                ]
                print("waiting for futures")
                for idx, sample in enumerate(as_completed(futures)):
                    print(idx)
                    result = sample.result()
                    print(result)
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
        print("Starting data sampler")
        while len(self.batch_cache) == 0:
            pass

        return self.batch_cache.pop(0)

    def __len__(self):
        return len(self.dataset) // self.batch_size

    def __iter__(self):
        return self
