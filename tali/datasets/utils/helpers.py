import logging
import os
import pathlib
import signal
from functools import wraps
import concurrent.futures

import numpy as np
import torch
from torch.utils.data import dataloader

from preprocessing_scripts.convert_audiofiles_to_npz import path_to_string

log = logging.getLogger(__name__)


class SubSampleVideoFrames(torch.nn.Module):
    """Crops the given image at the center.
    If the image is torch Tensor, it is expected
    to have [..., H, W] shape, where ... means an arbitrary number of leading dimensions.
    If image size is smaller than output size along any edge, image is padded with 0 and then center cropped.

    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made. If provided a sequence of length 1, it will be interpreted as (size[0], size[0]).
    """

    def __init__(self, num_frames):
        self.num_frames = num_frames
        super().__init__()

    def forward(self, sequence_of_frames):
        """
        Args:
            img (PIL Image or Tensor): Image to be cropped.

        Returns:
            PIL Image or Tensor: Cropped image.
        """
        total_num_frames = sequence_of_frames.shape[0]
        maximum_start_point = total_num_frames - self.num_frames

        if maximum_start_point < 0:
            padding = torch.zeros(
                size=(
                    -maximum_start_point,
                    sequence_of_frames.shape[1],
                    sequence_of_frames.shape[2],
                    sequence_of_frames.shape[3],
                )
            )
            sequence_of_frames = torch.cat([sequence_of_frames, padding], dim=0)

        else:
            choose_start_point = torch.randint(
                low=0, high=maximum_start_point + 1, size=(1,)
            )[0]

            sequence_of_frames = sequence_of_frames[
                choose_start_point : choose_start_point + self.num_frames
            ]

        # log.debug(sequence_of_frames.shape)

        return sequence_of_frames

    def __repr__(self):
        return self.__class__.__name__ + "(size={0})".format(self.size)


class SubSampleAudioFrames(torch.nn.Module):
    """Crops the given image at the center.
    If the image is torch Tensor, it is expected
    to have [..., H, W] shape, where ... means an arbitrary number of leading dimensions.
    If image size is smaller than output size along any edge, image is padded with 0 and then center cropped.

    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made. If provided a sequence of length 1, it will be interpreted as (size[0], size[0]).
    """

    def __init__(self, num_frames):
        self.num_frames = num_frames
        super().__init__()

    def forward(self, sequence_of_audio_frames):
        """
        Args:
            img (PIL Image or Tensor): Image to be cropped.

        Returns:
            PIL Image or Tensor: Cropped image.
        """
        # logging.info(f"{sequence_of_audio_frames.shape} {self.num_frames}")
        total_num_frames = sequence_of_audio_frames.shape[1]

        if self.num_frames <= total_num_frames:
            return sequence_of_audio_frames[:, 0 : self.num_frames]

        padding_size = self.num_frames - total_num_frames

        sequence_of_audio_frames = torch.cat(
            [
                sequence_of_audio_frames,
                torch.zeros(sequence_of_audio_frames.shape[0], padding_size),
            ],
            dim=1,
        )
        return sequence_of_audio_frames

    def __repr__(self):
        return self.__class__.__name__ + "(size={0})".format(self.size)


def timeout(timeout_secs: int):
    def wrapper(func):
        @wraps(func)
        def time_limited(*args, **kwargs):
            # Register an handler for the timeout
            def handler(signum, frame):
                raise Exception(f"Timeout for function '{func.__name__}'")

            # Register the signal function handler
            signal.signal(signal.SIGALRM, handler)

            # Define a timeout for your function
            signal.alarm(timeout_secs)

            result = None
            try:
                result = func(*args, **kwargs)
            except Exception as exc:
                logging.error(f"Exploded due to time out on {args, kwargs}")
                raise exc
            finally:
                # disable the signal alarm
                signal.alarm(0)

            return result

        return time_limited

    return wrapper


def collect_subclip_data(input_tuple):

    filepath, json_filepath = input_tuple

    if not isinstance(filepath, pathlib.Path):
        filepath = pathlib.Path(filepath)

    if not isinstance(json_filepath, pathlib.Path):
        json_filepath = pathlib.Path(json_filepath)

    frame_list = list(filepath.glob("**/*.jpg"))

    # log.info(f"{len(frame_list)} frames found in {filepath}")

    if len(frame_list) > 0:
        frame_idx_to_filepath = {
            int(frame_filepath.name.split("_")[-1].replace(".jpg", "")): path_to_string(
                frame_filepath
            )
            for frame_filepath in frame_list
        }

        frame_idx_to_filepath = {
            k: v for k, v in sorted(list(frame_idx_to_filepath.items()))
        }
        frame_list = list(frame_idx_to_filepath.values())
        audio_data_filepath = filepath.with_suffix(".npz")
        audio_data_raw_filepath = filepath.with_suffix(".aac")

        if (
            filepath.exists()
            and (audio_data_filepath.exists() or audio_data_raw_filepath.exists())
            and json_filepath.exists()
        ):
            data_tuple = (
                frame_list,
                path_to_string(filepath),
                path_to_string(audio_data_filepath),
                path_to_string(json_filepath),
            )

            return data_tuple

    return None


import multiprocessing as mp


def collect_files(args):
    # sourcery skip: identity-comprehension, simplify-len-comparison, use-named-expression
    json_file_path, training_set_size_fraction_value = args
    video_files = list(pathlib.Path(json_file_path.parent).glob("**/*.frames"))
    video_files_new = []

    for file in video_files:
        roll = np.random.random()
        log.info(
            f"{roll} {training_set_size_fraction_value} {roll <= training_set_size_fraction_value}"
        )
        if roll <= training_set_size_fraction_value:
            video_files_new.append(file)

    log.info(f"{len(video_files)} {len(video_files_new)}")

    video_key = json_file_path.parent.stem
    media_tuples = []
    multiprocessing_tuple = [(filepath, json_file_path) for filepath in video_files_new]
    with concurrent.futures.ProcessPoolExecutor(max_workers=2) as executor:
        for data_tuple in executor.map(collect_subclip_data, multiprocessing_tuple):
            if data_tuple is not None:
                media_tuples.append(data_tuple)
            # log.info(f"{len(video_files)} {len(video_files_new)} {len(media_tuples)}")

    return video_key, media_tuples


def collate_resample_none(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return dataloader.default_collate(batch)
