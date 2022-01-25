import logging
import os
import pathlib
import signal
from functools import wraps
from xml.etree.ElementTree import ElementTree, fromstring

import numpy as np
import torch
from torch.utils.data import dataloader

from tali.utils.storage import load_json, save_json


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
        # logging.info(f'{sequence_of_audio_frames.shape}')
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


def load_text_into_language_time_stamps(filepath):
    filepath = f"{filepath}".replace("\n", "")
    meta_data = load_json(filepath)
    caption_data_filepath = (
        f'{filepath.replace("meta_data", "start_timestamp_to_caption_dict")}'
    )

    captions = meta_data["captions"]

    captions_matched = {
        key: value for key, value in captions.items() if key in ["a.en", "en"]
    }

    if len(captions_matched) > 1:
        selected_key = "en"
    else:
        selected_key = list(captions_matched.keys())[0]

    if os.path.exists(caption_data_filepath):
        return load_json(caption_data_filepath), selected_key

    selected_captions = captions_matched[selected_key]
    xml_tree = ElementTree(fromstring(selected_captions))

    root = list(xml_tree.iter())
    timestamp_to_caption_dict = {}

    for item in root:
        if selected_key == "a.en":
            children_text = [
                child.text.replace("\n", " ")
                for child in item
                if child.text is not None
            ]
            if item.tag == "p" and children_text:
                timestamp_to_caption_dict[
                    float(item.attrib["t"]) / 1000
                ] = children_text

        elif selected_key == "en":
            if item.tag == "p" and len(item.items()) == 2:
                [(_, start), (_, dur)] = item.items()

                timestamp_to_caption_dict[float(start) / 1000] = (
                    item.text.replace("\n", " ") if item.text is not None else ""
                )

    save_json(
        filepath=caption_data_filepath,
        metrics_dict=timestamp_to_caption_dict,
        overwrite=True,
    )

    return timestamp_to_caption_dict, selected_key


def get_text_tokens(meta_data_filepath, start_timestamp, end_timestamp):
    # logging.info(f'{start_timestamp} {end_timestamp}')
    timestamp_to_caption_dict, selected_key = load_text_into_language_time_stamps(
        filepath=meta_data_filepath
    )
    start_timestamp = float(np.floor(start_timestamp))
    end_timestamp = float(np.floor(end_timestamp))

    time_stamp_to_caption_dict = {}

    for current_start_timestamp in sorted(timestamp_to_caption_dict.keys()):
        current_start_timestamp_float = float(current_start_timestamp)
        if start_timestamp <= current_start_timestamp_float <= end_timestamp:
            time_stamp_to_caption_dict[
                current_start_timestamp_float
            ] = timestamp_to_caption_dict[current_start_timestamp]

        if current_start_timestamp_float > end_timestamp:
            break

    # logging.info(f"{merged_caption_text}")
    return time_stamp_to_caption_dict


def get_text_tokens_from_store(
    caption_store_dict, video_id, start_timestamp, end_timestamp
):
    caption_item = caption_store_dict[video_id]

    caption_store_time_indexes = caption_item["time_idx_to_caption_store_idx"][
        start_timestamp:end_timestamp
    ]

    return "".join(
        [caption_item["captions"][key] for key in set(caption_store_time_indexes)]
    )


def sample_frame_indexes_to_collect(
    video_length_in_frames,
    video_fps,
    num_video_frames_per_datapoint,
    num_audio_frames_per_datapoint,
    num_audio_sample_rate,
    start_point_in_seconds=None,
    end_point_in_seconds=None,
    rng=None,
):
    # sourcery skip: or-if-exp-identity

    if rng is None:
        rng = np.random.RandomState()

    clip_start_point_in_seconds = 0
    clip_end_point_in_seconds = int(np.floor(video_length_in_frames / video_fps))

    if clip_end_point_in_seconds == 0:
        clip_end_point_in_seconds = 1

    # logging.info(f'Init {clip_start_point_in_seconds} {clip_end_point_in_seconds}, '
    #              f'{start_point_in_seconds} {end_point_in_seconds}')

    start_point_in_seconds = (
        start_point_in_seconds
        if start_point_in_seconds is not None
        else rng.randint(
            low=clip_start_point_in_seconds,
            high=clip_end_point_in_seconds,
        )
    )
    end_point_in_seconds = (
        end_point_in_seconds
        if end_point_in_seconds is not None
        else rng.randint(
            low=start_point_in_seconds + 1,
            high=clip_end_point_in_seconds + 1,
        )
    )

    duration = end_point_in_seconds - start_point_in_seconds

    start_video_frame = int(start_point_in_seconds * video_fps)

    end_video_frame = int(end_point_in_seconds * video_fps)

    # logging.info(f'{start_point_in_seconds} {end_point_in_seconds}'
    #              f' {start_video_frame} {end_video_frame}')

    possible_frames = list(range(start_video_frame, end_video_frame))

    if len(possible_frames) < num_video_frames_per_datapoint:
        video_frames_to_collect = possible_frames
    else:
        video_frames_to_collect = sorted(
            list(
                rng.choice(
                    possible_frames,
                    size=(num_video_frames_per_datapoint,),
                    replace=False,
                )
            )
        )

    # logging.info(f'video_frames_to_collect: {video_frames_to_collect}, '
    #              f'{video_length_in_frames} '
    #              f'{start_point_in_seconds}'
    #              f' {end_point_in_seconds}')

    # logging.info(f'video_frames_to_collect: {video_frames_to_collect} {start_video_frame}, {end_video_frame}')

    total_audio_frames_in_selected_time_clip = duration * num_audio_sample_rate

    audio_frame_collection_interval = int(
        np.floor(
            total_audio_frames_in_selected_time_clip / num_audio_frames_per_datapoint
        )
    )

    start_audio_frame = int(start_point_in_seconds * num_audio_sample_rate)

    duration_audio_frame = duration * num_audio_sample_rate

    audio_frames_to_collect = list(
        range(
            0,
            int(duration_audio_frame),
            int(audio_frame_collection_interval)
            if int(audio_frame_collection_interval) > 0
            else 1,
        )
    )[:num_audio_frames_per_datapoint]

    return (
        start_point_in_seconds,
        end_point_in_seconds,
        start_video_frame,
        end_video_frame,
        video_frames_to_collect,
        start_audio_frame,
        duration_audio_frame,
        audio_frames_to_collect,
    )


def collect_files(args):
    json_file_path, training_set_size_fraction_value = args
    video_files = list(pathlib.Path(json_file_path.parent).glob("**/*.mp4"))
    video_key = json_file_path.parent.stem
    folder_list = []
    for file in video_files:
        video_data_filepath = os.fspath(file.resolve())
        audio_data_filepath = os.fspath(file.resolve()).replace(".mp4", ".aac")
        meta_data_filepath = os.fspath(json_file_path.resolve())

        if (
            pathlib.Path(video_data_filepath).exists()
            and pathlib.Path(meta_data_filepath).exists()
            and pathlib.Path(audio_data_filepath).exists()
        ) and np.random.random() <= training_set_size_fraction_value:
            folder_list.append(
                (video_data_filepath, audio_data_filepath, meta_data_filepath)
            )

    return video_key, folder_list


def collate_resample_none(batch):
    none_batch = list(filter(lambda x: x is None, batch))
    batch = list(filter(lambda x: x is not None, batch))
    return dataloader.default_collate(batch), dataloader.default_collate(none_batch)
