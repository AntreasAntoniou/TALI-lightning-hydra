import concurrent.futures
import itertools
import logging
import multiprocessing as mp
import os
import pathlib
import re
import time
from typing import Callable, Dict, List, Union

import numpy as np
import torch
import tqdm
from base import utils
from tali.config_repository import TALIDatasetConfig
from torch.utils.data import Dataset

from tali.datasets.utils import audio_utils
from tali.datasets.utils.audio_utils import prevent_error_kill
from tali.datasets.utils.helpers import (
    collect_files,
    get_text_tokens,
    sample_frame_indexes_to_collect,
    timeout,
)
from tali.datasets.utils.video_utils import get_frames_opencv_cpu, get_meta_data_opencv
from tali.utils.arg_parsing import DictWithDotNotation
from tali.utils.storage import load_json, save_json

log = utils.get_logger(__name__)

class TALIMultiModalDataset(Dataset):
    def __init__(
        self,
        config: TALIDatasetConfig,
        set_name: str,
        transforms: Dict[str, Union[List[Callable], Callable]],
    ):
        super(TALIMultiModalDataset, self).__init__()

        self.config = config
        self.dataset_root = config.dataset_root
        self.set_name = set_name
        self.transforms = transforms
        self.training_set_fraction_value = {
            "micro": 1 / 2 ** 8,
            "milli": 1 / 2 ** 5,
            "centi": 1 / 2 ** 4,
            "deci": 1 / 2 ** 3,
            "base": 1 / 2 ** 2,
            "deka": 1 / 2 ** 1,
            "hecta": 1 / 2 ** 0,
        }[self.config.training_set_size_identifier]

        self.dataset_dir = os.path.join(self.dataset_root, self.set_name)

        self.pre_scanned_dataset_json_filepath = os.path.join(
            self.dataset_dir,
            f"tali_path_cache_{self.config.training_set_size_identifier}.json",
        )

        logging.info(self.pre_scanned_dataset_json_filepath)

        if self._check_if_paths_already_scanned():
            self.path_dict = load_json(self.pre_scanned_dataset_json_filepath)
        else:
            self.path_dict = self._scan_paths_return_dict(
                training_set_fraction_value=self.training_set_fraction_value)
            save_json(
                filepath=self.pre_scanned_dataset_json_filepath,
                metrics_dict=self.path_dict,
                overwrite=True,
            )

        logging.info(
            f"{np.sum(len(value) for key, value in self.path_dict.items())} "
            f"files found"
        )

        self.index_to_video_path = [video_path
                             for folder_list in self.path_dict.values()
                             for video_path in folder_list]

        logging.info(f"num video paths: {len(self.index_to_video_path)}")

    def get_frames(self, data_dict, filepath, rng):
        (
            total_frames,
            video_frames_per_second,
            clip_duration_in_seconds,
        ) = get_meta_data_opencv(filepath)


        (
            start_point_in_seconds,
            end_point_in_seconds,
            start_video_frame,
            end_video_frame,
            video_frames_to_collect,
            start_audio_frame,
            duration_audio_frame,
            audio_frames_to_collect,
        ) = sample_frame_indexes_to_collect(
            video_length_in_frames=total_frames,
            video_fps=video_frames_per_second,
            num_video_frames_per_datapoint=self.config.num_video_frames_per_datapoint,
            num_audio_frames_per_datapoint=self.config.num_audio_frames_per_datapoint,
            num_audio_sample_rate=self.config.num_audio_sample_rate,
            start_point_in_seconds=data_dict.start_point_in_seconds,
            end_point_in_seconds=data_dict.end_point_in_seconds,
            rng=rng,
        )

        frames_dict = DictWithDotNotation(
            dict(
                total_frames=total_frames,
                video_frames_per_second=video_frames_per_second,
                clip_duration_in_seconds=clip_duration_in_seconds,
                start_point_in_seconds=start_point_in_seconds,
                end_point_in_seconds=end_point_in_seconds,
                start_video_frame=start_video_frame,
                end_video_frame=end_video_frame,
                video_frames_to_collect=video_frames_to_collect,
                start_audio_frame=end_video_frame,
                duration_audio_frame=duration_audio_frame,
                audio_frames_to_collect=audio_frames_to_collect,
            )
        )

        frames_dict.video = None
        frames_dict.audio = None
        frames_dict.image = None

        if len(video_frames_to_collect) == 0:
            video_path = pathlib.Path(filepath)
            audio_path = video_path.with_suffix(".aac")
            video_path.unlink()
            audio_path.unlink()
            log.error(f'No video frames found in {filepath}')

        if self.config.modality_config.video:
            frames_dict.video = get_frames_opencv_cpu(
                filepath=filepath,
                frame_indexes_to_collect=video_frames_to_collect,
                image_height=self.config.image_shape.height,
                image_width=self.config.image_shape.width,
                image_channels=self.config.image_shape.channels,
            )

            frames_dict.video = frames_dict.video.permute([0, 3, 1, 2])

        if self.config.modality_config.audio:
            audio_filepath = filepath.replace(".mp4", ".aac")

            if not pathlib.Path(audio_filepath).exists():
                return None

            frames_dict.audio = audio_utils.load(
                filename=audio_filepath,
                sr=self.config.num_audio_sample_rate,
                mono=False,
                normalize=False,
                in_type=np.float32,
                out_type=np.float32,
                log_time=False,
                start_point_in_seconds=start_point_in_seconds,
                duration_in_seconds=duration_audio_frame
                / self.config.num_audio_sample_rate,
                frames_to_collect=audio_frames_to_collect,
            )
            frames_dict.audio = torch.from_numpy(frames_dict.audio).view(-1, 2)
            frames_dict.audio = frames_dict.audio.permute([1, 0])

        if self.config.modality_config.image:
            if frames_dict.video is not None:
                frames_dict.image = frames_dict.video[
                    rng.choice(len(video_frames_to_collect))
                ]
            else:
                frames_dict.image = get_frames_opencv_cpu(
                    filepath=filepath,
                    frame_indexes_to_collect=[rng.choice(video_frames_to_collect)],
                    image_height=self.config.image_shape.height,
                    image_width=self.config.image_shape.width,
                    image_channels=self.config.image_shape.channels,
                )

                frames_dict.image = frames_dict.image[0].permute([2, 0, 1])

        return frames_dict

    def get_text_data_tensors(
        self,
        rng,
        data_dict,
        meta_data_filepath,
        start_time_relative_to_full_video,
        duration_in_seconds,
    ):
        data_dict.text = get_text_tokens(
            meta_data_filepath=meta_data_filepath,
            start_timestamp=start_time_relative_to_full_video,
            end_timestamp=start_time_relative_to_full_video + duration_in_seconds,
        )

        if len(data_dict.text) == 0:
            data_dict.text = "No detected speech"
            return None
        else:
            data_dict.text_start_point_in_seconds = rng.choice(
                list(data_dict.text.keys())
            ).item()

            data_dict.text = [
                value.split(" ") if isinstance(value, str) else value
                for key, value in data_dict.text.items()
                if data_dict.text_start_point_in_seconds >= key
            ]

            data_dict.text = list(itertools.chain.from_iterable(data_dict.text))
            data_dict.text = data_dict.text[: self.config.text_context_length]
            data_dict.text = [token.replace(" ", "") for token in data_dict.text]
            data_dict.text = (
                " ".join(data_dict.text) if len(data_dict.text) > 1 else data_dict.text
            )

            data_dict.start_point_in_seconds = (
                data_dict.text_start_point_in_seconds
                - start_time_relative_to_full_video
            )
            data_dict.end_point_in_seconds = duration_in_seconds

        return data_dict.text

    @prevent_error_kill
    def __getitem__(self, index):
        # we have:
        # audio with exact clipping of whatever start-end segment we want
        # video structured as a per second list of tensors, each tensor containing a
        # series of images
        # text structured as a list of sentences, as well as a list that maps seconds to
        # sentence whose parts play at that second
        # video_key, value_idx = self.index_to_item_address[
        #     index % len(self.index_to_item_address)
        # ]
        current_time_rng = int(time.time_ns() % 100000)
        rng = np.random.RandomState(current_time_rng)
        torch_rng = torch.Generator()
        torch_rng.manual_seed(current_time_rng)

        (video_data_filepath,
         audio_data_filepath,
         meta_data_filepath) = self.index_to_video_path[index]
        # sub_video_idx = rng.choice(len((self.path_dict[video_key])))
        # (video_data_filepath, audio_data_filepath, meta_data_filepath) = self.path_dict[
        #     video_key
        # ][sub_video_idx]

        video_segment_idx = int(
            re.match(
                r"full_video_360p(.*).mp4", video_data_filepath.split("/")[-1]
            ).groups()[0]
        )

        total_frames, fps, duration_in_seconds = get_meta_data_opencv(
            video_data_filepath
        )

        start_time_relative_to_full_video = int(video_segment_idx * 10)

        data_dict = DictWithDotNotation()
        data_dict.start_point_in_seconds = None
        data_dict.end_point_in_seconds = None
        data_dict.text = None
        data_dict.video = None
        data_dict.audio = None
        data_dict.image = None

        # write script that cleans up clips without any captions, write script that
        # cleans something in video space (removes videos smaller than 10 seconds)

        if self.config.modality_config.text:
            data_dict.text = self.get_text_data_tensors(
                rng,
                data_dict,
                meta_data_filepath,
                start_time_relative_to_full_video,
                duration_in_seconds,
            )

            if data_dict.text is not None:
                data_dict.text = self.apply_transforms_if_available(
                    modality_name="text", data=data_dict.text
                )

                data_dict.text = data_dict.text.type(torch.int32)

        frames_dict = self.get_frames(
            data_dict=data_dict, filepath=video_data_filepath, rng=rng
        )

        if frames_dict is not None:
            data_dict.update(frames_dict)

        if self.config.modality_config.video and data_dict.video is not None:
            data_dict.video = self.apply_transforms_if_available(
                modality_name="video", data=data_dict.video
            )

            data_dict.video = data_dict.video.type(torch.float32)

        if self.config.modality_config.image and data_dict.image is not None:
            data_dict.image = self.apply_transforms_if_available(
                modality_name="image", data=data_dict.image
            )

            data_dict.image = data_dict.image.type(torch.float32)

        if self.config.modality_config.audio and data_dict.audio is not None:
            data_dict.audio = self.apply_transforms_if_available(
                modality_name="audio", data=data_dict.audio
            )
            data_dict.audio = data_dict.audio.type(torch.float32)

        data_dict = {
            key: value
            for key, value in data_dict.items()
            if isinstance(value, torch.Tensor)
        }

        if self.config.modality_config.image and 'image' not in data_dict:
            return None

        if self.config.modality_config.video and 'video' not in data_dict:
            return None

        if self.config.modality_config.audio and 'audio' not in data_dict:
            return None

        if self.config.modality_config.text and 'text' not in data_dict:
            return None

        return data_dict

    def __len__(self):

        return len(self.index_to_video_path)

    def apply_transforms_if_available(self, modality_name, data):
        if self.transforms[modality_name]:
            if isinstance(self.transforms[modality_name], list):
                for transform in self.transforms[modality_name]:
                    data = transform(data)
            else:
                data = self.transforms[modality_name](data)

        return data

    def _check_if_paths_already_scanned(self):

        if self.config.rescan_paths:
            return False

        return pathlib.Path(self.pre_scanned_dataset_json_filepath).exists()

    def _scan_paths_return_dict(self, training_set_fraction_value):

        path_dict = {}
        # caption_store_dict = {}
        logging.info(self.dataset_dir)

        matched_meta_data_files = list(
            pathlib.Path(self.dataset_dir).glob("**/meta_data.json")
        )

        logging.info(f"Found {len(matched_meta_data_files)} matched meta_data files")

        args = [
            (item, training_set_fraction_value) for item in matched_meta_data_files
        ]

        logging.info(f"Scanning folders for media files")

        with concurrent.futures.ProcessPoolExecutor(
            max_workers=mp.cpu_count()
        ) as executor:
            with tqdm.tqdm(total=len(matched_meta_data_files), smoothing=0.0) as pbar:
                for video_key, folder_list in executor.map(collect_files, args):
                    if len(folder_list) > 0:
                        path_dict[video_key] = folder_list

                    pbar.update(1)

        return path_dict
