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
import tqdm.rich as tqdm
from torch.utils.data import Dataset

from base import utils
from tali.config_repository import TALIDatasetConfig
from tali.datasets.utils import audio
from tali.datasets.utils.audio import prevent_error_kill
from tali.datasets.utils.helpers import (
    collect_files,
)
from tali.datasets.utils.text import get_text_tokens
from tali.datasets.utils.video import load_frames
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
        self.training_set_fraction_value = (
            {
                "milli": 1 / 10 ** 3,
                "centi": 1 / 10 ** 2,
                "deci": 1 / 10 ** 1,
                "base": 1 / 10 ** 0,
            }[self.config.training_set_size_identifier]
            if set_name == "train"
            else 1
        )

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
                training_set_fraction_value=self.training_set_fraction_value
            )
            save_json(
                filepath=self.pre_scanned_dataset_json_filepath,
                metrics_dict=self.path_dict,
                overwrite=True,
            )

        logging.info(
            f"{np.sum(len(value) for key, value in self.path_dict.items())} "
            f"files found"
        )

        self.index_to_video_path = [
            video_path
            for folder_list in self.path_dict.values()
            for video_path in folder_list
        ]

        self.num_video_clips = len(self.index_to_video_path)
        logging.info(f"num video clips: {self.num_video_clips}")

    def get_frames(
        self,
        frame_list,
        video_filepath,
        audio_filepath,
        rng,
    ):

        num_frames_to_sample_for_video = self.config.num_video_frames_per_datapoint
        # log.info(f"{len(frame_list)}, {num_frames_to_sample_for_video}")
        if len(frame_list) < num_frames_to_sample_for_video:
            selected_frame_list_idx = sorted(
                list(
                    rng.choice(
                        len(frame_list),
                        size=(num_frames_to_sample_for_video,),
                        replace=True,
                    )
                )
            )
        else:
            selected_frame_list_idx = sorted(
                list(
                    rng.choice(
                        len(frame_list),
                        size=(num_frames_to_sample_for_video,),
                        replace=False,
                    )
                )
            )
        # log.info(f"selected_frame_list_idx {selected_frame_list_idx}")
        frames_dict = DictWithDotNotation()

        frames_dict.video = None
        frames_dict.audio = None
        frames_dict.image = None

        if self.config.modality_config.video:
            frames_dict.video = load_frames(
                image_height=self.config.image_shape.height,
                image_width=self.config.image_shape.width,
                image_channels=self.config.image_shape.channels,
                selected_frame_list=[
                    frame_list[idx] for idx in selected_frame_list_idx
                ],
            )

        if self.config.modality_config.audio:
            if not pathlib.Path(audio_filepath).exists():
                return None

            frames_dict.audio = audio.load_to_tensor(
                filename=audio_filepath,
                sample_rate=self.config.num_audio_sample_rate,
                mono=False,
                in_type=np.float32,
                out_type=np.float32,
                video_frame_idx_list=selected_frame_list_idx,
                total_video_frames=len(frame_list),
            )
            frames_dict.audio = frames_dict.audio.view(-1, 2)
            frames_dict.audio = frames_dict.audio.permute([1, 0])

        if self.config.modality_config.image:
            frames_dict.image = load_frames(
                image_height=self.config.image_shape.height,
                image_width=self.config.image_shape.width,
                image_channels=self.config.image_shape.channels,
                selected_frame_list=rng.choice(frame_list, (1,)),
            )

            frames_dict.image = frames_dict.image[0]

        return frames_dict

    def get_text_data_tensors(
        self,
        rng,
        meta_data_filepath,
        start_time_relative_to_full_video,
        duration_in_seconds,
    ):
        text = get_text_tokens(
            meta_data_filepath=meta_data_filepath,
            start_timestamp=start_time_relative_to_full_video,
            end_timestamp=start_time_relative_to_full_video + duration_in_seconds,
        )

        if not text:
            text = "No detected speech"
        else:
            text = [
                value.split(" ") if isinstance(value, str) else value
                for key, value in text.items()
            ]
            text = list(itertools.chain.from_iterable(text))
            text = [token.replace(" ", "") for token in text]
            text = " ".join(text) if len(text) > 1 else text

        return text

    @prevent_error_kill
    def __getitem__(self, index):
        actual_index = index % self.num_video_clips
        current_time_rng = int(time.time_ns() % 100000)
        rng = np.random.RandomState(current_time_rng)
        torch_rng = torch.Generator()
        torch_rng.manual_seed(current_time_rng)

        (
            frame_list,
            video_filepath,
            audio_filepath,
            meta_data_filepath,
        ) = self.index_to_video_path[actual_index]

        video_segment_idx = int(
            re.match(
                r"full_video_360p(.*).frames", video_filepath.split("/")[-1]
            ).groups()[0]
        )

        total_frames = len(frame_list)
        fps = 8
        duration_in_seconds = total_frames / float(fps)

        start_time_relative_to_full_video = int(video_segment_idx * 10)

        data_dict = DictWithDotNotation()
        data_dict.text = None
        data_dict.video = None
        data_dict.audio = None
        data_dict.image = None

        # write script that cleans up clips without any captions, write script that
        # cleans something in video space (removes videos smaller than 10 seconds)

        if self.config.modality_config.text:
            data_dict.text = self.get_text_data_tensors(
                rng,
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
            frame_list=frame_list,
            video_filepath=video_filepath,
            audio_filepath=audio_filepath,
            rng=rng,
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

        if self.config.modality_config.image and "image" not in data_dict:
            video_path = pathlib.Path(video_filepath)
            audio_path = video_path.with_suffix(".aac")
            video_path.unlink()
            audio_path.unlink()
            return None

        if self.config.modality_config.video and "video" not in data_dict:
            video_path = pathlib.Path(video_filepath)
            audio_path = video_path.with_suffix(".aac")
            video_path.unlink()
            audio_path.unlink()
            return None

        if self.config.modality_config.audio and "audio" not in data_dict:
            video_path = pathlib.Path(video_filepath)
            audio_path = video_path.with_suffix(".aac")
            video_path.unlink()
            audio_path.unlink()
            return None

        if self.config.modality_config.text and "text" not in data_dict:
            video_path = pathlib.Path(video_filepath)
            audio_path = video_path.with_suffix(".aac")
            video_path.unlink()
            audio_path.unlink()
            return None

        data_dict["filepath"] = video_filepath

        return data_dict

    def __len__(self):
        # use 25000 to keep training very long to ensure even val
        # intervals no matter what the size of the dataset
        return 4252731 if self.set_name == "train" else self.num_video_clips

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
        logging.info(self.dataset_dir)

        matched_meta_data_files = []

        for file in pathlib.Path(self.dataset_dir).glob("**/meta_data.json"):
            matched_meta_data_files.append(file)
            log.info(f"{len(matched_meta_data_files)}")

        logging.info(f"Found {len(matched_meta_data_files)} matched meta_data files")

        args = [(item, training_set_fraction_value) for item in matched_meta_data_files]

        logging.info("Scanning folders for media files")

        with concurrent.futures.ProcessPoolExecutor(
            max_workers=int(mp.cpu_count() / 2)
        ) as executor:
            with tqdm.tqdm(total=len(matched_meta_data_files), smoothing=0.0) as pbar:
                for video_key, folder_list in executor.map(collect_files, args):
                    if len(folder_list) > 0:
                        path_dict[video_key] = folder_list

                    pbar.update(1)

        return path_dict
