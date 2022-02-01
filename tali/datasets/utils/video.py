import datetime
import logging
import pathlib
import subprocess

import cv2
import numpy as np
import torch

log = logging.getLogger(__name__)


def get_meta_data_opencv(filepath):
    vid_capture = False
    try:
        vid_capture = cv2.VideoCapture(filepath)
        total_frames = vid_capture.get(cv2.CAP_PROP_FRAME_COUNT)
        fps = vid_capture.get(cv2.CAP_PROP_FPS)
        duration_in_seconds = total_frames / fps
        vid_capture.release()
        return total_frames, fps, duration_in_seconds

    except Exception:
        video_path = pathlib.Path(filepath)
        audio_path = video_path.with_suffix(".aac")
        if video_path.exists():
            video_path.unlink()
        if audio_path.exists():
            audio_path.unlink()
        log.exception("OpenCV reading gone wrong 🤦")
        if vid_capture:
            vid_capture.release()
        return None


def silent_error_handler(status, func_name, err_msg, file_name, line):
    pass


def load_frames(
    selected_frame_list,
    image_height,
    image_width,
    image_channels,
):
    image_tensor = torch.zeros(
        (len(selected_frame_list), image_channels, image_height, image_width)
    )

    for idx, frame_filepath in enumerate(selected_frame_list):

        image = cv2.imread(frame_filepath)
        image = (
            cv2.resize(
                image, (image_height, image_width), interpolation=cv2.INTER_CUBIC
            )
            / 255.0
        )
        image = torch.Tensor(image).permute([2, 0, 1])
        image_tensor[idx] = image

    return image_tensor
