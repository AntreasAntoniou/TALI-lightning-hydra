import datetime
import logging
import pathlib
import subprocess

import cv2
import numpy as np
import torch

log = logging.getLogger(__name__)


def get_meta_data_opencv(filepath):
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
        vid_capture.release()
        return None


def get_frames_opencv_cpu(
    filepath,
    video_frame_idx_list,
    image_height,
    image_width,
    **kwargs,
):
    # Create a video capture object, in this case we are reading the video from a file
    frames = np.zeros(shape=(len(video_frame_idx_list), image_height, image_width, 3))
    vid_capture = cv2.VideoCapture(filepath)

    frame_successfully_acquired = True

    frames_collected = 0
    frames_read = 0
    # if len(video_frame_idx_list) > 0:
    #     if video_frame_idx_list[0] > 0:
    #         vid_capture.set(cv2.CAP_PROP_POS_FRAMES, video_frame_idx_list[0])
    #         frames_read = video_frame_idx_list[0]
    # else:
    #     log.info(f"{get_meta_data_opencv(filepath)}")

    while (
        frame_successfully_acquired
        and frames_collected <= len(video_frame_idx_list) - 1
    ):
        frame_successfully_acquired, image = vid_capture.read()

        if (
            frame_successfully_acquired
            and frames_read == video_frame_idx_list[frames_collected]
        ):
            img_frame = cv2.resize(
                image, (image_height, image_width), interpolation=cv2.INTER_CUBIC
            )
            frames[frames_collected] = img_frame / 255.0

            frames_collected += 1
        frames_read += 1

    vid_capture.release()

    return torch.from_numpy(frames)
