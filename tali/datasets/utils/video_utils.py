import datetime
import logging
import subprocess

import cv2
import numpy as np
import torch

log = logging.getLogger(__name__)


def get_meta_data_opencv(filepath):
    vid_capture = cv2.VideoCapture(filepath)
    try:
        total_frames = vid_capture.get(cv2.CAP_PROP_FRAME_COUNT)
        fps = vid_capture.get(cv2.CAP_PROP_FPS)
        duration_in_seconds = total_frames / fps
        vid_capture.release()
        return total_frames, fps, duration_in_seconds

    except Exception:
        log.exception("OpenCV reading gone wrong 🤦")
        vid_capture.release()
        return None


def get_frames_opencv_cpu(
    filepath,
    frame_indexes_to_collect,
    image_height,
    image_width,
    **kwargs,
):
    # Create a video capture object, in this case we are reading the video from a file
    frames = (
        np.zeros(shape=(len(frame_indexes_to_collect), image_height, image_width, 3))
        # if np.random.randint(0, 2) == 0
        # else np.random.normal(
        #     size=(len(frame_indexes_to_collect), image_height, image_width, 3)
        # )
    )
    vid_capture = cv2.VideoCapture(filepath)

    frame_successfully_acquired = True

    frames_collected = 0
    frames_read = 0
    if len(frame_indexes_to_collect) > 0:
        if frame_indexes_to_collect[0] > 0:
            vid_capture.set(cv2.CAP_PROP_POS_FRAMES, frame_indexes_to_collect[0])
            frames_read = frame_indexes_to_collect[0]
    else:
        log.info(f'{get_meta_data_opencv(filepath)}')

    while (
        frame_successfully_acquired
        and frames_collected <= len(frame_indexes_to_collect) - 1
    ):
        frame_successfully_acquired, image = vid_capture.read()

        # logging.info(f'{frame_successfully_acquired} '
        #              f'{frames_collected <= len(frame_indexes_to_collect) - 1}, '
        #              f'frames_read {frames_read}')
        # logging.info(
        #     f'{np.mean(image)}, {np.std(image)}, {np.max(image)}, {np.min(image)}')

        if (
            frame_successfully_acquired
            and frames_read == frame_indexes_to_collect[frames_collected]
        ):
            img_frame = cv2.resize(
                image, (image_height, image_width), interpolation=cv2.INTER_CUBIC
            )
            frames[frames_collected] = img_frame / 255.0
            # logging.info(f'here')

            frames_collected += 1
        frames_read += 1

    vid_capture.release()

    # logging.info(f"Frames read: {frames_read}, frames collected: {frames_collected}")

    if len(frames) == 0:
        log.error(f"No frames were extracted from the video {filepath}")

    return torch.from_numpy(frames)


class PinnedMem(object):
    def __init__(self, size, dtype=np.uint8):
        self.array = np.empty(size, dtype)
        cv2.cuda.registerPageLocked(self.array)
        self.pinned = True

    def __del__(self):
        cv2.cuda.unregisterPageLocked(self.array)
        self.pinned = False

    def __repr__(self):
        return f"pinned = {self.pinned}"


def get_frames_opencv_gpu(
    filepath,
    start_timestamp_seconds,
    duration_to_extract,
    fps_to_extract,
    image_height,
    image_width,
    image_channels,
    **kwargs,
):
    # Create a video capture object, in this case we are reading the video from a file
    video_capture = cv2.cudacodec.createVideoReader(filepath)
    frame_device = cv2.cuda_GpuMat(image_height, image_width, cv2.CV_8UC3)
    frame_host = PinnedMem((image_height, image_width, image_channels))
    video_stream = cv2.cuda_Stream()

    frames = []
    if video_capture.isOpened() == False:
        raise FileNotFoundError(f"Error opening the video file {filepath}")
    # Read fps and frame count
    # Get frame rate information
    # You can replace 5 with CAP_PROP_FPS as well, they are enumerations
    # logging.info(f'Frames per second :  {fps} FPS')
    # logging.info(f'Total frames :  {frame_count}')
    frame_succesfully_acquired = True
    frame_idx = 0
    num_extracted_frames = 0
    frame_succesfully_acquired, _ = video_capture.nextFrame(frame_device, video_stream)
    while frame_succesfully_acquired and len(frames) < duration_to_extract:
        # vid_capture.read() methods returns a tuple, first element is a bool
        # and the second is frame
        frame_succesfully_acquired, _ = video_capture.nextFrame(
            frame_device, video_stream
        )
        video_capture.set(cv2.CAP_PROP_POS_MSEC, (start_timestamp_seconds * 1000))  #
        # added this line
        frame_succesfully_acquired, image = video_capture.read()
        # print('Read a new frame: ', success)
        if frame_succesfully_acquired and frame_idx % (fps_to_extract) == 0:
            img_frame = cv2.resize(
                image, (image_height, image_width), interpolation=cv2.INTER_AREA
            )
            frames.append(img_frame)

        frame_idx += 1
        # logging.info(f'here {frame_idx} {len(frames)}')

    # Release the video capture object
    video_capture.release()
    return frames


def extract_video_frames_ffmpeg(
    video_filepath,
    start_timestamp,
    duration_to_extract,
    num_frames_to_extract,
    width,
    height,
    channels,
    total_fps,
    selected_fps,
):
    command_args = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "quiet",
        "-i",
        video_filepath,
        "-vf",
        f"scale={height}:{width}",
        "-ss",
        f"{datetime.timedelta(0, start_timestamp)}",
        "-t",
        f"{datetime.timedelta(0, duration_to_extract)}",
        "-frames:v",
        str(num_frames_to_extract),
        "-pix_fmt",
        "rgb24",
        "-vsync",
        "vfr",
        "-f",
        "rawvideo",
        "-",
    ]

    process = subprocess.Popen(
        command_args,
        stdin=None,
        stdout=subprocess.PIPE,
        stderr=None,
        cwd=None,
    )

    out, err = process.communicate(None)
    retcode = process.poll()
    if retcode:
        logging.exception("🤦")
        raise Exception(f"{retcode}")

    frames = np.frombuffer(out, np.uint8).reshape(-1, width, height, channels)

    frames = frames[
        [i for i in range(num_frames_to_extract) if i % (total_fps - selected_fps) == 0]
    ]

    if frames.shape[0] < num_frames_to_extract:
        frames = np.concatenate(
            [
                frames,
                np.zeros(
                    (num_frames_to_extract - frames.shape[0], width, height, channels)
                ),
            ],
            axis=0,
        )

    if frames.shape[0] > num_frames_to_extract:
        frames = frames[:num_frames_to_extract]

    return frames, err


def extract_video_frames_opencv(
    video_filepath,
    start_timestamp,
    duration_to_extract,
    num_frames_to_extract,
    width,
    height,
    channels,
    total_fps,
    selected_fps,
):
    command_args = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "quiet",
        "-i",
        video_filepath,
        "-vf",
        f"scale={height}:{width}",
        "-ss",
        f"{datetime.timedelta(0, start_timestamp)}",
        "-t",
        f"{datetime.timedelta(0, duration_to_extract)}",
        "-frames:v",
        str(num_frames_to_extract),
        "-pix_fmt",
        "rgb24",
        "-vsync",
        "vfr",
        "-f",
        "rawvideo",
        "-",
    ]

    process = subprocess.Popen(
        command_args,
        stdin=None,
        stdout=subprocess.PIPE,
        stderr=None,
        cwd=None,
    )

    out, err = process.communicate(None)
    retcode = process.poll()
    if retcode:
        log.exception("🤦")
        raise Exception(f"{retcode}")

    frames = np.frombuffer(out, np.uint8).reshape(-1, width, height, channels)

    frames = frames[
        [i for i in range(num_frames_to_extract) if i % (total_fps - selected_fps) == 0]
    ]

    if frames.shape[0] < num_frames_to_extract:
        frames = np.concatenate(
            [
                frames,
                np.zeros(
                    (num_frames_to_extract - frames.shape[0], width, height, channels)
                ),
            ],
            axis=0,
        )

    if frames.shape[0] > num_frames_to_extract:
        frames = frames[:num_frames_to_extract]

    return frames, err


def extract_audio_frames_ffmpeg(video_filepath, num_channels):
    command_args = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "quiet",
        "-i",
        video_filepath,
        "-f",
        "s24le",
        "-ac",
        f"{num_channels}",
        "-ar",
        "44.1k",
        "-acodec",
        "pcm_s24le",
        "-",
    ]

    process = subprocess.Popen(
        command_args,
        stdin=None,
        stdout=subprocess.PIPE,
        stderr=None,
        cwd=None,
    )

    out, err = process.communicate(None)

    frames = np.frombuffer(out, np.uint8).reshape(-1, num_channels)

    retcode = process.poll()
    if retcode:
        log.exception("🤦")
        raise Exception(f"{retcode}")
    return frames, err
