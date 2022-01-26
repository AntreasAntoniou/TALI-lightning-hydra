import datetime
import inspect
import logging
import subprocess
import time

import numpy as np
import torch

log = logging.getLogger(__name__)


def timeit(method):
    def timed(*args, **kwargs):
        ts = time.time()
        result = method(*args, **kwargs)
        te = time.time()

        if "log_time" in kwargs:
            if kwargs["log_time"] == True:
                log.info(f"{method.__name__} took {te - ts:.4f} sec")
        return result

    return timed


def prevent_error_kill(method):
    def try_catch_return(*args, **kwargs):
        try:
            result = method(*args, **kwargs)
            return result
        except Exception as e:
            log.exception(f"{method.__name__} error: {e}")
            return None

    return try_catch_return


# load_audio can not detect the input type
@timeit
def load(
    filename: str,
    sample_rate: int = 44100,
    num_audio_frames_per_datapoint: int = 88200,
    mono: bool = False,
    in_type=np.float32,
    out_type=np.float32,
    video_frame_idx_list=None,
    total_video_frames=1,
):
    # logging.info(
    #     f'load "{filename}", start_point_in_seconds: {start_point_in_seconds}, duration_in_seconds: {duration_in_seconds}, sr: {sr}, mono: {mono}, normalize: {normalize}, in_type: {in_type}, out_type: {out_type}, log_time: {log_time}, frames_to_collect: {len(frames_to_collect)}'
    # )
    channels = 1 if mono else 2
    format_strings = {
        np.float64: "f64le",
        np.float32: "f32le",
        np.int16: "s16le",
        np.int32: "s32le",
        np.uint32: "u32le",
    }
    format_string = format_strings[in_type]
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",  # if log.level >= logging.DEBUG else "quiet",
        "-i",
        filename,
        # "-ss",
        # f"{datetime.timedelta(0, start_point_in_seconds)}",
        # "-t",
        # f"{datetime.timedelta(0, duration_in_seconds)}",
        "-f",
        format_string,
        "-acodec",
        f"pcm_{format_string}",
        "-ac",
        str(channels),
        "-ar",
        str(sample_rate),
        "-",
    ]
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=None, stdin=None)
    out, err = process.communicate(None)

    if retcode := process.poll():
        log.exception(f"Error loading audio file {filename}")
        raise Exception(
            f"{inspect.stack()[0][3]} " f"returned non-zero exit code {retcode}"
        )

    audio = np.frombuffer(out, dtype=in_type).astype(out_type)

    audio = audio.reshape(-1, channels)

    num_audio_frames = audio.shape[0]

    audio_frames_per_video_frame_actual = int(
        np.floor(num_audio_frames / total_video_frames)
    )

    audio_frames_per_video_frame_to_sample = int(
        np.floor(num_audio_frames_per_datapoint / len(video_frame_idx_list))
    )

    if video_frame_idx_list is not None:
        audio_frames_collected = []
        for video_frame_idx in video_frame_idx_list:
            audio_frame_idx_range = (
                video_frame_idx * audio_frames_per_video_frame_to_sample,
                (video_frame_idx + 1) * audio_frames_per_video_frame_to_sample,
            )
            audio_frames_collected.extend(
                audio[audio_frame_idx_range[0] : audio_frame_idx_range[1]]
            )
        audio = np.array(audio_frames_collected)
        log.info(audio.shape)
        if audio.shape[0] < num_audio_frames_per_datapoint:
            audio = torch.cat(
                [
                    audio,
                    torch.zeros(
                        num_audio_frames_per_datapoint - audio.shape[0], audio.shape[1]
                    ),
                ],
                dim=0,
            )

    return audio
