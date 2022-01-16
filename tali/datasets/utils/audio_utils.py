import datetime
import inspect
import logging
import subprocess
import time

import numpy as np


def timeit(method):
    def timed(*args, **kwargs):
        ts = time.time()
        result = method(*args, **kwargs)
        te = time.time()

        if "log_time" in kwargs:
            if kwargs["log_time"] == True:
                logging.info(f"{method.__name__} took {te - ts:.4f} sec")
        return result

    return timed


def prevent_error_kill(method):
    def try_catch_return(*args, **kwargs):
        try:
            result = method(*args, **kwargs)
            return result
        except Exception as e:
            logging.exception(f"{method.__name__} error: {e}")
            return None

    return try_catch_return


# load_audio can not detect the input type
@timeit
def load(
    filename: str,
    start_point_in_seconds: int,
    duration_in_seconds: int,
    sr: int = 44100,
    mono: bool = False,
    normalize=True,
    in_type=np.int16,
    out_type=np.float32,
    log_time=True,
    frames_to_collect=None,
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
        "error",
        "-i",
        filename,
        "-ss",
        f"{datetime.timedelta(0, start_point_in_seconds)}",
        "-t",
        f"{datetime.timedelta(0, duration_in_seconds)}",
        "-f",
        format_string,
        "-acodec",
        f"pcm_{format_string}",
        "-ac",
        str(channels),
        "-ar",
        str(sr),
        "-",
    ]
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=None, stdin=None)
    out, err = process.communicate(None)
    retcode = process.poll()

    if retcode:
        logging.exception(f"Error loading audio file {filename}")
        raise Exception(
            f"{inspect.stack()[0][3]} " f"returned non-zero exit code {retcode}"
        )

    audio = np.frombuffer(out, dtype=in_type).astype(out_type)

    loading_shape = audio.shape

    audio = audio.reshape(-1, channels)

    reshape_shape = audio.shape

    frames_to_collect = np.array(frames_to_collect)

    frames_to_collect_original_shape = frames_to_collect.shape

    frames_to_collect = frames_to_collect[frames_to_collect < audio.shape[0]]

    if frames_to_collect is not None:
        audio = audio[frames_to_collect]

    # logging.info(f'{audio.shape}, {loading_shape}, {reshape_shape}, '
    #              f'{frames_to_collect.shape}, {frames_to_collect_original_shape}')

    # logging.debug(f"{filename} loaded into array of shape {audio.shape}")
    return audio
