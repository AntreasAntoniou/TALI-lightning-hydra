import cv2


def extract_frames_from_video_file(filepath):
    cap = cv2.VideoCapture(filepath)

    # Get the frames per second
    num_total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    fps = cap.get(cv2.CAP_PROP_FPS)

    frame_list = [i for i in range(int(num_total_frames))]
    image_list = []

    for frame_number in frame_list:

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)  # optional
        success, image = cap.read()

        while success and frame_number <= num_total_frames:
            # do stuff

            frame_number += fps
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            success, image = cap.read()
            image_list.append(image)

    return image_list, fps, num_total_frames


def extract_specific_frame_from_video(filepath, frame_list, cap=None):

    if not cap:
        cap = cv2.VideoCapture(filepath)

    # Get the frames per second
    num_total_frames = cap.stream.get(cv2.CAP_PROP_FRAME_COUNT)
    fps = cap.get(cv2.CAP_PROP_FPS)

    image_list = []

    for frame_number in frame_list:

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)  # optional
        success, image = cap.read()

        while success and frame_number <= num_total_frames:
            # do stuff

            frame_number += fps
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            success, image = cap.read()
            image_list.append(image)

    return image_list, fps, num_total_frames


def get_num_total_frames(filepath, cap=None):
    if not cap:
        cap = cv2.VideoCapture(filepath)

    return cap.stream.get(cv2.CAP_PROP_FRAME_COUNT)


def get_frames_per_second(filepath, cap=None):
    if not cap:
        cap = cv2.VideoCapture(filepath)

    return cap.get(cv2.CAP_PROP_FPS)
