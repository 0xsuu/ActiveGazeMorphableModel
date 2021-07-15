
import cv2
import numpy as np

from constants import *
from utils.face_landmark_detector import FaceLandmarkDetector


def to_numpy(arr):
    return np.array([[float(j) for j in i.split(";")] for i in arr])


def read_camera_calibrations(filename):
    with open(filename, "r") as f:
        content = f.read().splitlines()
    intrinsics, R, T = content[3:6], content[7:10], content[11:14]

    intrinsics = to_numpy(intrinsics)
    R = to_numpy(R)
    T = to_numpy(T)
    return intrinsics, R, T


def read_screen_calibrations(filename):
    with open(filename, "r") as f:
        content = f.read().splitlines()
    k_x, k_y, R, T = float(content[1]), float(content[3]), content[5:8], content[9:12]
    R = to_numpy(R)
    T = to_numpy(T)
    return k_x, k_y, R, T


def read_labels(filename):
    with open(filename, "r") as f:
        content = f.read().splitlines()
    label_list = [[float(j) for j in line.split(";")[1:]] for line in content[1:]]
    proper_len = len(max(label_list, key=lambda x:len(x)))
    for idx, i in enumerate(label_list):
        if len(i) != proper_len:
            label_list[idx] = [0.] * proper_len
    return np.array(label_list)


def world_to_img(coordinates, intrinsics, R, T):
    projected_coord = (((coordinates - T.T) @ R.T) @ intrinsics.T)
    projected_coord = projected_coord[:, :2] / projected_coord[:, 2].reshape(-1, 1)
    return projected_coord


def draw_line(frame, point_1, point_2, colour=(0, 255, 255), thickness=2):
    cv2.line(frame, (int(point_1[0]), int(point_1[1])), (int(point_2[0]), int(point_2[1])),
             colour, lineType=cv2.LINE_AA, thickness=thickness)


def draw_face_result(frame, face_bbox, face_lms):
    for lm in np.concatenate(face_lms, axis=0):
        cv2.circle(frame, (int(lm[0]), int(lm[1])), 2, (0, 0, 255))
    cv2.rectangle(frame, (face_bbox[0], face_bbox[1]), (face_bbox[2], face_bbox[3]), (0, 0, 255))
    centre = face_lms[2][-1].astype(np.int)
    cv2.rectangle(frame, (centre[0] - 48, centre[1] - 48), (centre[0] + 48, centre[1] + 48), (0, 255, 255))


def process_single_subject(subject_id, experiment_id, experiment_type, head_movement, feed="hd", preview=True):
    """
    Process a single folder, the folder name can be decomposed to
        "subject_id"_"experiment_id"_"experiment_type"_"head_movement".

    :param subject_id: "1"-"16".
    :param experiment_id: "A" or "B".
    :param experiment_type: "CS", "DS" or "FT",
        correspond to dynamic screen target, static screen target and floating ball.
    :param head_movement: "M" or "S".
    :param feed: "vga", "hd" or "depth".
    :param preview: Show marked video.

    :return:
    """
    # Initiate paths.
    data_folder_name = subject_id + "_" + experiment_id + "_" + experiment_type + "_" + head_movement
    data_folder_path = EYEDIAP_PATH + "Data/" + data_folder_name + "/"

    # Read calibrations.
    # Gaze state annotations.
    if experiment_type == "DS":
        gaze_state = None
    else:
        with open(EYEDIAP_PATH + "Annotations/" + data_folder_name + "/gaze_state.txt", "r") as f:
            gaze_state = f.readlines()
            gaze_state = [line.strip().split("\t")[1] == "OK" for line in gaze_state]

    # Camera calibration.
    if feed == "vga":
        cam_intrinsics, cam_R, cam_T = read_camera_calibrations(data_folder_path + "rgb_vga_calibration.txt")
    elif feed == "hd":
        cam_intrinsics, cam_R, cam_T = read_camera_calibrations(data_folder_path + "rgb_hd_calibration.txt")
    elif feed == "depth":
        raise NotImplemented
    else:
        raise ValueError

    scr_k_x, scr_k_y, scr_R, scr_T = read_screen_calibrations(
        EYEDIAP_PATH + "Metadata/ScreenCalibration/screen_calibration.txt")

    # Read labels.
    if experiment_type in ["CS", "DS"]:
        target_content = read_labels(data_folder_path + "screen_coordinates.txt")
        scr_2d_coord = target_content[:, :2]
        scr_3d_coord = target_content[:, 3:]
    elif experiment_type == "FT":
        target_content = read_labels(data_folder_path + "ball_tracking.txt")
        ball_3d_coord = target_content[:, -3:]
        ball_proj = world_to_img(ball_3d_coord, cam_intrinsics, cam_R, cam_T)
        if feed == "vga":
            ball_2d_coord = target_content[:, :2]
        elif feed == "hd":
            ball_2d_coord = target_content[:, 4:6]
        elif feed == "depth":
            raise NotImplemented
        else:
            raise ValueError
    else:
        raise ValueError

    eye_content = read_labels(data_folder_path + "eye_tracking.txt")
    eye_l_3d_coord = eye_content[:, -6:-3]
    eye_r_3d_coord = eye_content[:, -3:]
    eye_l_proj = world_to_img(eye_l_3d_coord, cam_intrinsics, cam_R, cam_T)
    eye_r_proj = world_to_img(eye_r_3d_coord, cam_intrinsics, cam_R, cam_T)

    if feed == "vga":
        eye_l_2d_coord = eye_content[:, :2]
        eye_r_2d_coord = eye_content[:, 2:4]
    elif feed == "hd":
        eye_l_2d_coord = eye_content[:, 8:10]
        eye_r_2d_coord = eye_content[:, 10:12]
    elif feed == "depth":
        raise NotImplemented
    else:
        raise ValueError

    assert len(eye_content) == len(target_content)

    # Read data frames.
    if feed == "vga":
        cap = cv2.VideoCapture(data_folder_path + "rgb_vga.mov")
    elif feed == "hd":
        cap = cv2.VideoCapture(data_folder_path + "rgb_hd.mov")
    elif feed == "depth":
        raise NotImplemented
    else:
        raise ValueError

    if gaze_state is not None:
        frame_valid_list = gaze_state.copy()[:len(eye_content)]
    else:
        frame_valid_list = [True] * len(eye_content)

    fld = FaceLandmarkDetector()
    for frame_idx in range(len(eye_content)):
        success, img = cap.read()
        if not success:
            break

        if not all(i == 0 for i in eye_content[frame_idx]) and \
                not all(i == 0 for i in target_content[frame_idx]) and \
                frame_valid_list[frame_idx] and \
                ((experiment_type in ["CS", "DS"] and not all(j == 0 for j in scr_2d_coord[frame_idx])) or
                    (experiment_type == "FT" and not all(j == 0 for j in ball_2d_coord[frame_idx]))):
            valid_frame = True
        else:
            valid_frame = False
            frame_valid_list[frame_idx] = False

        if valid_frame:
            face_bbox, face_lms = fld.process_face(img)
            draw_face_result(img, face_bbox, face_lms)
            if experiment_type in ["CS", "DS"]:
                # draw_line(img, eye_l_2d_coord[frame_idx], scr_2d_coord[frame_idx])
                # draw_line(img, eye_r_2d_coord[frame_idx], scr_2d_coord[frame_idx])
                draw_line(img, eye_l_proj[frame_idx], scr_2d_coord[frame_idx], thickness=1)
                draw_line(img, eye_r_proj[frame_idx], scr_2d_coord[frame_idx], thickness=1)
            elif experiment_type == "FT":
                draw_line(img, eye_l_2d_coord[frame_idx], ball_2d_coord[frame_idx], colour=(0, 0, 255))
                draw_line(img, eye_r_2d_coord[frame_idx], ball_2d_coord[frame_idx], colour=(0, 0, 255))
                draw_line(img, eye_l_proj[frame_idx], ball_proj[frame_idx], thickness=1)
                draw_line(img, eye_r_proj[frame_idx], ball_proj[frame_idx], thickness=1)

        if preview:
            cv2.imshow("Preview", img)
            cv2.waitKey(1)

    cap.release()

    # Pack and return.


if __name__ == '__main__':
    process_single_subject("2", "A", "FT", "M", feed="vga")
