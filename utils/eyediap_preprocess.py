
import cv2
import numpy as np
from tqdm import tqdm

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


def world_to_img(coordinates_3d, intrinsics, R, T):
    projected_coord = (((coordinates_3d - T.T) @ R.T) @ intrinsics.T)
    projected_coord[:, :2] = projected_coord[:, :2] / projected_coord[:, 2].reshape(-1, 1)
    return projected_coord


def img_to_world(coordinates_2dz, intrinsics, R, T):
    coordinates_2dz[:, :2] = coordinates_2dz[:, :2] * coordinates_2dz[:, 2].reshape(-1, 1)
    coordinates_2dz = coordinates_2dz @ np.linalg.inv(intrinsics.T) @ np.linalg.inv(R.T) + T.T
    return coordinates_2dz


def draw_line(frame, point_1, point_2, colour=(0, 255, 255), thickness=2):
    cv2.line(frame, (int(point_1[0]), int(point_1[1])), (int(point_2[0]), int(point_2[1])),
             colour, lineType=cv2.LINE_AA, thickness=thickness)


def draw_face_result(frame, face_bbox, face_lms):
    draw_face_landmarks(frame, np.concatenate(face_lms, axis=0))
    cv2.rectangle(frame, (face_bbox[0], face_bbox[1]), (face_bbox[2], face_bbox[3]), (0, 0, 255))
    centre = face_lms[2][-1].astype(np.int)
    cv2.rectangle(frame,
                  (centre[0] - int(FACE_CROP_SIZE / 2), centre[1] - int(FACE_CROP_SIZE / 2)),
                  (centre[0] + int(FACE_CROP_SIZE / 2), centre[1] + int(FACE_CROP_SIZE / 2)), (0, 255, 255))


def draw_face_landmarks(frame, face_lms):
    for lm in face_lms:
        cv2.circle(frame, (int(lm[0]), int(lm[1])), 2, (0, 0, 255))


def crop_after_3d_point(point, crop_tl, camera_parameters):
    intrinsics, R, T = camera_parameters
    point_2dz = world_to_img(point, intrinsics, R, T)
    point_2dz[:, :2] -= crop_tl
    return img_to_world(point_2dz, intrinsics, R, T)[0]


def process_single_subject(result_dict, subject_id, experiment_id, experiment_type, head_movement,
                           feed="vga", preview=True):
    """
    Process a single folder, the folder name can be decomposed to
        "subject_id"_"experiment_id"_"experiment_type"_"head_movement".

    :param result_dict:
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

    cam_K = np.array([[cam_intrinsics[0, 0], 0., cam_intrinsics[0, 2], 0.],
                      [0., cam_intrinsics[1, 1], cam_intrinsics[1, 2], 0.],
                      [0., 0., 0., 1.],
                      [0., 0., 1., 0.]])

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
        ball_proj = world_to_img(ball_3d_coord, cam_intrinsics, cam_R, cam_T)[:, :2]
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
    eye_l_proj = world_to_img(eye_l_3d_coord, cam_intrinsics, cam_R, cam_T)[:, :2]
    eye_r_proj = world_to_img(eye_r_3d_coord, cam_intrinsics, cam_R, cam_T)[:, :2]

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

    marked_video_writer = cv2.VideoWriter(data_folder_path + feed + "_marked.mov",
                                          cv2.VideoWriter_fourcc("m", "p", "4", "v"), 20.0, (640, 480))
    fld = FaceLandmarkDetector()
    for frame_idx in tqdm(range(len(eye_content))):
        # Read video stream.
        success, img = cap.read()
        if not success:
            break

        # Check if frame is valid.
        if not all(i == 0 for i in eye_content[frame_idx]) and \
                not all(i == 0 for i in target_content[frame_idx]) and \
                frame_valid_list[frame_idx] and \
                ((experiment_type in ["CS", "DS"] and not all(j == 0 for j in scr_2d_coord[frame_idx])) or
                    (experiment_type == "FT" and not all(j == 0 for j in ball_2d_coord[frame_idx]))):
            valid_frame = True
        else:
            valid_frame = False
            frame_valid_list[frame_idx] = False

        # Face detection and landmarks.
        if valid_frame:
            face_bbox, face_lms = fld.process_face(img)
            if face_bbox is None or face_lms is None:
                valid_frame = False
                frame_valid_list[frame_idx] = False

        # Preprocess data if valid frame.
        if valid_frame:
            face_top_left_coord = (face_lms[2][-1] - FACE_CROP_SIZE / 2).astype(np.int)
            cropped_face = img[face_top_left_coord[1]:face_top_left_coord[1] + FACE_CROP_SIZE,
                               face_top_left_coord[0]:face_top_left_coord[0] + FACE_CROP_SIZE]
            cropped_face_copy = cropped_face.copy()
            result_dict["subject_id"].append(subject_id)
            result_dict["cam_R"].append(cam_R)
            result_dict["cam_T"].append(cam_T.T[0])
            result_dict["cam_K"].append(cam_K)
            result_dict["frames"].append(cropped_face.copy())
            result_dict["face_box_tl"].append(face_top_left_coord)
            result_dict["face_landmarks"].append(np.concatenate(face_lms))
            draw_face_result(img, face_bbox, face_lms)
            draw_face_landmarks(cropped_face_copy, np.concatenate(face_lms) - face_top_left_coord)

            # Gazes.
            crop_eye_l_3d = crop_after_3d_point(eye_l_3d_coord[frame_idx],
                                                face_top_left_coord, (cam_intrinsics, cam_R, cam_T))
            crop_eye_r_3d = crop_after_3d_point(eye_r_3d_coord[frame_idx],
                                                face_top_left_coord, (cam_intrinsics, cam_R, cam_T))
            crop_eye_l_2d_proj = world_to_img(crop_eye_l_3d, cam_intrinsics, cam_R, cam_T)[0]
            crop_eye_r_2d_proj = world_to_img(crop_eye_r_3d, cam_intrinsics, cam_R, cam_T)[0]
            crop_eye_l_2d = eye_l_2d_coord[frame_idx] - face_top_left_coord
            crop_eye_r_2d = eye_r_2d_coord[frame_idx] - face_top_left_coord

            result_dict["left_eyeball_3d"].append(eye_r_3d_coord[frame_idx])  # Right eye is the left eye on image.
            result_dict["right_eyeball_3d"].append(eye_l_3d_coord[frame_idx])
            result_dict["left_eyeball_2d"].append(eye_l_2d_coord[frame_idx])
            result_dict["right_eyeball_2d"].append(eye_l_2d_coord[frame_idx])
            if experiment_type in ["CS", "DS"]:
                draw_line(img, eye_l_2d_coord[frame_idx], scr_2d_coord[frame_idx])
                draw_line(img, eye_r_2d_coord[frame_idx], scr_2d_coord[frame_idx])
                draw_line(img, eye_l_proj[frame_idx], scr_2d_coord[frame_idx], thickness=1)
                draw_line(img, eye_r_proj[frame_idx], scr_2d_coord[frame_idx], thickness=1)

                result_dict["target_screen"].append(scr_2d_coord[frame_idx])
            elif experiment_type == "FT":
                draw_line(img, eye_l_2d_coord[frame_idx], ball_2d_coord[frame_idx], colour=(0, 0, 255))
                draw_line(img, eye_r_2d_coord[frame_idx], ball_2d_coord[frame_idx], colour=(0, 0, 255))
                draw_line(img, eye_l_proj[frame_idx], ball_proj[frame_idx], thickness=1)
                draw_line(img, eye_r_proj[frame_idx], ball_proj[frame_idx], thickness=1)

                crop_target_3d = crop_after_3d_point(ball_3d_coord[frame_idx],
                                                     face_top_left_coord, (cam_intrinsics, cam_R, cam_T))
                crop_target_2d_proj = world_to_img(crop_target_3d, cam_intrinsics, cam_R, cam_T)[0]
                crop_target_2d = ball_2d_coord[frame_idx] - face_top_left_coord

                result_dict["target_3d"].append(ball_3d_coord[frame_idx])
                result_dict["target_2d"].append(ball_2d_coord[frame_idx])
                draw_line(cropped_face_copy, crop_eye_l_2d, crop_target_2d, colour=(0, 0, 255))
                draw_line(cropped_face_copy, crop_eye_r_2d, crop_target_2d, colour=(0, 0, 255))
                draw_line(cropped_face_copy, crop_eye_l_2d_proj, crop_target_2d_proj, thickness=1)
                draw_line(cropped_face_copy, crop_eye_r_2d_proj, crop_target_2d_proj, thickness=1)

            gaze_l = ball_3d_coord[frame_idx] - eye_l_3d_coord[frame_idx]
            rotation_l = np.array([np.arctan2(gaze_l[1], gaze_l[2]),
                                   np.arctan2(gaze_l[2], gaze_l[0]) - np.arctan2(1, 0)])
            gaze_r = ball_3d_coord[frame_idx] - eye_r_3d_coord[frame_idx]
            rotation_r = np.array([np.arctan2(gaze_r[1], gaze_r[2]),
                                   np.arctan2(gaze_r[2], gaze_r[0]) - np.arctan2(1, 0)])
            result_dict["left_eyeball_rotation"].append(rotation_r)
            result_dict["right_eyeball_rotation"].append(rotation_l)

            marked_video_writer.write(img)

        # Visualise
        if preview:
            cv2.imshow("Preview", img)
            if valid_frame:
                cv2.imshow("Cropped", cropped_face_copy)
            cv2.waitKey(1)
            del img

    cap.release()
    marked_video_writer.release()


if __name__ == '__main__':
    for d in os.listdir(EYEDIAP_PATH + "Data/"):
        print("Processing", d, "...")
        results = {"subject_id": [], "frames": [], "face_box_tl": [], "face_landmarks": [],
                   "target_3d": [], "left_eyeball_3d": [], "right_eyeball_3d": [],
                   "target_2d": [], "left_eyeball_2d": [], "right_eyeball_2d": [],
                   "left_eyeball_rotation": [], "right_eyeball_rotation": [], "cam_R": [], "cam_T": [], "cam_K": []}
        subject_id, experiment_id, experiment_type, head_movement = d.split("_")
        if experiment_id == "A" and experiment_type == "FT" and head_movement == "S":
            process_single_subject(results, subject_id, experiment_id, experiment_type, head_movement,
                                   feed="vga", preview=False)
            lengths = []
            for key, value in results.items():
                if key == "subject_id":
                    results[key] = np.array(results[key], dtype=np.int)
                else:
                    results[key] = np.stack(results[key])
                lengths.append(results[key].shape[0])
            assert np.unique(lengths).shape[0] == 1
            np.save(DATASETS_PATH + "eyediap/" + d, results)
