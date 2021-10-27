import os

import cv2
import h5py
import numpy as np
from tqdm import tqdm

CAM_NUMBER = 18
XGAZE_PATH = "../datasets/eth-xgaze/"


def cam_to_img(coordinates_3d, intrinsics, R=None, T=None):
    if R is None and T is None:
        projected_coord = coordinates_3d @ intrinsics.T
    else:
        projected_coord = (((coordinates_3d - T.T) @ R.T) @ intrinsics.T)
    projected_coord[:, :2] = projected_coord[:, :2] / projected_coord[:, 2].reshape(-1, 1)
    return projected_coord


def normalise_face(img, face_model, landmarks, hr, ht, gc, cam):
    """
    Copied and modified from ETH-XGaze repository, under the license of CC BY-NC-SA 4.0 license.

    :param img:
    :param face_model:
    :param landmarks:
    :param hr:
    :param ht:
    :param gc:
    :param cam:
    :return:
    """
    # Normalized camera parameters.
    focal_norm = 960  # Focal length of normalized camera.
    distance_norm = 600  # Normalized distance between eye and camera.
    roi_size = (224, 224)  # Size of cropped eye image.

    # Compute estimated 3D positions of the landmarks.
    ht = ht.reshape((3, 1))
    hR = cv2.Rodrigues(hr)[0]  # rotation matrix
    Fc = np.dot(hR, face_model.T) + ht
    face_centre = get_face_centre(Fc)

    # Normalise image.
    distance = np.linalg.norm(face_centre)  # Actual distance between eye and original camera.

    z_scale = distance_norm / distance
    cam_norm = np.array([
        [focal_norm, 0, roi_size[0] / 2],
        [0, focal_norm, roi_size[1] / 2],
        [0, 0, 1.0],
    ])
    S = np.array([  # Scaling matrix.
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, z_scale],
    ])

    hRx = hR[:, 0]
    forward = (face_centre / distance).reshape(3)
    down = np.cross(forward, hRx)
    down /= np.linalg.norm(down)
    right = np.cross(down, forward)
    right /= np.linalg.norm(right)
    R = np.c_[right, down, forward].T  # Rotation matrix R for head pose normalisation.

    W = np.dot(np.dot(cam_norm, S), np.dot(R, np.linalg.inv(cam)))  # Transformation matrix.

    img_warped = None
    # img_warped = cv2.warpPerspective(img, W, roi_size)  # Image normalization.

    # Normalise rotation.
    hR_norm = np.dot(R, hR)  # Rotation matrix in normalized space.
    hr_norm = cv2.Rodrigues(hR_norm)[0]  # Convert rotation matrix to rotation vectors.

    # Normalise gaze vector.
    gc = gc.reshape((3, 1))
    gc_normalised = gc - face_centre  # Gaze vector.
    gc_normalised = np.dot(R, gc_normalised)
    gc_normalised = gc_normalised / np.linalg.norm(gc_normalised)

    # Warp the facial landmarks.
    landmarks_warped = cv2.perspectiveTransform(landmarks[:, None, :], W)
    landmarks_warped = landmarks_warped.reshape(-1, 2)

    # Return normalised and cropped image,
    #        head pose rotation,
    return img_warped, hr_norm, gc_normalised, landmarks_warped, R, face_centre.flatten(), W


def transform_face_model_3d(head_r, head_T, face_lm_model):
    head_T = head_T.reshape((3, 1))
    head_R = cv2.Rodrigues(head_r)[0]
    fm_transformed = np.dot(head_R, face_lm_model.T) + head_T

    return fm_transformed.T


def get_face_centre(Fc):
    two_eye_center = np.mean(Fc[:, 0:4], axis=1).reshape((3, 1))
    mouth_center = np.mean(Fc[:, 4:6], axis=1).reshape((3, 1))
    face_centre = np.mean(np.concatenate((two_eye_center, mouth_center), axis=1), axis=1).reshape((3, 1))

    return face_centre


def process_subjects(partition="train"):
    # Load camera calibration.
    cam_param = []
    cam_intrinsics_list = []
    for i in range(CAM_NUMBER):
        fs = cv2.FileStorage(XGAZE_PATH + "calibration/cam_calibration/cam%02d.xml" % i, cv2.FILE_STORAGE_READ)
        cam_intrinsics = fs.getNode("Camera_Matrix").mat()
        distortion_coefficients = fs.getNode("Distortion_Coefficients").mat()
        cam_rotation = fs.getNode("cam_rotation").mat()
        cam_translation = fs.getNode("cam_translation").mat()
        cam_param.append([cam_intrinsics, cam_rotation, cam_translation])  # Transpose the camera rotation here.
        cam_intrinsics_list.append(cam_intrinsics)
        fs.release()

    # Load face model.
    face_model = np.loadtxt(XGAZE_PATH + "calibration/face_model.txt")
    face_lm_model = face_model[[20, 23, 26, 29, 15, 19], :]

    # Load annotation.
    subject_id_list = []
    frame_id_list = []
    cam_id_list = []
    gaze_point_camera_list = []
    face_centre_camera_list = []
    face_landmarks_crop_list = []
    warp_matrix_list = []
    for subject_file in tqdm(os.listdir(XGAZE_PATH + "data/annotation_" + partition)):
        subject_id = int(subject_file[7:11])
        with open(XGAZE_PATH + "data/annotation_train/subject%04d.csv" % subject_id, "r") as f:
            lines = f.readlines()
            for li in lines:
                li = li.split(",")
                frame_id = int(li[0][5:])
                cam_id = int(li[1][3:5])
                rest = np.array(li[2:], dtype=np.float)
                gaze_point_screen = rest[:2]
                gaze_point_camera = rest[2:5]
                head_pose_R_camera = rest[5:8]
                head_pose_T_camera = rest[8:11]
                face_landmarks = rest[11:].reshape(-1, 2)

                _, _, _, landmarks_warped, _, face_centre, W = \
                    normalise_face(None, face_lm_model, face_landmarks, head_pose_R_camera, head_pose_T_camera,
                                   gaze_point_camera, cam_param[cam_id][0])

                subject_id_list.append(subject_id)
                frame_id_list.append(frame_id)
                cam_id_list.append(cam_id)
                gaze_point_camera_list.append(gaze_point_camera)
                face_centre_camera_list.append(face_centre)
                face_landmarks_crop_list.append(landmarks_warped)
                warp_matrix_list.append(W)

    # Save stuff.
    np.save(XGAZE_PATH + "processed/cam_intrinsics", np.stack(cam_intrinsics_list))
    np.save(XGAZE_PATH + "processed/additional_labels_" + partition,
            {"subject_id_list": np.stack(subject_id_list),
             "frame_id_list": np.stack(frame_id_list), "cam_id_list": np.stack(cam_id_list),
             "gaze_point_camera_list": np.stack(gaze_point_camera_list),
             "face_centre_camera_list": np.stack(face_centre_camera_list),
             "face_landmarks_crop_list": np.stack(face_landmarks_crop_list),
             "warp_matrix_list": np.stack(warp_matrix_list)})

    # # Load annotation old.
    # annotation_train = {}
    # with open(XGAZE_PATH + "data/annotation_train/subject%04d.csv" % subject_id, "r") as f:
    #     lines = f.readlines()
    #     for li in lines:
    #         li = li.split(",")
    #         frame_name = li[0]
    #         img_cam_name = li[1]
    #         rest = np.array(li[2:], dtype=np.float)
    #         annotation = {"gaze_point_screen": rest[:2], "gaze_point_camera": rest[2:5],
    #                       "head_pose_R_camera": rest[5:8], "head_pose_T_camera": rest[8:11],
    #                       "face_landmarks": rest[11:].reshape(-1, 2)}
    #         if frame_name not in annotation_train:
    #             annotation_train[frame_name] = {}
    #         annotation_train[frame_name][img_cam_name] = annotation

    # # Initialise write file.
    # for frame_folder_name in os.listdir(XGAZE_PATH + "data/" + partition + "/subject%04d" % subject_id):
    #     frame_id = int(frame_folder_name[5:])
    #     for i in range(CAM_NUMBER):
    #         image_file_name = "cam%02d.JPG" % i
    #
    #         # gaze_point_screen = annotation_train[frame_folder_name][image_file_name]["gaze_point_screen"]
    #         gaze_point_camera = annotation_train[frame_folder_name][image_file_name]["gaze_point_camera"]
    #         head_pose_R_camera = annotation_train[frame_folder_name][image_file_name]["head_pose_R_camera"]
    #         head_pose_T_camera = annotation_train[frame_folder_name][image_file_name]["head_pose_T_camera"]
    #         face_landmarks = annotation_train[frame_folder_name][image_file_name]["face_landmarks"]
    #
    #         frame = cv2.imread(XGAZE_PATH + "data/train/subject%04d/" % subject_id +
    #                            frame_folder_name + "/" + image_file_name)
    #         if i in [3, 6, 13]:
    #             frame = cv2.flip(frame, -1)
    #         # frame = None
    #         frame_warped, hr_norm, gc_normalised, landmarks_warped, R, face_centre, W = \
    #             normalise_face(frame, face_lm_model, face_landmarks, head_pose_R_camera, head_pose_T_camera,
    #                            gaze_point_camera, cam_param[i][0])
    #         # gaze_theta = np.arcsin((-1) * gc_normalised[1])
    #         # gaze_phi = np.arctan2((-1) * gc_normalised[0], (-1) * gc_normalised[2])
    #         # gaze_norm_2d = np.asarray([gaze_theta, gaze_phi])
    #
    #         # Points from camera space to image space, then crop.
    #         face_centre = face_centre.reshape(1, 3)
    #         face_centre = cam_to_img(face_centre, cam_param[i][0])
    #         face_centre = np.concatenate([cv2.perspectiveTransform(face_centre[:, None, :2], W, (224, 224))[:, 0, :],
    #                                       face_centre[:, 2, None]],
    #                                      axis=1)
    #         face_centre = face_centre.flatten()

    #         fm = cam_to_img(transform_face_model_3d(head_pose_R_camera, head_pose_T_camera, face_model), cam_param[i][0])
    #         fm_crop = np.concatenate([cv2.perspectiveTransform(fm[:, None, :2], W, (224, 224))[:, 0, :], fm[:, 2, None]],
    #                                  axis=1)
    #
    #         fc_crop = get_face_centre(fm_crop[[20, 23, 26, 29, 15, 19], :].T)[:, 0]
    #
    #         gaze_point_image = cam_to_img(gaze_point_camera[None, :], cam_param[i][0])
    #         gaze_point_image_crop = np.concatenate([
    #                 cv2.perspectiveTransform(gaze_point_image[:, None, :2], W, (224, 224))[:, 0, :],
    #                 gaze_point_image[:, 2, None]], axis=1)[0]
    #
    #         for j in landmarks_warped:
    #             cv2.drawMarker(frame_warped, (int(j[0]), int(j[1])), (0, 255, 0),
    #                            markerSize=4, markerType=cv2.MARKER_TILTED_CROSS, thickness=1)
    #         for j in fm_crop:
    #             cv2.drawMarker(frame_warped, (int(j[0]), int(j[1])), (0, 0, 255),
    #                            markerSize=4, markerType=cv2.MARKER_TILTED_CROSS, thickness=1)
    #         # for j in face_landmarks:
    #         #     cv2.drawMarker(frame, (int(j[0]), int(j[1])), (0, 255, 0),
    #         #                    markerSize=20, markerType=cv2.MARKER_TILTED_CROSS, thickness=5)
    #         # for j in fm:
    #         #     cv2.drawMarker(frame, (int(j[0]), int(j[1])), (0, 0, 255),
    #         #                    markerSize=20, markerType=cv2.MARKER_TILTED_CROSS, thickness=5)
    #
    #         cv2.arrowedLine(frame_warped,
    #                         (int(face_centre[0]), int(face_centre[1])),
    #                         (int(gaze_point_image_crop[0]), int(gaze_point_image_crop[1])),
    #                         (0, 0, 255), thickness=2)
    #         cv2.arrowedLine(frame_warped,
    #                         (int(fc_crop[0]), int(fc_crop[1])),
    #                         (int(fc_crop[0] + gc_normalised[0] * 100), int(fc_crop[1] + gc_normalised[1] * 100)),
    #                         (0, 255, 0), thickness=2)
    #
    #         frame = cv2.resize(frame, (frame.shape[1] // 5, frame.shape[0] // 5))
    #         cv2.imshow("", frame_warped)
    #         cv2.waitKey()


if __name__ == '__main__':
    process_subjects()
