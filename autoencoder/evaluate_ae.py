import logging

import cv2
import torch
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader

import numpy as np
from torchvision.transforms import Resize
from tqdm import tqdm

from autoencoder.model import Autoencoder
from constants import *
from utils.eyediap_dataset import EYEDIAP
from utils.eyediap_preprocess import world_to_img, img_to_world

NAME = "v1_m_server"


def evaluate(qualitative=False):
    logging.info(NAME)
    
    model = Autoencoder()
    model.load_state_dict(torch.load(LOGS_PATH + NAME + "/model_best.pt"))
    model.eval()

    test_data = EYEDIAP(partition="test", head_movement=["M"])
    test_loader = DataLoader(test_data, batch_size=1, shuffle=True, num_workers=0)

    l_gaze_angle_errors_rot = []
    r_gaze_angle_errors_rot = []
    l_gaze_rot_axis_pred = []
    r_gaze_rot_axis_pred = []
    l_gaze_rot_axis_gt = []
    r_gaze_rot_axis_gt = []
    l_gaze_angle_errors_tgt = []
    r_gaze_angle_errors_tgt = []
    f_gaze_angle_errors_tgt = []
    for data in tqdm(test_loader):
        gt_img = data["frames"].to(torch.float32) / 255.
        camera_parameters = (data["cam_R"], data["cam_T"], data["cam_K"])

        # Preprocess.
        resize_transformation = Resize(224)
        gt_img_input = resize_transformation(gt_img.permute(0, 3, 1, 2))

        # Forward.
        with torch.no_grad():
            results = model(gt_img_input, camera_parameters)

        # Revert to original position.
        face_box_tl = data["face_box_tl"].cpu().numpy()
        l_eyeball_centre = results["l_eyeball_centre"][0].cpu().numpy()
        r_eyeball_centre = results["r_eyeball_centre"][0].cpu().numpy()
        l_gaze_rot = results["left_gaze"][0].cpu().numpy() + l_eyeball_centre
        r_gaze_rot = results["right_gaze"][0].cpu().numpy() + r_eyeball_centre
        # l_gaze_rot = Autoencoder.apply_eyeball_rotation(
        #     torch.tensor([[[0., 0., 1.]]], device=device),
        #     torch.tensor([[[0., 0., 0.]]], device=device),
        #     results["left_eye_rotation"] + torch.tensor([[-0.00849474, -0.02305407]],
        #                                                 device=device))[0].cpu().numpy() \
        #     + l_eyeball_centre
        # r_gaze_rot = Autoencoder.apply_eyeball_rotation(
        #     torch.tensor([[[0., 0., 1.]]], device=device),
        #     torch.tensor([[[0., 0., 0.]]], device=device),
        #     results["right_eye_rotation"] + torch.tensor([[-0.01840461, -0.03251669]],
        #                                                  device=device))[0].cpu().numpy() \
        #     + r_eyeball_centre
        target = results["gaze_point_mid"][0].cpu().numpy()
        cam_R, cam_T, cam_K = camera_parameters
        cam_intrinsics = cam_K[0, :3, :3]
        cam_intrinsics[2, 2] = 1.
        cam_R, cam_T, cam_intrinsics = cam_R[0].cpu().numpy(), cam_T[0].cpu().numpy(), cam_intrinsics.cpu().numpy()

        l_gaze_rot_axis_pred.append(results["left_eye_rotation"][0].cpu().numpy())
        r_gaze_rot_axis_pred.append(results["right_eye_rotation"][0].cpu().numpy())
        l_gaze_rot_axis_gt.append(data["left_eyeball_rotation_crop"][0].cpu().numpy())
        r_gaze_rot_axis_gt.append(data["right_eyeball_rotation_crop"][0].cpu().numpy())
        l_eyeball_centre = revert_to_original_position(l_eyeball_centre, face_box_tl, cam_intrinsics, cam_R, cam_T)
        r_eyeball_centre = revert_to_original_position(r_eyeball_centre, face_box_tl, cam_intrinsics, cam_R, cam_T)
        l_gaze_rot = revert_to_original_position(l_gaze_rot, face_box_tl, cam_intrinsics, cam_R, cam_T)
        r_gaze_rot = revert_to_original_position(r_gaze_rot, face_box_tl, cam_intrinsics, cam_R, cam_T)
        target = revert_to_original_position(target, face_box_tl, cam_intrinsics, cam_R, cam_T)

        # Calculate angle errors.
        l_gaze_angle_errors_rot.append(
            get_angle(l_gaze_rot - l_eyeball_centre,
                      data["target_3d"].cpu().numpy() - data["left_eyeball_3d"].cpu().numpy()))
        r_gaze_angle_errors_rot.append(
            get_angle(r_gaze_rot - r_eyeball_centre,
                      data["target_3d"].cpu().numpy() - data["right_eyeball_3d"].cpu().numpy()))

        l_gaze_angle_errors_tgt.append(
            get_angle(target - l_eyeball_centre,
                      data["target_3d"].cpu().numpy() - data["left_eyeball_3d"].cpu().numpy()))
        r_gaze_angle_errors_tgt.append(
            get_angle(target - r_eyeball_centre,
                      data["target_3d"].cpu().numpy() - data["right_eyeball_3d"].cpu().numpy()))

        face_point = (data["right_eyeball_3d"].cpu().numpy() + data["left_eyeball_3d"].cpu().numpy()) / 2
        f_gaze_angle_errors_tgt.append(
            get_angle(target - face_point,
                      data["target_3d"].cpu().numpy() - face_point))

    logging.info("Left gaze rot error, mean: " + str(np.mean(l_gaze_angle_errors_rot)) +
                 ", std: " + str(np.std(l_gaze_angle_errors_rot)) +
                 ", median: " + str(np.median(l_gaze_angle_errors_rot)))
    logging.info("Right gaze rot error, mean: " + str(np.mean(r_gaze_angle_errors_rot)) +
                 ", std: " + str(np.std(r_gaze_angle_errors_rot)) +
                 ", median: " + str(np.median(r_gaze_angle_errors_rot)))
    logging.info("Total gaze rot error, mean: " + str(np.mean(l_gaze_angle_errors_rot + r_gaze_angle_errors_rot)) +
                 ", std: " + str(np.std(l_gaze_angle_errors_rot + r_gaze_angle_errors_rot)) +
                 ", median: " + str(np.median(l_gaze_angle_errors_rot + r_gaze_angle_errors_rot)))

    logging.info("Left gaze tgt error, mean: " + str(np.mean(l_gaze_angle_errors_tgt)) +
                 ", std: " + str(np.std(l_gaze_angle_errors_tgt)) +
                 ", median: " + str(np.median(l_gaze_angle_errors_tgt)))
    logging.info("Right gaze tgt error, mean: " + str(np.mean(r_gaze_angle_errors_tgt)) +
                 ", std: " + str(np.std(r_gaze_angle_errors_tgt)) +
                 ", median: " + str(np.median(r_gaze_angle_errors_tgt)))
    logging.info("Total gaze tgt error, mean: " + str(np.mean(l_gaze_angle_errors_tgt + r_gaze_angle_errors_tgt)) +
                 ", std: " + str(np.std(l_gaze_angle_errors_tgt + r_gaze_angle_errors_tgt)) +
                 ", median: " + str(np.median(l_gaze_angle_errors_tgt + r_gaze_angle_errors_tgt)))

    logging.info("Face gaze tgt error, mean: " + str(np.mean(f_gaze_angle_errors_tgt)) +
                 ", std: " + str(np.std(f_gaze_angle_errors_tgt)) +
                 ", median: " + str(np.median(f_gaze_angle_errors_tgt)))

    l_gaze_rot_axis_pred = np.stack(l_gaze_rot_axis_pred)
    r_gaze_rot_axis_pred = np.stack(r_gaze_rot_axis_pred)
    l_gaze_rot_axis_gt = np.stack(l_gaze_rot_axis_gt)
    r_gaze_rot_axis_gt = np.stack(r_gaze_rot_axis_gt)
    draw_distribution_scatter(l_gaze_rot_axis_pred, l_gaze_rot_axis_gt)
    draw_distribution_scatter(r_gaze_rot_axis_pred, r_gaze_rot_axis_gt)


def revert_to_original_position(point_3d, face_box_tl, cam_intrinsics, cam_R, cam_T):
    point2dz = world_to_img(point_3d, cam_intrinsics, cam_R, cam_T)
    point2dz[:, :2] += face_box_tl
    point3d_orig = img_to_world(point2dz, cam_intrinsics, cam_R, cam_T)
    return point3d_orig


def get_angle(vec_a, vec_b):
    return np.rad2deg(np.arccos(np.dot(vec_a[0], vec_b[0]) / (np.linalg.norm(vec_a[0]) * np.linalg.norm(vec_b[0]))))


def draw_distribution_scatter(pred, gt):
    plt.figure()
    plt.scatter(pred[:, 0], pred[:, 1], marker=".")
    plt.scatter(gt[:, 0], gt[:, 1], marker=".")
    plt.show()
    plt.figure()
    plt.scatter((gt - pred)[:, 0], (gt - pred)[:, 1], marker=".")
    plt.scatter(np.mean((gt - pred)[:, 0]), np.mean((gt - pred)[:, 1]), marker="x")
    print(np.mean(gt - pred, axis=0))
    plt.show()


if __name__ == '__main__':
    if os.path.exists(LOGS_PATH + NAME + "/eval.txt"):
        os.remove(LOGS_PATH + NAME + "/eval.txt")

    console_handler = logging.StreamHandler()
    logging.basicConfig(
        handlers=[logging.FileHandler(filename=LOGS_PATH + NAME + "/eval.txt"), console_handler],
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(message)s", datefmt="[%Y-%m-%d %H:%M:%S]")
    console_handler.setLevel(logging.INFO)

    evaluate()
