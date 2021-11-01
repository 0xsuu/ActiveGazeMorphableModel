
import json
import logging
from types import SimpleNamespace

import cv2
import torch
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
import numpy as np
from torchvision.transforms import Resize, Grayscale
from tqdm import tqdm

from autoencoder.model import Autoencoder, AutoencoderBaseline
from constants import *
from utils.eyediap_dataset import EYEDIAP
from utils.camera_model import world_to_img, img_to_world
from utils.xgaze_dataset import XGazeDataset

NAME = "v5_swin_xgaze_lb"
EPOCH = "best"
PARTITION = "cv"


def evaluate(qualitative=False):
    logging.info(NAME)
    with open(LOGS_PATH + NAME + "/config.json", "r") as f:
        args = SimpleNamespace(**json.load(f))

        # Backward compatibility.
        if not hasattr(args, "dataset"):
            args.dataset = "eyediap"
        if not hasattr(args, "auto_weight_loss"):
            args.auto_weight_loss = False
            if "lb" in NAME:
                args.auto_weight_loss = True

    model = Autoencoder(args)
    saved_state_dict = torch.load(LOGS_PATH + NAME + "/model_" + EPOCH + ".pt")

    # Load checkpoint and set to evaluate.
    model.load_state_dict(saved_state_dict)
    model.eval()

    # Load test dataset.
    test_data = XGazeDataset(partition=PARTITION)
    test_loader = DataLoader(test_data, batch_size=256, shuffle=False, num_workers=0)

    if PARTITION == "test":
        rot_pred_list = []
        rot_pred_ogt_list = []
    rot_angle_errors = []
    rot_angle_errors_ogt = []
    f_gaze_angle_errors_tgt = []
    f_gaze_angle_errors_tgt_ogt = []
    progress = tqdm(test_loader)
    # counter = 0
    for data in progress:
        # if counter >= 256:
        #     break
        # counter += 1
        gt_img = data["frames"].to(torch.float32) / 255.
        camera_parameters = data["cam_intrinsics"]
        warp_matrices = data["warp_matrices"]
        head_rotation = data["head_rotations"]

        # Preprocess.
        gt_img_input = gt_img.permute(0, 3, 1, 2)

        # Forward.
        with torch.no_grad():
            results = model(gt_img_input, camera_parameters, warp_matrices)

        # Extract forward results.
        target_pred = results["gaze_point_mid"][:, 0]
        origin_pred = results["face_centre"]
        rot_pred = vector_to_pitchyaw(
            ((origin_pred - target_pred).unsqueeze(1) @ torch.transpose(head_rotation, 1, 2))[:, 0].cpu().numpy())

        origin_gt = data["gaze_origins"]
        rot_pred_ogt = vector_to_pitchyaw(
            ((origin_gt - target_pred).unsqueeze(1) @ torch.transpose(head_rotation, 1, 2))[:, 0].cpu().numpy())

        if qualitative:
            pass

        # Calculate angle errors.
        if PARTITION != "test":
            target_gt = data["target_3d_crop"]
            rot_gt = data["face_gazes"].cpu().numpy()

            rot_angle_errors.append(angular_error(pitchyaw_to_vector(rot_pred), pitchyaw_to_vector(rot_gt)))
            rot_angle_errors_ogt.append(angular_error(pitchyaw_to_vector(rot_pred_ogt), pitchyaw_to_vector(rot_gt)))
            f_gaze_angle_errors_tgt.append(angular_error((target_pred - origin_pred).cpu().numpy(),
                                                         (target_gt - origin_gt).cpu().numpy()))
            f_gaze_angle_errors_tgt_ogt.append(angular_error((target_pred - origin_gt).cpu().numpy(),
                                                             (target_gt - origin_gt).cpu().numpy()))

            progress.set_description("rot: %.05f, rot_ogt: %.05f, tgt: %.05f, tgt_ogt: %.05f" %
                                     (np.mean([j.mean() for j in rot_angle_errors]),
                                      np.mean([j.mean() for j in rot_angle_errors_ogt]),
                                      np.mean([j.mean() for j in f_gaze_angle_errors_tgt]),
                                      np.mean([j.mean() for j in f_gaze_angle_errors_tgt_ogt])))
        else:
            rot_pred_list.append(rot_pred)
            rot_pred_ogt_list.append(rot_pred_ogt)

    if PARTITION == "test":
        rot_pred_list = np.concatenate(rot_pred_list, axis=0)
        rot_pred_ogt_list = np.concatenate(rot_pred_ogt_list, axis=0)

        np.savetxt(LOGS_PATH + NAME + "/within_eva_results.txt", rot_pred_list, delimiter=",")
        np.savetxt(LOGS_PATH + NAME + "/within_eva_results_ogt.txt", rot_pred_ogt_list, delimiter=",")
    else:
        rot_angle_errors = np.concatenate(rot_angle_errors, axis=0)
        f_gaze_angle_errors_tgt = np.concatenate(f_gaze_angle_errors_tgt, axis=0)
        f_gaze_angle_errors_tgt_ogt = np.concatenate(f_gaze_angle_errors_tgt_ogt, axis=0)
        logging.info("Face gaze tgt error, mean: " + str(np.mean(rot_angle_errors)) +
                     ", std: " + str(np.std(rot_angle_errors)) +
                     ", median: " + str(np.median(rot_angle_errors)))
        logging.info("Face gaze tgt error, mean: " + str(np.mean(f_gaze_angle_errors_tgt)) +
                     ", std: " + str(np.std(f_gaze_angle_errors_tgt)) +
                     ", median: " + str(np.median(f_gaze_angle_errors_tgt)))
        logging.info("Face gaze tgt gt error, mean: " + str(np.mean(f_gaze_angle_errors_tgt_ogt)) +
                     ", std: " + str(np.std(f_gaze_angle_errors_tgt_ogt)) +
                     ", median: " + str(np.median(f_gaze_angle_errors_tgt_ogt)))


def vector_to_pitchyaw(vectors):
    r"""Convert given gaze vectors to yaw (:math:`\theta`) and pitch (:math:`\phi`) angles.
    Args:
        vectors (:obj:`numpy.array`): gaze vectors in 3D :math:`(n\times 3)`.
    Returns:
        :obj:`numpy.array` of shape :math:`(n\times 2)` with values in radians.
    """
    n = vectors.shape[0]
    out = np.empty((n, 2))
    vectors = np.divide(vectors, np.linalg.norm(vectors, axis=1).reshape(n, 1))
    out[:, 0] = np.arcsin(vectors[:, 1])  # theta
    out[:, 1] = np.arctan2(vectors[:, 0], vectors[:, 2])  # phi
    return out


def pitchyaw_to_vector(pitchyaws):
    r"""Convert given yaw (:math:`\theta`) and pitch (:math:`\phi`) angles to unit gaze vectors.
    Args:
        pitchyaws (:obj:`numpy.array`): yaw and pitch angles :math:`(n\times 2)` in radians.
    Returns:
        :obj:`numpy.array` of shape :math:`(n\times 3)` with 3D vectors per row.
    """
    n = pitchyaws.shape[0]
    sin = np.sin(pitchyaws)
    cos = np.cos(pitchyaws)
    out = np.empty((n, 3))
    out[:, 0] = np.multiply(cos[:, 0], sin[:, 1])
    out[:, 1] = sin[:, 0]
    out[:, 2] = np.multiply(cos[:, 0], cos[:, 1])
    return out


def angular_error(a, b):
    """Calculate angular error (via cosine similarity)."""
    a = pitchyaw_to_vector(a) if a.shape[1] == 2 else a
    b = pitchyaw_to_vector(b) if b.shape[1] == 2 else b

    ab = np.sum(np.multiply(a, b), axis=1)
    a_norm = np.linalg.norm(a, axis=1)
    b_norm = np.linalg.norm(b, axis=1)

    # Avoid zero-values (to avoid NaNs)
    a_norm = np.clip(a_norm, a_min=1e-7, a_max=None)
    b_norm = np.clip(b_norm, a_min=1e-7, a_max=None)

    similarity = np.divide(ab, np.multiply(a_norm, b_norm))

    return np.arccos(similarity) * 180.0 / np.pi


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


def draw_gaze(image, l_eyeball_centre_screen_gt, target_screen_gt, r_eyeball_centre_screen_gt,
              l_eyeball_centre_screen, target_screen_l, r_eyeball_centre_screen, target_screen_r):
    """
    Draw the ground truth and prediction gaze direction in 2D on the given image.

    Ground truth gaze is in green, predicted gaze is in red.

    :param image: Image to draw.
    :param l_eyeball_centre_screen_gt: Ground truth left eyeball centre coordinate in screen coordinate system.
    :param target_screen_gt: Ground truth target coordinate in screen coordinate system.
    :param r_eyeball_centre_screen_gt: Ground truth right eyeball centre coordinate in screen coordinate system.
    :param l_eyeball_centre_screen: Predicted left eyeball centre in screen coordinate system.
    :param target_screen_l: Predicted left target coordinate in screen coordinate system.
    :param r_eyeball_centre_screen: Predicted right eyeball centre coordinate in screen coordinate system.
    :param target_screen_r: Predicted right target coordinate in screen coordinate system.
    :return: None
    """
    def draw_line(origin, target, colour):
        cv2.line(image,
                 (int(origin[0] / FACE_CROP_SIZE * image.shape[0]),
                  int(origin[1] / FACE_CROP_SIZE * image.shape[1])),
                 (int(target[0] / FACE_CROP_SIZE * image.shape[0]),
                  int(target[1] / FACE_CROP_SIZE * image.shape[1])),
                 color=colour, lineType=cv2.LINE_AA, thickness=1)
    draw_line(l_eyeball_centre_screen_gt, target_screen_gt, (0, 255, 0))
    draw_line(r_eyeball_centre_screen_gt, target_screen_gt, (0, 255, 0))
    draw_line(l_eyeball_centre_screen, target_screen_l, (0, 0, 255))
    draw_line(r_eyeball_centre_screen, target_screen_r, (0, 0, 255))


if __name__ == '__main__':
    if os.path.exists(LOGS_PATH + NAME + "/eval" + EPOCH + ".txt"):
        os.remove(LOGS_PATH + NAME + "/eval" + EPOCH + ".txt")
        while os.path.exists(LOGS_PATH + NAME + "/eval" + EPOCH + ".txt"):
            pass

    console_handler = logging.StreamHandler()
    logging.basicConfig(
        handlers=[logging.FileHandler(filename=LOGS_PATH + NAME + "/eval" + EPOCH + ".txt"), console_handler],
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(message)s", datefmt="[%Y-%m-%d %H:%M:%S]")
    console_handler.setLevel(logging.INFO)

    evaluate(qualitative=True)
