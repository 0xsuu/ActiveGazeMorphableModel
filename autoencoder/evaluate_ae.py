import json
import logging
from collections import OrderedDict
from types import SimpleNamespace

import cv2
import torch
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
import numpy as np
from torchvision.transforms import Resize, Grayscale
from tqdm import tqdm
from psbody.mesh import Mesh

from autoencoder.model import Autoencoder, AutoencoderBaseline
from constants import *
from utils.eyediap_dataset import EYEDIAP
from utils.camera_model import world_to_img, img_to_world

NAME = "v5_swin_baseline"


def evaluate(qualitative=False):
    logging.info(NAME)
    with open(LOGS_PATH + NAME + "/config.json", "r") as f:
        args = SimpleNamespace(**json.load(f))

    if "baseline" in NAME:
        model = AutoencoderBaseline(args)
    else:
        model = Autoencoder(args)
    saved_state_dict = torch.load(LOGS_PATH + NAME + "/model_best.pt")

    # # Version change fixing. Modify the saved state dict for backward compatibility.
    # new_saved_state_dict = OrderedDict()
    # for k, v in saved_state_dict.items():
    #     if "encoder" in k and "net" not in k:
    #         new_saved_state_dict[".".join(["encoder", "net"] + k.split(".")[1:])] = v
    #     else:
    #         new_saved_state_dict[k] = v
    # saved_state_dict = new_saved_state_dict

    # Load checkpoint and set to evaluate.
    model.load_state_dict(saved_state_dict)
    model.eval()

    # Load test dataset.
    test_data = EYEDIAP(partition="test", eval_subject=16, head_movement=["M", "S"])
    test_loader = DataLoader(test_data, batch_size=1, shuffle=False, num_workers=0)

    frame_transform = Resize(224)
    l_eye_patch_transformation = Grayscale()
    r_eye_patch_transformation = Grayscale()

    # Initiate video writer. The qualitative result will be saved in a video.
    rendered_video_writer = cv2.VideoWriter(LOGS_PATH + NAME + "/result_full.mov",
                                            cv2.VideoWriter_fourcc("m", "p", "4", "v"), 20.0, (1024 + 512, 512))

    l_gaze_angle_errors_rot = []  # Gaze angle error calculated by rotation ground truth.
    r_gaze_angle_errors_rot = []
    l_gaze_angle_errors_rot_gt = []  # Gaze angle error calculated by rotation ground truth, gaze vector only,
    r_gaze_angle_errors_rot_gt = []  # in cropped space.
    l_gaze_rot_axis_pred = []
    r_gaze_rot_axis_pred = []
    l_gaze_rot_axis_gt = []
    r_gaze_rot_axis_gt = []
    l_gaze_angle_errors_tgt = []  # Gaze angle error calculated by eye centre to target vector ground truth.
    r_gaze_angle_errors_tgt = []
    # Gaze angle error calculated by eye centre to target vector ground truth. Eye centre ground truth is available.
    l_gaze_angle_errors_tgt_gt = []
    r_gaze_angle_errors_tgt_gt = []
    f_gaze_angle_errors_tgt = []
    for data in tqdm(test_loader):
        gt_img = data["frames"].to(torch.float32) / 255.
        camera_parameters = (data["cam_R"], data["cam_T"], data["cam_K"])

        # Preprocess.
        gt_img_input = frame_transform(gt_img.permute(0, 3, 1, 2))

        # Forward.
        with torch.no_grad():
            results = model(gt_img_input, camera_parameters)

        # Extract forward results.
        face_box_tl = data["face_box_tl"].cpu().numpy()
        l_eyeball_centre = results["l_eyeball_centre"][0].cpu().numpy()
        r_eyeball_centre = results["r_eyeball_centre"][0].cpu().numpy()
        l_gaze_rot = results["left_gaze"][0].cpu().numpy() + l_eyeball_centre
        r_gaze_rot = results["right_gaze"][0].cpu().numpy() + r_eyeball_centre

        # # Fix for Kapa angle.
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
        target_l = results["gaze_point_l"][0].cpu().numpy()
        target_r = results["gaze_point_r"][0].cpu().numpy()

        # Process camera parameters.
        cam_R, cam_T, cam_K = camera_parameters
        cam_intrinsics = cam_K[0, :3, :3]
        cam_intrinsics[2, 2] = 1.
        cam_R, cam_T, cam_intrinsics = cam_R[0].cpu().numpy(), cam_T[0].cpu().numpy(), cam_intrinsics.cpu().numpy()

        l_gaze_rot_axis_pred.append(results["left_eye_rotation"][0].cpu().numpy())
        r_gaze_rot_axis_pred.append(results["right_eye_rotation"][0].cpu().numpy())
        l_gaze_rot_axis_gt.append(data["left_eyeball_rotation_crop"][0].cpu().numpy())
        r_gaze_rot_axis_gt.append(data["right_eyeball_rotation_crop"][0].cpu().numpy())

        # Revert cropping.
        l_eyeball_centre_orig = revert_to_original_position(l_eyeball_centre, face_box_tl, cam_intrinsics, cam_R, cam_T)
        r_eyeball_centre_orig = revert_to_original_position(r_eyeball_centre, face_box_tl, cam_intrinsics, cam_R, cam_T)
        l_gaze_rot = revert_to_original_position(l_gaze_rot, face_box_tl, cam_intrinsics, cam_R, cam_T)
        r_gaze_rot = revert_to_original_position(r_gaze_rot, face_box_tl, cam_intrinsics, cam_R, cam_T)
        target_orig = revert_to_original_position(target, face_box_tl, cam_intrinsics, cam_R, cam_T)
        target_orig_l = revert_to_original_position(target_l, face_box_tl, cam_intrinsics, cam_R, cam_T)
        target_orig_r = revert_to_original_position(target_r, face_box_tl, cam_intrinsics, cam_R, cam_T)

        if qualitative:
            raw = data["frames"][0].cpu().numpy()
            raw_gaze = raw.copy()

            l_eyeball_centre_screen_gt = data["left_eyeball_2d_crop"][0].cpu().numpy()
            r_eyeball_centre_screen_gt = data["right_eyeball_2d_crop"][0].cpu().numpy()
            target_screen_gt = data["target_2d_crop"][0].cpu().numpy()
            l_eyeball_centre_screen = world_to_img(l_eyeball_centre, cam_intrinsics, cam_R, cam_T)[0, :2]
            r_eyeball_centre_screen = world_to_img(r_eyeball_centre, cam_intrinsics, cam_R, cam_T)[0, :2]
            target_screen = world_to_img(target, cam_intrinsics, cam_R, cam_T)[0, :2]
            target_screen_l = world_to_img(target_l, cam_intrinsics, cam_R, cam_T)[0, :2]
            target_screen_r = world_to_img(target_r, cam_intrinsics, cam_R, cam_T)[0, :2]

            draw_gaze(raw_gaze, l_eyeball_centre_screen_gt, target_screen_gt, r_eyeball_centre_screen_gt,
                      l_eyeball_centre_screen, target_screen_l, r_eyeball_centre_screen, target_screen_r)

            if "img" in results:
                rendered = (results["img"][0].cpu().numpy() * 255).astype(np.uint8)
                rendered_none_idx = np.all(rendered == [0, 255, 0], axis=2)
                rendered[rendered_none_idx] = cv2.resize(raw, (224, 224))[rendered_none_idx]

                # draw_gaze(rendered, l_eyeball_centre_screen_gt, target_screen_gt, r_eyeball_centre_screen_gt,
                #           l_eyeball_centre_screen, target_screen_l, r_eyeball_centre_screen, target_screen_r)

                rendered = cv2.resize(rendered, (512, 512))
            else:
                rendered = np.zeros((512, 512, 3))

            raw = cv2.resize(raw, (512, 512))
            raw_gaze = cv2.resize(raw_gaze, (512, 512))
            combined = np.concatenate([raw, rendered, raw_gaze], axis=1)
            rendered_video_writer.write(combined)

            # cv2.imshow("1", combined)
            # cv2.waitKey(100)

        # Calculate angle errors.
        l_gaze_angle_errors_rot.append(
            get_angle(l_gaze_rot - l_eyeball_centre_orig,
                      data["target_3d"].cpu().numpy() - data["left_eyeball_3d"].cpu().numpy()))
        r_gaze_angle_errors_rot.append(
            get_angle(r_gaze_rot - r_eyeball_centre_orig,
                      data["target_3d"].cpu().numpy() - data["right_eyeball_3d"].cpu().numpy()))
        l_gaze_angle_errors_rot_gt.append(
            get_angle(results["left_gaze"][0].cpu().numpy(),
                      data["target_3d_crop"].cpu().numpy() - data["left_eyeball_3d_crop"].cpu().numpy()))
        r_gaze_angle_errors_rot_gt.append(
            get_angle(results["right_gaze"][0].cpu().numpy(),
                      data["target_3d_crop"].cpu().numpy() - data["right_eyeball_3d_crop"].cpu().numpy()))

        l_gaze_angle_errors_tgt.append(
            get_angle(target_orig - l_eyeball_centre_orig,
                      data["target_3d"].cpu().numpy() - data["left_eyeball_3d"].cpu().numpy()))
        r_gaze_angle_errors_tgt.append(
            get_angle(target_orig - r_eyeball_centre_orig,
                      data["target_3d"].cpu().numpy() - data["right_eyeball_3d"].cpu().numpy()))

        l_gaze_angle_errors_tgt_gt.append(
            get_angle(target_orig - data["left_eyeball_3d"].cpu().numpy(),
                      data["target_3d"].cpu().numpy() - data["left_eyeball_3d"].cpu().numpy()))
        r_gaze_angle_errors_tgt_gt.append(
            get_angle(target_orig - data["right_eyeball_3d"].cpu().numpy(),
                      data["target_3d"].cpu().numpy() - data["right_eyeball_3d"].cpu().numpy()))

        face_point = (data["right_eyeball_3d"].cpu().numpy() + data["left_eyeball_3d"].cpu().numpy()) / 2
        f_gaze_angle_errors_tgt.append(
            get_angle(target_orig - face_point,
                      data["target_3d"].cpu().numpy() - face_point))

    rendered_video_writer.release()

    report_two_eye_gaze_angle_error("rot", l_gaze_angle_errors_rot, r_gaze_angle_errors_rot)
    report_two_eye_gaze_angle_error("rot_crop_gt", l_gaze_angle_errors_rot_gt, r_gaze_angle_errors_rot_gt)
    report_two_eye_gaze_angle_error("tgt", l_gaze_angle_errors_tgt, r_gaze_angle_errors_tgt)
    report_two_eye_gaze_angle_error("tgt_gt", l_gaze_angle_errors_tgt_gt, r_gaze_angle_errors_tgt_gt)

    logging.info("Face gaze tgt error, mean: " + str(np.mean(f_gaze_angle_errors_tgt)) +
                 ", std: " + str(np.std(f_gaze_angle_errors_tgt)) +
                 ", median: " + str(np.median(f_gaze_angle_errors_tgt)))

    # l_gaze_rot_axis_pred = np.stack(l_gaze_rot_axis_pred)
    # r_gaze_rot_axis_pred = np.stack(r_gaze_rot_axis_pred)
    # l_gaze_rot_axis_gt = np.stack(l_gaze_rot_axis_gt)
    # r_gaze_rot_axis_gt = np.stack(r_gaze_rot_axis_gt)
    # draw_distribution_scatter(l_gaze_rot_axis_pred, l_gaze_rot_axis_gt)
    # draw_distribution_scatter(r_gaze_rot_axis_pred, r_gaze_rot_axis_gt)


def report_two_eye_gaze_angle_error(name, l_gaze_errors, r_gaze_errors):
    logging.info("Left gaze " + name + " error, mean: " + str(np.mean(l_gaze_errors)) +
                 ", std: " + str(np.std(l_gaze_errors)) +
                 ", median: " + str(np.median(l_gaze_errors)))
    logging.info("Right gaze " + name + " error, mean: " + str(np.mean(r_gaze_errors)) +
                 ", std: " + str(np.std(r_gaze_errors)) +
                 ", median: " + str(np.median(r_gaze_errors)))
    logging.info("Total gaze " + name + " error, mean: " + str(np.mean(l_gaze_errors + r_gaze_errors)) +
                 ", std: " + str(np.std(l_gaze_errors + r_gaze_errors)) +
                 ", median: " + str(np.median(l_gaze_errors + r_gaze_errors)))


def revert_to_original_position(point_3d, face_box_tl, cam_intrinsics, cam_R, cam_T):
    """
    Project world coordinate system 3D points onto image plane, apply cropping offset,
    then project back to world coordinate system.

    :param point_3d:
    :param face_box_tl:
    :param cam_intrinsics:
    :param cam_R:
    :param cam_T:
    :return:
    """
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
    if os.path.exists(LOGS_PATH + NAME + "/eval.txt"):
        os.remove(LOGS_PATH + NAME + "/eval.txt")

    console_handler = logging.StreamHandler()
    logging.basicConfig(
        handlers=[logging.FileHandler(filename=LOGS_PATH + NAME + "/eval.txt"), console_handler],
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(message)s", datefmt="[%Y-%m-%d %H:%M:%S]")
    console_handler.setLevel(logging.INFO)

    evaluate(qualitative=True)
