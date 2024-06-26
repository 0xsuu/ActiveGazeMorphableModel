import json
import logging
from collections import OrderedDict
from types import SimpleNamespace

import cv2
import torch
from matplotlib import pyplot as plt
from pytorch3d.io import save_obj
from pytorch3d.renderer import PerspectiveCameras
from torch.utils.data import DataLoader
import numpy as np
from torchvision import transforms
from torchvision.transforms import Resize, Grayscale
from tqdm import tqdm

from autoencoder.model import Autoencoder, AutoencoderBaseline
from constants import *
from utils.eyediap_dataset import EYEDIAP
from utils.camera_model import world_to_img, img_to_world

NAME = "v9_l1_lmd7x50_run2"
EPOCH = "70"
# SUBJECT_IDS = [15]
# HERO_FRAME = 2731
# SUBJECT_IDS = [16]
# HERO_FRAME = 95
SUBJECT_IDS = [4]
HERO_FRAME = 1240


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

    if "baseline" in NAME:
        model = AutoencoderBaseline(args)
    else:
        model = Autoencoder(args)
    saved_state_dict = torch.load(LOGS_PATH + NAME + "/model_" + EPOCH + ".pt")

    # Load checkpoint and set to evaluate.
    model.load_state_dict(saved_state_dict)
    model.eval()

    # Load test dataset.
    test_data = EYEDIAP(partition="test", eval_subjects=SUBJECT_IDS, head_movement=["M", "S"])
    test_loader = DataLoader(test_data, batch_size=1, shuffle=False, num_workers=0)

    frame_transform = transforms.Compose([
        Resize(224),
        # transforms.ColorJitter(brightness=0.4, contrast=0.2, saturation=0.1, hue=0),
        # AddGaussianNoise(0, 0.2),
        transforms.Normalize(mean=[0.2630, 0.2962, 0.4256], std=[0.1957, 0.1928, 0.2037])
    ])
    l_eye_patch_transformation = Grayscale()
    r_eye_patch_transformation = Grayscale()

    # Initiate video writer. The qualitative result will be saved in a video.
    if HERO_FRAME is None:
        rendered_video_writer = cv2.VideoWriter(LOGS_PATH + NAME + "/result_" + EPOCH + "_" +
                                                "_".join([str(i) for i in SUBJECT_IDS]) + ".mov",
                                                cv2.VideoWriter_fourcc("m", "p", "4", "v"), 20.0, (1024 + 512, 512))
    else:
        rendered_video_writer = None

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
    eye_region_model_eval = {"vert": [], "lm": [], "lm_gt": [], "sid": []}
    for idx, data in tqdm(enumerate(test_loader), total=len(test_loader)):
        if HERO_FRAME is not None and idx != HERO_FRAME:
            continue
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

        angle_eror_tgt = \
            (get_angle(
                target_orig - l_eyeball_centre_orig,
                data["target_3d"].cpu().numpy() - data["left_eyeball_3d"].cpu().numpy()) +
             get_angle(target_orig - r_eyeball_centre_orig,
                       data["target_3d"].cpu().numpy() - data["right_eyeball_3d"].cpu().numpy())) / 2
        eye_region_model_eval["vert"].append(results["vert_masked"][0].cpu().numpy())
        eye_region_model_eval["lm"].append(results["face_landmarks"][0].cpu().numpy())
        eye_region_model_eval["lm_gt"].append(data["face_landmarks_crop"][0].cpu().numpy())
        eye_region_model_eval["sid"].append(data["subject_id"][0].cpu().numpy())
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
                # # Normalisation correction.
                # rendered_frame = results["img"][0].cpu().numpy()
                # mask = np.all(rendered_frame != [0., 1., 0.], axis=2)
                # rendered_frame[mask] = rendered_frame[mask] * np.array([0.1957, 0.1928, 0.2037])[None, :] + \
                #                        np.array([0.2630, 0.2962, 0.4256])[None, :]
                # rendered = (rendered_frame * 255).astype(np.uint8)
                # No normalisation correction.
                rendered = (results["img"][0].cpu().numpy() * 255).astype(np.uint8)

                rendered_none_idx = np.all(rendered == [0, 255, 0], axis=2)
                rendered[rendered_none_idx] = cv2.resize(raw, (224, 224))[rendered_none_idx]

                # draw_gaze(rendered, l_eyeball_centre_screen_gt, target_screen_gt, r_eyeball_centre_screen_gt,
                #           l_eyeball_centre_screen, target_screen_l, r_eyeball_centre_screen, target_screen_r)

                rendered = cv2.resize(rendered, (512, 512))
            else:
                rendered = np.zeros((512, 512, 3))

            if idx == HERO_FRAME:
                # Save textured mesh.
                save_obj(LOGS_PATH + NAME + "/eval" + EPOCH + "_" +
                         "_".join([str(i) for i in SUBJECT_IDS]) + "_" +
                         str(HERO_FRAME) + ".obj", verts=results["vert_masked"][0], faces=model.face_model.faces,
                         texture_map=results["tex"][0].flip(2), faces_uvs=model.face_model.albedo_ft,
                         verts_uvs=model.face_model.albedo_vt)

                # Save landmarks and eyes.
                white_img = np.ones((512, 512, 3), dtype=np.uint8) * 255
                for lm in results["face_landmarks"][0] / test_data.face_crop_size * 512:
                    cv2.drawMarker(white_img, (int(lm[0]), int(lm[1])), (255, 0, 0),
                                   markerType=cv2.MARKER_TILTED_CROSS, markerSize=12, thickness=3,
                                   line_type=cv2.LINE_AA)

                cameras = PerspectiveCameras(R=camera_parameters[0], T=camera_parameters[1], K=camera_parameters[2],
                                             device=device)

                def draw_eye_gazes(l_c, r_c, l_t, r_t, colour, length_l=0.2, length_r=0.2):
                    l_eb_centre = cameras.transform_points(l_c)[0] / test_data.face_crop_size * 512
                    r_eb_centre = cameras.transform_points(r_c)[0] / test_data.face_crop_size * 512
                    cv2.circle(white_img, (int(l_eb_centre[0]), int(l_eb_centre[1])), 7, color=colour, thickness=2,
                               lineType=cv2.LINE_AA)
                    cv2.circle(white_img, (int(r_eb_centre[0]), int(r_eb_centre[1])), 7, color=colour, thickness=2,
                               lineType=cv2.LINE_AA)

                    tgt_l = (cameras.transform_points(l_t)[0]
                             / test_data.face_crop_size * 512 - l_eb_centre) * length_l + l_eb_centre
                    tgt_r = (cameras.transform_points(r_t)[0]
                             / test_data.face_crop_size * 512 - r_eb_centre) * length_r + r_eb_centre
                    cv2.arrowedLine(white_img, (int(l_eb_centre[0]), int(l_eb_centre[1])), (int(tgt_l[0]), int(tgt_l[1])),
                                    colour, thickness=2, line_type=cv2.LINE_AA, tipLength=0.05)
                    cv2.arrowedLine(white_img, (int(r_eb_centre[0]), int(r_eb_centre[1])), (int(tgt_r[0]), int(tgt_r[1])),
                                    colour, thickness=2, line_type=cv2.LINE_AA, tipLength=0.05)

                    return l_eb_centre, r_eb_centre

                draw_eye_gazes(data["left_eyeball_3d_crop"], data["right_eyeball_3d_crop"],
                               data["target_3d_crop"], data["target_3d_crop"], (0, 255, 0))
                l_eb_centre, r_eb_centre = draw_eye_gazes(
                    results["l_eyeball_centre"][0], results["r_eyeball_centre"][0],
                    results["gaze_point_l"][0], results["gaze_point_r"][0], (0, 0, 255))

                # Save gaze on raw image.
                raw_gaze_demo = cv2.resize(raw.copy(), (512, 512))
                tgt = cameras.transform_points(results["gaze_point_mid"][0])[0] / test_data.face_crop_size * 512
                tgt_gt = cameras.transform_points(data["target_3d_crop"])[0] / test_data.face_crop_size * 512
                l_gt = cameras.transform_points(data["left_eyeball_3d_crop"])[0] / test_data.face_crop_size * 512
                r_gt = cameras.transform_points(data["right_eyeball_3d_crop"])[0] / test_data.face_crop_size * 512
                cv2.arrowedLine(raw_gaze_demo,
                                (int(l_gt[0]), int(l_gt[1])), (int(tgt_gt[0]), int(tgt_gt[1])),
                                (0, 255, 0), thickness=4, line_type=cv2.LINE_AA, tipLength=0.05)
                cv2.arrowedLine(raw_gaze_demo,
                                (int(r_gt[0]), int(r_gt[1])), (int(tgt_gt[0]), int(tgt_gt[1])),
                                (0, 255, 0), thickness=4, line_type=cv2.LINE_AA, tipLength=0.05)
                cv2.arrowedLine(raw_gaze_demo,
                                (int(l_eb_centre[0]), int(l_eb_centre[1])), (int(tgt[0]), int(tgt[1])),
                                (0, 0, 255), thickness=4, line_type=cv2.LINE_AA, tipLength=0.05)
                cv2.arrowedLine(raw_gaze_demo,
                                (int(r_eb_centre[0]), int(r_eb_centre[1])), (int(tgt[0]), int(tgt[1])),
                                (0, 0, 255), thickness=4, line_type=cv2.LINE_AA, tipLength=0.05)

                cv2.imwrite(LOGS_PATH + NAME + "/eval" + EPOCH + "_" +
                            "_".join([str(i) for i in SUBJECT_IDS]) + "_" +
                            str(HERO_FRAME) + "_landmarks.png", white_img)
                cv2.imwrite(LOGS_PATH + NAME + "/eval" + EPOCH + "_" +
                            "_".join([str(i) for i in SUBJECT_IDS]) + "_" +
                            str(HERO_FRAME) + "_raw.png", raw)
                cv2.imwrite(LOGS_PATH + NAME + "/eval" + EPOCH + "_" +
                            "_".join([str(i) for i in SUBJECT_IDS]) + "_" +
                            str(HERO_FRAME) + "_rendered.png", rendered)
                cv2.imwrite(LOGS_PATH + NAME + "/eval" + EPOCH + "_" +
                            "_".join([str(i) for i in SUBJECT_IDS]) + "_" +
                            str(HERO_FRAME) + "_gaze.png", raw_gaze_demo)

            raw = cv2.resize(raw, (512, 512))
            raw_gaze = cv2.resize(raw_gaze, (512, 512))
            combined = np.concatenate([raw, rendered, raw_gaze], axis=1).astype(np.uint8)
            cv2.putText(combined, "frame: %d, angle error tgt: %.04f" % (idx, angle_eror_tgt),
                        (10, 10), cv2.FONT_HERSHEY_PLAIN, 1.0, (255, 255, 255))
            if rendered_video_writer is not None:
                rendered_video_writer.write(combined)

        # Calculate angle errors.
        l_gaze_angle_errors_rot.append(
            get_angle(l_gaze_rot - l_eyeball_centre_orig,
                      data["target_3d"].cpu().numpy() - data["left_eyeball_3d"].cpu().numpy()))
        r_gaze_angle_errors_rot.append(
            get_angle(r_gaze_rot - r_eyeball_centre_orig,
                      data["target_3d"].cpu().numpy() - data["right_eyeball_3d"].cpu().numpy()))
        l_gaze_angle_errors_rot_gt.append(
            get_angle(results["left_gaze"][:, 0].cpu().numpy(),
                      data["target_3d_crop"].cpu().numpy() - data["left_eyeball_3d_crop"].cpu().numpy()))
        r_gaze_angle_errors_rot_gt.append(
            get_angle(results["right_gaze"][:, 0].cpu().numpy(),
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

    if rendered_video_writer is not None:
        rendered_video_writer.release()

    report_two_eye_gaze_angle_error("rot", l_gaze_angle_errors_rot, r_gaze_angle_errors_rot)
    report_two_eye_gaze_angle_error("rot_crop_gt", l_gaze_angle_errors_rot_gt, r_gaze_angle_errors_rot_gt)
    report_two_eye_gaze_angle_error("tgt", l_gaze_angle_errors_tgt, r_gaze_angle_errors_tgt)
    report_two_eye_gaze_angle_error("tgt_gt", l_gaze_angle_errors_tgt_gt, r_gaze_angle_errors_tgt_gt)

    if HERO_FRAME is None:
        np.save(LOGS_PATH + NAME + "/result_" + EPOCH + "_" + "_".join([str(i) for i in SUBJECT_IDS]) + "tgt",
                np.concatenate([l_gaze_angle_errors_tgt, r_gaze_angle_errors_tgt]))
        np.save(LOGS_PATH + NAME + "/result_" + EPOCH + "_" + "_".join([str(i) for i in SUBJECT_IDS]) +
                "eye_region_model_stat", {k: np.stack(v) for k, v in eye_region_model_eval.items()})

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
    assert vec_a.shape == (1, 3) and vec_b.shape == (1, 3)
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
    if HERO_FRAME is None:
        if os.path.exists(LOGS_PATH + NAME + "/eval" + EPOCH + "_" + "_".join([str(i) for i in SUBJECT_IDS]) + ".txt"):
            os.remove(LOGS_PATH + NAME + "/eval" + EPOCH + "_" + "_".join([str(i) for i in SUBJECT_IDS]) + ".txt")
            while os.path.exists(LOGS_PATH + NAME + "/eval" + EPOCH +
                                 "_" + "_".join([str(i) for i in SUBJECT_IDS]) + ".txt"):
                pass

        console_handler = logging.StreamHandler()
        logging.basicConfig(
            handlers=[
                logging.FileHandler(
                    filename=LOGS_PATH + NAME + "/eval" + EPOCH + "_" + "_".join([str(i) for i in SUBJECT_IDS]) + ".txt"),
                console_handler],
            level=logging.INFO,
            format="%(asctime)s %(levelname)-8s %(message)s", datefmt="[%Y-%m-%d %H:%M:%S]")
        console_handler.setLevel(logging.INFO)

    evaluate(qualitative=True)
