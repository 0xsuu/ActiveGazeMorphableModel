
import json
import logging
from collections import OrderedDict
from types import SimpleNamespace

import cv2
import torch
from matplotlib import pyplot as plt
from pytorch3d.renderer import PerspectiveCameras
from torch.utils.data import DataLoader
import numpy as np
from torchvision import transforms
from torchvision.transforms import Resize, Grayscale
from tqdm import tqdm
from psbody.mesh import Mesh

from autoencoder.model import Autoencoder, AutoencoderBaseline
from constants import *
from utils.evaluate_xgaze import angular_error, pitchyaw_to_vector, vector_to_pitchyaw
from utils.eyediap_dataset import EYEDIAP
from utils.camera_model import world_to_img, img_to_world
from utils.find_flame_initialisation import compute_rt
from utils.xgaze_dataset import XGazeDataset, cam_to_img, perspective_transform

NAME = "v8"
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

    if "baseline" in NAME:
        model = AutoencoderBaseline(args)
    else:
        model = Autoencoder(args)
    saved_state_dict = torch.load(LOGS_PATH + NAME + "/model_" + EPOCH + ".pt")
    model.load_state_dict(saved_state_dict)
    model.eval()

    test_data_eydiap = EYEDIAP(partition="test", eval_subjects=[16], head_movement=["M", "S"])
    eyediap_data = test_data_eydiap[0]
    camera_parameters_eyediap = (eyediap_data["cam_R"].unsqueeze(0),
                                 eyediap_data["cam_T"].unsqueeze(0),
                                 eyediap_data["cam_K"].unsqueeze(0))
    cameras = PerspectiveCameras(R=camera_parameters_eyediap[0],
                                 T=camera_parameters_eyediap[1],
                                 K=camera_parameters_eyediap[2], device=device)

    test_data = XGazeDataset(partition=PARTITION)
    test_loader = DataLoader(test_data, batch_size=1, shuffle=False, num_workers=0)

    xgaze_mean, xgaze_std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    frame_transform = transforms.Compose([
        Resize(224),
        # transforms.ColorJitter(brightness=0.4, contrast=0.2, saturation=0.1, hue=0),
        # AddGaussianNoise(0, 0.2),
        transforms.Normalize(mean=xgaze_mean, std=xgaze_std)
    ])

    losses = []
    rot_pred_list = []
    progress = tqdm(test_loader)
    for data in progress:
        gt_img = data["frames"].to(torch.float32) / 255.
        cam_parameters = data["cam_intrinsics"]
        warp_matrices = data["warp_matrices"]
        head_rotation = data["head_rotations"]
        frame_raw = gt_img.cpu().numpy()[0]
        gt_img_input = frame_transform(gt_img.permute(0, 3, 1, 2))
        with torch.no_grad():
            results = model(gt_img_input, camera_parameters_eyediap)

        cam_R, cam_T, cam_K = camera_parameters_eyediap
        cam_intrinsics = cam_K[0, :3, :3]
        cam_intrinsics[2, 2] = 1.
        cam_R, cam_T, cam_intrinsics = cam_R[0].cpu().numpy(), cam_T[0].cpu().numpy(), cam_intrinsics.cpu().numpy()
        # l_eyeball_centre = results["l_eyeball_centre"]
        # r_eyeball_centre = results["r_eyeball_centre"]
        # l_gaze_rot = results["left_gaze"] + l_eyeball_centre
        # r_gaze_rot = results["right_gaze"] + r_eyeball_centre

        target_pred = results["gaze_point_mid"][:, 0]
        origin_pred = results["face_centre"]

        l_eyeball_centre = results["l_eyeball_centre"][0].cpu().numpy()
        r_eyeball_centre = results["r_eyeball_centre"][0].cpu().numpy()
        # l_gaze_rot = results["left_gaze"][0].cpu().numpy() + l_eyeball_centre
        # r_gaze_rot = results["right_gaze"][0].cpu().numpy() + r_eyeball_centre

        l_eyeball_centre_screen = world_to_img(l_eyeball_centre, cam_intrinsics, cam_R, cam_T)[0, :2]
        r_eyeball_centre_screen = world_to_img(r_eyeball_centre, cam_intrinsics, cam_R, cam_T)[0, :2]
        target_screen = world_to_img(target_pred.cpu().numpy(), cam_intrinsics, cam_R, cam_T)[0, :2]
        origin_screen = world_to_img(origin_pred.cpu().numpy(), cam_intrinsics, cam_R, cam_T)[0, :2]

        def draw_line(origin, target, colour):
            cv2.line(frame_raw,
                     (int(origin[0] / FACE_CROP_SIZE * frame_raw.shape[0]),
                      int(origin[1] / FACE_CROP_SIZE * frame_raw.shape[1])),
                     (int(target[0] / FACE_CROP_SIZE * frame_raw.shape[0]),
                      int(target[1] / FACE_CROP_SIZE * frame_raw.shape[1])),
                     color=colour, lineType=cv2.LINE_AA, thickness=1)
        draw_line(l_eyeball_centre_screen, target_screen, (0, 0, 255))
        draw_line(r_eyeball_centre_screen, target_screen, (0, 0, 255))
        draw_line(origin_screen, target_screen, (0, 0, 255))

        # cameras.transform_points(l_eyeball_centre)

        # target_pred = img_to_world(world_to_img(target_pred.cpu().numpy(), cam_intrinsics, cam_R, cam_T), cam_parameters[0].cpu().numpy())
        # origin_pred = img_to_world(world_to_img(origin_pred.cpu().numpy(), cam_intrinsics, cam_R, cam_T), cam_parameters[0].cpu().numpy())
        # target_pred = target_pred.cpu().numpy()
        # origin_pred = origin_pred.cpu().numpy()

        # Matching.
        face_landmarks_3d_pred = results["face_landmarks_3d"][0]
        face_landmarks_3d_gt = data["face_landmarks_3d"][0]
        fm_pred_mean = face_landmarks_3d_pred.mean(0)
        scale = float(torch.norm(face_landmarks_3d_gt - face_landmarks_3d_gt.mean(0), dim=1).mean() /
                      torch.norm(face_landmarks_3d_pred - face_landmarks_3d_pred.mean(0), dim=1).mean())
        R, T = compute_rt((face_landmarks_3d_pred - fm_pred_mean).cpu().numpy() * scale,
                          face_landmarks_3d_gt.cpu().numpy())

        target_pred = ((target_pred - fm_pred_mean).cpu().numpy() * scale) @ R.T + T
        origin_pred = ((origin_pred - fm_pred_mean).cpu().numpy() * scale) @ R.T + T
        gaze_pred = (origin_pred - target_pred) @ head_rotation.cpu().numpy()[0].T
        gaze_pred = gaze_pred / np.linalg.norm(gaze_pred)
        rot_gt = data["face_gazes"].cpu().numpy()
        # print(angular_error(gaze_pred, pitchyaw_to_vector(rot_gt)))
        losses.append(angular_error(gaze_pred, pitchyaw_to_vector(rot_gt)))
        progress.set_description("%.04f" % np.mean(losses))

        rot_pred_list.append(vector_to_pitchyaw(gaze_pred))

        # o = perspective_transform(cam_to_img(torch.from_numpy(origin_pred).unsqueeze(0).to(device),
        #                                      cam_parameters), warp_matrices)
        # t = perspective_transform(cam_to_img(torch.from_numpy(target_pred).unsqueeze(0).to(device),
        #                                      cam_parameters), warp_matrices)
        # o_gt = perspective_transform(cam_to_img(data["gaze_origins"], cam_parameters), warp_matrices)
        # t_gt = perspective_transform(cam_to_img(data["target_3d_crop"], cam_parameters), warp_matrices)

        # cv2.arrowedLine(frame_raw,
        #                 (int(o[0, 0, 0]), int(o[0, 0, 1])),
        #                 (int(t[0, 0, 0]), int(t[0, 0, 1])),
        #                 (0, 0, 255), thickness=2)
        # cv2.arrowedLine(frame_raw,
        #                 (int(o_gt[0, 0, 0]), int(o_gt[0, 0, 1])),
        #                 (int(t_gt[0, 0, 0]), int(t_gt[0, 0, 1])),
        #                 (0, 255, 0), thickness=2)
        #
        # cv2.imshow("", frame_raw)
        # cv2.waitKey(0)
    #rot_pred_list = np.concatenate(rot_pred_list, axis=0)
    #np.savetxt(LOGS_PATH + NAME + "/within_eva_results.txt", rot_pred_list, delimiter=",")


if __name__ == '__main__':
    evaluate()
