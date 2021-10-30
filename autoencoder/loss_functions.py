
import torch
import torch.nn.functional as F
import numpy as np


def pixel_loss(image_pred, image_gt):
    mask = torch.all(image_pred != torch.tensor([0., 1., 0.]).cuda(), dim=3)
    if torch.any(mask):
        return F.mse_loss(image_pred[mask], image_gt[mask])
    else:
        return torch.tensor(0., requires_grad=True).cuda()


def landmark_loss(landmarks_pred, landmarks_gt):
    return F.mse_loss(landmarks_pred, landmarks_gt)


def eye_loss(left_eye_centre_pred, right_eye_centre_pred,
             left_eye_centre_gt, right_eye_centre_gt):
    return F.mse_loss(left_eye_centre_pred, left_eye_centre_gt) + F.mse_loss(right_eye_centre_pred, right_eye_centre_gt)


def gaze_target_loss(gaze_target_pred, gaze_target_gt):
    return F.mse_loss(gaze_target_pred, gaze_target_gt)


def gaze_divergence_loss(gaze_divergence_distance):
    return torch.mean(gaze_divergence_distance ** 2)


def gaze_pose_loss(left_gaze_rot_pred, left_gaze_rot_gt, right_gaze_rot_pred, right_gaze_rot_gt):
    return F.mse_loss(left_gaze_rot_pred, left_gaze_rot_gt) + F.mse_loss(right_gaze_rot_pred, right_gaze_rot_gt)


def parameters_regulariser(shape_parameters):
    return torch.mean(shape_parameters ** 2)


def gaze_degree_error(left_eye_centre_pred, right_eye_centre_pred, left_eye_centre_gt, right_eye_centre_gt,
                      gaze_target_pred, gaze_target_gt):
    with torch.no_grad():
        gaze_vec_pred = gaze_target_pred.detach() - (left_eye_centre_pred.detach() + right_eye_centre_pred.detach()) / 2
        gaze_vec_gt = gaze_target_gt.detach() - (left_eye_centre_gt.detach() + right_eye_centre_gt.detach()) / 2
        gaze_vec_pred = gaze_vec_pred.cpu().numpy()
        gaze_vec_gt = gaze_vec_gt.cpu().numpy()

        gaze_rad_error = np.arccos(np.sum(gaze_vec_pred * gaze_vec_gt, axis=1) /
                                   (np.linalg.norm(gaze_vec_pred, axis=1) * np.linalg.norm(gaze_vec_gt, axis=1)))
        return np.rad2deg(np.nan_to_num(gaze_rad_error)).mean()
