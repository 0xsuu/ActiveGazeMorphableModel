
import torch
import torch.nn.functional as F


def pixel_loss(image_pred, image_gt):
    mask = torch.all(image_pred != torch.tensor([0., 1., 0.]).cuda(), dim=3)
    if torch.any(mask):
        return F.mse_loss(image_pred[mask], image_gt[mask])
    else:
        return torch.tensor(0).cuda()


def landmark_loss(landmarks_pred, landmarks_gt):
    return F.mse_loss(landmarks_pred, landmarks_gt)


def eye_loss(left_eye_centre_pred, right_eye_centre_pred,
             left_eye_centre_gt, right_eye_centre_gt):
    return F.mse_loss(left_eye_centre_pred, left_eye_centre_gt) + F.mse_loss(right_eye_centre_pred, right_eye_centre_gt)


def gaze_target_loss(gaze_target_pred, gaze_target_gt):
    return F.mse_loss(gaze_target_pred, gaze_target_gt)


def gaze_divergence_loss(gaze_divergence_distance):
    return torch.mean(gaze_divergence_distance)


def gaze_pose_loss(left_gaze_rot_pred, left_gaze_rot_gt, right_gaze_rot_pred, right_gaze_rot_gt):
    return F.mse_loss(left_gaze_rot_pred, left_gaze_rot_gt) + F.mse_loss(right_gaze_rot_pred, right_gaze_rot_gt)


def parameters_regulariser(shape_parameters):
    return torch.mean(shape_parameters ** 2)
