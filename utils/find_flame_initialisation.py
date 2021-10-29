
import cv2
import torch
import numpy as np

from face_model.flame_model import FlameModel
from constants import *
from utils.xgaze_dataset import XGazeDataset, generate_face_gaze_vector, cam_to_img, perspective_transform, \
    generate_gaze_rotation_angle


def compute_rt(probe_v, model_v):
    """
    Find correct rigid transformation given corresponded point sets using SVD.

    :param probe_v: The moving part.
    :param model_v: The fixed part.
    :return:
    """
    if probe_v.shape[0] != model_v.shape[0] or probe_v.shape[1] != model_v.shape[1]:
        raise ValueError("Probe and model have different numbers of points.")
    if probe_v.shape[1] != 2 and probe_v.shape[1] != 3:
        raise ValueError("Probe and model have wrong number of dimensions (only 2 or 3 allowed).")

    probe_mean = np.mean(probe_v, axis=0)
    probe_pts_zm = probe_v - probe_mean
    model_mean = np.mean(model_v, axis=0)
    model_pts_zm = model_v - model_mean

    B = probe_pts_zm.T @ model_pts_zm
    U, _, VH = np.linalg.svd(B)
    V = VH.T
    R = V @ U.T

    if np.linalg.det(R) < 0:
        if probe_v.shape[1] == 3:
            R = V @ np.diag([1, 1, -1]) @ U.T
        else:
            R = V @ np.diag([1, -1]) @ U.T

    T = model_mean - R @ probe_mean

    return R, T


if __name__ == '__main__':
    face_model = FlameModel(FLAME_PATH + "FLAME2020/generic_model.pkl",
                            FLAME_PATH + "albedoModel2020_FLAME_albedoPart.npz",
                            FLAME_PATH + "FLAME_masks/FLAME_masks.pkl", None,
                            masks=["right_eyeball", "left_eyeball", "nose", "eye_region"]).to(device)
    face_model = face_model.template_v[0, face_model.landmarks]

    # xgaze_face_model = np.loadtxt(XGAZE_PATH + "calibration/face_model.txt")
    # xgaze_face_model = torch.from_numpy(xgaze_face_model[None, :31, :]).to(dtype=torch.float32, device=device)
    xgaze_face_model = torch.tensor([[-19.0997256 , 100.81622091, 880.27077615],
                                     [-74.50425729, -30.02453538, 904.87560044],
                                     [-69.23948095, -34.18457525, 891.30491371],
                                     [-59.40698336, -34.52833854, 877.73788232],
                                     [-51.05582709, -35.05233876, 870.55987344],
                                     [-37.53944131, -31.04368672, 865.19055775],
                                     [ -7.20573382, -29.84276127, 861.92172257],
                                     [  5.81373027, -33.85270862, 863.9485397 ],
                                     [ 15.74916997, -33.69976231, 868.92281196],
                                     [ 28.11368561, -32.46657095, 878.08249289],
                                     [ 36.3685867 , -27.61418645, 888.41837639],
                                     [-20.57680537, -13.43081466, 867.27078096],
                                     [-20.41146364,   4.28355055, 865.66606771],
                                     [-21.17134646,  17.41401047, 860.29494896],
                                     [-21.37911998,  28.29535573, 857.13916519],
                                     [-30.92184898,  35.21080726, 877.13433785],
                                     [-25.83986333,  38.45677762, 873.64164571],
                                     [-20.07926631,  40.21565647, 871.76967191],
                                     [-14.34175043,  38.83213453, 872.48562053],
                                     [ -9.15209672,  36.05551328, 874.70575845],
                                     [-65.14502959, -16.61498009, 893.88405063],
                                     [-56.50289917, -17.91117127, 883.23961135],
                                     [-46.25467343, -18.19860973, 880.65432706],
                                     [-37.46492038, -10.60946945, 883.34458445],
                                     [-48.2548333 ,  -9.3142541 , 884.16704134],
                                     [-54.83791304, -11.36816932, 885.91520481],
                                     [ -1.9187863 ,  -9.06458954, 879.01917572],
                                     [  5.08615305, -15.85526067, 875.36155817],
                                     [ 14.69813139, -14.51125019, 875.80731206],
                                     [ 25.48662451, -12.62462676, 883.21104955],
                                     [ 14.64845371,  -7.55204093, 878.01549926],
                                     [  8.20663212,  -6.21570591, 877.41166251],
                                     [-39.68615195,  57.98329048, 882.34894736],
                                     [-35.41134328,  52.84286659, 877.65040585],
                                     [-25.23999248,  50.98706915, 871.12615478],
                                     [-20.12627194,  51.90435182, 870.20516131],
                                     [-15.10877035,  51.33119511, 870.02929857],
                                     [ -4.42030133,  53.3464316 , 873.5238077 ],
                                     [  1.44427089,  59.14905842, 877.04151935],
                                     [ -4.09878653,  63.33108477, 873.6874737 ],
                                     [-12.70456745,  67.0183853 , 871.43076597],
                                     [-20.08051133,  67.44992024, 872.53165344],
                                     [-26.89704414,  66.17340516, 873.45227794],
                                     [-34.48952051,  62.50158937, 878.57328782],
                                     [-29.73588966,  56.81019593, 876.63296964],
                                     [-19.51361711,  57.91343064, 873.40101759],
                                     [ -9.3799875 ,  57.6240947 , 874.15645729],
                                     [ -9.28989796,  58.74448566, 874.31693918],
                                     [-19.69009121,  59.18487263, 873.05117357],
                                     [-29.89107056,  57.95468174, 876.7855306 ],]
                                    , dtype=torch.float32, device=device)[1:32, ]
    xgaze_face_model_mean = xgaze_face_model.mean(0)
    xgaze_face_model = (xgaze_face_model - xgaze_face_model_mean) + \
        xgaze_face_model_mean

    scale = float(torch.norm(xgaze_face_model - xgaze_face_model.mean(0), dim=1).mean() /
                  torch.norm(face_model - face_model.mean(0), dim=1).mean())
    R, T = compute_rt((face_model - face_model.mean(0)).cpu().numpy() * scale, xgaze_face_model.cpu().numpy())
    print(R, T, scale)

    face_model = torch.from_numpy(((face_model - face_model.mean(0)).cpu().numpy() * scale) @ R.T + T).to(torch.float32).to(device)

    xgaze_face_centre = (torch.mean(xgaze_face_model[[19, 22, 25, 28]], dim=0) + torch.mean(xgaze_face_model[[14, 18]], dim=0)) / 2
    fm_face_centre = (torch.mean(face_model[[19, 22, 25, 28]], dim=0) + torch.mean(face_model[[14, 18]], dim=0)) / 2
    print(xgaze_face_centre, fm_face_centre)

    dt = XGazeDataset()[0]

    frame = dt["frames"].cpu().numpy() / 255.
    lms = dt["face_landmarks_crop"]
    face_centre = dt["gaze_origins"].unsqueeze(0)
    gaze_target = dt["target_3d_crop"].unsqueeze(0)
    R = dt["head_rotations"].unsqueeze(0)
    cam_intr = dt["cam_intrinsics"].unsqueeze(0)
    W = dt["warp_matrices"].unsqueeze(0)

    xgaze_face_model_img = cam_to_img(xgaze_face_model.unsqueeze(0), cam_intr)
    xgaze_face_model_img = perspective_transform(xgaze_face_model_img, W)
    face_model_img = cam_to_img(face_model.unsqueeze(0), cam_intr)
    face_model_img = perspective_transform(face_model_img, W)

    gaze_target_rot = generate_face_gaze_vector(dt["face_gazes"].unsqueeze(0), dt["head_rotations"].unsqueeze(0))
    gaze_target_rot = face_centre + gaze_target_rot * 100.

    gaze_vec = gaze_target - face_centre
    gaze_vec_rot = gaze_target_rot - face_centre
    gaze_vec = gaze_vec / torch.linalg.norm(gaze_vec, dim=1, keepdim=True)
    gaze_vec_rot = gaze_vec_rot / torch.linalg.norm(gaze_vec_rot, dim=1, keepdim=True)
    print(torch.arccos(torch.sum(gaze_vec * gaze_vec_rot)))
    face_centre_img = cam_to_img(face_centre, cam_intr)
    gaze_target_img = cam_to_img(gaze_target, cam_intr)
    gaze_target_rot_img = cam_to_img(gaze_target_rot, cam_intr)

    a = cv2.perspectiveTransform(face_centre_img[:, 0:1, :2].cpu().numpy(), W.cpu().numpy())
    b = perspective_transform(face_centre_img, W)
    c = perspective_transform(gaze_target_img, W)
    d = perspective_transform(gaze_target_rot_img, W)
    # print(a, b)

    gaze_rot_angle = generate_gaze_rotation_angle(gaze_target, face_centre, dt["head_rotations"].unsqueeze(0))[0]
    print(gaze_rot_angle, dt["face_gazes"])
    # xgaze_face_model_img[0, ]
    for idx, lm in enumerate(xgaze_face_model_img[0]):
        cv2.putText(frame, str(idx), (int(lm[0]), int(lm[1])), cv2.FONT_HERSHEY_PLAIN, 0.5, color=(0, 1, 0))

    for idx, lm in enumerate(face_model_img[0]):
        cv2.putText(frame, str(idx), (int(lm[0]), int(lm[1])), cv2.FONT_HERSHEY_PLAIN, 0.5, color=(0, 0, 1))

    for idx, lm in enumerate(dt["face_landmarks_crop"]):
        cv2.putText(frame, str(idx), (int(lm[0]), int(lm[1])), cv2.FONT_HERSHEY_PLAIN, 0.5, color=(1, 0, 0))

    cv2.arrowedLine(frame,
                    (int(b[0, 0, 0]), int(b[0, 0, 1])),
                    (int(c[0, 0, 0]), int(c[0, 0, 1])),
                    (0, 0, 255), thickness=2)
    cv2.arrowedLine(frame,
                    (int(b[0, 0, 0]), int(b[0, 0, 1])),
                    (int(d[0, 0, 0]), int(d[0, 0, 1])),
                    (0, 255, 0), thickness=2)

    cv2.imshow("", cv2.resize(frame, (512, 512)))
    cv2.waitKey(0)
