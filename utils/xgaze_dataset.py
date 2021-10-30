
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from constants import *


def cam_to_img(points, cam_intrinsic):
    if len(points.shape) == 2:
        points = points.unsqueeze(1)
    points = torch.bmm(points, torch.transpose(cam_intrinsic, 1, 2))
    points = torch.cat([points[:, :, :2] / points[:, :, 2, None], points[:, :, 2, None]], dim=2)

    return points


def perspective_transform(points, W):
    if len(points.shape) == 2:
        points = points.unsqueeze(1)
    points_2d = points[:, :, :2]  # Ignore z-axis.
    # Append 1 to make it homogeneous.
    homo_points = torch.cat([points_2d, torch.ones(points_2d.shape[0], points_2d.shape[1], 1, device=device)], dim=2)
    points_transformed = torch.bmm(homo_points, torch.transpose(W, 1, 2))
    points_transformed = points_transformed[:, :, :2] / points_transformed[:, :, 2, None]
    points_transformed = torch.cat([points_transformed, points[:, :, 2, None]], dim=2)

    return points_transformed


def generate_gaze_rotation_angle(gaze_target, gaze_origin, head_rotation):
    gaze_vector = gaze_target - gaze_origin
    gaze_vector = torch.bmm(gaze_vector.unsqueeze(1), torch.transpose(head_rotation, 1, 2))[:, 0, :]
    gaze_vector = gaze_vector / torch.linalg.norm(gaze_vector, dim=1, keepdim=True)
    gaze_theta = torch.arcsin((-1) * gaze_vector[:, 1])
    gaze_phi = torch.atan2((-1) * gaze_vector[:, 0], (-1) * gaze_vector[:, 2])
    gaze_rot_angle = torch.stack([gaze_theta, gaze_phi], dim=1)

    return gaze_rot_angle


def generate_face_gaze_vector(face_gaze_pitch_yaw, head_rotation):
    """
    Returns the unit gaze vectors starting from face point.

    :param face_gaze_pitch_yaw:
    :param head_rotation:
    :return:
    """
    sin = torch.sin(face_gaze_pitch_yaw)
    cos = torch.cos(face_gaze_pitch_yaw)
    out = torch.stack([cos[:, 0] * sin[:, 1], sin[:, 0], cos[:, 0] * cos[:, 1]], dim=1) * -1.
    gaze_target_rot = torch.bmm(out.unsqueeze(1), torch.transpose(torch.linalg.inv(head_rotation), 1, 2))[:, 0, :]

    return gaze_target_rot


class XGazeDataset(Dataset):
    def __init__(self, partition="train", ratio_sampling=1.):
        # Sample a subset of the whole dataset, uniformly across subjects and cams.
        self.ratio_sampling = ratio_sampling

        # Load dataset info.
        self.subject_files = {}
        if partition == "train" or partition == "cv":
            for i in os.listdir(XGAZE_PATH + "processed/train/"):
                sid = int(i[7:11])
                if (partition == "train" and sid < 92) or (partition == "cv" and sid >= 92):
                    if os.name == "nt":
                        self.subject_files[sid] = h5py.File(XGAZE_PATH + "processed/train/" + i, "r", swmr=True)
                    else:
                        self.subject_files[sid] = h5py.File(XGAZE_PATH + "processed/train/" + i, "r", driver="core")
                # if (partition == "train" and sid == 92) or (partition == "cv" and sid == 92):
                #     self.subject_files[sid] = h5py.File(XGAZE_PATH + "processed/train/" + i, "r", driver="core")

            # Load additional labels.
            add_labels = np.load(XGAZE_PATH + "processed/additional_labels_train.npy", allow_pickle=True)[()]
            self.subject_ids = add_labels["subject_id_list"]
            if partition == "train":
                availability_mask = self.subject_ids < 92
            else:
                availability_mask = self.subject_ids >= 92
            # if partition == "train":
            #     availability_mask = self.subject_ids == 92
            # else:
            #     availability_mask = self.subject_ids == 92
            self.subject_ids = torch.from_numpy(
                self.subject_ids[availability_mask]).to(device=device, dtype=torch.long)
            self.frame_ids = torch.from_numpy(
                add_labels["frame_id_list"][availability_mask]).to(device=device, dtype=torch.long)
            self.cam_ids = torch.from_numpy(
                add_labels["cam_id_list"][availability_mask]).to(device=device, dtype=torch.long)
            self.gaze_targets = torch.from_numpy(
                add_labels["gaze_point_camera_list"][availability_mask]).to(device=device, dtype=torch.float32)
            self.gaze_origins = torch.from_numpy(
                add_labels["face_centre_camera_list"][availability_mask]).to(device=device, dtype=torch.float32)
            self.landmarks = torch.from_numpy(
                add_labels["face_landmarks_crop_list"][availability_mask]).to(device=device, dtype=torch.float32)
            self.landmarks_3d = torch.from_numpy(
                add_labels["face_landmarks_3d_list"][availability_mask]).to(device=device, dtype=torch.float32)
            self.head_rotations = torch.from_numpy(
                add_labels["head_rotation_list"][availability_mask]).to(device=device, dtype=torch.float32)
            self.warp_matrices = torch.from_numpy(
                add_labels["warp_matrix_list"][availability_mask]).to(device=device, dtype=torch.float32)
        elif partition == "test":
            raise NotImplemented

        # Flip z-axis.
        # self.gaze_targets[:, 2] *= -1
        # self.gaze_origins[:, 2] *= -1
        # self.landmarks_3d[:, :, 2] *= -1

        # Load camera info.
        self.cam_intrinsics = torch.from_numpy(
            np.load(XGAZE_PATH + "processed/cam_intrinsics.npy")).to(device=device, dtype=torch.float32)

        # Construct mapping from full-data index to key and person-specific index.
        self.idx_to_sid = []
        for sid, h5_file in self.subject_files.items():
            n = h5_file["face_patch"].shape[0]
            self.idx_to_sid += [(sid, i) for i in range(n)]

    def __len__(self):
        return int(len(self.idx_to_sid) * self.ratio_sampling)

    def __getitem__(self, item):
        if self.ratio_sampling < 1.:
            all_subject_ids = torch.unique(self.subject_ids)
            sid = all_subject_ids[torch.randint(high=all_subject_ids.size(0), size=(1,))[0]]
            random_cam = torch.randint(high=18, size=(1,))[0]
            all_items = torch.where(torch.logical_and(self.subject_ids == sid, self.cam_ids == random_cam))[0]
            item = all_items[torch.randint(high=all_items.shape[0], size=(1,))[0]]

        sid, idx = self.idx_to_sid[item]

        h5_file = self.subject_files[sid]

        # Get face image.
        image = torch.from_numpy(h5_file["face_patch"][idx, :]).to(device, dtype=torch.float32)
        # image = image[:, :, ::-1]  # From BGR to RGB.

        # Get gaze.
        face_gaze = torch.from_numpy(h5_file["face_gaze"][idx, :]).to(device, dtype=torch.float32)

        data = {"subject_ids": self.subject_ids[item], "frame_ids": self.frame_ids[item], "cam_ids": self.cam_ids[item],
                "target_3d_crop": self.gaze_targets[item], "gaze_origins": self.gaze_origins[item],
                "face_landmarks_crop": self.landmarks[item][17:48], "face_landmarks_3d": self.landmarks_3d[item][1:32],
                "head_rotations": self.head_rotations[item],
                "warp_matrices": self.warp_matrices[item], "cam_intrinsics": self.cam_intrinsics[self.cam_ids[item]],
                "frames": image, "face_gazes": face_gaze}

        return data

    @property
    def face_crop_size(self):
        return self[0]["frames"].shape[0]


if __name__ == '__main__':
    """
    Some frame ids are not aligned, refer to self.frame_ids.
    h5_file["frame_index"]: processed dataset provided by the author.
    self.frame_ids: raw dataset.
    """
    import cv2
    dataset = XGazeDataset()
    for i in tqdm(range(len(dataset))):
        dt = dataset[i]

        frame = dt["frames"].cpu().numpy() / 255.
        lms = dt["face_landmarks_crop"]
        face_centre = dt["gaze_origins"].unsqueeze(0)
        gaze_target = dt["target_3d_crop"].unsqueeze(0)
        gaze_target_rot = generate_face_gaze_vector(dt["face_gazes"].unsqueeze(0), dt["head_rotations"].unsqueeze(0))
        gaze_target_rot = face_centre + gaze_target_rot * 100.

        gaze_vec = gaze_target - face_centre
        gaze_vec_rot = gaze_target_rot - face_centre
        gaze_vec = gaze_vec / torch.linalg.norm(gaze_vec, dim=1, keepdim=True)
        gaze_vec_rot = gaze_vec_rot / torch.linalg.norm(gaze_vec_rot, dim=1, keepdim=True)
        print(torch.arccos(torch.sum(gaze_vec * gaze_vec_rot)))
        face_centre_img = cam_to_img(face_centre, dt["cam_intrinsics"].unsqueeze(0))
        gaze_target_img = cam_to_img(gaze_target, dt["cam_intrinsics"].unsqueeze(0))
        gaze_target_rot_img = cam_to_img(gaze_target_rot, dt["cam_intrinsics"].unsqueeze(0))

        a = cv2.perspectiveTransform(face_centre_img[:, 0:1, :2].cpu().numpy(), dt["warp_matrices"].cpu().numpy())
        b = perspective_transform(face_centre_img, dt["warp_matrices"].unsqueeze(0))
        c = perspective_transform(gaze_target_img, dt["warp_matrices"].unsqueeze(0))
        d = perspective_transform(gaze_target_rot_img, dt["warp_matrices"].unsqueeze(0))
        print(a, b)

        gaze_rot_angle = generate_gaze_rotation_angle(gaze_target, face_centre, dt["head_rotations"].unsqueeze(0))[0]
        print(gaze_rot_angle, dt["face_gazes"])

        for lm in lms:
            cv2.drawMarker(frame, (int(lm[0]), int(lm[1])), (0, 1, 0),
                           markerSize=4, markerType=cv2.MARKER_TILTED_CROSS, thickness=1)

        cv2.arrowedLine(frame,
                        (int(b[0, 0, 0]), int(b[0, 0, 1])),
                        (int(c[0, 0, 0]), int(c[0, 0, 1])),
                        (0, 0, 255), thickness=2)
        cv2.arrowedLine(frame,
                        (int(b[0, 0, 0]), int(b[0, 0, 1])),
                        (int(d[0, 0, 0]), int(d[0, 0, 1])),
                        (0, 255, 0), thickness=2)

        cv2.imshow("", frame)
        cv2.waitKey(0)
