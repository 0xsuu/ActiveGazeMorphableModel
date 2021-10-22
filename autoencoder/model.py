
import torch
import torch.nn as nn
from pytorch3d.transforms import axis_angle_to_matrix

from autoencoder.encoder import Encoder
from face_model.flame_model import FlameModel
from constants import *
from pt_renderer import PTRenderer
from utils.rotations import rotation_matrix_from_axis_combined


class Autoencoder(nn.Module):
    def __init__(self, render_image=True):
        super().__init__()

        self.render_image = render_image

        self.encoder = Encoder(400 + 145 + 3 + 3 + 1 + 2 + 2)

        self.face_model = FlameModel(FLAME_PATH + "FLAME2020/generic_model.pkl",
                                     FLAME_PATH + "albedoModel2020_FLAME_albedoPart.npz",
                                     FLAME_PATH + "FLAME_masks/FLAME_masks.pkl", None,
                                     masks=["right_eyeball", "left_eyeball", "nose", "eye_region"]).to(device)
        self.renderer = PTRenderer(self.face_model.faces,
                                   self.face_model.albedo_ft.unsqueeze(0),
                                   self.face_model.albedo_vt.unsqueeze(0)).to(device)

    def forward(self, x, camera_parameters):
        # Encoder.
        latent_code = self.encoder(x)
        shape_parameters = latent_code[:, :400]
        albedo_parameters = latent_code[:, 400:545]
        face_rotation = latent_code[:, 545:548]
        face_translation = latent_code[:, 548:551]
        face_scale = latent_code[:, 551:552]
        left_eye_rotation = latent_code[:, 552:554]
        right_eye_rotation = latent_code[:, 554:556]

        face_scale = face_scale + 1.

        # 3D face reconstruction.
        vert, tex = self.face_model(shape_parameters, albedo_parameters)
        # Rigid translation with scale. i.e. apply face pose.
        # vert = torch.bmm(vert - vert.mean(1, keepdims=True),
        #                  torch.transpose(axis_angle_to_matrix(face_rotation), 1, 2)) \
        #     * face_scale.unsqueeze(1) + face_translation.unsqueeze(1)
        vert = torch.bmm(vert - vert.mean(1, keepdims=True), rotation_matrix_from_axis_combined(face_rotation)) \
            * face_scale.unsqueeze(1) + face_translation.unsqueeze(1)  # Why use this?

        # # TODO: Debug
        # vert = vert - torch.mean(vert[:, self.face_model.left_eyeball_mask], dim=1).unsqueeze(1) + dbg1.unsqueeze(1)
        # # print(vert.mean([0, 1]))
        # left_eye_rotation = dbg2
        # right_eye_rotation = dbg3

        # Extracting key points in world coordinate.
        l_eyeball = vert[:, self.face_model.left_eyeball_mask]
        r_eyeball = vert[:, self.face_model.right_eyeball_mask]
        l_eyeball_centre = torch.mean(l_eyeball, dim=1, keepdim=True)
        r_eyeball_centre = torch.mean(r_eyeball, dim=1, keepdim=True)

        # Get initial gaze direction caused by head movement.
        l_eyeball_gaze_init = vert[:, None, 4051] - l_eyeball_centre
        r_eyeball_gaze_init = vert[:, None, 4597] - r_eyeball_centre
        l_eyeball_gaze_init = l_eyeball_gaze_init / torch.norm(l_eyeball_gaze_init, dim=2, keepdim=True)
        r_eyeball_gaze_init = r_eyeball_gaze_init / torch.norm(r_eyeball_gaze_init, dim=2, keepdim=True)

        # Apply rotations to eyeballs.
        # Inplace version.
        # vert[:, self.face_model.left_eyeball_mask] = \
        #     self.apply_eyeball_rotation(l_eyeball, l_eyeball_centre, left_eye_rotation)
        # vert[:, self.face_model.right_eyeball_mask] = \
        #     self.apply_eyeball_rotation(r_eyeball, r_eyeball_centre, right_eye_rotation)
        # Non-inplace version.
        vert_eyes = torch.zeros_like(vert)
        vert_eyes[:, self.face_model.left_eyeball_mask] = vert[:, self.face_model.left_eyeball_mask]
        vert_eyes[:, self.face_model.right_eyeball_mask] = vert[:, self.face_model.right_eyeball_mask]
        vert_no_eyes = vert - vert_eyes
        # Normalise eyeball vertices to make it look at [0, 0, 1].
        vert_eyes[:, self.face_model.left_eyeball_mask] = \
            torch.bmm((vert_eyes[:, self.face_model.left_eyeball_mask] - l_eyeball_centre),
                      self.find_gaze_axis_rotation_matrix(l_eyeball_gaze_init)) \
            + l_eyeball_centre
        vert_eyes[:, self.face_model.right_eyeball_mask] = \
            torch.bmm((vert_eyes[:, self.face_model.right_eyeball_mask] - r_eyeball_centre),
                      self.find_gaze_axis_rotation_matrix(r_eyeball_gaze_init)) \
            + r_eyeball_centre

        # Apply predicted gaze direction.
        vert_eyes[:, self.face_model.left_eyeball_mask] = \
            self.apply_eyeball_rotation(vert_eyes[:, self.face_model.left_eyeball_mask],
                                        l_eyeball_centre, left_eye_rotation)
        vert_eyes[:, self.face_model.right_eyeball_mask] = \
            self.apply_eyeball_rotation(vert_eyes[:, self.face_model.right_eyeball_mask],
                                        r_eyeball_centre, right_eye_rotation)
        vert = vert_no_eyes + vert_eyes

        # Calculate gaze information.
        left_gaze = self.apply_eyeball_rotation(torch.tensor([[[0., 0., 1.]]], device=device),
                                                torch.tensor([[0., 0., 0.]], device=device), left_eye_rotation)
        right_gaze = self.apply_eyeball_rotation(torch.tensor([[[0., 0., 1.]]], device=device),
                                                 torch.tensor([[0., 0., 0.]], device=device), right_eye_rotation)

        lr_cross = torch.cross(left_gaze, right_gaze, dim=2)
        lin_solve = torch.bmm(torch.inverse(torch.cat([left_gaze, -right_gaze, lr_cross], dim=1).transpose(1, 2)),
                              (r_eyeball_centre - l_eyeball_centre).transpose(1, 2))
        gaze_point_l = l_eyeball_centre + lin_solve[:, 0:1, :] * left_gaze
        gaze_point_r = r_eyeball_centre + lin_solve[:, 1:2, :] * right_gaze
        gaze_point_mid = (gaze_point_l + gaze_point_r) / 2
        gaze_point_dist = torch.linalg.norm(gaze_point_l - gaze_point_r, dim=2).squeeze(1)

        # Mask the regions for rendering.
        vert = vert[:, self.face_model.mask]

        # Differentiable render.
        img, vert_img = self.renderer(vert, tex, camera_parameters)
        # img = torch.stack([im[fc[1]:fc[1] + FACE_CROP_SIZE, fc[0] - 80:fc[0] + FACE_CROP_SIZE - 80, :]
        #                    for im, fc in zip(img, face_box_tl)])

        face_landmarks = vert_img[:, self.face_model.masked_landmarks, :2]

        ret_dict = {"img": img,  # Rendered images.
                    "face_landmarks": face_landmarks,  # Face landmarks in image coordinate, before crop.
                    "l_eyeball_centre": l_eyeball_centre,  # Left eyeball centre in world coordinate.
                    "r_eyeball_centre": r_eyeball_centre,  # Right eyeball centre in world coordinate.
                    "left_eye_rotation": left_eye_rotation,  # Left and Right gaze rotation.
                    "right_eye_rotation": right_eye_rotation,
                    "left_gaze": left_gaze,  # Left and Right gazes.
                    "right_gaze": right_gaze,
                    "gaze_point_l": gaze_point_l,  # Predicted left gaze point in world coordinate.
                    "gaze_point_r": gaze_point_r,  # Predicted right gaze point in world coordinate.
                    "gaze_point_mid": gaze_point_mid,  # Predicted gaze point in world coordinate.
                    "gaze_point_dist": gaze_point_dist,  # Sum of the distances between the gaze points to gaze vector.
                    "shape_parameters": shape_parameters,  # FLAME shape parameters.
                    "albedo_parameters": albedo_parameters,  # FLAME albedo parameters.
                    }
        
        return ret_dict

    @staticmethod
    def apply_eyeball_rotation(eyeball, eyeball_centre, eyeball_rotation_axis_angle):
        """
        Rotate eyeball with azimuth and elevation.

        :param eyeball:
        :param eyeball_centre: Provided to reduce calculation.
        :param eyeball_rotation_axis_angle:
        :return:
        """
        return torch.einsum(
            "bvl,bli->bvi", eyeball - eyeball_centre,
            axis_angle_to_matrix(torch.cat([eyeball_rotation_axis_angle,
                                            torch.zeros_like(eyeball_rotation_axis_angle[:, 0:1])], dim=1))) \
            + eyeball_centre

    @staticmethod
    def find_gaze_axis_rotation_matrix(init_gaze):
        """
        Find the azimuth and elevation axis angles from init_gaze to [0, 0, 1]

        v1 -> v2 = [0, 0, 1]
        azimuth = atan2(v2.z,v2.y) - atan2(v1.z,v1.y)
        elevation = atan2(v2.z,v2.x) - atan2(v1.z,v1.x)

        :param init_gaze:
        :return:
        """
        # # Old shit.
        # azimuth = - torch.atan2(init_gaze[:, 0, 1], init_gaze[:, 0, 2])
        # elevation = 1.5707963267948966 - torch.atan2(init_gaze[:, 0, 2], init_gaze[:, 0, 0])  # atan2(1, 0) - ...
        #
        # return torch.stack([azimuth, elevation], dim=1)

        # Idiot for-loop version.
        # rot_stacked = []
        # for i in range(init_gaze.shape[0]):
        #     a = init_gaze[i, 0]
        #     b = torch.tensor([0., 0., 1.]).to(device)
        #
        #     a = a / torch.norm(a)
        #
        #     v = torch.cross(a, b)
        #     v_cross = torch.tensor([[0., -v[2], v[1]],
        #                             [v[2], 0., -v[0]],
        #                             [-v[1], v[0], 0.]], device=device)
        #     rot = torch.eye(3).to(device) + v_cross + v_cross @ v_cross * (1 / (1 + torch.sum(a * b)))
        #     rot_stacked.append(rot.T)
        # rot_stacked = torch.stack(rot_stacked)
        #
        # return rot_stacked

        batch_size = init_gaze.shape[0]
        a = init_gaze[:, 0, :]
        b = torch.tensor([[0., 0., 1.]]).to(device).repeat(batch_size, 1)

        a = a / torch.norm(a, dim=1).unsqueeze(1)

        v = torch.cross(a, b)
        v_cross = torch.stack([torch.stack([torch.zeros_like(v[:, 0]), -v[:, 2], v[:, 1]], dim=1),
                               torch.stack([v[:, 2], torch.zeros_like(v[:, 0]), -v[:, 0]], dim=1),
                               torch.stack([-v[:, 1], v[:, 0], torch.zeros_like(v[:, 0])], dim=1)], dim=1)
        rot = torch.eye(3).to(device).repeat(batch_size, 1, 1) + \
            v_cross + torch.bmm(v_cross, v_cross) * (1 / (1 + torch.sum(a * b, dim=1)))[:, None, None]
        rot = torch.transpose(rot, 1, 2)

        return rot
