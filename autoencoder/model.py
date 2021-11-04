import cv2
import torch
import torch.nn as nn
from pytorch3d.transforms import axis_angle_to_matrix
from torchvision.transforms import Resize
from torchvision.transforms.functional import crop

from autoencoder.encoder import Encoder, EncoderEyePatch
from face_model.flame_model import FlameModel
from constants import *
from pt_renderer import PTRenderer
from utils.rotations import rotation_matrix_from_axis_combined


class Autoencoder(nn.Module):
    def __init__(self, args, face_crop_size=FACE_CROP_SIZE):
        super().__init__()

        self.args = args
        self.render_image = args.pixel_loss
        self.face_crop_size = face_crop_size

        if args.dataset == "eyediap":
            self.gaze_direction = 1.
        else:
            self.gaze_direction = -1.

        base_feature_channels = 400 + 145 + 3 + 3 + 1 + 2 + 2
        if args.dataset == "xgaze":
            # Add face gaze azimuth and elevation.
            base_feature_channels += 2

            # Add initial pose.
            initial_R = torch.tensor([[ 0.99494505, -0.00189795, -0.1004036 ],
                                      [ 0.02820983, -0.9542791 ,  0.297583  ],
                                      [-0.09637786, -0.29891106, -0.9494016 ]], dtype=torch.float32, device=device)
            initial_T = torch.tensor([[-20.613983,   -7.9618144, 876.4872]], dtype=torch.float32, device=device)
            initial_scale = 994.7109985351562
        else:
            initial_R, initial_T, initial_scale = None, None, None

        if args.eye_patch:
            self.encoder = EncoderEyePatch(base_feature_channels, args.network).to(device)
        else:
            self.encoder = Encoder(base_feature_channels, args.network, args.auto_weight_loss).to(device)

        self.face_model = FlameModel(FLAME_PATH + "FLAME2020/generic_model.pkl",
                                     FLAME_PATH + "albedoModel2020_FLAME_albedoPart.npz",
                                     FLAME_PATH + "FLAME_masks/FLAME_masks.pkl", None,
                                     masks=["right_eyeball", "left_eyeball", "nose", "eye_region"],
                                     initial_R=initial_R, initial_T=initial_T, initial_scale=initial_scale).to(device)
        self.renderer = PTRenderer(self.face_model.faces,
                                   self.face_model.albedo_ft.unsqueeze(0),
                                   self.face_model.albedo_vt.unsqueeze(0),
                                   face_crop_size=face_crop_size).to(device)

    def forward(self, x, camera_parameters, warp_matrices=None):
        # Encoder.
        latent_code = self.encoder(x)
        shape_parameters = latent_code[:, :400]
        albedo_parameters = latent_code[:, 400:545]
        face_rotation = latent_code[:, 545:548]
        face_translation = latent_code[:, 548:551]
        face_scale = latent_code[:, 551:552]
        left_eye_rotation = latent_code[:, 552:554]
        right_eye_rotation = latent_code[:, 554:556]
        if self.args.dataset == "xgaze":
            face_gaze_az_el = latent_code[:, 556:558]
        else:
            face_gaze_az_el = None

        face_scale = face_scale + 1.

        # 3D face reconstruction.
        vert, tex = self.face_model(shape_parameters, albedo_parameters)
        # Rigid translation with scale. i.e. apply face pose.
        # vert = torch.bmm(vert - vert.mean(1, keepdims=True),
        #                  torch.transpose(axis_angle_to_matrix(face_rotation), 1, 2)) \
        #     * face_scale.unsqueeze(1) + face_translation.unsqueeze(1)
        # print(face_rotation, face_translation, face_scale)
        vert_mean = vert.mean(1, keepdims=True)
        vert = torch.bmm(vert - vert_mean, rotation_matrix_from_axis_combined(face_rotation)) \
            * face_scale.unsqueeze(1) + face_translation.unsqueeze(1)  # Why use this?
        if warp_matrices is not None:
            # Stands for x-gaze dataset.
            vert += vert_mean  # Haven't test on eyediap.

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
        face_centre = (vert[:, self.face_model.landmarks[[19, 22, 25, 28]]].mean(1) +
                       vert[:, self.face_model.landmarks[[14, 18]]].mean(1)) / 2

        # Get initial gaze direction caused by head movement.
        l_eyeball_gaze_init = vert[:, None, 4051] - l_eyeball_centre
        r_eyeball_gaze_init = vert[:, None, 4597] - r_eyeball_centre
        l_eyeball_gaze_init = l_eyeball_gaze_init / torch.norm(l_eyeball_gaze_init, dim=2, keepdim=True)
        r_eyeball_gaze_init = r_eyeball_gaze_init / torch.norm(r_eyeball_gaze_init, dim=2, keepdim=True)

        # Apply rotations to eyeballs.
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
        left_gaze = self.apply_eyeball_rotation(torch.tensor([[[0., 0., self.gaze_direction]]], device=device),
                                                torch.tensor([[0., 0., 0.]], device=device), left_eye_rotation)
        right_gaze = self.apply_eyeball_rotation(torch.tensor([[[0., 0., self.gaze_direction]]], device=device),
                                                 torch.tensor([[0., 0., 0.]], device=device), right_eye_rotation)

        lr_cross = torch.cross(right_gaze, left_gaze, dim=2)
        lin_solve = torch.linalg.solve(torch.cat([left_gaze, -right_gaze, lr_cross], dim=1).transpose(1, 2),
                                       (r_eyeball_centre - l_eyeball_centre).transpose(1, 2))
        gaze_point_l = l_eyeball_centre + lin_solve[:, 0:1, :] * left_gaze
        gaze_point_r = r_eyeball_centre + lin_solve[:, 1:2, :] * right_gaze
        gaze_point_mid = (gaze_point_l + gaze_point_r) / 2
        gaze_point_dist = torch.linalg.norm(gaze_point_l - gaze_point_r, dim=2).squeeze(1)

        # Mask the regions for rendering.
        vert_full = vert.detach().clone()
        vert = vert[:, self.face_model.mask]
        face_landmarks_3d = vert[:, self.face_model.masked_landmarks, :]

        # Differentiable render.
        # Note: the ver_img is in the (FACE_CROP_SIZE, FACE_CROP_SIZE) image space.
        img, vert_img = self.renderer(vert, tex, camera_parameters, warp_matrices)
        # img = torch.stack([im[fc[1]:fc[1] + FACE_CROP_SIZE, fc[0] - 80:fc[0] + FACE_CROP_SIZE - 80, :]
        #                    for im, fc in zip(img, face_box_tl)])

        face_landmarks = vert_img[:, self.face_model.masked_landmarks, :2]

        # Get eye patch img.
        if self.args.eye_patch:
            left_eye_contour = vert_img.detach().clone()[:, self.face_model.masked_landmarks[-12:-6]] \
                / self.face_crop_size * self.renderer.image_size[0]
            right_eye_contour = vert_img.detach().clone()[:, self.face_model.masked_landmarks[-6:]] \
                / self.face_crop_size * self.renderer.image_size[0]
            left_contour_centre = left_eye_contour.mean(1)
            right_contour_centre = right_eye_contour.mean(1)
            l_half_width = torch.max(
                torch.abs(
                    left_eye_contour - left_contour_centre[:, None, :]).reshape(left_eye_contour.shape[0], -1),
                dim=1)[0].to(torch.long)
            r_half_width = torch.max(
                torch.abs(
                    right_eye_contour - right_contour_centre[:, None, :]).reshape(right_eye_contour.shape[0], -1),
                dim=1)[0].to(torch.long)
            left_contour_centre = torch.clip(left_contour_centre, l_half_width[:, None] + 1,
                                             (self.renderer.image_size[1] - l_half_width - 1)[:, None]).to(torch.long)
            right_contour_centre = torch.clip(right_contour_centre, r_half_width[:, None] + 1,
                                              (self.renderer.image_size[1] - r_half_width - 1)[:, None]).to(torch.long)

            left_eye_patch = []
            right_eye_patch = []
            for i in range(img.shape[0]):
                left_img = Resize(56)(img[i,
                                      int(left_contour_centre[i, 0] - l_half_width[i]):
                                      int(left_contour_centre[i, 0] + l_half_width[i]),
                                      int(left_contour_centre[i, 1] - l_half_width[i]):
                                      int(left_contour_centre[i, 1] + l_half_width[i])].permute(2, 0, 1)) \
                    .permute(1, 2, 0)
                right_img = Resize(56)(img[i,
                                           int(right_contour_centre[i, 0] - r_half_width[i]):
                                           int(right_contour_centre[i, 0] + r_half_width[i]),
                                           int(right_contour_centre[i, 1] - r_half_width[i]):
                                           int(right_contour_centre[i, 1] + r_half_width[i])].permute(2, 0, 1)) \
                    .permute(1, 2, 0)
                left_eye_patch.append(left_img)
                right_eye_patch.append(right_img)
            left_eye_patch = torch.stack(left_eye_patch)
            right_eye_patch = torch.stack(right_eye_patch)
        else:
            left_eye_patch = None
            right_eye_patch = None

        ret_dict = {"img": img,  # Rendered images.
                    "face_landmarks": face_landmarks,  # Face landmarks in image coordinate, before crop.
                    "face_landmarks_3d": face_landmarks_3d,  # Face landmarks in world coordinate.
                    "l_eyeball_centre": l_eyeball_centre,  # Left eyeball centre in world coordinate.
                    "r_eyeball_centre": r_eyeball_centre,  # Right eyeball centre in world coordinate.
                    "face_centre": face_centre,  # Face centre world coordinate.
                    "left_eye_rotation": left_eye_rotation,  # Left and Right gaze rotation.
                    "right_eye_rotation": right_eye_rotation,
                    "left_gaze": left_gaze,  # Left and Right gazes.
                    "right_gaze": right_gaze,
                    "face_gaze_az_el": face_gaze_az_el,  # Face gaze azimuth and elevation.
                    "gaze_point_l": gaze_point_l,  # Predicted left gaze point in world coordinate.
                    "gaze_point_r": gaze_point_r,  # Predicted right gaze point in world coordinate.
                    "gaze_point_mid": gaze_point_mid,  # Predicted gaze point in world coordinate.
                    "gaze_point_dist": gaze_point_dist,  # Sum of the distances between the gaze points to gaze vector.
                    "shape_parameters": shape_parameters,  # FLAME shape parameters.
                    "albedo_parameters": albedo_parameters,  # FLAME albedo parameters.
                    "vert_full": vert_full,  # Full FLAME head.
                    "vert_masked": vert,  # Masked FLAME head.
                    "left_eye_patch": left_eye_patch,
                    "right_eye_patch": right_eye_patch,
                    "sb": l_eyeball_centre + left_gaze * 1000.,
                    "sb2": r_eyeball_centre + right_gaze * 1000.,
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

    def find_gaze_axis_rotation_matrix(self, init_gaze):
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
        b = torch.tensor([[0., 0., self.gaze_direction]]).to(device).repeat(batch_size, 1)

        a = a / torch.norm(a, dim=1).unsqueeze(1)

        v = torch.cross(a, b)
        v_cross = torch.stack([torch.stack([torch.zeros_like(v[:, 0]), -v[:, 2], v[:, 1]], dim=1),
                               torch.stack([v[:, 2], torch.zeros_like(v[:, 0]), -v[:, 0]], dim=1),
                               torch.stack([-v[:, 1], v[:, 0], torch.zeros_like(v[:, 0])], dim=1)], dim=1)
        rot = torch.eye(3).to(device).repeat(batch_size, 1, 1) + \
            v_cross + torch.bmm(v_cross, v_cross) * (1 / (1 + torch.sum(a * b, dim=1)))[:, None, None]
        rot = torch.transpose(rot, 1, 2)

        return rot


class AutoencoderBaseline(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.args = args

        if args.eye_patch:
            self.encoder = EncoderEyePatch(10, args.network).to(device)
        else:
            self.encoder = Encoder(10, args.network).to(device)

    def forward(self, x, camera_parameters=None, warp_matrices=None):
        # Encoder.
        latent_code = self.encoder(x)
        l_eyeball_centre = latent_code[:, None, 0:3]
        r_eyeball_centre = latent_code[:, None, 3:6]
        left_eye_rotation = latent_code[:, 6:8]
        right_eye_rotation = latent_code[:, 8:10]

        # Calculate gaze information.
        left_gaze = self.apply_eyeball_rotation(torch.tensor([[[0., 0., 1.]]], device=device),
                                                torch.tensor([[0., 0., 0.]], device=device), left_eye_rotation)
        right_gaze = self.apply_eyeball_rotation(torch.tensor([[[0., 0., 1.]]], device=device),
                                                 torch.tensor([[0., 0., 0.]], device=device), right_eye_rotation)

        lr_cross = torch.cross(right_gaze, left_gaze, dim=2)
        lin_solve = torch.linalg.solve(torch.cat([left_gaze, -right_gaze, lr_cross], dim=1).transpose(1, 2),
                                       (r_eyeball_centre - l_eyeball_centre).transpose(1, 2))
        gaze_point_l = l_eyeball_centre + lin_solve[:, 0:1, :] * left_gaze
        gaze_point_r = r_eyeball_centre + lin_solve[:, 1:2, :] * right_gaze
        gaze_point_mid = (gaze_point_l + gaze_point_r) / 2
        gaze_point_dist = torch.linalg.norm(gaze_point_l - gaze_point_r, dim=2).squeeze(1)

        ret_dict = {"l_eyeball_centre": l_eyeball_centre,  # Left eyeball centre in world coordinate.
                    "r_eyeball_centre": r_eyeball_centre,  # Right eyeball centre in world coordinate.
                    "left_eye_rotation": left_eye_rotation,  # Left and Right gaze rotation.
                    "right_eye_rotation": right_eye_rotation,
                    "left_gaze": left_gaze,  # Left and Right gazes.
                    "right_gaze": right_gaze,
                    "gaze_point_l": gaze_point_l,  # Predicted left gaze point in world coordinate.
                    "gaze_point_r": gaze_point_r,  # Predicted right gaze point in world coordinate.
                    "gaze_point_mid": gaze_point_mid,  # Predicted gaze point in world coordinate.
                    "gaze_point_dist": gaze_point_dist  # Sum of the distances between the gaze points to gaze vector.
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
