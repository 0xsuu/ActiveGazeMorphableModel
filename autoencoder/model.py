
import torch
import torch.nn as nn
import torchvision.models as models
from pytorch3d.transforms import axis_angle_to_matrix
from torchvision.models.resnet import load_state_dict_from_url, BasicBlock, model_urls

from face_model.flame_model import FlameModel
from constants import *
from pt_renderer import PTRenderer
from utils.rotations import rotation_matrix_from_axis_combined


class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = models.ResNet(BasicBlock, [2, 2, 2, 2], num_classes=400 + 145 + 3 + 3 + 1 + 2 + 2).to(device)
        pretrain_state_dict = load_state_dict_from_url(model_urls["resnet18"], progress=True)
        pretrain_state_dict.pop("fc.weight")
        pretrain_state_dict.pop("fc.bias")
        self.encoder.load_state_dict(pretrain_state_dict, strict=False)
        nn.init.normal_(self.encoder.fc.weight, 0., 0.001)
        nn.init.zeros_(self.encoder.fc.bias)

        self.face_model = FlameModel(FLAME_PATH + "FLAME2020/generic_model.pkl",
                                     FLAME_PATH + "albedoModel2020_FLAME_albedoPart.npz",
                                     FLAME_PATH + "FLAME_masks/FLAME_masks.pkl", None,
                                     masks=["right_eyeball", "left_eyeball", "nose", "eye_region"]).to(device)
        self.renderer = PTRenderer(self.face_model.faces,
                                   self.face_model.albedo_ft.unsqueeze(0),
                                   self.face_model.albedo_vt.unsqueeze(0)).to(device)

    def forward(self, x, camera_parameters):
        # Encoder.
        hidden_vector = self.encoder(x)
        shape_parameters = hidden_vector[:, :400]
        albedo_parameters = hidden_vector[:, 400:545]
        face_rotation = hidden_vector[:, 545:548]
        face_translation = hidden_vector[:, 548:551]
        face_scale = hidden_vector[:, 551:552]
        left_eye_rotation = hidden_vector[:, 552:554]
        right_eye_rotation = hidden_vector[:, 554:556]

        face_scale = face_scale + 1.

        # 3D face reconstruction.
        vert, tex = self.face_model(shape_parameters, albedo_parameters)
        # Rigid translation with scale. i.e. apply face pose.
        # vert = torch.bmm(vert - vert.mean(1, keepdims=True), axis_angle_to_matrix(face_rotation)) \
        #     * face_scale.unsqueeze(1) + face_translation.unsqueeze(1)
        vert = torch.bmm(vert - vert.mean(1, keepdims=True), rotation_matrix_from_axis_combined(face_rotation)) \
            * face_scale.unsqueeze(1) + face_translation.unsqueeze(1)

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
        vert_eyes[:, self.face_model.left_eyeball_mask] = \
            self.apply_eyeball_rotation(l_eyeball, l_eyeball_centre, left_eye_rotation)
        vert_eyes[:, self.face_model.right_eyeball_mask] = \
            self.apply_eyeball_rotation(r_eyeball, r_eyeball_centre, right_eye_rotation)
        vert = vert_no_eyes + vert_eyes

        # Calculate gaze information.
        left_gaze = self.apply_eyeball_rotation(torch.tensor([[[0., 0., 1.]]], device=device),
                                                torch.tensor([[0., 0., 0.]], device=device), left_eye_rotation)
        right_gaze = self.apply_eyeball_rotation(torch.tensor([[[0., 0., 1.]]], device=device),
                                                 torch.tensor([[0., 0., 0.]], device=device), right_eye_rotation)

        lr_cross = torch.cross(left_gaze, right_gaze, dim=2)
        results = torch.bmm(torch.inverse(torch.cat([left_gaze, -right_gaze, lr_cross], dim=1).transpose(1, 2)),
                            (r_eyeball_centre - l_eyeball_centre).transpose(1, 2))
        gaze_point_l = l_eyeball_centre + results[:, 0:1, :] * left_gaze
        gaze_point_r = r_eyeball_centre + results[:, 1:2, :] * right_gaze
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
                    "right_eye_rotation": right_eye_rotation,  #
                    "left_gaze": left_gaze,  # Left and Right gazes.
                    "right_gaze": right_gaze,
                    "gaze_point_mid": gaze_point_mid,  # Predicted gaze point in world coordinate.
                    "gaze_point_dist": gaze_point_dist,  # Sum of the distances between the gaze points to gaze vector.
                    "shape_parameters": shape_parameters,  # FLAME shape parameters.
                    "albedo_parameters": albedo_parameters  # FLAME albedo parameters.
                    }
        
        return ret_dict

    @staticmethod
    def apply_eyeball_rotation(eyeball, eyeball_centre, eyeball_rotation_euler):
        return torch.einsum(
            "bvl,bli->bvi", eyeball - eyeball_centre,
            axis_angle_to_matrix(torch.cat([eyeball_rotation_euler,
                                            torch.zeros_like(eyeball_rotation_euler[:, 0:1])], dim=1))) \
            + eyeball_centre
