import numpy as np
import torch
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torchvision.transforms import Resize
from tqdm import tqdm

from autoencoder.encoder import Encoder
from autoencoder.loss_functions import gaze_pose_loss, gaze_target_loss
from autoencoder.model import Autoencoder
from utils.eyediap_dataset import EYEDIAP
from constants import *


def pretrain():
    train_data = EYEDIAP(partition="train", head_movement=["S", "M"])
    test_data = EYEDIAP(partition="test", head_movement=["S", "M"])
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_data, batch_size=32, shuffle=True, num_workers=0)

    encoder = Encoder(400 + 145 + 3 + 3 + 1 + 2 + 2)
    optimiser = torch.optim.Adam(encoder.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = StepLR(optimiser, 25, 0.9)

    best_loss = np.inf
    for epoch in range(1, 100):
        average_loss = 0
        for data in tqdm(train_loader):
            # Load forward information.
            gt_img = data["frames"].to(torch.float32) / 255.

            # Preprocess.
            resize_transformation = Resize(224)
            gt_img_input = resize_transformation(gt_img.permute(0, 3, 1, 2))

            # Forward.
            optimiser.zero_grad()
            latent_code = encoder(gt_img_input)
            left_eye_rotation = latent_code[:, 552:554]
            right_eye_rotation = latent_code[:, 554:556]

            left_gaze = Autoencoder.apply_eyeball_rotation(
                torch.tensor([[[0., 0., 1.]]], device=device),
                torch.tensor([[0., 0., 0.]], device=device),
                left_eye_rotation)
            right_gaze = Autoencoder.apply_eyeball_rotation(
                torch.tensor([[[0., 0., 1.]]], device=device),
                torch.tensor([[0., 0., 0.]], device=device),
                right_eye_rotation)

            l_eyeball_centre = data["left_eyeball_3d"].unsqueeze(1)
            r_eyeball_centre = data["right_eyeball_3d"].unsqueeze(1)
            lr_cross = torch.cross(left_gaze, right_gaze, dim=2)
            lin_solve = torch.bmm(torch.inverse(torch.cat([left_gaze, -right_gaze, lr_cross], dim=1).transpose(1, 2)),
                                  (r_eyeball_centre - l_eyeball_centre).transpose(1, 2))
            gaze_point_l = l_eyeball_centre + lin_solve[:, 0:1, :] * left_gaze
            gaze_point_r = r_eyeball_centre + lin_solve[:, 1:2, :] * right_gaze
            gaze_point_mid = (gaze_point_l + gaze_point_r) / 2

            loss_gaze_pose = gaze_pose_loss(left_eye_rotation, data["left_eyeball_rotation_crop"],
                                            right_eye_rotation, data["right_eyeball_rotation_crop"])
            loss_gaze_target = gaze_target_loss(gaze_point_mid.squeeze(1), data["target_3d_crop"])

            # Calculate losses.
            loss = loss_gaze_pose * 10. + loss_gaze_target * 50.
            average_loss += loss.item() * gt_img.shape[0]

            # Back-prop.
            loss.backward()
            optimiser.step()

        average_loss /= len(train_loader)
        if average_loss <= best_loss:
            print("Model saved.")
            torch.save(encoder.state_dict(), "encoder_pretrain.pt")
            best_loss = average_loss

        scheduler.step()


if __name__ == '__main__':
    pretrain()
