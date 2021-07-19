
import cv2
import numpy as np
import time
import torch
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torchvision.transforms import Resize
from tqdm import tqdm
import argparse
import logging
import shutil

from autoencoder.loss_functions import pixel_loss, landmark_loss, eye_loss, gaze_target_loss, gaze_divergence_loss, \
    parameters_regulariser, gaze_pose_loss
from autoencoder.model import Autoencoder
from utils.eyediap_dataset import EYEDIAP
from utils.logger import TrainingLogger
from constants import *


def train():
    model = Autoencoder()
    train_data = EYEDIAP(partition="train", head_movement=["S", "M"])
    test_data = EYEDIAP(partition="test", head_movement=["S", "M"])
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_data, batch_size=args.test_batch_size, shuffle=True, num_workers=0)

    optimiser = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
    if args.lr_scheduler == "step":
        scheduler = StepLR(optimiser, 25, 0.9)
    else:
        scheduler = None

    def calculate_losses(data_, results_, partition_):
        loss_pixel = pixel_loss(results_["img"], gt_img_input.permute(0, 2, 3, 1))
        loss_landmark = landmark_loss(results_["face_landmarks"], data_["face_landmarks_crop"])
        loss_eye = eye_loss(results_["l_eyeball_centre"].squeeze(1), results_["r_eyeball_centre"].squeeze(1),
                            data_["left_eyeball_3d_crop"], data_["right_eyeball_3d_crop"])
        loss_gaze_target = gaze_target_loss(results_["gaze_point_mid"].squeeze(1), data_["target_3d_crop"])
        loss_gaze_div = gaze_divergence_loss(results_["gaze_point_dist"])
        loss_gaze_pose = gaze_pose_loss(results_["right_eye_rotation"], data_["left_eyeball_rotation_crop"],
                                        results_["right_eye_rotation"], data_["right_eyeball_rotation_crop"])
        reg_shape_param = parameters_regulariser(results_["shape_parameters"])
        reg_albedo_param = parameters_regulariser(results_["albedo_parameters"])

        batch_size = results_["img"].shape[0]
        training_logger.log_batch_loss("pixel_loss", loss_pixel.item(), partition_, batch_size)
        training_logger.log_batch_loss("landmark_loss", loss_landmark.item(), partition_, batch_size)
        training_logger.log_batch_loss("eye_loss", loss_eye.item(), partition_, batch_size)
        training_logger.log_batch_loss("gaze_tgt_loss", loss_gaze_target.item(), partition_, batch_size)
        training_logger.log_batch_loss("gaze_div_loss", loss_gaze_div.item(), partition_, batch_size)
        training_logger.log_batch_loss("gaze_pose_loss", loss_gaze_pose.item(), partition_, batch_size)
        training_logger.log_batch_loss("shape_param_reg", reg_shape_param.item(), partition_, batch_size)
        training_logger.log_batch_loss("albedo_para_reg", reg_albedo_param.item(), partition_, batch_size)

        total_loss_weighted = \
            loss_pixel * args.lambda1 + loss_landmark * args.lambda2 + loss_eye * args.lambda3 + \
            loss_gaze_target * args.lambda4 + loss_gaze_div * args.lambda5 + loss_gaze_pose * args.lambda6 + \
            reg_shape_param * args.lambda7 + reg_albedo_param * args.lambda8

        training_logger.log_batch_loss("loss", loss_eye.item() + loss_gaze_target.item() + loss_gaze_div.item(),
                                       partition_, batch_size)

        return total_loss_weighted

    logging.info("Start training...")
    full_start_time = time.time()
    for epoch in range(1, args.epochs + 1):
        logging.info("*** Epoch " + str(epoch) + " ***")
        epoch_start_time = time.time()

        """ Train.
        """
        for data in tqdm(train_loader):
            # Load forward information.
            gt_img = data["frames"].to(torch.float32) / 255.
            camera_parameters = (data["cam_R"], data["cam_T"], data["cam_K"])

            # Preprocess.
            resize_transformation = Resize(224)
            gt_img_input = resize_transformation(gt_img.permute(0, 3, 1, 2))

            # Forward.
            optimiser.zero_grad()
            results = model(gt_img_input, camera_parameters)

            # Calculate losses.
            loss = calculate_losses(data, results, "train")

            # Back-prop.
            loss.backward()
            optimiser.step()

        if scheduler is not None:
            scheduler.step()

        """ Evaluate.
        """
        model.eval()
        for data in tqdm(test_loader):
            # Load forward information.
            gt_img = data["frames"].to(torch.float32) / 255.
            camera_parameters = (data["cam_R"], data["cam_T"], data["cam_K"])

            # Preprocess.
            resize_transformation = Resize(224)
            gt_img_input = resize_transformation(gt_img.permute(0, 3, 1, 2))

            # Forward.
            with torch.no_grad():
                results = model(gt_img_input, camera_parameters)

                # Calculate losses.
                calculate_losses(data, results, "eval")

        j = 0
        gt_img_np = gt_img_input.permute(0, 2, 3, 1)[j].detach().cpu().numpy()
        result_img_np = results["img"][j].detach().cpu().numpy()

        result_img_np_none = np.all(result_img_np == [0., 1., 0.], axis=2)
        result_img_np[result_img_np_none] = gt_img_np[result_img_np_none]

        # tl = data["face_box_tl"][j]
        # for lm in data["face_landmarks"][j]:
        #     cv2.circle(result_img_np, (int(lm[0] - tl[0]), int(lm[1] - tl[1])), 1, (0, 255, 0))
        # for lm in results["face_landmarks"][j]:
        #     cv2.circle(result_img_np, (int(lm[0] - tl[0]), int(lm[1] - tl[1])), 1, (0, 0, 255))

        cv2.imshow("1", cv2.resize(gt_img_np, (512, 512)))
        cv2.imshow("2", cv2.resize(result_img_np, (512, 512)))
        cv2.waitKey(1)
        model.train()

        """ Epoch ends.
        """
        training_logger.log_epoch({"epoch": epoch,
                                   "model_weights": model.state_dict(),
                                   "model": model,
                                   "optimiser_weights": optimiser.state_dict()})
        epoch_end_time = time.time()
        logging.info("Time left: " + time_format((epoch_end_time - epoch_start_time) * (args.epochs - epoch)))

    full_end_time = time.time()
    logging.info("Total training time: " + time_format(full_end_time - full_start_time))


def time_format(time_diff):
    hours, rem = divmod(time_diff, 3600)
    minutes, seconds = divmod(rem, 60)
    return "{:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train Autoencoder")
    """ Experiment configurations.
    """
    parser.add_argument("-n", "--name", type=str, default="Temporary Experiment",
                        help="Name of the experiment.")
    parser.add_argument("--seed", type=int, default=1,
                        help="Random seed (default: 1).")
    parser.add_argument("-o", "--override", type=bool, default=False,
                        help="Override log directory.")

    """ Hyper-parameters.
    """
    parser.add_argument("-e", "--epochs", type=int, default=200, metavar="N",
                        help="Number of episode to train.")
    parser.add_argument("-b", "--batch_size", type=int, default=32, metavar="N",
                        help="Batch size for training.")
    parser.add_argument("--test_batch_size", type=int, default=32, metavar="N",
                        help="Batch size for evaluating.")

    # Loss function hyper-parameters.
    parser.add_argument("--lambda1", type=float, nargs="*", default=1.,
                        help="Lambda to balance loss function.")
    parser.add_argument("--lambda2", type=float, nargs="*", default=0.1,
                        help="Lambda to balance loss functions.")
    parser.add_argument("--lambda3", type=float, nargs="*", default=100.,
                        help="Lambda to balance loss function.")
    parser.add_argument("--lambda4", type=float, nargs="*", default=50.,
                        help="Lambda to balance loss functions.")
    parser.add_argument("--lambda5", type=float, nargs="*", default=10.,
                        help="Lambda to balance loss functions.")
    parser.add_argument("--lambda6", type=float, nargs="*", default=10.,
                        help="Lambda to balance loss functions.")
    parser.add_argument("--lambda7", type=float, nargs="*", default=0.05,
                        help="Lambda to balance loss functions.")
    parser.add_argument("--lambda8", type=float, nargs="*", default=0.01,
                        help="Lambda to balance loss functions.")
    parser.add_argument("--lr", type=float, default=5e-4,
                        help="Learning rate.")
    parser.add_argument("--lr_scheduler", type=str, default=None,
                        help="Learning rate scheduler, choose from: (cos) for (CosineAnnealingLR), "
                             "(step) for (StepLR).")

    """ Misc
    """
    parser.add_argument("--logging_level", type=str, default="INFO",
                        help="Logging level, choose from: (CRITICAL|ERROR|WARNING|INFO|DEBUG).")

    args = parser.parse_args()

    """ Insert argument override here. """
    # args.name = "v1_m"
    args.epochs = 150
    args.seed = 1
    args.lr = 1e-4
    args.lr_scheduler = "step"

    args.batch_size = 32

    args.override = True
    # """ End of argument override. """

    # Set PyTorch manual random seed.
    # torch.manual_seed(args.seed)

    # Setup logging.
    # Initialise log folder.
    log_name_dir = LOGS_PATH + args.name + "/"
    if os.path.exists(log_name_dir):
        if args.name == "Temporary Experiment" or args.override:
            shutil.rmtree(log_name_dir)
            shutil.rmtree(LOGS_PATH + "tf_board/" + args.name)
        else:
            raise ValueError("Name has been used.")
    os.mkdir(log_name_dir)

    # Initialise logging config.
    assert args.logging_level in ["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"]
    logging_level = eval("logging." + args.logging_level)

    console_handler = logging.StreamHandler()
    logging.basicConfig(
        handlers=[logging.FileHandler(filename=log_name_dir + "log.txt"), console_handler],
        level=logging.DEBUG,
        format="%(asctime)s %(levelname)-8s %(message)s", datefmt="[%Y-%m-%d %H:%M:%S]")
    console_handler.setLevel(logging.INFO)

    logging.info(vars(args))

    training_logger = TrainingLogger(log_name_dir, args)
    train()
