
import cv2
import numpy as np
import time
import torch
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torchvision.transforms import Resize, Grayscale, transforms, Normalize
import torch.nn.functional as F
from tqdm import tqdm
import argparse
import logging
import shutil

from autoencoder.loss_functions import pixel_loss, landmark_loss, eye_loss, gaze_target_loss, gaze_divergence_loss, \
    parameters_regulariser, gaze_pose_loss, gaze_degree_error
from autoencoder.model import Autoencoder, AutoencoderBaseline
from utils.eyediap_dataset import EYEDIAP
from utils.logger import TrainingLogger
from constants import *
from utils.xgaze_dataset import XGazeDataset, cam_to_img, perspective_transform


def train():
    # Load datasets.
    if args.dataset == "eyediap":
        train_data = EYEDIAP(partition="train", eval_subjects=[1, 15, 16], head_movement=["S", "M"])
        validation_data = EYEDIAP(partition="test", eval_subjects=[1, 15], head_movement=["S", "M"])
    else:
        train_data = XGazeDataset(partition="train", ratio_sampling=0.1)
        validation_data = XGazeDataset(partition="cv", ratio_sampling=0.1)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=0)
    validation_loader = DataLoader(validation_data, batch_size=args.test_batch_size, shuffle=True, num_workers=0)

    # Initialise model.
    if "baseline" in args.name:
        model = AutoencoderBaseline(args)
    else:
        model = Autoencoder(args, face_crop_size=train_data.face_crop_size)

    # Log source code for records.
    logging_source_code(model)

    # Initialise optimiser.
    optimiser = torch.optim.Adam(model.encoder.parameters(), lr=1e-4, weight_decay=1e-4)
    if args.lr_scheduler == "step":
        scheduler = StepLR(optimiser, 25, 0.9)
    else:
        scheduler = None

    # Loss function calculation wrap-up.
    def calculate_losses(data_, results_, partition_):
        batch_size = results_["left_gaze"].shape[0]

        loss_weights = model.encoder.sigmas

        total_loss_weighted = 0
        total_loss_weighted += torch.sum(loss_weights)
        if args.pixel_loss:
            loss_pixel = pixel_loss(results_["img"], gt_img_input.permute(0, 2, 3, 1))
            if args.eye_patch:
                loss_pixel += pixel_loss(
                    l_eye_patch_transformation(results_["left_eye_patch"].permute(0, 3, 1, 2)).permute(0, 2, 3, 1),
                    left_eye_img.permute(0, 2, 3, 1))
                loss_pixel += pixel_loss(
                    r_eye_patch_transformation(results_["right_eye_patch"].permute(0, 3, 1, 2)).permute(0, 2, 3, 1),
                    right_eye_img.permute(0, 2, 3, 1))
            training_logger.log_batch_loss("pixel_loss", loss_pixel.item(), partition_, batch_size)
            total_loss_weighted += loss_pixel * torch.exp(-loss_weights[0]) #* args.lambda1

        if args.landmark_loss:
            loss_landmark = landmark_loss(results_["face_landmarks"], data_["face_landmarks_crop"])
            training_logger.log_batch_loss("landmark_loss", loss_landmark.item(), partition_, batch_size)
            total_loss_weighted += loss_landmark * torch.exp(-loss_weights[1])#* args.lambda2

        if args.eye_loss:
            if args.dataset == "eyediap":
                loss_eye = eye_loss(results_["l_eyeball_centre"].squeeze(1), results_["r_eyeball_centre"].squeeze(1),
                                    data_["left_eyeball_3d_crop"], data_["right_eyeball_3d_crop"])
            else:
                # Using face point loss instead of eye positions for x-gaze dataset.
                # loss_eye = F.l1_loss(results_["face_centre"], data_["gaze_origins"])
                # loss_eye = F.l1_loss(results_["face_landmarks_3d"], data_["face_landmarks_3d"])
                loss_eye = F.mse_loss(results_["face_landmarks_3d"], data_["face_landmarks_3d"])
            training_logger.log_batch_loss("eye_loss", loss_eye.item(), partition_, batch_size)
            total_loss_weighted += loss_eye * torch.exp(-loss_weights[2])#* args.lambda3

        if args.gaze_tgt_loss:
            if args.dataset == "eyediap":
                loss_gaze_target = gaze_target_loss(results_["gaze_point_mid"].squeeze(1), data_["target_3d_crop"])
            else:
                loss_gaze_target = F.mse_loss(results_["gaze_point_mid"].squeeze(1), data_["target_3d_crop"])
            training_logger.log_batch_loss("gaze_tgt_loss", loss_gaze_target.item(), partition_, batch_size)
            total_loss_weighted += loss_gaze_target * torch.exp(-loss_weights[3])#* args.lambda4

        if args.gaze_div_loss:
            loss_gaze_div = gaze_divergence_loss(results_["gaze_point_dist"])
            training_logger.log_batch_loss("gaze_div_loss", loss_gaze_div.item(), partition_, batch_size)
            total_loss_weighted += loss_gaze_div * torch.exp(-loss_weights[4])#* args.lambda5

        face_gaze_pred = None
        if args.gaze_pose_loss:
            if args.dataset == "eyediap":
                loss_gaze_pose = gaze_pose_loss(results_["left_eye_rotation"], data_["left_eyeball_rotation_crop"],
                                                results_["right_eye_rotation"], data_["right_eyeball_rotation_crop"])
            else:
                # loss_gaze_pose = F.mse_loss(results_["face_gaze_az_el"], data_["face_gazes"])
                # Additional gaze vector loss.
                face_gaze_pred = results_["gaze_point_mid"].squeeze(1) - results_["face_centre"]
                face_gaze_pred_n = face_gaze_pred / torch.linalg.norm(face_gaze_pred, dim=1, keepdim=True)
                face_gaze_gt = data_["target_3d_crop"] - data_["gaze_origins"]
                face_gaze_gt_n = face_gaze_gt / torch.linalg.norm(face_gaze_gt, dim=1, keepdim=True)
                loss_gaze_pose = F.mse_loss(face_gaze_pred_n, face_gaze_gt_n)
                # training_logger.log_batch_loss("gaze_vector_loss", loss_gaze_vector.item(), partition_, batch_size)
                # total_loss_weighted += loss_gaze_vector * args.lambda6

            training_logger.log_batch_loss("gaze_pose_loss", loss_gaze_pose.item(), partition_, batch_size)
            total_loss_weighted += loss_gaze_pose * torch.exp(-loss_weights[5])#* args.lambda6
        # else:
        #     # TODO: look forward regulariser. Range from [-45, 45] for all azimuth and elevation. Not working.
        #     l_rot = results_["left_eye_rotation"]
        #     r_rot = results_["right_eye_rotation"]
        #     deg = 1.5708 / 2
        #     look_forward_reg = ((l_rot[l_rot > deg] - deg) ** 2).sum() + \
        #                        ((l_rot[l_rot < -deg] + deg) ** 2).sum() + \
        #                        ((r_rot[r_rot > deg] - deg) ** 2).sum() + \
        #                        ((r_rot[r_rot < deg] + deg) ** 2).sum()
        #     training_logger.log_batch_loss("look_forward_reg", look_forward_reg.item(), partition_, batch_size)
        #     total_loss_weighted += look_forward_reg * 10000.

        if args.parameters_regulariser:
            reg_shape_param = parameters_regulariser(results_["shape_parameters"])
            reg_albedo_param = parameters_regulariser(results_["albedo_parameters"])
            training_logger.log_batch_loss("shape_param_reg", reg_shape_param.item(), partition_, batch_size)
            training_logger.log_batch_loss("albedo_para_reg", reg_albedo_param.item(), partition_, batch_size)
            total_loss_weighted += reg_shape_param * args.lambda7
            total_loss_weighted += reg_albedo_param * args.lambda8

        training_logger.log_batch_loss("total_loss_weighted", total_loss_weighted.item(), partition_, batch_size)
        if args.dataset == "eyediap":
            early_stopping_criteria = gaze_degree_error(results_["l_eyeball_centre"].squeeze(1),
                                                        results_["r_eyeball_centre"].squeeze(1),
                                                        data_["left_eyeball_3d_crop"],
                                                        data_["right_eyeball_3d_crop"],
                                                        results_["gaze_point_mid"].squeeze(1),
                                                        data_["target_3d_crop"])
        elif face_gaze_pred is not None:
            early_stopping_criteria = \
                np.rad2deg(np.nan_to_num(np.arccos(
                    np.sum(
                        face_gaze_pred_n.detach().cpu().numpy() *
                        face_gaze_gt_n.detach().cpu().numpy(), axis=1)))).mean()
        else:
            early_stopping_criteria = 0.

        if early_stopping_criteria == 0:
            early_stopping_criteria = 361.  # Duo yi du re ai.

        training_logger.log_batch_loss("loss", early_stopping_criteria,
                                       partition_, batch_size)

        return total_loss_weighted

    logging.info("Start training...")
    full_start_time = time.time()
    frame_transform = Resize(224)
    l_eye_patch_transformation = Grayscale()
    r_eye_patch_transformation = Grayscale()
    for epoch in range(1, args.epochs + 1):
        logging.info("*** Epoch " + str(epoch) + " ***")
        epoch_start_time = time.time()

        """ Train.
        """
        gt_img_input, results = None, None
        for data in tqdm(train_loader):
            # Load forward information.
            gt_img = data["frames"].to(torch.float32) / 255.
            if args.dataset == "eyediap":
                camera_parameters = (data["cam_R"], data["cam_T"], data["cam_K"])
                warp_matrices = None
            else:
                camera_parameters = data["cam_intrinsics"]
                warp_matrices = data["warp_matrices"]

            # Preprocess images.
            if args.eye_patch:
                left_eye_img = data["left_eye_images"].to(torch.float32).permute(0, 3, 1, 2) / 255.
                right_eye_img = data["right_eye_images"].to(torch.float32).permute(0, 3, 1, 2) / 255.
                left_eye_img = l_eye_patch_transformation(left_eye_img)
                right_eye_img = r_eye_patch_transformation(right_eye_img)

            gt_img_input = frame_transform(gt_img.permute(0, 3, 1, 2))

            # Forward.
            optimiser.zero_grad()
            if args.eye_patch:
                results = model((gt_img_input, left_eye_img, right_eye_img), camera_parameters, warp_matrices)
            else:
                results = model(gt_img_input, camera_parameters, warp_matrices)

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
        for data in tqdm(validation_loader):
            # Load forward information.
            gt_img = data["frames"].to(torch.float32) / 255.
            if args.dataset == "eyediap":
                camera_parameters = (data["cam_R"], data["cam_T"], data["cam_K"])
                warp_matrices = None
            else:
                camera_parameters = data["cam_intrinsics"]
                warp_matrices = data["warp_matrices"]

            # Preprocess images.
            if args.eye_patch:
                left_eye_img = data["left_eye_images"].to(torch.float32).permute(0, 3, 1, 2) / 255.
                right_eye_img = data["right_eye_images"].to(torch.float32).permute(0, 3, 1, 2) / 255.
                left_eye_img = l_eye_patch_transformation(left_eye_img)
                right_eye_img = r_eye_patch_transformation(right_eye_img)

            gt_img_input = frame_transform(gt_img.permute(0, 3, 1, 2))

            # Forward.
            with torch.no_grad():
                if args.eye_patch:
                    results = model((gt_img_input, left_eye_img, right_eye_img), camera_parameters, warp_matrices)
                else:
                    results = model(gt_img_input, camera_parameters, warp_matrices)

                # Calculate losses.
                calculate_losses(data, results, "eval")

        if os.name == "nt" and "img" in results:
            j = 1
            gt_img_np = gt_img_input.permute(0, 2, 3, 1)[j].detach().cpu().numpy()
            result_img_np = results["img"][j].detach().cpu().numpy()

            result_img_np_none = np.all(result_img_np == [0., 1., 0.], axis=2)
            result_img_np[result_img_np_none] = gt_img_np[result_img_np_none]

            if args.dataset == "xgaze":
                lm_f3d = cam_to_img(data["face_landmarks_3d"], data["cam_intrinsics"])
                lm_f3d = perspective_transform(lm_f3d, data["warp_matrices"])

                lm_f3d_pred = cam_to_img(results["face_landmarks_3d"], data["cam_intrinsics"])
                lm_f3d_pred = perspective_transform(lm_f3d_pred, data["warp_matrices"])
                #
                # for idx, lm in enumerate(data["face_landmarks_crop"][j]):
                #     cv2.putText(gt_img_np, str(idx), (int(lm[0]), int(lm[1])), cv2.FONT_HERSHEY_PLAIN, 0.5, color=(1, 1, 0))
                # for idx, lm in enumerate(results["face_landmarks"][j]):
                #     cv2.putText(gt_img_np, str(idx), (int(lm[0]), int(lm[1])), cv2.FONT_HERSHEY_PLAIN, 0.5, color=(1, 0, 1))
                for idx, lm in enumerate(lm_f3d[j]):
                    cv2.putText(gt_img_np, str(idx), (int(lm[0]), int(lm[1])), cv2.FONT_HERSHEY_PLAIN, 0.5, color=(0, 1, 0))
                for idx, lm in enumerate(lm_f3d_pred[j]):
                    cv2.putText(gt_img_np, str(idx), (int(lm[0]), int(lm[1])), cv2.FONT_HERSHEY_PLAIN, 0.5, color=(0, 0, 1))

                fc = cam_to_img(results["face_centre"], data["cam_intrinsics"])
                fc = perspective_transform(fc, data["warp_matrices"])[j, 0]
                gc = cam_to_img(data["target_3d_crop"], data["cam_intrinsics"])
                gc = perspective_transform(gc, data["warp_matrices"])[j, 0]
                lc = cam_to_img(results["l_eyeball_centre"], data["cam_intrinsics"])
                lc = perspective_transform(lc, data["warp_matrices"])[j, 0]
                rc = cam_to_img(results["r_eyeball_centre"], data["cam_intrinsics"])
                rc = perspective_transform(rc, data["warp_matrices"])[j, 0]
                ft = cam_to_img(results["sb"], data["cam_intrinsics"])
                ft = perspective_transform(ft, data["warp_matrices"])[j, 0]
                ft2 = cam_to_img(results["sb2"], data["cam_intrinsics"])
                ft2 = perspective_transform(ft2, data["warp_matrices"])[j, 0]
                cv2.arrowedLine(gt_img_np,
                                (int(lc[0]), int(lc[1])),
                                (int(ft[0]), int(ft[1])),
                                (0, 0, 1), thickness=1)
                cv2.arrowedLine(gt_img_np,
                                (int(rc[0]), int(rc[1])),
                                (int(ft2[0]), int(ft2[1])),
                                (0, 1, 1), thickness=1)
                cv2.arrowedLine(gt_img_np,
                                (int(fc[0]), int(fc[1])),
                                (int(gc[0]), int(gc[1])),
                                (0, 1, 0), thickness=1)

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

    torch.save(model.state_dict(), log_name_dir + "model_final.pt")
    full_end_time = time.time()
    logging.info("Total training time: " + time_format(full_end_time - full_start_time))


def time_format(time_diff):
    hours, rem = divmod(time_diff, 3600)
    minutes, seconds = divmod(rem, 60)
    return "{:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds)


def logging_source_code(ae):
    # Logging file and model information.
    logging.debug(ae)
    with open(PROJECT_PATH + "train_autoencoder.py", "r", encoding="utf-8") as f:
        logging.debug("*** File content for train_autoencoder.py ***")
        logging.debug("".join(f.readlines()))


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

    parser.add_argument("--dataset", type=str, default="eyediap",
                        help="Dataset to train.")
    parser.add_argument("--network", type=str, default="ResNet18",
                        help="Backbone network.")
    parser.add_argument("--eye_patch", type=bool, default=False,
                        help="Backbone network.")

    """ Hyper-parameters.
    """
    parser.add_argument("-e", "--epochs", type=int, default=200, metavar="N",
                        help="Number of episode to train.")
    parser.add_argument("-b", "--batch_size", type=int, default=32, metavar="N",
                        help="Batch size for training.")
    parser.add_argument("--test_batch_size", type=int, default=32, metavar="N",
                        help="Batch size for evaluating.")

    # Loss function hyper-parameters.
    parser.add_argument("--lambda1", type=float, default=1.,
                        help="Lambda to balance loss function.")
    parser.add_argument("--lambda2", type=float, default=1.,
                        help="Lambda to balance loss functions.")
    parser.add_argument("--lambda3", type=float, default=100.,
                        help="Lambda to balance loss function.")
    parser.add_argument("--lambda4", type=float, default=50.,
                        help="Lambda to balance loss functions.")
    parser.add_argument("--lambda5", type=float, default=10.,
                        help="Lambda to balance loss functions.")
    parser.add_argument("--lambda6", type=float, default=10.,
                        help="Lambda to balance loss functions.")
    parser.add_argument("--lambda7", type=float, default=0.05,
                        help="Lambda to balance loss functions.")
    parser.add_argument("--lambda8", type=float, default=0.01,
                        help="Lambda to balance loss functions.")
    parser.add_argument("--lr", type=float, default=5e-4,
                        help="Learning rate.")
    parser.add_argument("--lr_scheduler", type=str, default=None,
                        help="Learning rate scheduler, choose from: (cos) for (CosineAnnealingLR), "
                             "(step) for (StepLR).")

    """ Ablation studies.
    """
    parser.add_argument("--pixel_loss", type=bool, default=True,
                        help="")
    parser.add_argument("--landmark_loss", type=bool, default=True,
                        help="")
    parser.add_argument("--eye_loss", type=bool, default=True,
                        help="")
    parser.add_argument("--gaze_tgt_loss", type=bool, default=True,
                        help="")
    parser.add_argument("--gaze_div_loss", type=bool, default=True,
                        help="")
    parser.add_argument("--gaze_pose_loss", type=bool, default=True,
                        help="")
    parser.add_argument("--parameters_regulariser", type=bool, default=True,
                        help="")

    """ Misc
    """
    parser.add_argument("--logging_level", type=str, default="INFO",
                        help="Logging level, choose from: (CRITICAL|ERROR|WARNING|INFO|DEBUG).")

    args = parser.parse_args()

    """ Insert argument override here. """
    # args.name = "v5_swin_cv115t16_lb"
    args.name = "v5_swin_xgaze_lb"
    args.epochs = 1500
    args.seed = 1
    args.lr = 5e-5
    args.lr_scheduler = None

    args.dataset = "xgaze"

    args.network = "Swin"
    # args.network = "ResNet18"
    args.eye_patch = False

    args.pix_loss = True
    args.landmark_loss = True
    args.eye_loss = True
    args.gaze_tgt_loss = True
    args.gaze_div_loss = True
    args.gaze_pose_loss = True

    args.lambda1 = 1.
    args.lambda2 = 0.5

    args.lambda3 *= 10.
    args.lambda4 *= 10.

    args.lambda7 *= 10.
    args.lambda8 *= 10.

    args.batch_size = 32
    args.test_batch_size = 256

    if args.dataset == "xgaze":
        args.lambda1 *= 10.
        args.lambda2 *= 1
        args.lambda3 *= 1e-3
        args.lambda4 *= 1e-3
        args.lambda5 *= 2
        args.lambda6 *= 100.
        args.lambda7 *= 500.
        # args.lambda8 *= 10.
        args.lambda8 *= 500.

    args.override = True

    # # Baseline.
    # args.pixel_loss = False
    # args.landmark_loss = False
    # args.eye_loss = False
    # args.gaze_tgt_loss = False
    # args.gaze_div_loss = False
    # args.gaze_pose_loss = True
    # args.parameters_regulariser = False

    # # Baseline_2.
    # args.pixel_loss = False
    # args.landmark_loss = False
    # args.gaze_pose_loss = False
    # args.parameters_regulariser = False

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

    training_logger = TrainingLogger(log_name_dir, args, True)
    train()

    """
    baseline: only eye rotation L2 loss.
    baseline_2: only eye position and rotation.
    v1: initial version.
    v2: added eyeball rotation correction regarding head pose.
    v3: fix bugs from v2.
    v4: v3 + nosteplr_lrby2_lmd1by10_lmd2d05_ltgteyex10_l7x10_nogal and new dataset without outliers.
    v5: +eye patch option for training. FIXED EYE POSE LOSS.
    """
