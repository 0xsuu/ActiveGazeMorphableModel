
import logging
import json

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from constants import *


def log_table(dict_list, col_list=None):
    if not col_list:
        col_list = list(dict_list[0].keys() if dict_list else [])
    my_list = [col_list]
    for item in dict_list:
        my_list.append([str(item[col] if item[col] is not None else '') for col in col_list])
    colSize = [max(map(len, col)) for col in zip(*my_list)]
    format_str = ' | '.join(["{{:<{}}}".format(i) for i in colSize])
    my_list.insert(1, ['-' * i for i in colSize])
    for item in my_list:
        logging.info(format_str.format(*item))


class TrainingLogger(object):
    def __init__(self, log_name_dir, args, save=True):
        self.log_name_dir = log_name_dir
        self.args = args
        self.save = save

        with open(self.log_name_dir + "config.json", "w") as f:
            json.dump(vars(args), f, indent=4)

        # Initialise running config.
        self.running_metrics = {"best_eval": np.inf, "epochs_trained": 0}

        # Initialise in-epoch training.
        self.epoch_logs = {}
        self.total_size = {}

        # Initialise Tensorboard.
        self.tfb_writer = SummaryWriter(log_dir=LOGS_PATH + "tf_board/" + args.name)
        self.hparam_dict = vars(args)
        del self.hparam_dict["logging_level"]

    def log_batch_loss(self, name, value, partition, size):
        key = name + "/" + partition
        if key in self.epoch_logs:
            self.epoch_logs[key] += value * size  # Assume value is averaged regarding the batch.
            self.total_size[key] += size
        else:
            self.epoch_logs[key] = value * size
            self.total_size[key] = size

    def log_epoch(self, log_dict):
        epoch = log_dict["epoch"]
        self.running_metrics["epochs_trained"] = epoch

        # Log Console and Tensorboard.
        train_dict = {"partition": "train"}
        eval_dict = {"partition": "eval"}
        for key in self.epoch_logs:
            self.epoch_logs[key] /= self.total_size[key]
            if "train" in key:
                train_dict[key.split("/")[0]] = "%.7f" % self.epoch_logs[key]
            if "eval" in key:
                eval_dict[key.split("/")[0]] = "%.7f" % self.epoch_logs[key]
        log_table([train_dict, eval_dict])

        scalars = self.epoch_logs
        for key, value in scalars.items():
            self.tfb_writer.add_scalar(key, value, epoch)

        if self.save:
            # Log weights.
            if self.epoch_logs["loss/eval"] <= self.running_metrics["best_eval"]:
                self.running_metrics["best_eval"] = self.epoch_logs["loss/eval"]
                torch.save(log_dict["model_weights"], self.log_name_dir + "model_best.pt")
                # torch.save(log_dict["optimiser_weights"], self.log_name_dir + "optimiser_best.pt")
                logging.info("Best model weights saved.")

                self.running_metrics.update(self.epoch_logs)
                self.tfb_writer.add_hparams(self.hparam_dict, self.running_metrics, run_name="Best eval")

            if epoch % 10 == 0:
                torch.save(log_dict["model_weights"], self.log_name_dir + "model_" + str(epoch) + ".pt")
                logging.info("Model weights saved.")

        # Reset for next epoch.
        self.epoch_logs.clear()
        self.total_size.clear()
