
import torch
from torch.utils.data import Dataset, DataLoader

import numpy as np

from constants import *


class EYEDIAP(Dataset):
    def __init__(self, partition="train", eval_subject=16, exp_id="A", exp_type="FT", head_movement="S",
                 load_device="cuda"):
        self.load_device = load_device
        self.dataset = None

        for i in range(1, 17):
            for hdm in head_movement:
                if partition == "train" and i == eval_subject:
                    continue
                if partition == "test" and i != eval_subject:
                    continue
                if not os.path.exists(DATASETS_PATH + "eyediap/" + str(i) +
                                      "_" + exp_id + "_" + exp_type + "_" + hdm + ".npy"):
                    continue

                data = np.load(DATASETS_PATH + "eyediap/" + str(i) +
                               "_" + exp_id + "_" + exp_type + "_" + hdm + ".npy",
                               allow_pickle=True)[()]
                if self.dataset is None:
                    self.dataset = data
                else:
                    for key, value in data.items():
                        self.dataset[key] = np.concatenate([self.dataset[key], value])

        for key, value in self.dataset.items():
            if key in ["subject_id", "frames", "frame_id"]:
                d_type = torch.uint8
            elif key == "face_box_tl":
                d_type = torch.long
            else:
                d_type = torch.float32
            if load_device == "cpu":
                self.dataset[key] = torch.from_numpy(value).to(d_type)
            else:
                self.dataset[key] = torch.from_numpy(value).to(device, d_type)

    def __getitem__(self, index):
        ret_dict = {}
        for key, value in self.dataset.items():
            if self.load_device == "cpu":
                ret_dict[key] = value[index].to(device)
            else:
                ret_dict[key] = value[index]

        return ret_dict

    def __len__(self):
        return self.dataset["frames"].shape[0]


if __name__ == '__main__':
    dl = DataLoader(EYEDIAP(), batch_size=32, shuffle=True, num_workers=0)
    for k in dl:
        break
    print()
