
from torch import device, cuda
from torch.backends.cudnn import benchmark
import os


""" PyTorch related """
device = device("cuda" if cuda.is_available() else "cpu")
benchmark = True  # Speed up for static CNNs.


""" Paths """
PROJECT_PATH = os.path.abspath(os.path.join(os.path.realpath(__file__), os.pardir)) + "/"

DATASETS_PATH = PROJECT_PATH + "datasets/"

EYEDIAP_PATH = "Z:/research/datasets/EYEDIAP/"
FLAME_PATH = "Z:/research/datasets/FLAME/"
