
import torch.nn as nn
import torchvision.models as models
from torchvision.models.resnet import load_state_dict_from_url, BasicBlock, model_urls, Bottleneck

from constants import *


class Encoder(nn.Module):
    def __init__(self, out_channel):
        super().__init__()

        # # ResNet18.
        # self.net = models.ResNet(BasicBlock, [2, 2, 2, 2], num_classes=out_channel).to(device)
        # pretrain_state_dict = \
        #     load_state_dict_from_url(model_urls["resnet18"], model_dir="autoencoder/", progress=True)
        # ResNet50.
        self.net = models.ResNet(Bottleneck, [3, 4, 6, 3], num_classes=out_channel).to(device)
        pretrain_state_dict = load_state_dict_from_url(model_urls["resnet50"], model_dir="autoencoder/", progress=True)

        pretrain_state_dict.pop("fc.weight")
        pretrain_state_dict.pop("fc.bias")
        self.net.load_state_dict(pretrain_state_dict, strict=False)
        nn.init.normal_(self.net.fc.weight, 0., 0.001)
        nn.init.zeros_(self.net.fc.bias)

    def forward(self, x):
        return self.net(x)
