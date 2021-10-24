
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models.resnet import load_state_dict_from_url, BasicBlock, model_urls, Bottleneck

from autoencoder.swin_transformer import SwinTransformer
from constants import *


class DropoutLayer(nn.Module):
    def __init__(self, in_channel, out_channel, dropout):
        super(DropoutLayer, self).__init__()
        self.pre_dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(in_channel, out_channel)

        nn.init.normal_(self.fc.weight, 0., 0.001)
        nn.init.zeros_(self.fc.bias)

    def forward(self, x):
        return self.fc(self.pre_dropout(x))


class Encoder(nn.Module):
    def __init__(self, out_channel, network="ResNet18"):
        super().__init__()

        if network == "ResNet18":
            # ResNet18.
            self.net = models.ResNet(BasicBlock, [2, 2, 2, 2], num_classes=out_channel)
            nn.init.normal_(self.net.fc.weight, 0., 0.001)
            nn.init.zeros_(self.net.fc.bias)
            pretrain_state_dict = \
                load_state_dict_from_url(model_urls["resnet18"], model_dir="./", progress=True)
            pretrain_state_dict.pop("fc.weight")
            pretrain_state_dict.pop("fc.bias")
            self.net.load_state_dict(pretrain_state_dict, strict=False)
            nn.init.normal_(self.net.fc.weight, 0., 0.001)
            nn.init.zeros_(self.net.fc.bias)
        elif network == "Swin":
            # Swin Transformer.
            self.net = SwinTransformer(
                224, num_classes=out_channel, embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24],
                window_size=7, drop_path_rate=0.2)
            pretrain_state_dict = torch.load(PROJECT_PATH + "autoencoder/swin_tiny_patch4_window7_224.pth")["model"]
            pretrain_state_dict.pop("head.weight")
            pretrain_state_dict.pop("head.bias")
            self.net.load_state_dict(pretrain_state_dict, strict=False)
            nn.init.normal_(self.net.head.weight, 0., 0.001)
            nn.init.zeros_(self.net.head.bias)
        else:
            raise ValueError("Unknown backbone network type.")

    def forward(self, x):
        return self.net(x)
