
import torch.nn as nn
import torchvision.models as models
from torchvision.models.resnet import load_state_dict_from_url, BasicBlock, model_urls, Bottleneck

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
    def __init__(self, out_channel, dropout=0.):
        super().__init__()

        if dropout == 0.:
            # ResNet18.
            self.net = models.ResNet(BasicBlock, [2, 2, 2, 2], num_classes=out_channel)
            # # ResNet50.
            # self.net = models.ResNet(Bottleneck, [3, 4, 6, 3], num_classes=out_channel)
        else:
            self.net = models.ResNet(BasicBlock, [2, 2, 2, 2], num_classes=1024)

        pretrain_state_dict = \
            load_state_dict_from_url(model_urls["resnet18"], model_dir="./", progress=True)
        # pretrain_state_dict = \
        #     load_state_dict_from_url(model_urls["resnet50"], model_dir="./", progress=True)
        pretrain_state_dict.pop("fc.weight")
        pretrain_state_dict.pop("fc.bias")
        self.net.load_state_dict(pretrain_state_dict, strict=False)

        nn.init.normal_(self.net.fc.weight, 0., 0.001)
        nn.init.zeros_(self.net.fc.bias)

        if dropout > 0.:
            dropout_layer = DropoutLayer(1024, out_channel, dropout)
            self.net = nn.Sequential(self.net, dropout_layer)

    def forward(self, x):
        return self.net(x)
