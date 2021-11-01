
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models.resnet import load_state_dict_from_url, BasicBlock, model_urls, Bottleneck, conv1x1

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


class ResNet2(nn.Module):

    def __init__(
            self,
            block,
            layers,
            num_classes: int = 1000,
            zero_init_residual: bool = False,
            groups: int = 1,
            width_per_group: int = 64,
            replace_stride_with_dilation=None,
            norm_layer=None
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(2, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(self, block, planes: int, blocks: int,
                    stride: int = 1, dilate: bool = False) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x):
        return self._forward_impl(x)


class EyePatchLayer(nn.Module):
    def __init__(self, img_features_channel):
        super().__init__()

        # self.conv1 = nn.Conv2d(2, 64, (3, 3))
        # self.conv2 = nn.Conv2d(2, 128, (3, 3))
        # self.conv3 = nn.Conv2d(2, 256, (3, 3))
        self.conv_net = ResNet2(BasicBlock, [2, 2, 2, 2], num_classes=32)
        self.fc_l = nn.Linear(img_features_channel + 32, 2)
        self.fc_r = nn.Linear(img_features_channel + 32, 2)

        nn.init.normal_(self.fc_l.weight, 0., 0.001)
        nn.init.zeros_(self.fc_l.bias)
        nn.init.normal_(self.fc_r.weight, 0., 0.001)
        nn.init.zeros_(self.fc_r.bias)

    def forward(self, img_features, l_img, r_img):
        x = self.conv_net(torch.cat([l_img, r_img], dim=1))
        in_feature = torch.cat([img_features, x], dim=1)
        return torch.cat([self.fc_l(in_feature), self.fc_r(in_feature)], dim=1)


class Encoder(nn.Module):
    def __init__(self, out_channel, network="ResNet18", loss_weights=False):
        super().__init__()
        self.network = network
        if loss_weights:
            self.sigmas = nn.Parameter(torch.ones(6))

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


class EncoderEyePatch(Encoder):
    def __init__(self, out_channel, network="ResNet18"):
        super(EncoderEyePatch, self).__init__(out_channel - 4, network)

        self.eye_path_layer = EyePatchLayer(768)

    def forward(self, x):
        img, eye_img_l, eye_img_r = x
        if self.network == "Swin":
            img_features = self.net.forward_features(img)
            res_0 = self.net.head(img_features)

            res_1 = self.eye_path_layer(img_features, eye_img_l, eye_img_r)

            return torch.cat([res_0, res_1], dim=1)
        else:
            raise NotImplemented
