import sys

sys.path.append("../../")
import torch.nn as nn
import torch
from .shufflenetv2 import ShuffleNetV2
from .resnet import ResNet, BasicBlock
from torch.nn.modules.batchnorm import _BatchNorm
import torch.nn.functional as F
import numpy as np
import cv2

from .tcn import TemporalConvNet
# -- auxiliary functions


def threeD_to_2D_tensor(x):
    n_batch, n_channels, s_time, sx, sy = x.shape
    x = x.transpose(1, 2)
    return x.reshape(n_batch * s_time, n_channels, sx, sy)


def _average_batch(x, lengths, B):
    return torch.stack([torch.mean(x[index][:, 0:i], 1) for index, i in enumerate(lengths)], 0)


class VideoModel(nn.Module):
    def __init__(self,
                hidden_dim=256,
                backbone_type="resnet",
                relu_type="prelu",
                width_mult=1.0,
                pretrain=None):
        super(VideoModel, self).__init__()
        self.backbone_type = backbone_type
        if self.backbone_type == "resnet":
            self.frontend_nout = 64
            self.backend_out = 512
            self.trunk = ResNet(BasicBlock, [2, 2, 2 ,2], relu_type=relu_type)
        elif self.backbone_type == "shufflenet":
            assert width_mult in [0.5, 1.0, 1.5, 2.0], "Width multiplier not correct"
            shufflenet = ShuffleNetV2(input_size=96, width_mult=width_mult)
            self.trunk = nn.Sequential(
                shufflenet.features, shufflenet.conv_last, shufflenet.globalpool
            )
            self.frontend_nout = 24
            self.backend_out = 1024 if width_mult != 2.0 else 2048
            self.stage_out_channels = shufflenet.stage_out_channels[-1]

        frontend_relu = (
            nn.PReLU(num_parameters=self.frontend_nout) if relu_type == "prelu" else nn.ReLU()
        )
        self.frontend3D = nn.Sequential(
            nn.Conv3d(
                1,
                self.frontend_nout,
                kernel_size=(5, 7, 7),
                stride=(1, 2, 2),
                padding=(2, 3, 3),
                bias=False,
            ),
            nn.BatchNorm3d(self.frontend_nout),
            frontend_relu,
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),
        )
        self.pretrain = pretrain
        # self.fc = nn.Linear(512 , 256)
        # self.bnfc = nn.BatchNorm1d(130)
        # self.conv_1D = nn.Conv1d(in_channels=1,out_channels=1,kernel_size=2,stride=1,padding=0)
        # self.prelu = nn.PReLU()
        # self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        # self.linear = nn.Linear(in_features=512, out_features=65)
        if pretrain:
            self.init_from(pretrain)

        self.linear = nn.Linear(in_features=512, out_features=65)
        self.pool = nn.AdaptiveAvgPool2d((50,1))

    def forward(self, mouth):
        mouth = mouth.cuda()
        B, C, T, H, W = mouth.size()#2,2,50,88,88


        x1 = self.frontend3D(mouth[:, :1, :, :, :])  #[:, :1, :, :, :]
        Tnew = x1.shape[2]  # outpu should be B x C2 x Tnew x H x W
        x1 = threeD_to_2D_tensor(x1)
        x1 = self.trunk(x1)

        x2 = self.frontend3D(mouth[:, 1:, :, :, :])
        x2 = threeD_to_2D_tensor(x2)
        x2 = self.trunk(x2)

        # if self.backbone_type == "shufflenet":
        #     x = x.view(-1, self.stage_out_channels)
        x1 = x1.view(B, Tnew, x1.size(1))
        # x1 = self.linear(x1)
        x2 = x2.view(B, Tnew, x2.size(1))
        # x2 = self.linear(x2)
        x = torch.stack([x1, x2], dim=1)  #3,2,50,65 50是时间，65是特征
        x_loss = self.pool(x)

        return x, x_loss

        # return upsampled_tensor

    def init_from(self, path):
        pretrained_dict = torch.load(path, map_location="cpu")["model_state_dict"]
        update_frcnn_parameter(self, pretrained_dict)

    def train(self, mode=True):
        super().train(mode)
        if mode:    # freeze BN stats
            for m in self.modules():
                if isinstance(m, _BatchNorm):
                    m.eval()


def check_parameters(net):
    """
    Returns module parameters. Mb
    """
    parameters = sum(param.numel() for param in net.parameters())
    return parameters / 10 ** 6


def update_frcnn_parameter(model, pretrained_dict):
    model_dict = model.state_dict()
    update_dict = {}
    for k, v in pretrained_dict.items():
        if "tcn" in k:
            pass
        else:
            update_dict[k] = v
    model_dict.update(update_dict)
    model.load_state_dict(model_dict)
    for p in model.parameters():
        p.requires_grad = False
    return model


if __name__ == "__main__":
    frames = torch.randn(1, 1, 50, 96, 96)
