import sys

sys.path.append("../../")
import torch.nn as nn
import torch
from .shufflenetv2 import ShuffleNetV2
from .resnet import ResNet, BasicBlock
from torch.nn.modules.batchnorm import _BatchNorm
import torchvision
import torch.nn.functional as F
import numpy as np
import cv2

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)


class Resnet18(nn.Module):
    def __init__(self, original_resnet, pool_type='maxpool', input_channel=3, with_fc=False, fc_in=512, fc_out=512):
        super(Resnet18, self).__init__()
        self.pool_type = pool_type
        self.input_channel = input_channel
        self.with_fc = with_fc

        # customize first convolution layer to handle different number of channels for images and spectrograms
        self.conv1 = nn.Conv2d(self.input_channel, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.conv2 = nn.Conv2d(self.input_channel*2, 64, kernel_size=7, stride=2, padding=3, bias=False)
        layers = [self.conv1]
        layers1 = [self.conv2]
        layers.extend(list(original_resnet.children())[1:-2])
        layers1.extend(list(original_resnet.children())[1:-2])
        self.feature_extraction = nn.Sequential(*layers)  # features before pooling
        self.feature_extraction1 = nn.Sequential(*layers1)

        if with_fc:
            self.fc = nn.Linear(fc_in, fc_out)
            self.fc.apply(weights_init)

    def forward(self, x):
        (B, T, C, H, W) = x.size()
        frame =  x[:, 0, :, :, :]
        # features = np.ones((B, 512, 1, 1),dtype=np.float32)
        # features = torch.from_numpy(features).cuda()

        flow1 = np.zeros((B, 1, C, H, W, 2))
        for j in range(C):
            frame1 = x[0, 0, j].cpu().detach().numpy()
            frame2 = x[0, 1, j].cpu().detach().numpy()
            flow1[:, :, j, :,:] = cv2.calcOpticalFlowFarneback(
                prev=frame1, next=frame2,
                flow=None, pyr_scale=0.5,
                levels=3, winsize=15,
                iterations=3, poly_n=5,
                poly_sigma=1.2, flags=0
            )
        dim_tuple = (B, C * 2, H, W)
        flow1 = flow1.reshape(dim_tuple)
        flow1 = torch.from_numpy(flow1).float().to('cuda')
        features = self.feature_extraction1(flow1)  # 将特征沿时间维度堆叠


        features = features.contiguous()  # 49,512,7,7



        x = self.feature_extraction(frame)

        if self.pool_type == 'avgpool':
            x = F.adaptive_avg_pool2d(x, 1)
        elif self.pool_type == 'maxpool':
            x = F.adaptive_max_pool2d(x, 1)
            # features = F.adaptive_max_pool2d(features, 1)
        else:
            return x

        if self.with_fc:
            x = x.view(x.size(0), -1)
            x = self.fc(x)
            return x.view(x.size(0), -1, 1, 1)
        else:
            return x.view(x.size(0), -1, 1, 1), features

    def forward_multiframe(self, x, pool=True):
        (B, T, C, H, W) = x.size()
        flow = np.zeros((B, T - 1, C, H, W, 2))
        for i in range(B):
            for t in range(T - 1):
                for j in range(C):
                    frame1 = x[i, t, j].cpu().detach().numpy()
                    frame2 = x[i, t + 1, j].cpu().detach().numpy()
                    flow[i, t, j] = cv2.calcOpticalFlowFarneback(
                        prev=frame1, next=frame2,
                        flow=None, pyr_scale=0.5,
                        levels=3, winsize=15,
                        iterations=3, poly_n=5,
                        poly_sigma=1.2, flags=0
                    )
        flow = torch.from_numpy(flow).float().cuda() #1，49，3，224，224，2
        flow1 = flow.view(B * (T-1) , C*2, H, W)
        features = self.feature_extraction1(flow1) # 将特征沿时间维度堆叠

        # features = torch.stack(features, dim=1) #1,49,512,7,7
        features = features.contiguous() #49,512,7,7
        (_,  C1, H1, W1) = features.size()
        features = features.view(B, T-1, C1, H1, W1)
        features = features.permute(0, 2, 1, 3, 4)

        x = x.contiguous()

        x = x.view(B * T, C, H, W) #50,3,224,224
        x = self.feature_extraction(x) #50,512,7,7

        (_, C, H, W) = x.size()
        x = x.view(B, T, C, H, W)
        x = x.permute(0, 2, 1, 3, 4)

        if not pool:
            return x

        if self.pool_type == 'avgpool':
            x = F.adaptive_avg_pool3d(x, 1)
            features = F.adaptive_avg_pool3d(features, 1)
        elif self.pool_type == 'maxpool':
            x = F.adaptive_max_pool3d(x, 1)
            features = F.adaptive_avg_pool3d(features, 1)
        # if self.with_fc:
        #     x = x.view(x.size(0), -1)
        #     x = self.fc(x)
        #     return x.view(x.size(0), -1, 1, 1)
        # else:
        #     return x.view(x.size(0), -1, 1, 1)


        return x, features


class build_facial(nn.Module):
    def __init__(self,
                pool_type='maxpool', input_channel=3, fc_out=512, with_fc=False, weights=''):
        super(build_facial, self).__init__()
        pretrained = False
        self.original_resnet = torchvision.models.resnet18(pretrained)
        self.net = Resnet18(self.original_resnet, pool_type=pool_type, with_fc=with_fc, fc_in=512, fc_out=fc_out)

        self.weights = weights
    def forward(self, x):
        pretrained_state = torch.load(self.weights)
        model_state = self.net.state_dict()
        pretrained_state = {k: v for k, v in pretrained_state.items() if
                            k in model_state and v.size() == model_state[k].size()}
        model_state.update(pretrained_state)
        self.net.load_state_dict(model_state)
        x,features =self.net(x)
        x = F.normalize(x, p=2, dim=1)
        # features = F.normalize(features, p=2, dim=1)
        return x, features