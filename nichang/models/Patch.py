import os
os.environ['PATH'] += '/root/data1/anaconda/envs/nichang/bin/ninja'
import math
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple
import sys

sys.path.append('/root/data1/LZR/CTCNet-main/nichang/layers')
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torch.nn.parameter import Parameter
# from sru import SRU
import numpy as np
# from espnet2.enh.decoder.stft_decoder import STFTDecoder
# from espnet2.enh.encoder.stft_encoder import STFTEncoder
# from espnet2.enh.layers.complex_utils import new_complex_like
# from espnet2.enh.separator.abs_separator import AbsSeparator
# from espnet2.torch_utils.get_layer_from_string import get_layer
from packaging.version import parse as V
from torch_complex.tensor import ComplexTensor
from ..layers.STFT import STFTDecoder,STFTEncoder
from ..layers.conv2d_cplx import ComplexConv2d_Encoder,ComplexConv2d_Decoder,ComplexGRU,ComplexLSTM,ComplexConvTranspose1d
from complexPyTorch.complexLayers import ComplexConv2d
from ..layers.sparse_attention import SparseAttention
from ..layers.Mamba import ExBimamba,Mamba1
from ..layers.resnet import ResNet, BasicBlock
from ..layers.lstm_cell import MultiModalLSTMCell
from ..layers.CBAM import BasicBlock
from ..layers.triple_attention import TripletAttention
from ..layers.tcn import TemporalConvNet,MultiscaleMultibranchTCN,TCN
# from complexPyTorch.complexLayers import C
# import cvnn.layers as cvnn_layers
# from espnet2.layers.stft import Stft
# from espnet2.enh.layers.complex_utils import is_torch_complex_tensor
from einops import rearrange
import difflib
from sklearn.decomposition import PCA
import librosa
# from ..layers.gla import GatedLinearAttention
from memory_profiler import profile
import gc
# from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
# device = torch.device('cuda'if torch.cuda.is_available() else'cpu')

EPSILON = torch.finfo(torch.float32).eps
is_torch_1_10_plus = V(torch.__version__) >= V("1.10.0")
is_torch_1_9_plus = V(torch.__version__) >= V("1.9.0")
def to_complex(c):
    # Convert to torch native complex
    if isinstance(c, ComplexTensor):
        c = c.real + 1j * c.imag
        return c
    elif torch.is_complex(c):
        return c
    else:
        return torch.view_as_complex(c)
from ..layers.patch_embed import ConvPatchEmbed, DeformablePatchTransformer
from ..layers.mamba_layer import PlainMambaLayer
from abc import ABC, abstractmethod
tcn_options = {'num_layers': 3,
                       'kernel_size': 3,
                       'dropout': 0,
                       'dwpw': False,
                       "tcn_width_mult": 2,
                       "width_mult": 1.0}
def is_torch_complex_tensor(c):
    return not isinstance(c, ComplexTensor) and torch.is_complex(c)

def get_layer(l_name, library=torch.nn):
    """Return layer object handler from library e.g. from torch.nn

    E.g. if l_name=="elu", returns torch.nn.ELU.

    Args:
        l_name (string): Case insensitive name for layer in library (e.g. .'elu').
        library (module): Name of library/module where to search for object handler
        with l_name e.g. "torch.nn".

    Returns:
        layer_handler (object): handler for the requested layer e.g. (torch.nn.ELU)

    """

    all_torch_layers = [x for x in dir(torch.nn)]
    match = [x for x in all_torch_layers if l_name.lower() == x.lower()]
    if len(match) == 0:
        close_matches = difflib.get_close_matches(
            l_name, [x.lower() for x in all_torch_layers]
        )
        raise NotImplementedError(
            "Layer with name {} not found in {}.\n Closest matches: {}".format(
                l_name, str(library), close_matches
            )
        )
    elif len(match) > 1:
        close_matches = difflib.get_close_matches(
            l_name, [x.lower() for x in all_torch_layers]
        )
        raise NotImplementedError(
            "Multiple matchs for layer with name {} not found in {}.\n "
            "All matches: {}".format(l_name, str(library), close_matches)
        )
    else:
        # valid
        layer_handler = getattr(library, match[0])
        return layer_handler
def new_complex_like(ref,real_imag):
    if isinstance(ref, ComplexTensor):
        return ComplexTensor(*real_imag)
    elif is_torch_complex_tensor(ref):
        return torch.complex(*real_imag)
    else:
        raise ValueError(
            "Please update your PyTorch version to 1.9+ for complex support."
        )
params = {
    "d_model": 128,
    "d_state": 64,
    "expand": 2,
    "dt_rank": "auto",
    "dt_min": 0.001,
    "dt_max": 0.1,
    "dt_init": "random",
    "dt_scale": 1.0,
    "dt_init_floor": 1e-4,
    "conv_size": 7,
    "conv_bias": True,
    "bias": False,
    "init_layer_scale": None,
    "default_hw_shape": None
}

norm_cfg = {
    "type": "LayerNorm",
    "layer_args": {
        # Add any specific arguments needed for LayerNorm here
        "eps": 1e-5,
        "elementwise_affine": True
    },
    "requires_grad": True  # Set to False if you don't want to update gradients
}

from mamba_ssm.modules.mamba_simple import Mamba


class SRU(nn.Module):
    def __init__(self,
                 oup_channels: int,
                 group_num: int = 16,
                 gate_treshold: float = 0.5,
                 torch_gn: bool = True
                 ):
        super().__init__()

        self.gn = nn.GroupNorm(num_channels=oup_channels, num_groups=group_num) if torch_gn else GroupBatchnorm2d(
            c_num=oup_channels, group_num=group_num)
        self.gate_treshold = gate_treshold
        self.sigomid = nn.Sigmoid()

    def forward(self, x):
        gn_x = self.gn(x)
        w_gamma = self.gn.weight / sum(self.gn.weight)
        w_gamma = w_gamma.view(1, -1, 1, 1)
        reweigts = self.sigomid(gn_x * w_gamma)
        # Gate
        w1 = torch.where(reweigts > self.gate_treshold, torch.ones_like(reweigts), reweigts)  # 大于门限值的设为1，否则保留原值
        w2 = torch.where(reweigts > self.gate_treshold, torch.zeros_like(reweigts), reweigts)  # 大于门限值的设为0，否则保留原值
        x_1 = w1 * x
        x_2 = w2 * x
        y = self.reconstruct(x_1, x_2)
        return y

    def reconstruct(self, x_1, x_2):
        x_11, x_12 = torch.split(x_1, x_1.size(1) // 2, dim=1)
        x_21, x_22 = torch.split(x_2, x_2.size(1) // 2, dim=1)
        return torch.cat([x_11 + x_22, x_12 + x_21], dim=1)
class CRU(nn.Module):
    '''
    alpha: 0<alpha<1
    '''
    def __init__(self,
                 op_channel:int,
                 alpha: float = 1 / 2,
                 squeeze_radio:int = 2 ,
                 group_size:int = 2,
                 group_kernel_size:int = 3,
                 ):
        super().__init__()

        self.up_channel     = up_channel   =   int(alpha*op_channel)
        self.low_channel    = low_channel  =   op_channel-up_channel
        self.squeeze1       = nn.Conv2d(up_channel,up_channel//squeeze_radio,kernel_size=1,bias=False)
        self.squeeze2       = nn.Conv2d(low_channel,low_channel//squeeze_radio,kernel_size=1,bias=False)
        #up
        self.GWC            = nn.Conv2d(up_channel//squeeze_radio, op_channel,kernel_size=group_kernel_size, stride=1,padding=group_kernel_size//2, groups = group_size)
        self.PWC1           = nn.Conv2d(up_channel//squeeze_radio, op_channel,kernel_size=1, bias=False)
        #low
        self.PWC2           = nn.Conv2d(low_channel//squeeze_radio, op_channel-low_channel//squeeze_radio,kernel_size=1, bias=False)
        self.advavg         = nn.AdaptiveAvgPool2d(1)

    def forward(self,x):
        # Split
        up,low  = torch.split(x,[self.up_channel,self.low_channel],dim=1)
        up,low  = self.squeeze1(up),self.squeeze2(low)
        # Transform
        Y1      = self.GWC(up) + self.PWC1(up)
        Y2      = torch.cat( [self.PWC2(low), low], dim= 1 )
        # Fuse
        out     = torch.cat( [Y1,Y2], dim= 1 )
        out     = F.softmax( self.advavg(out), dim=1 ) * out
        out1,out2 = torch.split(out,out.size(1)//2,dim=1)
        return out1+out2


class GatedMultimodalLayer(nn.Module):
    """ Gated Multimodal Layer based on 'Gated multimodal networks, Arevalo1 et al.' (https://arxiv.org/abs/1702.01992) """

    def __init__(self, size_in1, size_in2, size_out):
        super(GatedMultimodalLayer, self).__init__()
        self.size_in1, self.size_in2, self.size_out = size_in1, size_in2, size_out

        self.hidden1 = nn.Linear(size_in1, size_out, bias=False)
        self.hidden2 = nn.Linear(size_in2, size_out, bias=False)
        self.hidden_sigmoid = nn.Linear(size_out * 2, 1, bias=False)

        # Channel transformation parameters (GCT-style scale and bias)
        self.scale1 = nn.Parameter(torch.ones(size_out))  # Scale for modality 1
        self.bias1 = nn.Parameter(torch.zeros(size_out))  # Bias for modality 1
        self.scale2 = nn.Parameter(torch.ones(size_out))  # Scale for modality 2
        self.bias2 = nn.Parameter(torch.zeros(size_out))  # Bias for modality 2

        # Activation functions
        self.tanh_f = nn.Tanh()
        self.tanh_f2 = nn.Tanh()
        self.sigmoid_f = nn.Sigmoid()

    def forward(self, x1, x2):
        h1 = self.tanh_f(self.hidden1(x1))
        h2 = self.tanh_f2(self.hidden2(x2))

        # Apply channel transformation (GCT style) to each modality
        h1_transformed = self.scale1 * h1 + self.bias1
        h2_transformed = self.scale2 * h2 + self.bias2

        x = torch.cat((x1, x2), dim=-1)
        z = self.sigmoid_f(self.hidden_sigmoid(x))
        return z * h1_transformed + (1 - z) * h2_transformed


from torch_geometric.nn import GCNConv  # GCNConv 是图卷积网络的实现
from torch_geometric.data import Data


class GatedMultimodalLayerGNN(nn.Module):
    """ GMU with GCN to capture complex nonlinear relationships between consistency and complementary modalities """

    def __init__(self, size_in1, size_in2, size_out, num_nodes):
        super(GatedMultimodalLayerGNN, self).__init__()
        self.size_in1, self.size_in2, self.size_out = size_in1, size_in2, size_out
        self.num_nodes = num_nodes  # 节点数，这里是模态数量，2个模态

        # 一致性特征与互补性特征的线性变换
        self.hidden1 = nn.Linear(size_in1, size_out, bias=False)  # 一致性特征
        self.hidden2 = nn.Linear(size_in2, size_out, bias=False)  # 互补性特征

        # GCN用于学习模态间的非线性关系
        self.gcn1 = GCNConv(size_in1, size_out)  # 第一个模态的图卷积
        self.gcn2 = GCNConv(size_out, size_out)  # 第二个模态的图卷积

        # 门控机制的sigmoid层，用于控制模态的加权
        self.hidden_sigmoid = nn.Linear(size_out * 2, 1, bias=False)

        # 激活函数
        self.tanh_f = nn.Tanh()
        self.sigmoid_f = nn.Sigmoid()

    def forward(self, audio,video,x1, x2):
        """
        x1: 互补性特征1 (consistency features)
        x2: 互补性特征2 (complementary features)
        edge_index: 模态间的图结构
        """
        # 对一致性特征应用非线性变换
        concat = torch.cat([audio,video,x1, x2], dim=-1)

        # 定义图结构（edge_index），即哪些节点之间有连接
        edge_index = torch.tensor([
            [0, 1], [1, 0],  # 音频和视频连接
            [0, 2], [2, 0],  # 音频和一致性特征连接
            [0, 3], [3, 0],  # 音频和互补性特征连接
            [1, 2], [2, 1],  # 视频和一致性特征连接
            [1, 3], [3, 1],  # 视频和互补性特征连接
            [2, 3], [3, 2]  # 一致性和互补性特征连接
        ], dtype=torch.long).t().contiguous().cuda()

        # 创建图数据对象
        data = Data(x=concat, edge_index=edge_index).cuda()

        x, edge_index = data.x, data.edge_index
        x = self.gcn1(x, edge_index)
        x = F.relu(x)
        x = self.gcn2(x, edge_index)

        # h1 = self.tanh_f(self.hidden1(x1))
        # # 对互补性特征应用非线性变换
        # h2 = self.tanh_f(self.hidden2(x2))
        #
        # # 将两个模态的特征连接起来，用于门控机制
        # combined = torch.cat((h1, h2), dim=1)
        #
        # # 使用GCN捕捉模态间的非线性关系
        # h1_gcn = self.gcn1(h1, edge_index)  # 对一致性特征进行图卷积
        # h2_gcn = self.gcn2(h2, edge_index)  # 对互补性特征进行图卷积
        #
        # # 门控机制：通过sigmoid控制一致性和互补性特征的加权
        # z = self.sigmoid_f(self.hidden_sigmoid(combined))
        #
        # # 融合特征，结合GCN处理后的模态特征
        # out = z* h1_gcn + (1 - z) * h2_gcn
        return x


class GridNetBlock(nn.Module):
    def __getitem__(self, key):
        return getattr(self, key)

    def __init__(
            self,
            emb_dim,
            emb_ks,
            emb_hs,
            n_freqs,
            hidden_channels,
            n_head=4,
            approx_qk_dim=512,
            activation="prelu",
            eps=1e-5,
            n_layers=None,
    ):
        super(GridNetBlock,self).__init__()

        in_channels = emb_dim * emb_ks
        self.conv2d = nn.Sequential(
            nn.Conv2d(2, 48, (3, 3), padding=(1, 1)),
            nn.GroupNorm(1, emb_dim, eps=eps),
        )
        self.intra_norm = LayerNormalization4D(emb_dim, eps=eps)
        self.intra_rnn = nn.LSTM(in_channels, hidden_channels, 1, batch_first=True, bidirectional=True
            #in_channels, hidden_channels, 1, batch_first=True, bidirectional=True,48
        )
        self.intra_gru_cpx = ComplexLSTM(48, hidden_channels, 1, batch_first=True, bidirectional=True
                                 # in_channels, hidden_channels, 1, batch_first=True, bidirectional=True
                                 )
        self.inter_gru_cpx = ComplexLSTM(
            48, hidden_channels, 1, batch_first=True, bidirectional=True
        )

        self.intra_linear = nn.ConvTranspose1d(
            hidden_channels * 2, emb_dim, emb_ks, stride=emb_hs
        )
        self.intra_linear_cpx = ComplexConvTranspose1d(
            hidden_channels * 2, emb_dim, 1, stride=emb_hs
        )


        reduction = 5
        channel = 65
        # self.fc = nn.Sequential(
        #     nn.Linear(channel, channel // reduction, bias=False),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(channel // reduction, channel, bias=False),
        #     nn.Sigmoid()
        # )
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.inter_norm = LayerNormalization4D(emb_dim, eps=eps)
        self.inter_rnn = nn.LSTM(
            in_channels, hidden_channels, 1, batch_first=True, bidirectional=True
        )

        self.inter_linear = nn.ConvTranspose1d(
            hidden_channels * 2, emb_dim, emb_ks, stride=emb_hs
        )
        self.inter_linear_cpx =ComplexConvTranspose1d(
            hidden_channels * 2, emb_dim, 1, stride=emb_hs
        )

        E = math.ceil(
            approx_qk_dim * 1.0 / n_freqs
        )  # approx_qk_dim is only approximate,math.ceil向上取整，512//65
        assert emb_dim % n_head == 0
        for ii in range(n_head):
            self.add_module(
                "attn_conv_Q_%d" % ii,
                nn.Sequential(
                    nn.Conv2d(emb_dim, E, 1),
                    get_layer(activation)(),
                    LayerNormalization4DCF((E, n_freqs), eps=eps),
                ),
            )
            self.add_module(
                "attn_conv_K_%d" % ii,
                nn.Sequential(
                    nn.Conv2d(emb_dim, E, 1),
                    get_layer(activation)(),
                    LayerNormalization4DCF((E, n_freqs), eps=eps),
                ),
            )
            self.add_module(
                "attn_conv_V_%d" % ii,
                nn.Sequential(
                    nn.Conv2d(emb_dim, emb_dim // n_head, 1),
                    get_layer(activation)(),
                    LayerNormalization4DCF((emb_dim // n_head, n_freqs), eps=eps),
                ),
            )
        self.add_module(
            "attn_concat_proj",
            nn.Sequential(
                nn.Conv2d(emb_dim, emb_dim, 1),
                get_layer(activation)(),
                LayerNormalization4DCF((emb_dim, n_freqs), eps=eps),
            ),
        )

        self.emb_dim = emb_dim
        self.emb_ks = emb_ks
        self.emb_hs = emb_hs
        self.n_head = n_head
        self.linear = nn.Linear(130, 65)

        # self.SparseAttention = SparseAttention(heads=4,attn_mode='strided',local_attn_ctx=4,blocksize=4)
        # self.SparseAttention_inter = SparseAttention(heads=4, attn_mode='strided', local_attn_ctx=4, blocksize=4)

        # self.cpx_mamba = ExBimamba(d_model=None)

        self.intra_m = MultiModalLSTMCell(in_channels, hidden_channels, hidden_channels, hidden_channels)


        self.swish = nn.SiLU()
        self.swish_inter = nn.SiLU()
        self.dropout = nn.Dropout(0.1)
        self.dropout_inter = nn.Dropout(0.1)
        self.avgpool = nn.AdaptiveAvgPool2d((1,None))
        self.maxpool = nn.AdaptiveMaxPool2d((1,None))
        self.sigmoid = nn.Sigmoid()
        self.sigmoid1 = nn.Sigmoid()
        self.sigmoid2 = nn.Sigmoid()
        self.avgpool_intra = nn.AdaptiveAvgPool2d((None,1))
        self.maxpool_intra  = nn.AdaptiveMaxPool2d((None,1))
        self.sigmoid_intra = nn.Sigmoid()

        self.cbam = BasicBlock(planes=emb_dim)
        self.swish_all = nn.SiLU()
        self.dropout_all = nn.Dropout(0.1)
        self.fc = nn.Sequential(nn.Conv2d(emb_dim, emb_dim // 16, 1, bias=False),
                                nn.ReLU(),
                                nn.Conv2d(emb_dim // 16, emb_dim, 1, bias=False))
        self.fc_intra = nn.Sequential(nn.Conv2d(emb_dim, emb_dim // 16, 1, bias=False),
                                      nn.ReLU(),
                                      nn.Conv2d(emb_dim // 16, emb_dim, 1, bias=False))

        self.fc_inter = nn.Sequential(nn.Conv1d(emb_dim, emb_dim, 1, bias=False),
                                nn.SiLU(),
                                )
        self.fc_all = nn.Sequential(
            nn.Conv2d(emb_dim, emb_dim, (3, 3), padding=(1, 1)),
            nn.GroupNorm(1, emb_dim, eps=eps),
            nn.Sigmoid()
        )
        self.mouth_intranorm = LayerNormalization4D(emb_dim, eps=eps)
        self.mouth_internorm = LayerNormalization4D(emb_dim, eps=eps)

        self.cru = CRU(op_channel=emb_dim)
        self.sru = SRU(oup_channels=emb_dim,group_num=4)
        self.tripletattention = TripletAttention()
        self.gmu = GatedMultimodalLayer(size_in1=emb_dim,size_in2=emb_dim,size_out= emb_dim)

        #
        # self.tcn = TCN(input_size=emb_dim,
        #                num_channels=[48,48,48],
        #                num_classes=500,
        #                tcn_options=tcn_options,
        #                dropout=tcn_options['dropout'],
        #                relu_type='prelu',
        #                dwpw=tcn_options['dwpw'], )

    # @profile(precision=5,stream=open('/root/data1/ceshi/memory_profiler_GridNetBlock.log','w+'))
    def forward(self, x,mouth):
        """GridNetBlock Forward.
        Args:
            x: [B, C, T, Q]
            out: [B, C, T, 0Q]
        """
        B, C, T, Q = x.shape  #2,48,501,65
        # T = math.ceil((old_T - self.emb_ks) / self.emb_hs) * self.emb_hs + self.emb_ks
        # Q = math.ceil((old_Q - self.emb_ks) / self.emb_hs) * self.emb_hs + self.emb_ks
        # x = F.pad(x, (0, Q - old_Q, 0, T - old_T))
        # intra RNN
        input_ = x


        intra_rnn = self.intra_norm(input_)  # [B, C, T, Q]
        intra_rnn = (
            intra_rnn.transpose(1, 2).contiguous().view(B * T, C, Q) ## 1002,48,65
        )  # [BT, C, Q]

        intra_rnn = F.unfold(
            intra_rnn[..., None], (self.emb_ks, 1), stride=(self.emb_hs, 1)
        )  # [BT, C*emb_ks, -1]

        intra_rnn = intra_rnn.transpose(1, 2)  # [BT, -1, C*emb_ks]

        intra_rnn, _ = self.intra_rnn(intra_rnn)  # [BT, -1, H]
        intra_rnn = intra_rnn.transpose(1, 2)  # [BT, H, -1]
        intra_rnn = self.intra_linear(intra_rnn)  # [BT, C, Q]


        intra_rnn = intra_rnn.view([B, T, C, Q])
        intra_rnn = intra_rnn.transpose(1, 2).contiguous()  # [B, C, T, Q]
        # gate = self.sigmoid(mouth)
        # gate1 = self.sigmoid1(intra_rnn)
        # intra_rnn = intra_rnn + intra_rnn * gate
        # mouth_1 = mouth + mouth * gate1
        # mouth_1 = self.dropout(mouth_1)
        # mouth_1 = mouth_1.view(B * Q, C, T)
        # mouth_1 = self.fc_inter(mouth_1)
        # mouth_1 = mouth_1.view(B, C, T, Q)

        avgpool_result = self.fc_intra(self.avgpool_intra(mouth))
        maxpool_result = self.fc_intra(self.maxpool_intra(mouth))
        pool_sum = avgpool_result + maxpool_result
        activated_pool_sum = self.sigmoid_intra(pool_sum)
        residual_result = mouth * activated_pool_sum
        mouth = mouth + residual_result
        gate = self.swish(mouth)
        gate = self.mouth_intranorm(gate)

        x = intra_rnn * gate
        x = self.dropout(x)
        intra_rnn = x + intra_rnn + input_# [B, C, T, Q]
        mouth_1 = mouth + gate
        avgpool_result = self.fc(self.avgpool(mouth_1))
        maxpool_result = self.fc(self.maxpool(mouth_1))
        pool_sum = avgpool_result + maxpool_result
        activated_pool_sum = self.sigmoid(pool_sum)
        residual_result = mouth_1 * activated_pool_sum
        mouth_1 = mouth_1 + residual_result
        mouth_1 = self.mouth_internorm(mouth_1)

        # inter RNN
        input_ = intra_rnn

        inter_rnn = self.inter_norm(input_)  # [B, C, T, F]
        inter_rnn = (
            inter_rnn.permute(0, 3, 1, 2).contiguous().view(B * Q, C, T) #65,48,501
        )  # [BF, C, T]
        inter_rnn = F.unfold(
            inter_rnn[..., None], (self.emb_ks, 1), stride=(self.emb_hs, 1)
        )  # [BF, C*emb_ks, -1]
        inter_rnn = inter_rnn.transpose(1, 2)  # [BF, -1, C*emb_ks]
        inter_rnn, _ = self.inter_rnn(inter_rnn)  # [BF, -1, H]
        inter_rnn = inter_rnn.transpose(1, 2)  # [BF, H, -1]
        inter_rnn = self.inter_linear(inter_rnn)  # [BF, C, T]

        inter_rnn = inter_rnn.view([B, Q, C, T])
        inter_rnn = inter_rnn.permute(0, 2, 3, 1).contiguous()  # [B, C, T, Q]
        # gate1 = self.sigmoid2(inter_rnn)
        # inter_rnn = inter_rnn + inter_rnn * mouth_1
        # mouth = mouth_1 + mouth_1 * gate1
        # mouth = self.dropout_inter(mouth)
        # mouth = self.fc_all(mouth)
        gate_inter = self.swish_inter(mouth_1)
        x_inter = inter_rnn * gate_inter
        x_inter = self.dropout_inter(x_inter)
        mouth = self.tripletattention(mouth_1)
        inter_rnn = input_  + inter_rnn + x_inter+ mouth # [B, C, T, Q]

        # attention
        inter_rnn = inter_rnn[..., :T, :Q]
        batch = inter_rnn
        all_Q, all_K, all_V = [], [], []
        for ii in range(self.n_head):
            all_Q+=[self["attn_conv_Q_%d" % ii](batch)] # [B, C, T, Q]
            all_K+=[self["attn_conv_K_%d" % ii](batch)]  # [B, C, T, Q]
            all_V+=[self["attn_conv_V_%d" % ii](batch)]  # [B, C, T, Q]

        Q = torch.cat(all_Q, dim=0)  # [B', C, T, Q]
        K = torch.cat(all_K, dim=0)  # [B', C, T, Q]
        V = torch.cat(all_V, dim=0)  # [B', C, T, Q]

        Q = Q.transpose(1, 2)

        Q = Q.flatten(start_dim=2)  # [B', T, C*Q]
        K = K.transpose(1, 2)
        K = K.flatten(start_dim=2)  # [B', T, C*Q]
        V = V.transpose(1, 2)  # [B', T, C, Q]
        old_shape = V.shape
        V = V.flatten(start_dim=2)  # [B', T, C*Q]
        emb_dim = Q.shape[-1]

        attn_mat = torch.matmul(Q, K.transpose(1, 2)) / (emb_dim ** 0.5)  # [B', T, T]
        attn_mat = F.softmax(attn_mat, dim=2)  # [B', T, T]
        V = torch.matmul(attn_mat, V)  # [B', T, C*Q]
        V = V.reshape(old_shape)  # [B', T, C, Q]
        V = V.transpose(1, 2)  # [B', C, T, Q]
        emb_dim = V.shape[1]

        batch = V.view([self.n_head, B, emb_dim, T, -1])  # [n_head, B, C, T, Q])
        batch = batch.transpose(0, 1)  # [B, n_head, C, T, Q])
        batch = batch.contiguous().view(
            [B, self.n_head * emb_dim, T, -1]
        )  # [B, C, T, Q])
        batch = self["attn_concat_proj"](batch)  # [B, C, T, Q])

        out = batch + inter_rnn

        # gate_all = self.swish_all(out)
        # out = mouth_all * gate_all
        # out = self.dropout_all(out)

        return out,mouth

class LayerNormalization4D(nn.Module):
    def __init__(self, input_dimension, eps=1e-5):
        super().__init__()
        param_size = [1, input_dimension, 1, 1]
        self.gamma = Parameter(torch.Tensor(*param_size).to(torch.float32))
        self.beta = Parameter(torch.Tensor(*param_size).to(torch.float32))
        init.ones_(self.gamma)
        init.zeros_(self.beta)
        self.eps = eps

    def forward(self, x):
        if x.ndim == 4:
            _, C, _, _ = x.shape
            stat_dim = (1,)
        else:
            raise ValueError("Expect x to have 4 dimensions, but got {}".format(x.ndim))
        mu_ = x.mean(dim=stat_dim, keepdim=True)  # [B,1,T,F]
        std_ = torch.sqrt(
            x.var(dim=stat_dim, unbiased=False, keepdim=True) + self.eps
        )  # [B,1,T,F]
        x_hat = ((x - mu_) / std_) * self.gamma.cuda() + self.beta.cuda()
        return x_hat


class LayerNormalization4DCF(nn.Module):
    def __init__(self, input_dimension, eps=1e-5):
        super().__init__()
        assert len(input_dimension) == 2
        param_size = [1, input_dimension[0], 1, input_dimension[1]]
        self.gamma = Parameter(torch.Tensor(*param_size).to(torch.float32))
        self.beta = Parameter(torch.Tensor(*param_size).to(torch.float32))
        init.ones_(self.gamma)
        init.zeros_(self.beta)
        self.eps = eps

    def forward(self, x):
        if x.ndim == 4:
            stat_dim = (1, 3)
        else:
            raise ValueError("Expect x to have 4 dimensions, but got {}".format(x.ndim))
        mu_ = x.mean(dim=stat_dim, keepdim=True)  # [B,1,T,1]均值
        std_ = torch.sqrt(
            x.var(dim=stat_dim, unbiased=False, keepdim=True) + self.eps
        )  # [B,1,T,F]标准差
        x_hat = ((x - mu_) / std_) * self.gamma + self.beta
        return x_hat


class GatingMechanism(nn.Module):
    def __init__(self, input_dim):
        super(GatingMechanism, self).__init__()
        # 门控网络，输出2个值用于控制一致性和互补性特征的权重
        self.fc = nn.Linear(input_dim, 2)


    def forward(self, combined_input):
        # 计算一致性和互补性特征的权重
        gate_values = torch.sigmoid(self.fc(combined_input))
        return gate_values
class TF_Patch(nn.Module):
    """Offline TFGridNet

    Reference:
    [1] Z.-Q. Wang, S. Cornell, S. Choi, Y. Lee, B.-Y. Kim, and S. Watanabe,
    "TF-GridNet: Integrating Full- and Sub-Band Modeling for Speech Separation",
    in arXiv preprint arXiv:2211.12433, 2022.
    [2] Z.-Q. W                                                                                                ang, S. Cornell, S. Choi, Y. Lee, B.-Y. Kim, and S. Watanabe,
    "TF-GridNet: Making Time-Frequency Domain Models Great Again for Monaural
    Speaker Separation", in arXiv preprint arXiv:2209.03952, 2022.

    NOTES:
    As outlined in the Reference, this model works best when trained with variance
    normalized mixture input and target, e.g., with mixture of shape [batch, samples,
    microphones], you normalize it by dividing with torch.std(mixture, (1, 2)). You
    must do the same for the target signals. It is encouraged to do so when not using
    scale-invariant loss functions such as SI-SDR.

    Args:
        input_dim: placeholder, not used
        n_srcs: number of output sources/speakers.
        n_fft: stft window size.
        stride: stft stride.
        window: stft window type choose between 'hamming', 'hanning' or None.
        n_imics: number of microphones channels (only fixed-array geometry supported).
        n_layers: number of TFGridNet blocks.
        lstm_hidden_units: number of hidden units in LSTM.
        attn_n_head: number of heads in self-attention
        attn_approx_qk_dim: approximate dimention of frame-level key and value tensors
        emb_dim: embedding dimension
        emb_ks: kernel size for unfolding and deconv1D
        emb_hs: hop size for unfolding and deconv1D
        activation: activation function to use in the whole TFGridNet model,
            you can use any torch supported activation e.g. 'relu' or 'elu'.
        eps: small epsilon for normalization layers.
        use_builtin_complex: whether to use builtin complex type or not.
    """

    def __init__(
            self,
            n_srcs=2,
            n_fft=128,
            stride=64,
            window="hann",
            n_imics=1,
            n_layers=6,
            lstm_hidden_units=192,
            attn_n_head=4,
            attn_approx_qk_dim=512,
            emb_dim=48,
            emb_ks=4,
            emb_hs=1,
            activation="prelu",
            eps=1.0e-5,
            use_builtin_complex=False,
            ref_channel=-1,
    ):
        super().__init__()
        self.model_args = {
            'n_srcs': n_srcs,
            'n_fft': n_fft,
            'stride': stride,
            'window': window,
            'n_imics': n_imics,
            'n_layers': n_layers,
            'lstm_hidden_units': lstm_hidden_units,
            'attn_n_head': attn_n_head,
            'attn_approx_qk_dim': attn_approx_qk_dim,
            'emb_dim': emb_dim,
            'emb_ks': emb_ks,
            'emb_hs': emb_hs,
            'activation': activation,
            'eps': eps,
            'use_builtin_complex': use_builtin_complex,
            'ref_channel': ref_channel,
        }
        self.n_srcs = n_srcs
        self.n_layers = n_layers
        self.n_imics = n_imics
        self.n_fft = n_fft
        assert n_fft % 2 == 0
        n_freqs = n_fft // 2 + 1
        self.ref_channel = ref_channel
        self.enc = STFTEncoder(
            n_fft, n_fft, stride, window=window, use_builtin_complex=use_builtin_complex
        )
        self.dec = STFTDecoder(n_fft, n_fft, stride, window=window)
        t_ksize = 3
        ks, padding = (t_ksize, 3), (t_ksize // 2, 1)
        self.conv_yuan = nn.Sequential(
            nn.Conv2d(2 * n_imics, emb_dim, ks, padding=padding),
            nn.GroupNorm(1, emb_dim, eps=eps),
        )
        self.conv = nn.Sequential(
            nn.Conv2d(2 * n_imics, emb_dim, ks, padding=padding),
            nn.GroupNorm(1, emb_dim, eps=eps),
        )
        self.conv_ = nn.Sequential(
            nn.Conv2d(2 * n_imics, emb_dim, ks, padding=padding),
            nn.GroupNorm(1, emb_dim, eps=eps),
        )

        self.blocks = nn.ModuleList([])
        for _ in range(n_layers):
            self.blocks.append(
                GridNetBlock(
                    emb_dim,
                    emb_ks,
                    emb_hs,
                    n_freqs,
                    lstm_hidden_units,
                    n_head=attn_n_head,
                    approx_qk_dim=attn_approx_qk_dim,
                    activation=activation,
                    eps=eps,
                    n_layers = _,
                )
            )
        self.deconv = nn.ConvTranspose2d(emb_dim, n_srcs * 2, ks, padding=padding)  # 原来是n_srcs*2

        self.linear = nn.Linear(144, 48)
        self.linear2 = nn.Linear(512, 65)
        self.EX_Mamba = ExBimamba(d_model=65,d_state=128)
        self.Mamba1 = Mamba1(d_model=65, d_state=128)
        # self.Mamba2 = Mamba1(d_model=65, d_state=128)

        self.complex_conv = ComplexConv2d(in_channels=1, out_channels=emb_dim, kernel_size=(3, 3),
                                                  padding=(1, 1))
        # self.encoder_layer = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=32000,
        #                                            nhead=4,
        #                                            dim_feedforward=2048,
        #                                            dropout=0.1,
        #                                            activation=F.relu,
        #                                            norm_first=False),
        # num_layers = 1)

        self.conv1d =  nn.Sequential(
            nn.Conv2d(2 * n_imics, emb_dim, ks, padding=padding),
            nn.GroupNorm(1, emb_dim, eps=eps),
        )
        self.conv1d_ = nn.Sequential(  nn.Conv2d(2 * n_imics, emb_dim, ks, padding=padding),
                                        nn.GroupNorm(1, emb_dim, eps=eps))



        self.pool = nn.AdaptiveAvgPool1d(1)
        self.pool1 = nn.AdaptiveAvgPool2d((1 ,1))
        self.pool2 = nn.AdaptiveAvgPool2d((65, 1))
        # self.trunk = ResNet(BasicBlock, [1,1,1,1], relu_type="prelu")
        # self.trunk1 = ResNet(BasicBlock, [1,1,1,1], relu_type="prelu")
        # self.trunk2 = ResNet(BasicBlock, [2, 2, 2, 2], relu_type="prelu")
        self.softmax = nn.Softmax(dim=1)
        self.softmax1 = nn.Softmax(dim=1)
        self.weight_x = nn.Parameter(torch.ones(1, 48))
        self.weight_y = nn.Parameter(torch.ones(1, 48))
        self.alpha = nn.Parameter(torch.tensor(0.5))
        self.gate = GatingMechanism(96)
        self.swish = nn.SiLU()

        self.conv_g = nn.Sequential(
            nn.Conv2d(2, emb_dim, (3, 3), padding=(1, 1)),
            nn.GroupNorm(1, emb_dim, eps=eps),
        )
        self.conv_g_1 = nn.Sequential(
            nn.Conv2d(2, emb_dim, (3, 3), padding=(1, 1)),
            nn.GroupNorm(1, emb_dim, eps=eps),
        )
        self.conv_g_2 = nn.Sequential(
            nn.Conv2d(2, emb_dim, (3, 3), padding=(1, 1)),
            nn.GroupNorm(1, emb_dim, eps=eps),
        )
        self.mouth_norm = LayerNormalization4D(emb_dim, eps=eps)
        self.mouth_norm1 = LayerNormalization4D(emb_dim, eps=eps)
        self.mouth_norm2 = LayerNormalization4D(emb_dim, eps=eps)
        self.ggmu = GatedMultimodalLayerGNN(size_in1=emb_dim * 4, size_in2=emb_dim, size_out=emb_dim, num_nodes=4)
        self.gmu = GatedMultimodalLayer(size_in1=emb_dim, size_in2=emb_dim, size_out=emb_dim)
        self.a = nn.Parameter(torch.tensor(0.5))

    # @profile(precision=5,stream=open('/root/data1/ceshi/memory_profiler.log','w+'))
    def forward(self, input, mouth):
        # additional: Optional[Dict] = None,
        """Forward.
        Args:
            input (torch.Tensor): batched multi-channel audio tensor with
                    M audio channels and N samples [B, N, M]
            ilens (torch.Tensor): input lengths [B]
            additional (Dict or None): other data, currently unused in this model.

        Returns:
            enhanced (List[Union(torch.Tensor)]):
                    [(B, T), ...] list of len n_srcs
                    of mono audio tensors with T samples.
            ilens (torch.Tensor): (B,)
            additional (Dict or None): other data, currently unused in this model,
                    we return it also in output.
        """
        # 假设所有样本长度都是 N
        B = input.shape[0]
        N = input.shape[1]  # 获取第二个维度的大小，即样本长度
        ilens = torch.full((input.shape[0],), N, dtype=torch.long)  # 创建一个填充了 N 的张量，形状为 [B]
        n_samples = input.shape[1]
        if self.n_imics == 1:
            assert len(input.shape) == 2
            input = input[..., None]  # [B, N, M]
        mix_std_ = torch.std(input, dim=(1, 2), keepdim=True)  # [B, 1, 1]
        input = input / mix_std_  # RMS normalization
        batch = self.enc(input, ilens)[0]  # [B, T, M, F],F=(fft/2)+1,T=32000/stride +1
        batch0 = batch.transpose(1, 2)  #  [B, M, T, F]
        # batch_complex = torch.stack([batch0.real, batch0.imag], -1)
        # x_cpx = self.complex_conv(batch0)
        # batch_real = batch0.real
        # batch_imag = batch0.imag
        # mag, phase = (torch.sqrt(torch.clamp(batch_real ** 2 + batch_imag ** 2, EPSILON))
        #               , torch.atan2(batch_imag + EPSILON, batch_real))
        # mag_input = []
        # mag_input.append(mag)
        # inputs_real, inputs_imag = mag * torch.cos(phase), mag * torch.sin(phase)

        batch = torch.cat((batch0.real, batch0.imag), dim=1)  # [B, 2*M, T, F]
        batch1 = batch
        n_batch, _, n_frames, n_freqs = batch.shape
        mouth = F.interpolate(mouth, size=(n_frames, n_freqs))  # 2,2,499,512
        # mouth = mouth.view(-1, 512)
        # mouth = self.linear2(mouth)
        mouth = mouth.view(n_batch, _, n_frames,  n_freqs)

        batch, mouth_ssm = self.EX_Mamba(batch.view(n_batch*_, n_frames, n_freqs), mouth.view(n_batch*_, n_frames, n_freqs))
        batch_ceshi = batch
        batch_o = self.Mamba1(batch1.view(n_batch*_, n_frames, n_freqs))

        batch = batch + batch1.view(n_batch*_, n_frames, n_freqs)
        batch_A = batch_o + batch1.view(n_batch*_, n_frames, n_freqs)
        mouth_A = mouth_ssm + mouth.view(n_batch*_, n_frames, n_freqs)

        # batch_A, batch_A_pool= self.trunk(batch.view(n_batch, _, n_frames, n_freqs)) #2,48
        # batch_B, batch_B_pool = self.trunk1(mouth_ssm.view(n_batch, _, n_frames, n_freqs))

        # batch = self.linear(fused_features)

        # batch = torch.cat([batch, mouth_ssm, batch1.view(n_batch*_, n_frames, n_freqs)], dim=-1)
        # batch = self.linear(batch)
        #
        # batch_A = self.softmax(batch_A_pool)
        # batch_B = self.softmax(batch_B_pool)

        batch = batch.view(n_batch, _, n_frames, n_freqs)
        batch = self.conv(batch)
        #
        # mouth_ = self.conv_g_2(mouth)
        # mouth_ = self.mouth_norm2(mouth_)

        input_c = batch_A.view(n_batch, -1, n_frames, n_freqs)
        input_p = mouth_A.view(n_batch, -1, n_frames, n_freqs)

        input_p = self.conv_g(input_p)
        input_p = self.mouth_norm(input_p)

        input_c = self.conv_g_1(input_c)
        input_c = self.mouth_norm1(input_c)

        mouth = self.gmu(

                          input_c.view(n_batch, n_frames, n_freqs, -1),
                          input_p.view(n_batch, n_frames, n_freqs, -1))

        # mouth = self.ggmu( batch_o.view(n_batch, n_frames, n_freqs ,-1),
        #                     mouth_.view(n_batch, n_frames, n_freqs ,-1),
        #                     input_c.view(n_batch, n_frames, n_freqs ,-1),
        #                   input_p.view(n_batch, n_frames, n_freqs, -1))
        mouth = mouth.view(n_batch, -1, n_frames, n_freqs)
        # batch = self.a * batch + (1-self.a) * mouth

        for ii in range(3):
            batch,mouth = self.blocks[ii](batch,mouth)  # [B, -1, T, F]
            mouth, batch = self.blocks[ii](mouth, batch)

        # batch = batch[:, :, :, :65]
        batch = self.deconv(batch)  # [B, n_srcs*2, T, F]
        batch = batch.view([n_batch, self.n_srcs, 2, n_frames, n_freqs])
        batch = new_complex_like(batch0, (batch[:, :, 0], batch[:, :, 1]))

        batch = self.dec(batch.view(-1, n_frames, n_freqs), ilens)[0]  # [B, n_srcs, -1]
        batch = self.pad2(batch.view([n_batch, self.num_spk, -1]), n_samples)

        batch = batch * mix_std_  # reverse the RMS normalization
        batch = [batch[:, src] for src in range(self.num_spk)]
        batch = torch.stack(batch, dim=1)




        return batch,batch_o, mouth_ssm,self.a

    # def from_pretrain(pretrained_model_conf_or_path, *args, **kwargs):
    #     def get(identifier):
    #         if identifier is None:
    #             return None
    #         elif callable(identifier):
    #             return identifier
    #         elif isinstance(identifier, str):
    #             cls = globals().get(identifier)
    #             if cls is None:
    #                 raise ValueError("Could not interpret activation identifier: " + str(identifier))
    #             return cls
    #         else:
    #             raise ValueError("Could not interpret activation identifier: " + str(identifier))
    #
    #     conf = torch.load(pretrained_model_conf_or_path, map_location="cpu")
    #     # Attempt to find the model and instantiate it.
    #     # import pdb; pdb.set_trace()
    #     model_class = get("SP_audionet")
    #     # model_class = get("AVFRCNN")
    #     model = model_class(*args, **kwargs)
    #     model.load_state_dict(conf["state_dict"])
    #     # model = BaseAVModel.load_state_dict_in(model, conf["state_dict"])
    #     return model

    @property
    def num_spk(self):
        return self.n_srcs

    @staticmethod
    def pad2(input_tensor, target_len):
        input_tensor = torch.nn.functional.pad(
            input_tensor, (0, target_len - input_tensor.shape[-1])
        )
        return input_tensor
    def serialize(self):
        import pytorch_lightning as pl  # Not used in torch.hub

        model_conf = dict(
            model_name=self.__class__.__name__,
            state_dict=self.get_state_dict(),
            model_args=self.get_model_args(),
        )
        # Additional infos
        infos = dict()
        infos["software_versions"] = dict(
            torch_version=torch.__version__,
            pytorch_lightning_version=pl.__version__,
        )
        model_conf["infos"] = infos
        return model_conf

    def get_state_dict(self):
        """In case the state dict needs to be modified before sharing the model."""
        return self.state_dict()

    def get_model_args(self):
        """Arguments needed to re-instantiate the model."""

        return self.model_args
