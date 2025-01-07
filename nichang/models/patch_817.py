import os
os.environ['PATH'] += '/root/data1/anaconda/envs/nichang/bin/ninja'
import math
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple
import sys
from ..layers.triple_attention import TripletAttention
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
from mamba_ssm.ops.triton.layer_norm import RMSNorm
from ..layers.resnet import ResNet, BasicBlock
from ..layers.lstm_cell import MultiModalLSTMCell
from ..layers.CBAM import BasicBlock
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
# from ..layers.patch_embed import ConvPatchEmbed, DeformablePatchTransformer
# from ..layers.mamba_layer import PlainMambaLayer
# from abc import ABC, abstractmethod
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
        B, C, T, Q =   x1.shape
        x1 = x1.view( B, T, Q, C)
        x2 = x2.view(B, T, Q, C)
        h1 = self.tanh_f(self.hidden1(x1))
        h2 = self.tanh_f2(self.hidden2(x2))

        # Apply channel transformation (GCT style) to each modality
        h1_transformed = self.scale1 * h1 + self.bias1
        h2_transformed = self.scale2 * h2 + self.bias2

        x = torch.cat((x1, x2), dim=-1)
        z = self.sigmoid_f(self.hidden_sigmoid(x))
        z = z * h1_transformed + (1 - z) * h2_transformed
        return z.view(B, C, T, Q )


class GatedFeatureTransfer(nn.Module):
    def __init__(self, channels):
        super(GatedFeatureTransfer, self).__init__()
        # 定义用于计算门控权重的卷积层
        self.conv_comp = nn.Conv2d(channels, 1, kernel_size=1)
        self.conv_cons = nn.Conv2d(channels, 1, kernel_size=1)
        # 初始化偏置项
        nn.init.constant_(self.conv_comp.bias, 0)
        nn.init.constant_(self.conv_cons.bias, 0)

    def forward(self, feat_comp, feat_cons):
        """
        输入:
        - feat_comp: 互补性特征，[B, C, T, Q]
        - feat_cons: 一致性特征，[B, C, T, Q]
        输出:
        - fused_feat: 融合后的特征，[B, C, T, Q]
        """
        # 计算互补性特征的门控值
        gate_comp = self.conv_comp(feat_comp)  # [B, 1, T, Q]
        # 计算一致性特征的门控值
        gate_cons = self.conv_cons(feat_cons)  # [B, 1, T, Q]

        # 计算总的门控权重
        gate = torch.sigmoid(gate_comp + gate_cons)  # [B, 1, T, Q]

        # 扩展门控权重的通道维度，便于元素级乘法
        gate = gate.expand(-1, feat_comp.size(1), -1, -1)  # [B, C, T, Q]

        # 进行加权融合
        fused_feat = gate * feat_comp + (1 - gate) * feat_cons  # [B, C, T, Q]

        return fused_feat
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
        # self.intra_gru_cpx = ComplexLSTM(48, hidden_channels, 1, batch_first=True, bidirectional=True
        #                          # in_channels, hidden_channels, 1, batch_first=True, bidirectional=True
        #                          )
        # self.inter_gru_cpx = ComplexLSTM(
        #     48, hidden_channels, 1, batch_first=True, bidirectional=True
        # )

        self.intra_linear = nn.ConvTranspose1d(
            hidden_channels * 2, emb_dim, emb_ks, stride=emb_hs
        )
        self.intra_linear_mouth = nn.ConvTranspose1d(
            hidden_channels * 2, emb_dim, emb_ks, stride=emb_hs
        )


        # reduction = 5
        # channel = 65
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
        self.inter_linear_mouth = nn.ConvTranspose1d(
            hidden_channels * 2, emb_dim, emb_ks, stride=emb_hs
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
        for ii in range(n_head):
            self.add_module(
                "attn_conv_Q_%d1" % ii,
                nn.Sequential(
                    nn.Conv2d(emb_dim, E, 1),
                    get_layer(activation)(),
                    LayerNormalization4DCF((E, n_freqs), eps=eps),
                ),
            )
            self.add_module(
                "attn_conv_K_%d1" % ii,
                nn.Sequential(
                    nn.Conv2d(emb_dim, E, 1),
                    get_layer(activation)(),
                    LayerNormalization4DCF((E, n_freqs), eps=eps),
                ),
            )
            self.add_module(
                "attn_conv_V_%d1" % ii,
                nn.Sequential(
                    nn.Conv2d(emb_dim, emb_dim // n_head, 1),
                    get_layer(activation)(),
                    LayerNormalization4DCF((emb_dim // n_head, n_freqs), eps=eps),
                ),
            )
        self.add_module(
            "attn_concat_proj1",
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
        mid_channels =128
        embed_dims = 128
        self.embed_dims=embed_dims
        self.intra_m = MultiModalLSTMCell(in_channels, hidden_channels, hidden_channels, hidden_channels)

        self.conv = nn.Sequential(
            nn.Conv2d(2 , emb_dim, (3,3), padding=(1,1)),
            nn.GroupNorm(1, emb_dim, eps=eps),
        )
        self.conv_1 = nn.Sequential(
            nn.Conv2d(2, emb_dim, (3, 3), padding=(1, 1)),
            nn.GroupNorm(1, emb_dim, eps=eps),
        )
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
        self.mouth_norm = LayerNormalization4D(emb_dim, eps=eps)
        self.cbam = BasicBlock(planes=emb_dim)
        self.swish_all = nn.SiLU()
        self.dropout_all = nn.Dropout(0.1)
        # self.fc = nn.Sequential(nn.Conv2d(emb_dim, emb_dim // 16, 1, bias=False),
        #                         nn.ReLU(),
        #                         nn.Conv2d(emb_dim // 16, emb_dim, 1, bias=False))
        # self.fc_intra = nn.Sequential(nn.Conv2d(emb_dim, emb_dim // 16, 1, bias=False),
        #                               nn.ReLU(),
        #                               nn.Conv2d(emb_dim // 16, emb_dim, 1, bias=False))
        #
        # self.fc_inter = nn.Sequential(nn.Conv1d(emb_dim, emb_dim, 1, bias=False),
        #                         nn.SiLU(),
        #                         )
        # self.fc_all = nn.Sequential(
        #     nn.Conv2d(emb_dim, emb_dim, (3, 3), padding=(1, 1)),
        #     nn.GroupNorm(1, emb_dim, eps=eps),
        #     nn.Sigmoid()
        # )
        self.mouth_intranorm = LayerNormalization4D(emb_dim, eps=eps)
        self.mouth_internorm = LayerNormalization4D(emb_dim, eps=eps)
        # self.a = nn.Parameter(torch.tensor(0.5))
        # self.tripletattention = TripletAttention()
        # self.sru = SRU(oup_channels=emb_dim,group_num=4)
        # self.conv_double = nn.Sequential(
        #     nn.Conv2d(emb_dim, emb_dim *2, (3, 3), padding=(1, 1)),
        #     nn.GroupNorm(1, emb_dim *2, eps=eps),
        # )
        # self.gmu = GatedMultimodalLayer(size_in1=emb_dim,size_in2=emb_dim,size_out= emb_dim)
        # self.gmu1 = GatedMultimodalLayer(size_in1=emb_dim, size_in2=emb_dim, size_out=emb_dim)
        # self.attention_layer = nn.Sequential(
        #     nn.Conv2d(in_channels=emb_dim, out_channels=emb_dim // 2, kernel_size=1),
        #     nn.ReLU(),
        #     nn.Conv2d(in_channels=emb_dim // 2, out_channels=1, kernel_size=1)
        # )
        # self.sigmoid_ = nn.Sigmoid()
        # self.gate = GatingMechanism(emb_dim)
        # self.gate2 = GatingMechanism(emb_dim)
        # self.alpha = nn.Parameter(torch.tensor(1.0))
        # # 可选的激活函数
        # self.activation = nn.ReLU()
        # self.alpha1 = nn.Parameter(torch.tensor(1.0))
        # # 可选的激活函数
        # self.activation1 = nn.ReLU()
        self.intra_rnn_mouth = nn.LSTM(in_channels, hidden_channels, 1, batch_first=True, bidirectional=True
                                 # in_channels, hidden_channels, 1, batch_first=True, bidirectional=True,48
                                 )
        self.inter_rnn_mouth = nn.LSTM(
            in_channels, hidden_channels, 1, batch_first=True, bidirectional=True
        )
        #
        # self.tcn = TCN(input_size=emb_dim,
        #                num_channels=[48,48,48],
        #                num_classes=500,
        #                tcn_options=tcn_options,
        #                dropout=tcn_options['dropout'],
        #                relu_type='prelu',
        #                dwpw=tcn_options['dwpw'], )


    # @profile(precision=5,stream=open('/root/data1/ceshi/memory_profiler_GridNetBlock.log','w+'))
    def forward(self, x,complement2):
        """GridNetBlock Forward.

        Args:
            x: [B, C, T, Q]
            out: [B, C, T, Q]
        """
        B, C, T, Q = x.shape  #2,48,501,65
        # T = math.ceil((old_T - self.emb_ks) / self.emb_hs) * self.emb_hs + self.emb_ks
        # Q = math.ceil((old_Q - self.emb_ks) / self.emb_hs) * self.emb_hs + self.emb_ks
        # x = F.pad(x, (0, Q - old_Q, 0, T - old_T))
        # intra RNN
        # input_ = batch_yuan + complement1 +x

        # attention_weights = self.sigmoid_(self.attention_layer(x))  # [B, 1, T, Q]
        # # 动态聚合音频和视觉互补性特征
        input_ = x

        mouth = complement2
        # mouth =  complement2

        # mouth = (
        #     mouth.transpose(1, 2).contiguous().view(B * T, C, Q) ## 1002,48,65
        # )  # [BT, C, Q]
        # mouth = self.tcn(mouth)
        # mouth = mouth.view(B, C, T, Q)

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
        intra_rnn = intra_rnn + input_
        # gate = self.sigmoid(mouth)
        # gate1 = self.sigmoid1(intra_rnn)
        # intra_rnn = intra_rnn + intra_rnn * gate
        # mouth_1 = mouth + mouth * gate1
        # mouth_1 = self.dropout(mouth_1)
        # mouth_1 = mouth_1.view(B * Q, C, T)
        # mouth_1 = self.fc_inter(mouth_1)
        # mouth_1 = mouth_1.view(B, C, T, Q)\\
        intra_rnn_c = self.mouth_norm(mouth)  # [B, C, T, Q]
        intra_rnn_c = (
            intra_rnn_c.transpose(1, 2).contiguous().view(B * T, C, Q)  ## 1002,48,65
        )  # [BT, C, Q]

        intra_rnn_c = F.unfold(
            intra_rnn_c[..., None], (self.emb_ks, 1), stride=(self.emb_hs, 1)
        )  # [BT, C*emb_ks, -1]

        intra_rnn_c = intra_rnn_c.transpose(1, 2)  # [BT, -1, C*emb_ks]
        intra_rnn_c, _ = self.intra_rnn_mouth(intra_rnn_c)  # [BT, -1, H]
        intra_rnn_c = intra_rnn_c.transpose(1, 2)  # [BT, H, -1]
        intra_rnn_c =  self.intra_linear_mouth(intra_rnn_c)  # [BT, C, Q]

        intra_rnn_c = intra_rnn_c.view([B, T, C, Q])
        intra_rnn_c = intra_rnn_c.transpose(1, 2).contiguous()  # [B, C, T, Q]
        intra_rnn_c = intra_rnn_c + mouth

        # avgpool_result = self.fc_intra(self.avgpool_intra(mouth))
        # maxpool_result = self.fc_intra(self.maxpool_intra(mouth))
        # pool_sum = avgpool_result + maxpool_result
        # activated_pool_sum = self.sigmoid_intra(pool_sum)
        # residual_result = mouth * activated_pool_sum
        # mouth = mouth + residual_result
        # mouth = self.mouth_intranorm(mouth)
        # # gate = self.swish(mouth)
        # # x_intra =  self.gate(intra_rnn,mouth)
        # x_intra = self.activation(intra_rnn * mouth) * self.alpha
        #
        # # x = self.dropout(x)
        # # intra_rnn = intra_rnn + input_ + complement2 # [B, C, T, Q]
        # intra_rnn =x_intra + input_# [B, C, T, Q]
        # mouth_1 = x_intra + mouth
        # # complement1 = mouth_1
        # avgpool_result = self.fc(self.avgpool(mouth_1))
        #
        # maxpool_result = self.fc(self.maxpool(mouth_1))
        # pool_sum = avgpool_result + maxpool_result
        # activated_pool_sum = self.sigmoid(pool_sum)
        # residual_result = mouth_1 * activated_pool_sum
        # mouth_1 = mouth_1 + residual_result
        # mouth_1 = self.mouth_internorm(mouth_1)

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
        # gate_inter = self.swish_inter(mouth_1)
        # x_inter = inter_rnn * gate_inter
        # x_inter = self.dropout_inter(x_inter)
        # mouth = self.conv_double(mouth_1)
        # mouth = self.tripletattention(mouth)

        #
        # x_inter = self.activation1(inter_rnn * mouth_1) * self.alpha1
        #
        # complement2 = mouth_1 + x_inter


        inter_rnn = input_  + inter_rnn # [B, C, T, Q]+ x_inter+ complement1 + complement2

        input_ = intra_rnn_c

        inter_rnn_c = self.mouth_internorm(input_)  # [B, C, T, F]
        inter_rnn_c = (
            inter_rnn_c.permute(0, 3, 1, 2).contiguous().view(B * Q, C, T)  # 65,48,501
        )  # [BF, C, T]
        inter_rnn_c = F.unfold(
            inter_rnn_c[..., None], (self.emb_ks, 1), stride=(self.emb_hs, 1)
        )  # [BF, C*emb_ks, -1]
        inter_rnn_c = inter_rnn_c.transpose(1, 2)  # [BF, -1, C*emb_ks]
        inter_rnn_c, _ = self.inter_rnn_mouth(inter_rnn_c)  # [BF, -1, H]
        inter_rnn_c = inter_rnn_c.transpose(1, 2)  # [BF, H, -1]
        inter_rnn_c = self.inter_linear_mouth(inter_rnn_c)  # [BF, C, T]

        inter_rnn_c = inter_rnn_c.view([B, Q, C, T])
        inter_rnn_c = inter_rnn_c.permute(0, 2, 3, 1).contiguous()  # [B, C, T, Q]
        inter_rnn_c = input_ + inter_rnn_c
        batch_ = inter_rnn_c


        # attention
        inter_rnn = inter_rnn[..., :T, :Q]
        batch = inter_rnn
        all_Q, all_K, all_V = [], [], []
        for ii in range(self.n_head):
            all_Q+=[self["attn_conv_Q_%d" % ii](batch_)] # [B, C, T, Q]
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


        all_Q, all_K, all_V = [], [], []
        for ii in range(self.n_head):
            all_Q += [self["attn_conv_Q_%d1" % ii](batch)]  # [B, C, T, Q]
            all_K += [self["attn_conv_K_%d1" % ii](batch_)]  # [B, C, T, Q]
            all_V += [self["attn_conv_V_%d1" % ii](batch_)]  # [B, C, T, Q]

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

        batch_ = V.view([self.n_head, B, emb_dim, T, -1])  # [n_head, B, C, T, Q])
        batch_ = batch_.transpose(0, 1)  # [B, n_head, C, T, Q])
        batch_ = batch_.contiguous().view(
            [B, self.n_head * emb_dim, T, -1]
        )  # [B, C, T, Q])
        batch_ = self["attn_concat_proj"](batch_)  # [B, C, T, Q])

        out1 = batch_ + inter_rnn_c

        # gate_all = self.swish_all(out)
        # out = mouth_all * gate_all
        # out = self.dropout_all(out)

        return out,out1

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
        self.dec1 = STFTDecoder(n_fft, n_fft, stride, window=window)
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
        # self.blocks1 = nn.ModuleList([])
        # for _ in range(n_layers):
        #     self.blocks1.append(
        #         GridNetBlock(
        #             emb_dim,
        #             emb_ks,
        #             emb_hs,
        #             n_freqs,
        #             lstm_hidden_units,
        #             n_head=attn_n_head,
        #             approx_qk_dim=attn_approx_qk_dim,
        #             activation=activation,
        #             eps=eps,
        #             n_layers = _,
        #         )
        #     )
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
        self.deconv_s1 = nn.ConvTranspose2d(emb_dim, n_srcs * 2, ks, padding=padding)  # 原来是n_srcs*2
        self.deconv_s2 = nn.ConvTranspose2d(emb_dim, n_srcs * 2, ks, padding=padding)  # 原来是n_srcs*2
        self.linear = nn.Linear(144, 48)
        self.linear2 = nn.Linear(512, 65)
        self.EX_Mamba = ExBimamba(d_model=65,d_state=128)
        # self.Mamba1 = Mamba1(d_model=65, d_state=128)
        # self.Mamba2 = Mamba1(d_model=65, d_state=128)

        # self.complex_conv = ComplexConv2d(in_channels=1, out_channels=emb_dim, kernel_size=(3, 3),
        #                                           padding=(1, 1))
        # self.encoder_layer = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=32000,
        #                                            nhead=4,
        #                                            dim_feedforward=2048,
        #                                            dropout=0.1,
        #                                            activation=F.relu,
        #                                            norm_first=False),
        # num_layers = 1)

        # self.conv1d =  nn.Sequential(
        #     nn.Conv2d(2 * n_imics, emb_dim, ks, padding=padding),
        #     nn.GroupNorm(1, emb_dim, eps=eps),
        # )
        # self.conv1d_ = nn.Sequential(nn.Conv2d(2 * n_imics, 4 * n_imics, ks, padding=padding),
        #                                 nn.GroupNorm(2, 4, eps=eps))
        #
        #
        #
        # self.pool = nn.AdaptiveAvgPool1d(1)
        # self.pool1 = nn.AdaptiveAvgPool2d((1 ,1))
        # self.pool2 = nn.AdaptiveAvgPool2d((65, 1))
        # self.trunk = ResNet(BasicBlock, [1,1,1,1], relu_type="prelu")
        # self.trunk1 = ResNet(BasicBlock, [1,1,1,1], relu_type="prelu")
        # self.trunk2 = ResNet(BasicBlock, [2, 2, 2, 2], relu_type="prelu")
        # self.softmax = nn.Softmax(dim=1)
        # self.softmax1 = nn.Softmax(dim=1)
        # self.weight_x = nn.Parameter(torch.ones(1, 48))
        # self.weight_y = nn.Parameter(torch.ones(1, 48))
        # self.alpha = nn.Parameter(torch.tensor(0.5))

        # self.swish = nn.SiLU()
        self.conv_g = nn.Sequential(
            nn.Conv2d(2, emb_dim, (3, 3), padding=(1, 1)),
            nn.GroupNorm(1, emb_dim, eps=eps),
        )
        # self.conv_g_1 = nn.Sequential(
        #     nn.Conv2d(2, emb_dim, (3, 3), padding=(1, 1)),
        #     nn.GroupNorm(1, emb_dim, eps=eps),
        # )
        # self.conv_g_2 = nn.Sequential(
        #     nn.Conv2d(2, emb_dim, (3, 3), padding=(1, 1)),
        #     nn.GroupNorm(1, emb_dim, eps=eps),
        # )
        # self.mouth_norm = LayerNormalization4D(emb_dim, eps=eps)
        # self.mouth_norm1 = LayerNormalization4D(emb_dim, eps=eps)
        # self.mouth_norm2 = LayerNormalization4D(emb_dim, eps=eps)
        # self.mouth_pool = nn.AdaptiveAvgPool2d((50,1))
        # self.audio_pool = nn.AdaptiveAvgPool2d((50, 1))
        self.mamba_norm = RMSNorm(hidden_size=65)
        self.mamba_norm1 = RMSNorm(hidden_size=65)
        self.mamba_norm2 = RMSNorm(hidden_size=65)

        # self.audio_reconstructor = nn.Sequential(
        #     nn.Conv2d(in_channels=emb_dim, out_channels=emb_dim/2, kernel_size=3, padding=1),
        #     nn.ReLU(),
        #     nn.Conv2d(in_channels=emb_dim/2, out_channels=emb_dim, kernel_size=3, padding=1)
        # )
        #
        # self.visual_reconstructor = nn.Sequential(
        #     nn.Conv2d(in_channels=emb_dim, out_channels=emb_dim/2, kernel_size=3, padding=1),
        #     nn.ReLU(),
        #     nn.Conv2d(in_channels=emb_dim/2, out_channels=emb_dim, kernel_size=3, padding=1)
        # )

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

        for ii in range(3):
            batch, mouth= self.EX_Mamba(batch.view(n_batch*_, n_frames, n_freqs), mouth.view(n_batch*_, n_frames, n_freqs))
            batch = self.mamba_norm(batch)
            mouth = self.mamba_norm1(mouth)
            # batch_o = self.Mamba1(batch_o.view(n_batch*_, n_frames, n_freqs))
            # batch_o = self.mamba_norm2(batch_o)
            # mouth_ssm_ceshi = mouth.view(n_batch, _, n_frames, n_freqs)
            # batch_ceshi = batch.view(n_batch, _, n_frames, n_freqs)
            # batch_ceshi_o = torch.cat([batch_ceshi, batch_ceshi], dim=1)
            # batch_ceshi_v = self.audio_pool(batch_ceshi)
            # batch_o = batch_o.view(n_batch, _, n_frames, n_freqs)
        # mouth_o = self.Mamba2(mouth.view(n_batch*_, n_frames, n_freqs))
        # mouth_o = mouth_o.view(n_batch, _, n_frames, n_freqs)

        # batch = batch + batch1.view(n_batch*_, n_frames, n_freqs)
        #     batch_comp = torch.cat([batch_o,batch_o],dim=1)
        # batch_comp = self.audio_pool(batch_o.view(n_batch, _, n_frames, n_freqs))
        #     mouth_comp = self.mouth_pool(mouth.view(n_batch, _, n_frames, n_freqs))
        # mouth_ssm = mouth_ssm + mouth.view(n_batch*_, n_frames, n_freqs)
        # batch = torch.cat([batch.view(n_batch, _, n_frames, n_freqs),mouth_ssm.view(n_batch, _, n_frames, n_freqs)],dim=1)
            batch = batch.view(n_batch, _, n_frames, n_freqs)
            batch2 = batch
            batch2 = self.conv(batch2)
            batch_s1 = self.deconv_s1(batch2)  # [B, n_srcs*2, T, F]
            batch_s1 = batch_s1.view([n_batch, self.n_srcs, 2, n_frames, n_freqs])
            batch_s1 = new_complex_like(batch0, (batch_s1[:, :, 0], batch_s1[:, :, 1]))
            real_part1 = batch_s1[:, :1, :, :].real
            imag_part1 = batch_s1[:, :1, :, :].imag
            s1 = torch.cat((real_part1, imag_part1), dim=1)
            real_part2 = batch_s1[:, 1:, :, :].real
            imag_part2 = batch_s1[:, 1:, :, :].imag
            s2 = torch.cat((real_part2, imag_part2), dim=1)

            # 拼接实部和虚部，形成一个新的实数张量
            # 这里将实部和虚部沿着新的最后一个维度拼接
            batch_s1 = torch.cat((s1, s2), dim=1)
            mouth = mouth.view(n_batch, _, n_frames, n_freqs)
            # batch_yuan = self.conv_yuan(batch1)

        # assert not torch.any(torch.isnan(batch))

        # batch_ = self.conv_(batch).view(n_batch, n_frames, n_freqs,-1)
        # batch__ = self.conv(batch)  # [B, -1, T, F]
        #
        # mouth_ssm_ =self.conv1d(mouth_ssm).view(n_batch, n_frames, n_freqs,-1)
        # mouth_ssm__ = self.conv1d_(mouth_ssm)
        # batch = torch.mul(self.softmax(mouth_ssm__),batch__)
        # # batch = torch.mul(self.softmax1(mouth_ssm_),batch_)
        # # batch = torch.cat((mouth_ssm, mouth_ssm_, batch_), dim=-1)
        # gate_values = self.gate(torch.cat((mouth_ssm_,batch_), dim=-1))
        # consistent_weight, complementary_weight1= gate_values[:, :, :,:1], gate_values[:, :, :,1:2]
        # # #
        # # # 动态加权融合特征
        # mouth_ssm = consistent_weight * mouth_ssm_ + complementary_weight1 * batch_
        # mouth_ssm = mouth_ssm.view(n_batch, -1, n_frames, n_freqs)

        # batch_tri = batch.view(n_batch,  n_frames, n_freqs, -1)
        # mouth_ssm = self.conv1d(mouth_ssm)
        # batch = batch.view(n_batch,  n_frames, n_freqs, -1)
        # mouth_ssm = mouth_ssm.view(n_batch,  n_frames, n_freqs, -1)
        # # batch = torch.mul(batch,batch_A) + torch.mul(batch,batch_B)
        # weighted_x = batch * self.weight_x
        # weighted_y = mouth_ssm * self.weight_y
        #
        # # 增强正交性：通过减去投影部分，提取独特信息
        # orthogonal_x = weighted_x - (weighted_x * weighted_y).sum(dim=-1, keepdim=True) * weighted_y
        # orthogonal_y = weighted_y - (weighted_x * weighted_y).sum(dim=-1, keepdim=True) * weighted_x

        # # 拼接正交化后的模态特征
        # concat = torch.cat((orthogonal_x, orthogonal_y,batch_tri ), dim=-1)
        # batch = self.linear(concat)


        # 应用avgpool和maxpool
        #     input_c = batch_o
            input_p = mouth
            input_p = self.conv_g(input_p)
        #     input_c = self.conv_g_1(input_c)
            batch_s2 = self.deconv_s2(input_p)  # [B, n_srcs*2, T, F]
            batch_s2 = batch_s2.view([n_batch, self.n_srcs, 2, n_frames, n_freqs])
            batch_s2 = new_complex_like(batch0, (batch_s2[:, :, 0], batch_s2[:, :, 1]))
            real_part1 = batch_s2[:, :2, :, :].real
            imag_part1 = batch_s2[:, :2, :, :].imag
            s1 = torch.cat((real_part1, imag_part1), dim=1)
            real_part2 = batch_s2[:, 2:, :, :].real
            imag_part2 = batch_s2[:, 2:, :, :].imag
            s2 = torch.cat((real_part2, imag_part2), dim=1)

            batch_s2 = torch.cat((s1, s2), dim=1)

        batch_ceshi = batch2
        mouth_ceshi = input_p

        for ii in range(self.n_layers):
            batch2,batch_= self.blocks[ii](batch2,input_p)  # [B, -1, T, F]
            # batch_ = self.blocks1[ii](input_p)
            # reconstructed_audio = self.audio_reconstructor(batch)
            # reconstructed_visual = self.visual_reconstructor(batch)
        # batch = batch[:, :, :, :65]
        batch = self.deconv(batch2)  # [B, n_srcs*2, T, F]
        batch = batch.view([n_batch, self.n_srcs, 2, n_frames, n_freqs])
        batch = new_complex_like(batch0, (batch[:, :, 0], batch[:, :, 1]))

        batch = self.dec(batch.view(-1, n_frames, n_freqs), ilens)[0]  # [B, n_srcs, -1]
        batch = self.pad2(batch.view([n_batch, self.num_spk, -1]), n_samples)

        batch = batch * mix_std_  # reverse the RMS normalization
        batch = [batch[:, src] for src in range(self.num_spk)]
        batch = torch.stack(batch, dim=1)
        # batch_A = self.conv1d(batch[:,:1,:])
        # batch_A = self.pool(batch_A)
        # batch_B = self.conv1d(batch[:, 1:, :])
        # batch_B = self.pool(batch_B)
        # mouth_ssm = mouth_ssm.view(2,2,501,65)
        # mouth_ssm = mouth_ssm + mouth
        # mouth_ssm = mouth_ssm.permute(0, 1, 3, 2)
        # mouth_ssm_A =self.pool1(mouth_ssm[:,:1,:,:]).squeeze(1)
        # mouth_ssm_B = self.pool1(mouth_ssm[:, 1:, :, :]).squeeze(1)
        # batch_tri = batch_tri.view(2, 2, 501, 65)
        # batch_tri = batch_tri.permute(0, 1, 3, 2)
        # batch_tri_A = self.pool2(batch_tri[:, :1, :, :]).squeeze(1)
        # batch_tri_B = self.pool2(batch_tri[:, 1:, :, :]).squeeze(1)

        batch_ = self.deconv_s1(batch_)  # [B, n_srcs*2, T, F]
        batch_ = batch_.view([n_batch, self.n_srcs, 2, n_frames, n_freqs])
        batch_ = new_complex_like(batch0, (batch_[:, :, 0], batch_[:, :, 1]))

        batch_ = self.dec1(batch_.view(-1, n_frames, n_freqs), ilens)[0]  # [B, n_srcs, -1]
        batch_ = self.pad2(batch_.view([n_batch, self.num_spk, -1]), n_samples)

        batch_ = batch_ * mix_std_  # reverse the RMS normalization
        batch_ = [batch_[:, src] for src in range(self.num_spk)]
        batch_ = torch.stack(batch_, dim=1)



        return batch,batch_, batch_ceshi,mouth_ceshi,batch_s1,batch_s2

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
