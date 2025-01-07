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
# from ..layers.sparse_attention import SparseAttention
from ..layers.Mamba import ExBimamba,Mamba1
from mamba_ssm.ops.triton.layer_norm import RMSNorm
from ..layers.resnet import ResNet, BasicBlock
from ..layers.lstm_cell import MultiModalLSTMCell
from ..layers.CBAM import BasicBlock
from ..layers.tpa_lstm import TPA_LSTM_CrossAttention
from local_attention import LocalAttention
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
from speechbrain.inference.speaker import EncoderClassifier
classifier = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb")
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


class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.fc1 = nn.Linear(in_channels, in_channels // reduction, bias=False)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(in_channels // reduction, in_channels, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, x_vector):
        """
        x: [B*T, Q, C] -> [B, T, Q, C]
        x_vector: [B, C]
        """

        B = x_vector.size(0)
        T = x.size(0) // B
        Q = x.size(1)
        C = x.size(2)

        # Reshape x to [B, T, Q, C]
        x = x.view(B, T, Q, C)

        # Global Average Pooling over T and Q: [B, C]
        x_pooled = x.mean(dim=[1, 2])  # [B, C]

        # 使用 x-vector 和池化特征生成注意力权重
        combined = x_pooled + x_vector  # [B, C]
        attention = self.fc1(combined)  # [B, C//reduction]
        attention = self.relu(attention)
        attention = self.fc2(attention)  # [B, C]
        attention = self.sigmoid(attention)  # [B, C]

        # 扩展注意力权重以匹配 x 的形状
        attention = attention.view(B, 1, 1, C)  # [B, 1, 1, C]

        # 应用注意力权重
        x = x * attention  # 广播应用 [B, T, Q, C]

        # 重塑回 [B*T, Q, C]
        x = x.view(B * T, Q, C)

        return x
def generate_spectrogram_complex(audio, stft_frame, stft_hop, n_fft):
    spectro = librosa.core.stft(audio, hop_length=stft_hop, n_fft=n_fft, win_length=stft_frame, center=True)
    real = np.expand_dims(np.real(spectro), axis=0)
    imag = np.expand_dims(np.imag(spectro), axis=0)
    spectro_two_channel = np.concatenate((real, imag), axis=0)
    return spectro_two_channel
from mamba_ssm.modules.mamba_simple import Mamba

class Conv(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""
    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=0, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, groups=g, dilation=d, bias=False,padding=p)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """Perform transposed convolution of 2D data."""
        return self.act(self.conv(x))
class MatchNeck_Inner(nn.Module):
    def __init__(self, channels) -> None:
        super().__init__()

        self.gap = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            Conv(channels, channels)
        )
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.conv_hw = Conv(channels, channels, (3, 1))
        self.conv_pool_hw = Conv(channels, channels, 1)

    def forward(self, x):
        _, _, h, w = x.size()
        x_pool_h, x_pool_w, x_pool_ch = self.pool_h(x), self.pool_w(x).permute(0, 1, 3, 2), self.gap(x)
        x_pool_hw = torch.cat([x_pool_h, x_pool_w], dim=2)
        x_pool_h, x_pool_w = torch.split(x_pool_hw, [h, w], dim=2)
        x_pool_hw_weight = x_pool_hw.sigmoid()
        x_pool_h_weight, x_pool_w_weight = torch.split(x_pool_hw_weight, [h, w], dim=2)
        x_pool_h, x_pool_w = x_pool_h * x_pool_h_weight, x_pool_w * x_pool_w_weight
        x_pool_ch = x_pool_ch * torch.mean(x_pool_hw_weight, dim=2, keepdim=True)
        return x * x_pool_h.sigmoid() * x_pool_w.permute(0, 1, 3, 2).sigmoid() * x_pool_ch.sigmoid()


class MatchNeck(nn.Module):
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=1,p=(1, 1)):
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k, 1,p=p)
        self.cv2 = Conv(c_, c2, k, 1, g=g,p=p)
        self.cv3 = Conv(c1, c_, k, 1,p=p)
        self.cv4 = Conv(c_, c2, k, 1, g=g,p=p)
        self.add = shortcut and c1 == c2
        self.MN = MatchNeck_Inner(c2)

    def forward(self, x):
        return x + self.MN(self.cv2(self.cv1(x))) if self.add else self.MN(self.cv2(self.cv1(x)))


class MSFM(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, )
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(MatchNeck(self.c, self.c, shortcut, g, k=(3, 3), e=1.0) for _ in range(n))

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x):
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))
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
        self.intra_norm1 = LayerNormalization4D(emb_dim, eps=eps)
        self.intra_rnn = nn.LSTM(in_channels, hidden_channels, 1, batch_first=True, bidirectional=True
        )
        self.intra_rnn_ = nn.LSTM(in_channels, hidden_channels, 1, batch_first=True, bidirectional=True
                                 )

        self.intra_linear = nn.ConvTranspose1d(
            hidden_channels*4, emb_dim, emb_ks, stride=emb_hs
        )
        self.intra_linear_ = nn.ConvTranspose1d(
            hidden_channels * 4, emb_dim, emb_ks, stride=emb_hs
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
        self.inter_norm1 = LayerNormalization4D(emb_dim, eps=eps)
        self.inter_rnn = nn.LSTM(
            in_channels, hidden_channels, 1, batch_first=True, bidirectional=True
        )
        self.inter_rnn_ = nn.LSTM(
            in_channels, hidden_channels, 1, batch_first=True, bidirectional=True
        )

        self.inter_linear = nn.ConvTranspose1d(
            hidden_channels * 4, emb_dim, emb_ks, stride=emb_hs
        )
        self.inter_linear_ = nn.ConvTranspose1d(
            hidden_channels * 4, emb_dim, emb_ks, stride=emb_hs
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

            # self.add_module(
            #     "attn_conv_Q_%d1" % ii,
            #     nn.Sequential(
            #         nn.Conv2d(emb_dim, E, 1),
            #         get_layer(activation)(),
            #         LayerNormalization4DCF((E, n_freqs), eps=eps),
            #     ),
            # )
            # self.add_module(
            #     "attn_conv_K_%d1" % ii,
            #     nn.Sequential(
            #         nn.Conv2d(emb_dim, E, 1),
            #         get_layer(activation)(),
            #         LayerNormalization4DCF((E, n_freqs), eps=eps),
            #     ),
            # )
            # self.add_module(
            #     "attn_conv_V_%d1" % ii,
            #     nn.Sequential(
            #         nn.Conv2d(emb_dim, emb_dim // n_head, 1),
            #         get_layer(activation)(),
            #         LayerNormalization4DCF((emb_dim // n_head, n_freqs), eps=eps),
            #     ),
            )
        # self.add_module(
        #     "attn_concat_proj1",
        #     nn.Sequential(
        #         nn.Conv2d(emb_dim, emb_dim, 1),
        #         get_layer(activation)(),
        #         LayerNormalization4DCF((emb_dim, n_freqs), eps=eps),
        #     ),
        # )
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

        mid_channels =128
        embed_dims = 128
        self.embed_dims=embed_dims
        self.a = nn.Parameter(torch.tensor(0.5))
        self.b = nn.Parameter(torch.tensor(0.5))


        #
        # self.tcn = TCN(input_size=emb_dim,
        #                num_channels=[48,48,48],
        #                num_classes=500,
        #                tcn_options=tcn_options,
        #                dropout=tcn_options['dropout'],
        #                relu_type='prelu',
        #                dwpw=tcn_options['dwpw'], )


    # @profile(precision=5,stream=open('/root/data1/ceshi/memory_profiler_GridNetBlock.log','w+'))
    def forward(self, x, complement1):
        """GridNetBlock Forward.

        Args:
            x: [B, C, T, Q]
            out: [B, C, T, Q]
        """
        B, C, T, Q = x.shape  #2,48,501,65\
        input_ = x

        # complement1 = self.c1(complement1)
        complement11 = self.intra_norm1(complement1)  # [B, C, T, Q]
        complement11 = (
            complement11.transpose(1, 2).contiguous().view(B * T, C, Q)  ## 1002,48,65
        )  # [BT, C, Q]

        complement11 = F.unfold(
            complement11[..., None], (self.emb_ks, 1), stride=(self.emb_hs, 1)
        )  # [BT, C*emb_ks, -1]

        complement11 = complement11.transpose(1, 2)  # [BT, -1, C*emb_ks]
        # complement2 = self.c2(complement2)


        # T = math.ceil((old_T - self.emb_ks) / self.emb_hs) * self.emb_hs + self.emb_ks
        # Q = math.ceil((old_Q - self.emb_ks) / self.emb_hs) * self.emb_hs + self.emb_ks
        # x = F.pad(x, (0, Q - old_Q, 0, T - old_T))
        # intra RNN

        # mouth,z = self.gmu(complement1.view( B, T, Q, C),complement2.view(B, T, Q, C))
        # mouth = mouth.view(B, C, T, Q)

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


        # intra_rnn = self.attn(intra_rnn, intra_rnn, intra_rnn)
        # h0 = (
        #     complement1.transpose(1, 2).contiguous().view(B * T, C, Q) ## 1002,48,65
        # )  # [BT, C, Q]
        # h0 = self.conv_double(h0)
        # h0 = self.pool(h0)
        # h0 = h0.repeat(1, 1, B)
        # h0 = h0.view(B, B * T, -1)
        # c0 = h0

        #
        intra_rnn, _ = self.intra_rnn(intra_rnn)  # [BT, -1, H],(h0,c0)
        intra_rnn_, _ = self.intra_rnn_(complement11)  # [BT, -1, H],(h0,c0)
        intra_rnn = torch.cat([intra_rnn, intra_rnn_], dim=-1)
        # intra_rnn_flip = torch.flip(intra_rnn, [1])
        # complement11_flip = torch.flip(complement11, [1])
        # intra_rnn, _ = self.EX_Mamba_intra(intra_rnn,complement11)  # [BT, -1, H],(h0,c0)
        # intra_rnn_flip, _ = self.EX_Mamba_intra(intra_rnn_flip, complement11_flip)  # [BT, -1, H],(h0,c0)


        # intra_rnn = intra_rnn + complement1
        # complement1 = complement1.view(B,-1)
        # intra_rnn = self.channel(intra_rnn,complement1)


        # c1 = c1.transpose(1, 2)
        # c1 = self.intra_linear_cpx(c1)
        # c1 = c1.view([B, T, C, Q])
        # c1 = c1.transpose(1, 2).contiguous()

        intra_rnn = intra_rnn.transpose(1, 2)  # [BT, H, -1]
        complement11 = self.intra_linear_(intra_rnn)
        intra_rnn = self.intra_linear(intra_rnn)  # [BT, C, Q]
        #
        intra_rnn = intra_rnn.view([B, T, C, Q])
        intra_rnn = intra_rnn.transpose(1, 2).contiguous()  # [B, C, T, Q]

        #
        complement11 = complement11.view([B, T, C, Q])
        complement11 = complement11.transpose(1, 2).contiguous()  # [B, C, T, Q]
        complement11 = complement11 + complement1

        # gate = self.sigmoid(mouth)
        # gate1 = self.sigmoid1(intra_rnn)
        # intra_rnn = intra_rnn + intra_rnn * gate
        # mouth_1 = mouth + mouth * gate1
        # mouth_1 = self.dropout(mouth_1)
        # mouth_1 = mouth_1.view(B * Q, C, T)
        # mouth_1 = self.fc_inter(mouth_1)
        # mouth_1 = mouth_1.view(B, C, T, Q)

        # avgpool_result = self.fc_intra(self.avgpool_intra(mouth))
        # maxpool_result = self.fc_intra(self.maxpool_intra(mouth))
        # pool_sum = avgpool_result + maxpool_result
        # activated_pool_sum = self.sigmoid_intra(pool_sum)
        # residual_result = mouth * activated_pool_sum
        # mouth = mouth + residual_result
        # gate = self.swish(mouth)
        # gate = self.mouth_intranorm(gate)
        #
        # x = intra_rnn * gate
        # x = self.dropout(x)
        # gate = intra_rnn*complement1
        intra_rnn = intra_rnn + input_# [B, C, T, Q]+ complement2

        # intra_rnn = x + intra_rnn + input_# [B, C, T, Q]
        # mouth_1 = mouth + gate
        # avgpool_result = self.fc(self.avgpool(mouth_1))
        # maxpool_result = self.fc(self.maxpool(mouth_1))
        # pool_sum = avgpool_result + maxpool_result
        # activated_pool_sum = self.sigmoid(pool_sum)
        # residual_result = mouth_1 * activated_pool_sum
        # mouth_1 = mouth_1 + residual_result
        # mouth_1 = self.mouth_internorm(mouth_1)

        # inter RNN
        input_ = intra_rnn
        complement1 = complement11

        complement11 = self.inter_norm1(complement11)  # [B, C, T, Q]
        complement11 = (
            complement11.permute(0, 3, 1, 2).contiguous().view(B * Q, C, T)
        )  # [BT, C, Q]

        complement11 = F.unfold(
            complement11[..., None], (self.emb_ks, 1), stride=(self.emb_hs, 1)
        )  # [BT, C*emb_ks, -1]

        complement11 = complement11.transpose(1, 2)

        inter_rnn = self.inter_norm(input_)  # [B, C, T, F]
        inter_rnn = (
            inter_rnn.permute(0, 3, 1, 2).contiguous().view(B * Q, C, T) #65,48,501
        )  # [BF, C, T]
        inter_rnn = F.unfold(
            inter_rnn[..., None], (self.emb_ks, 1), stride=(self.emb_hs, 1)
        )  # [BF, C*emb_ks, -1]
        inter_rnn = inter_rnn.transpose(1, 2)  # [BF, -1, C*emb_ks]

        inter_rnn, _ = self.inter_rnn(inter_rnn)  # [BF, -1, H]
        inter_rnn_, _ = self.inter_rnn_(complement11)

        # complement2 = complement2.view(B, -1)
        # inter_rnn = self.channel_mouth(inter_rnn, complement2)
        inter_rnn = torch.cat([inter_rnn, inter_rnn_], dim=-1)
        inter_rnn = inter_rnn.transpose(1, 2)  # [BF, H, -1]
        complement11 = self.inter_linear_(inter_rnn)
        inter_rnn = self.inter_linear(inter_rnn)  # [BF, C, T]

        inter_rnn = inter_rnn.view([B, Q, C, T])
        inter_rnn = inter_rnn.permute(0, 2, 3, 1).contiguous()  # [B, C, T, Q]

        complement11 = complement11.view([B, Q, C, T])
        complement11 = complement11.permute(0, 2, 3, 1).contiguous()  # [B, C, T, Q]
        complement11 = complement11 + complement1
        # gate1 = inter_rnn*complement2
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
        # complement1 = mouth[:,:self.emb_dim,:,:]
        # complement2 = mouth[:,self.emb_dim:,:,:]


        inter_rnn = inter_rnn + input_  # [B, C, T, Q]+ x_inter+ complement1 + complement2

        # attention
        inter_rnn = inter_rnn[..., :T, :Q]
        batch = inter_rnn +  complement11
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
        complement11 = batch + complement11

        # x = self.x1(x,out)
        # out =torch.cat((out, complement1, complement2), dim=1)
        # out = self.fc(out)

        # x1 = x + out
        #
        # all_Q, all_K, all_V = [], [], []
        # for ii in range(self.n_head):
        #     all_Q += [self["attn_conv_Q_%d1" % ii](x1)]  # [B, C, T, Q]
        #     all_K += [self["attn_conv_K_%d1" % ii](x1)]  # [B, C, T, Q]
        #     all_V += [self["attn_conv_V_%d1" % ii](x1)]  # [B, C, T, Q]
        #
        # Q = torch.cat(all_Q, dim=0)  # [B', C, T, Q]
        # K = torch.cat(all_K, dim=0)  # [B', C, T, Q]
        # V = torch.cat(all_V, dim=0)  # [B', C, T, Q]
        #
        # Q = Q.transpose(1, 2)
        #
        # Q = Q.flatten(start_dim=2)  # [B', T, C*Q]
        # K = K.transpose(1, 2)
        # K = K.flatten(start_dim=2)  # [B', T, C*Q]
        # V = V.transpose(1, 2)  # [B', T, C, Q]
        # old_shape = V.shape
        # V = V.flatten(start_dim=2)  # [B', T, C*Q]
        # emb_dim = Q.shape[-1]
        #
        # attn_mat = torch.matmul(Q, K.transpose(1, 2)) / (emb_dim ** 0.5)  # [B', T, T]
        # attn_mat = F.softmax(attn_mat, dim=2)  # [B', T, T]
        # V = torch.matmul(attn_mat, V)  # [B', T, C*Q]
        # V = V.reshape(old_shape)  # [B', T, C, Q]
        # V = V.transpose(1, 2)  # [B', C, T, Q]
        # emb_dim = V.shape[1]
        #
        # x = V.view([self.n_head, B, emb_dim, T, -1])  # [n_head, B, C, T, Q])
        # x = x.transpose(0, 1)  # [B, n_head, C, T, Q])
        # x = x.contiguous().view(
        #     [B, self.n_head * emb_dim, T, -1]
        # )  # [B, C, T, Q])
        # x = self["attn_concat_proj1"](x)  # [B, C, T, Q])
        #
        # out1 = x + x1

        return out,complement11

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


class MultiSpeakerEmbeddingExtractor(nn.Module):
    def __init__(self, embedding_dim=384, num_speakers=2):
        super(MultiSpeakerEmbeddingExtractor, self).__init__()
        self.num_speakers = num_speakers
        self.conv1 = nn.Sequential(
            nn.Conv2d(2, embedding_dim, (3, 3), padding=(1, 1)),
            nn.GroupNorm(1, embedding_dim, eps=1.0e-5),
        )
        self.pool3 = nn.AdaptiveAvgPool2d((1, 1))
        self.linear_S = nn.Linear(64, 192)
        self.linear_f = nn.Linear(64, 192)


        # self.conv1 = nn.Conv2d(2, embedding_dim, kernel_size=3, padding=1)
        # self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 384, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(384)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(512)
        self.conv5 = nn.Conv2d(512, 1024, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(1024)

        self.pool = nn.MaxPool2d(2, 2)

        self.pool_ =nn.AdaptiveAvgPool2d((1 ,1))

    def forward(self, x):
        B =     x.size(0)
        x = self.conv1(x)
        x = self.pool3(x)
        x = x.squeeze()
        x = torch.stack([self.linear_S(x), self.linear_f(x)], dim=1)  # [batch, 2, 192]
        x = x.view(B, 2, -1)

        # x = F.relu(self.conv1(x)) # (batch_size, 64, 501, 65)
        # x = self.pool(x)  # (batch_size, 64, 250, 32)
        # x = F.relu(self.bn2(self.conv2(x)))  # (batch_size, 128, 250, 32)
        # x = self.pool(x)  # (batch_size, 128, 125, 16)
        # x = F.relu(self.bn3(self.conv3(x)))  # (batch_size, 256, 125, 16)
        # x = self.pool(x)  # (batch_size, 256, 62, 8)
        # x = F.relu(self.bn4(self.conv4(x)))  # (batch_size, 512, 62, 8)
        # x = self.pool(x)  # (batch_size, 512, 31, 4)
        # x = F.relu(self.bn5(self.conv5(x)))  # (batch_size, 512, 62, 8)
        # x = self.pool(x)  # (batch_size, 512, 31, 4)

        # x = self.pool_(x)
        # x = x.squeeze()
        #
        # embeddings = x.view(x.size(0), self.num_speakers, -1)  # (batch_size, num_speakers, embedding_dim)
        return x


import torch
import torch.nn as nn


class CrossAttentionFusion(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super(CrossAttentionFusion, self).__init__()
        self.cross_attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout)
        self.layer_norm = nn.LayerNorm(embed_dim )  # 用于融合后的残差连接
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x_vector, mixed_features):
        """
        Args:
            x_vector: [1002, 2, 384]
            mixed_features: [1002, 62, 384]
        Returns:
            fused_features: [1002, 62, 384]
        """
        # 转置以符合 nn.MultiheadAttention 的输入要求
        # nn.MultiheadAttention 期望的形状为 [Seq_len, Batch, Emb_dim]
        # 这里假设 Batch = 1002，查询序列长度 = 62，键/值序列长度 = 2
        # 所以，我们需要将 mixed_features 作为查询，x_vector 作为键和值

        # mixed_features: [1002, 62, 384] -> [62, 1002, 384]
        queries = mixed_features.permute(1, 0, 2)  # [62, 1002, 384]

        # x_vector: [1002, 2, 384] -> [2, 1002, 384]
        keys = x_vector.permute(1, 0, 2)  # [2, 1002, 384]
        values = x_vector.permute(1, 0, 2)  # [2, 1002, 384]

        # 应用交叉注意力
        attn_output, attn_weights = self.cross_attn(queries, keys, values)  # attn_output: [62, 1002, 384]

        # 转置回原始形状
        attn_output = attn_output.permute(1, 0, 2)  # [1002, 62, 384]

        # 将注意力输出与原始混合特征拼接
        fused = torch.cat((mixed_features, attn_output), dim=-1)  # [1002, 62, 768]

        # 通过全连接层恢复维度
        fused = self.ffn(fused)  # [1002, 62, 384]
        fused = self.dropout(fused)

        # 残差连接和层归一化
        fused = self.layer_norm(fused + mixed_features)  # [1002, 62, 384]

        return fused
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
            nn.Conv2d(4 * n_imics, emb_dim, ks, padding=padding),
            nn.GroupNorm(1, emb_dim, eps=eps),
        )

        self.pool_conv= nn.MaxPool2d(2, 2)
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
        self.deconv = nn.ConvTranspose2d(emb_dim * 2, n_srcs * 2, ks, padding=padding)  # 原来是n_srcs*2
        self.deconv_s1 = nn.ConvTranspose2d(emb_dim, n_srcs * 2, ks, padding=padding)
        self.deconv_s2 = nn.ConvTranspose2d(emb_dim, n_srcs * 2, ks, padding=padding)
        self.deconv_s3 = nn.ConvTranspose2d(emb_dim, n_srcs * 2, ks, padding=padding)
        self.deconv_mouth = nn.ConvTranspose2d(emb_dim, n_srcs, ks, padding=padding)

        self.linear_E = nn.Linear(192, 384)

        self.EX_Mamba = ExBimamba(d_model=n_freqs,d_state=128)
        self.Mamba1 = Mamba1(d_model=n_freqs, d_state=128)
        self.Mamba2 = Mamba1(d_model=n_freqs, d_state=128)

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
        self.embedding = MultiSpeakerEmbeddingExtractor(embedding_dim=emb_dim)
        self.embedding_1 = MultiSpeakerEmbeddingExtractor(embedding_dim=emb_dim)
        self.pool_E = nn.AdaptiveAvgPool1d(1)

        self.pool1 = nn.AdaptiveAvgPool2d((1 ,1))
        self.pool2 = nn.AdaptiveAvgPool2d((1, 1))

        self.softmax = nn.Softmax(dim=1)
        self.softmax1 = nn.Softmax(dim=1)
        self.swish = nn.SiLU()
        self.conv_g = nn.Sequential(
            nn.Conv2d(2, emb_dim ,(3, 3), padding=(1, 1)),
            nn.GroupNorm(1, emb_dim, eps=eps),
        )
        self.conv_g_1 = nn.Sequential(
            nn.Conv2d(2, emb_dim, (3, 3), padding=(1, 1)),
            nn.GroupNorm(1, emb_dim, eps=eps),
        )

        self.mouth_pool = nn.AdaptiveAvgPool2d((50,1))
        self.mouth_norm =  nn.GroupNorm(1, emb_dim, eps=eps)

    # @profile(precision=5,stream=open('/root/data1/ceshi/memory_profiler.log','w+'))
    def forward(self, input, mouth, embedding):
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
        batch0 = batch.transpose(1, 2)  #  [B, M, T, F]2，1，501，65
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
        # mouth_embedding = mouth.view(B * 2, 50, -1).permute(0, 2, 1)
        # mouth_embedding = self.pool_E(mouth_embedding)
        mouth = F.interpolate(mouth, size=(n_frames, n_freqs))  # 2,2,499,512
        # batch_comp = self.embedding_1(batch.view(n_batch, _, n_frames, n_freqs))
        batch_a =batch + mouth
        batch_a = self.conv_g(batch_a)
        batch_com = batch_a
        batch = torch.cat([batch, mouth], dim=1)
        batch = self.conv(batch.view(n_batch,4, n_frames, n_freqs))
        batch_con = batch



        for ii in range(self.n_layers):
            batch,batch_a = self.blocks[ii](batch,batch_a)  # [B, -1, T, F]

        # batch = batch[:, :, :, :65]


        # batch_s2 = self.deconv_s1(batch)  # [B, n_srcs*2, T, F]
        # batch_s2 = batch_s2.view([n_batch, self.n_srcs, 2, n_frames, n_freqs])
        # batch_s2 = new_complex_like(batch0, (batch_s2[:, :, 0], batch_s2[:, :, 1]))
        # real_part1 = batch_s2[:, :1, :, :].real
        # imag_part1 = batch_s2[:, :1, :, :].imag
        # s1 = torch.cat((real_part1, imag_part1), dim=1)
        # real_part2 = batch_s2[:, 1:, :, :].real
        # imag_part2 = batch_s2[:, 1:, :, :].imag
        # s2 = torch.cat((real_part2, imag_part2), dim=1)
        # batch_s2 = torch.cat((s1, s2), dim=1)
        batch = torch.cat([batch, batch_a], dim=1)


        batch = self.deconv(batch)  # [B, n_srcs*2, T, F]
        batch = batch.view([n_batch, self.n_srcs, 2, n_frames, n_freqs])
        batch = new_complex_like(batch0, (batch[:, :, 0], batch[:, :, 1])) #2.2.501.65
        # real_part1 = batch[:, :1, :, :].real
        # imag_part1 = batch[:, :1, :, :].imag
        # s1 = torch.cat((real_part1, imag_part1), dim=1)
        # real_part2 = batch[:, 1:, :, :].real
        # imag_part2 = batch[:, 1:, :, :].imag
        # s2 = torch.cat((real_part2, imag_part2), dim=1)
        # batch_s2 = torch.cat((s1, s2), dim=1)

        batch = self.dec(batch.view(-1, n_frames, n_freqs), ilens)[0]  # [B, n_srcs, -1] 4,32000
        batch = self.pad2(batch.view([n_batch, self.num_spk, -1]), n_samples)#2,2,32000
        batch = batch * mix_std_  # reverse the RMS normalization
        # s1 = generate_spectrogram_complex(np.array(batch[:, 0].cpu().detach().numpy()),128, stft_hop=64,n_fft=128)
        # s2 = generate_spectrogram_complex(np.array(batch[:, 1].cpu().detach().numpy()), 128, stft_hop=64, n_fft=128)
        # s1 = torch.from_numpy(s1).transpose(2, 3)
        # s2 = torch.from_numpy(s2).transpose(2, 3)
        # # embeddings_0 = classifier.encode_batch(batch[:, 0])  # 1,1,192
        # # embeddings_1 = classifier.encode_batch(batch[:, 1])  # 1,1,192
        # # embedding = torch.cat([embeddings_0, embeddings_1], dim=1)  # 2，1,192
        # # embedding = embedding.squeeze()  # 2,192
        #
        # batch_s2 = torch.cat([s1, s2], dim=1).cuda()
        batch = [batch[:, src] for src in range(self.num_spk)]
        batch = torch.stack(batch, dim=1)
        # batch_s1 = self.deconv_s1(input_c)  # [B, n_srcs*2, T, F]
        # batch_s1 = batch_s1.view([n_batch, self.n_srcs, 2, n_frames, n_freqs])
        # batch_s1 = new_complex_like(batch0, (batch_s1[:, :, 0], batch_s1[:, :, 1]))
        # # batch_s1 = self.dec1(batch_s1.view(-1, n_frames, n_freqs), ilens)[0]  # [B, n_srcs, -1] 4,32000
        # # batch_s1 = self.pad2(batch_s1.view([n_batch, self.num_spk, -1]), n_samples)  # 2,2,32000
        #  # reverse the RMS normalization
        # real_part1 = batch_s1[:, :1, :, :].real
        # imag_part1 = batch_s1[:, :1, :, :].imag
        # s1 = torch.cat((real_part1, imag_part1), dim=1)
        # real_part2 = batch_s1[:, 1:, :, :].real
        # imag_part2 = batch_s1[:, 1:, :, :].imag
        # s2 = torch.cat((real_part2, imag_part2), dim=1)
        # batch_s1 = torch.cat((s1, s2), dim=1)

        # batch_s3 = self.deconv_s3(input_p)  # [B, n_srcs*2, T, F]
        # batch_s3 = batch_s3.view([n_batch, self.n_srcs, 2, n_frames, n_freqs])
        # batch_s3 = new_complex_like(batch0, (batch_s3[:, :, 0], batch_s3[:, :, 1]))
        # real_part1 = batch_s3[:, :1, :, :].real
        # imag_part1 = batch_s3[:, :1, :, :].imag
        # s1 = torch.cat((real_part1, imag_part1), dim=1)
        # real_part2 = batch_s3[:, 1:, :, :].real
        # imag_part2 = batch_s3[:, 1:, :, :].imag
        # s2 = torch.cat((real_part2, imag_part2), dim=1)
        # batch_s3 = torch.cat((s1, s2), dim=1)

        return batch,batch_com,batch_con

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
