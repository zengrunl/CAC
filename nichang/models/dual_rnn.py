import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import sys

sys.path.append('../')

import torch.nn.functional as F
from torch import nn
import torch
# from utils.util import check_parameters

import warnings

warnings.filterwarnings('ignore')


class GlobalLayerNorm(nn.Module):  # 归一化
    '''
       Calculate Global Layer Normalization
       dim: (int or list or torch.Size) –
          input shape from an expected input of size
       eps: a value added to the denominator for numerical stability.
       elementwise_affine: a boolean value that when set to True,
          this module has learnable per-element affine parameters
          initialized to ones (for weights) and zeros (for biases).
    '''

    def __init__(self, dim, shape, eps=1e-8, elementwise_affine=True):
        super(GlobalLayerNorm, self).__init__()
        self.dim = dim
        self.eps = eps
        self.elementwise_affine = elementwise_affine

        if self.elementwise_affine:
            if shape == 3:
                self.weight = nn.Parameter(torch.ones(self.dim, 1))
                self.bias = nn.Parameter(torch.zeros(self.dim, 1))
            if shape == 4:
                self.weight = nn.Parameter(torch.ones(self.dim, 1, 1))
                self.bias = nn.Parameter(torch.zeros(self.dim, 1, 1))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

    def forward(self, x):
        # x = N x C x K x S or N x C x L
        # N x 1 x 1
        # cln: mean,var N x 1 x K x S
        # gln: mean,var N x 1 x 1
        if x.dim() == 4:
            mean = torch.mean(x, (1, 2, 3), keepdim=True)
            var = torch.mean((x - mean) ** 2, (1, 2, 3), keepdim=True)
            if self.elementwise_affine:
                x = self.weight * (x - mean) / torch.sqrt(var + self.eps) + self.bias
            else:
                x = (x - mean) / torch.sqrt(var + self.eps)
        if x.dim() == 3:
            mean = torch.mean(x, (1, 2), keepdim=True)
            var = torch.mean((x - mean) ** 2, (1, 2), keepdim=True)
            if self.elementwise_affine:
                x = self.weight * (x - mean) / torch.sqrt(var + self.eps) + self.bias
            else:
                x = (x - mean) / torch.sqrt(var + self.eps)
        return x


class CumulativeLayerNorm(nn.LayerNorm):
    '''
       Calculate Cumulative Layer Normalization
       dim: you want to norm dim
       elementwise_affine: learnable per-element affine parameters
       这是一个使用PyTorch实现累积层归一化的类的初始化方法。其中，dim参数指定了归一化的维度，
       elementwise_affine参数指定是否使用可学习的仿射变换，eps参数指定了归一化时分母的平滑项。
    '''

    def __init__(self, dim, elementwise_affine=True):
        super(CumulativeLayerNorm, self).__init__(
            dim, elementwise_affine=elementwise_affine, eps=1e-8)

    def forward(self, x):
        # x: N x C x K x S or N x C x L
        # N x K x S x C
        if x.dim() == 4:
            x = x.permute(0, 2, 3, 1).contiguous()
            # N x K x S x C == only channel norm
            x = super().forward(x)
            # N x C x K x S
            x = x.permute(0, 3, 1, 2).contiguous()
        if x.dim() == 3:
            x = torch.transpose(x, 1, 2)
            # N x L x C == only channel norm
            x = super().forward(x)
            # N x C x L
            x = torch.transpose(x, 1, 2)
        return x


def select_norm(norm, dim, shape):
    if norm == 'gln':
        return GlobalLayerNorm(dim, shape, elementwise_affine=True)
    if norm == 'cln':
        return CumulativeLayerNorm(dim, elementwise_affine=True)
    if norm == 'ln':
        return nn.GroupNorm(1, dim, eps=1e-8)
    else:
        return nn.BatchNorm1d(dim)


class Encoder(nn.Module):
    '''
       Conv-Tasnet Encoder part
       kernel_size: the length of filters
       out_channels: the number of filters
    '''

    def __init__(self, kernel_size=2, out_channels=64):
        super(Encoder, self).__init__()
        self.conv1d = nn.Conv1d(in_channels=1, out_channels=out_channels,
                                kernel_size=kernel_size, stride=kernel_size//2, groups=1,
                                bias=False)  # Convolution 层的参数中有一个group参数，其意思是将对应的输入通道与输出通道数进行分组, 默认值为1, 也就是说默认输出输入的所有通道各为一组
        # 一维卷积

    def forward(self, x):
        """
          Input:
              x: [B, T], B is batch size, T is times
          Returns:
              x: [B, C, T_out]
              T_out is the number of time steps
              该编码器由一个一维卷积(Conv1d)层组成，将输入的一维信号转换为一组特征向量。该层具有以下参数：
in_channels: 输入通道数，此处为1

out_channels: 输出通道数，即特征向量的维度
        """
        # B x T -> B x 1 x T
        x = torch.unsqueeze(x, dim=1)
        # B x 1 x T -> B x C x T_out
        x = self.conv1d(x)
        x = F.relu(x)
        return x


'''
在进行一维卷积(Conv1d)操作时，输入的数据需要满足三个维度，分别表示batch size、通道数和时间步数。而原始的语音信号通常只有两个维度，
即时间步数和信号幅值，缺少了通道数的维度，因此需要通过unsqueeze函数在第二个维度上扩展为一个三维张量，以适配Conv1d层的输入要求。
具体来说，unsqueeze函数会在指定的维度上增加一个维度，例如在第二个维度上增加一个维度，可以将形状为(B, T)的输入张量扩展为形状为(B, 1, T)的三维张量，
其中1表示通道数，T表示时间步数。
然后将扩展后的三维张量作为输入传递到Conv1d层，进行卷积操作得到输出特征向量，其形状为(B, C, T_out)，
其中B为batch size，C为特征向量的维度，T_out为经过卷积计算后的时间步数。这样做的目的是为了将时间和特征信息进行分离，
并且在特征维度上进行卷积提取特征，从而更好地表达信号的语音特征。
'''


class Decoder(nn.ConvTranspose1d):
    '''
        Decoder of the TasNet
        This module can be seen as the gradient of Conv1d with respect to its input.
        It is also known as a fractionally-strided convolution
        or a deconvolution (although it is not an actual deconvolution operation).
    '''

    def __init__(self, *args, **kwargs):
        super(Decoder, self).__init__(*args, **kwargs)

    def forward(self, x):
        """
        x: [B, N, L]
        首先通过判断x的维度是否为3来确保输入符合要求。如果x的维度是2，则需要在第二个维度上插入一个新的维度，用来扩充张量。
        然后通过调用nn.ConvTranspose1d的forward函数对x进行逆卷积操作。接下来，通过判断x是否存在维度为1的数，
        来决定是否对其进行挤压操作。如果存在，则将维度为1的数挤压掉；否则不进行任何操作。最终返回重构后的语音信号。
        """
        if x.dim() not in [2, 3]:
            raise RuntimeError("{} accept 3/4D tensor as input".format(
                self.__name__))
        x = super().forward(x if x.dim() == 3 else torch.unsqueeze(x, 1))  # 插入新维度，用来插入新的维度扩充张量。

        if torch.squeeze(x).dim() == 1:
            x = torch.squeeze(x, dim=1)  # 挤掉维度为1的数
        else:
            x = torch.squeeze(x)
        return x

class SBTransformerBlock(nn.Module):
    def __init__(
        self,
        num_layers,
        d_model,  #N_encoder_output
        nhead,
        d_ffn=2048,
        dropout=0.0,
        activation="relu",
        norm_before=False,
    ):
        super(SBTransformerBlock, self).__init__()

        if activation == "relu":
            activation = nn.ReLU
            activation = F.relu
        elif activation == "gelu":
            activation = F.gelu
        else:
            raise ValueError("unknown activation")
        self.mdl = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(d_model=d_model,
                                                     nhead=nhead,
                                                     dim_feedforward=d_ffn,
                                                     dropout=dropout,
                                                     activation="relu",
                                                     ),
            num_layers=num_layers)



    def forward(self, x):
        """Returns the transformed output.

        Arguments
        ---------
        x : torch.Tensor
            Tensor shape [B, L, N],
            where, B = Batchsize,
                   L = time points
                   N = number of filters

        """
        x = x.permute(1, 0, 2)

        x = self.mdl(x)
        return x.permute(1, 0, 2)

class Dual_RNN_Block(nn.Module):
    '''
       Implementation of the intra-RNN and the inter-RNN
       input:
            in_channels: The number of expected features in the input x      想要的特征数
            out_channels: The number of features in the hidden state h        隐藏层特征数
            rnn_type: RNN, LSTM, GRU
            norm: gln = "Global Norm", cln = "Cumulative Norm", ln = "Layer Norm"
            dropout: If non-zero, introduces a Dropout layer on the outputs
                     of each LSTM layer except the last layer,
                     with dropout probability equal to dropout. Default: 0
            bidirectional: If True, becomes a bidirectional LSTM. Default: False
    '''

    def __init__(self, out_channels,
                 hidden_channels, rnn_type='RNN', norm='gln',
                 dropout=0, bidirectional=True, num_spks=2):
        super(Dual_RNN_Block, self).__init__()
        # RNN model
        # self.intra_rnn = getattr(nn, rnn_type)(
        #     out_channels, hidden_channels, 1, batch_first=True, dropout=dropout, bidirectional=bidirectional)
        self.inter_rnn = getattr(nn, rnn_type)(
            out_channels, hidden_channels, 1, batch_first=True, dropout=dropout, bidirectional=bidirectional)
        # Norm
        self.intra_norm = select_norm(norm, out_channels, 4)
        self.inter_norm = select_norm(norm, out_channels, 4)
        # self.conv1 = nn.Conv2d(in_channels=hidden_channels,out_channels = 1,kernel_size=1)
        # Linear
        self.IntraSeparator = SBTransformerBlock(num_layers=2,
                                            d_model=64,
                                            nhead=4,
                                            d_ffn=512,
                                            dropout=0,
                                            norm_before=False)
        self.intra_linear = nn.Linear(
            hidden_channels * 2 if bidirectional else hidden_channels, out_channels)  # 线性变换y=Ax+B
        self.inter_linear = nn.Linear(
            hidden_channels * 2 if bidirectional else hidden_channels, out_channels)

    def forward(self, x):
        '''
           x: [B, N, K, S]
           out: [Spks, B, N, K, S]

        '''
        B, N, K, S = x.shape

        # intra RNN
        # [BS, K, N]
        intra_rnn = x.permute(0, 3, 2, 1).contiguous().view(B * S, K, N)  # 交换N,S，再reshape
        # [BS, K, H]+

        # intra_rnn, _ = self.intra_rnn(intra_rnn)
        # # [BS, K, N]
        # intra_rnn = self.intra_linear(intra_rnn.contiguous().view(B * S * K, -1)).view(B * S, K, -1)
        intra_rnn = self.IntraSeparator(intra_rnn)
        # [B, S, K, N]
        intra_rnn = intra_rnn.view(B, S, K, N)  # view()的作用相当于numpy中的reshape，重新定义矩阵的形状。
        # [B, N, K, S]
        intra_rnn = intra_rnn.permute(0, 3, 2, 1).contiguous()
        intra_rnn = self.intra_norm(intra_rnn)

        # [B, N, K, S]
        intra_rnn = intra_rnn + x

        # inter RNN
        # [BK, S, N]
        inter_rnn = intra_rnn.permute(0, 2, 3, 1).contiguous().view(B * K, S, N)
        # [BK, S, H]
        inter_rnn, _ = self.inter_rnn(inter_rnn)
        # [BK, S, N]
        inter_rnn = self.inter_linear(inter_rnn.contiguous().view(B * S * K, -1)).view(B * K, S, -1)
        # [B, K, S, N]
        inter_rnn = inter_rnn.view(B, K, S, N)
        # [B, N, K, S]
        inter_rnn = inter_rnn.permute(0, 3, 1, 2).contiguous()
        inter_rnn = self.inter_norm(inter_rnn)
        # [B, N, K, S]
        out = inter_rnn + intra_rnn

        return out


class Dual_Path_RNN(nn.Module):
    '''
       Implementation of the Dual-Path-RNN model
       input:
            in_channels: The number of expected features in the input x
            out_channels: The number of features in the hidden state h
            rnn_type: RNN, LSTM, GRU
            norm: gln = "Global Norm", cln = "Cumulative Norm", ln = "Layer Norm"
            dropout: If non-zero, introduces a Dropout layer on the outputs
                     of each LSTM layer except the last layer,
                     with dropout probability equal to dropout. Default: 0
            bidirectional: If True, becomes a bidirectional LSTM. Default: False
            num_layers: number of Dual-Path-Block
            K:
            num_spks: the number of speakers
            首先对输入x进行归一化和卷积操作，再对其进行分段，然后通过多个Dual_RNN_Block模块进行RNN计算。
           接着经过PReLU激活和2维卷积操作，并将其重组为[Bspks, N, K, S]的形状。然后通过_over_add函数将其重组为[Bspks, N, L]的形状，
           并经过门控输出层和输出层得到最终的输出。最后将其重组为[B, spks, N, L]的形状，并经过激活函数处理后返回。
    '''

    def __init__(self, in_channels, out_channels, hidden_channels,
                 rnn_type='LSTM', norm='gln', dropout=0,
                 bidirectional=False, num_layers=4, K=200, num_spks=2):
        super(Dual_Path_RNN, self).__init__()
        self.K = K
        self.num_spks = num_spks
        self.num_layers = num_layers
        self.norm = select_norm(norm, in_channels, 3)
        self.conv1d = nn.Conv1d(in_channels, out_channels, 1, bias=False)

        self.dual_rnn = nn.ModuleList([])

        for i in range(num_layers):
            self.dual_rnn.append(Dual_RNN_Block(out_channels, hidden_channels,
                                                rnn_type=rnn_type, norm=norm, dropout=dropout,
                                                bidirectional=bidirectional))

        self.conv2d = nn.Conv2d(
            out_channels, out_channels * num_spks, kernel_size=1)
        self.end_conv1x1 = nn.Conv1d(out_channels, in_channels, 1, bias=False)
        self.prelu = nn.PReLU()
        self.activation = nn.ReLU()
        # gated output layer
        self.output = nn.Sequential(nn.Conv1d(out_channels, out_channels, 1),
                                    nn.Tanh()
                                    )
        self.output_gate = nn.Sequential(nn.Conv1d(out_channels, out_channels, 1),
                                         nn.Sigmoid()
                                         )

    def forward(self, x):
        '''
           x: [B, N, L]

        '''
        # [B, N, L]
        x = self.norm(x)
        # [B,  N, L]
        x = self.conv1d(x)
        # [B, N, K, S]
        x, gap = self._Segmentation(x, self.K)
        # [B, N*spks, K, S]
        for i in range(self.num_layers):
            x = self.dual_rnn[i](x)
        x = self.prelu(x)
        x = self.conv2d(x)
        # [B*spks, N, K, S]
        B, _, K, S = x.shape
        x = x.view(B * self.num_spks, -1, K, S)  # view中一个参数定为-1，代表动态调整这个维度上的元素个数，以保证元素的总数不变
        # [B*spks, N, L]
        x = self._over_add(x, gap)

        x = self.output(x) * self.output_gate(x)
        # [spks*B, N, L]
        x = self.end_conv1x1(x)
        # [B*spks, N, L] -> [B, spks, N, L]
        _, N, L = x.shape
        x = x.view(B, self.num_spks, N, L)
        x = self.activation(x)
        # [spks, B, N, L]
        x = x.transpose(0, 1)

        return x

    def _padding(self, input, K):
        '''
           padding the audio times
           K: chunks of length
           P: hop size
           input: [B, N, L]
           输入参数包括input表示输入的语音信号，K表示要处理的时间步数，即每次处理的语音片段长度。
           函数首先获取输入张量的形状(B, N, L)，其中B表示batch size，N表示特征向量的维度，L表示时间步数。
           然后计算出hop size P，即每次卷积的步长，以及需要补的零的数量gap。
           其中，gap的计算公式为K - (P + L % K) % K，保证了补零后的时间步数是K的倍数。
            接着，如果gap大于0，就创建一个形状为(B, N, gap)的全零张量pad，
            并将其与输入张量input在第三个维度上进行拼接，以进行补零操作。然后再创建一个形状为(B, N, P)的全零张量_pad，
            并将其分别拼接在输入张量input的前面和后面，以满足后续卷积操作的需要。最后，返回补零后的输入张量和补零的数量gap。
            这样做的目的是为了保证输入的语音信号长度可以被K整除，并且在进行卷积操作时，能够正确地处理边缘数据。
            P表示hop size，即每次卷积的步长。S表示分割后的小块数量
        '''
        B, N, L = input.shape
        P = K // 2
        gap = K - (P + L % K) % K
        if gap > 0:
            pad = torch.Tensor(torch.zeros(B, N, gap)).type(input.type()).cuda()
            input = torch.cat([input, pad], dim=2)

        _pad = torch.Tensor(torch.zeros(B, N, P)).type(input.type()).cuda()
        input = torch.cat([_pad, input, _pad], dim=2)

        return input, gap  # 程序看到是一个包含两个元素（由于被调用函数返回了两个值）的元组

    def _Segmentation(self, input, K):
        '''
           the segmentation stage splits
           K: chunks of length
           P: hop size
           input: [B, N, L]
           output: [B, N, K, S]
           接着，将输入张量input在第三个维度上进行切割，得到两个形状相同的张量input1和input2。其中，input1包含前L-K个时间步，input2包含后L-K个时间步。
           然后将input1和input2在第四个维度上进行拼接，得到一个形状为(B, N, S, 2K-P)的张量。其中，S表示分割后的小块数量。
           最后，通过view函数将拼接后的张量进行变形，得到形状为(B, N, K, S)的张量，
           并通过transpose函数将第三个维度和第四个维度进行交换，以满足后续卷积操作的需要。最终返回分割后的张量和补零的数量gap。
           这样做的目的是为了将长的语音信号分割成多个小块，以便在不损失重要信息的情况下进行卷积操作。
        '''
        B, N, L = input.shape
        P = K // 2
        input, gap = self._padding(input, K)
        # [B, N, K, S]
        input1 = input[:, :, :-P].contiguous().view(B, N, -1, K)
        input2 = input[:, :, P:].contiguous().view(B, N, -1, K)
        input = torch.cat([input1, input2], dim=3).view(
            B, N, -1, K).transpose(2, 3)

        return input.contiguous(), gap

    def _over_add(self, input, gap):
        '''
           Merge sequence
           input: [B, N, K, S]
           gap: padding length
           output: [B, N, L]
        '''
        B, N, K, S = input.shape
        P = K // 2
        # [B, N, S, K]
        input = input.transpose(2, 3).contiguous().view(B, N, -1, K * 2)

        input1 = input[:, :, :, :K].contiguous().view(B, N, -1)[:, :, P:]
        input2 = input[:, :, :, K:].contiguous().view(B, N, -1)[:, :, :-P]
        input = input1 + input2
        # [B, N, L]
        if gap > 0:
            input = input[:, :, :-gap]

        return input


class Dual_RNN_model(nn.Module):
    '''
       model of Dual Path RNN
       input:
            in_channels: The number of expected features in the input x
            out_channels: The number of features in the hidden state h
            hidden_channels: The hidden size of RNN
            kernel_size: Encoder and Decoder Kernel size
            rnn_type: RNN, LSTM, GRU
            norm: gln = "Global Norm", cln = "Cumulative Norm", ln = "Layer Norm"
            dropout: If non-zero, introduces a Dropout layer on the outputs
                     of each LSTM layer except the last layer,
                     with dropout probability equal to dropout. Default: 0
            bidirectional: If True, becomes a bidirectional LSTM. Default: False
            num_layers: number of Dual-Path-Block
            K: the length of chunk
            num_spks: the number of speakers
    '''

    def __init__(self, in_channels, out_channels, hidden_channels,
                 kernel_size=2, rnn_type='LSTM', norm='gln', dropout=0,
                 bidirectional=False, num_layers=4, K=200, num_spks=2):
        super(Dual_RNN_model, self).__init__()
        self.encoder = Encoder(kernel_size=kernel_size, out_channels=in_channels)
        self.separation = Dual_Path_RNN(in_channels, out_channels, hidden_channels,
                                        rnn_type=rnn_type, norm=norm, dropout=dropout,
                                        bidirectional=bidirectional, num_layers=num_layers, K=K, num_spks=num_spks)
        self.decoder = Decoder(in_channels=in_channels, out_channels=1, kernel_size=kernel_size,
                               stride=kernel_size//2, bias=False)
        self.num_spks = num_spks

    def forward(self, x, v):
        '''
           x: [B, L]
        '''
        # [B, N, L]
        e = self.encoder(x)
        e = e + v
        # [spks, B, N, L]
        s = self.separation(e)
        # [B, N, L] -> [B, L]
        out = [s[i] * e for i in range(self.num_spks)]  # s[i]是掩膜，分离出的语音
        audio = [self.decoder(out[i]) for i in range(self.num_spks)]
        audio = torch.stack(audio, dim=1)
        return audio




