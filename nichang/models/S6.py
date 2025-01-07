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
from espnet2.enh.layers.complex_utils import new_complex_like
from espnet2.enh.separator.abs_separator import AbsSeparator
from espnet2.torch_utils.get_layer_from_string import get_layer
from packaging.version import parse as V
from torch_complex.tensor import ComplexTensor
from espnet2.layers.stft import Stft
from ..layers.resnet_18_1D import ResNet, BasicBlock
from espnet2.enh.layers.complex_utils import is_torch_complex_tensor
from einops import rearrange
from ..layers.patch_embed import ConvPatchEmbed
from memory_profiler import profile
import gc

# device = torch.device('cuda'if torch.cuda.is_available() else'cpu')
device = torch.device('cuda:0')
is_torch_1_9_plus = V(torch.__version__) >= V("1.9.0")


class STFTEncoder(nn.Module):
    """STFT encoder for speech enhancement and separation"""

    def __init__(
            self,
            n_fft: int = 512,
            win_length: int = None,
            hop_length: int = 128,
            window="hann",
            center: bool = True,
            normalized: bool = False,
            onesided: bool = True,
            use_builtin_complex: bool = True,
    ):
        super().__init__()
        self.stft = Stft(
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            window=window,
            center=center,
            normalized=normalized,
            onesided=onesided,
        )

        self._output_dim = n_fft // 2 + 1 if onesided else n_fft
        self.use_builtin_complex = use_builtin_complex
        self.win_length = win_length if win_length else n_fft
        self.hop_length = hop_length
        self.window = window
        self.n_fft = n_fft
        self.center = center

    @property
    def output_dim(self) -> int:
        return self._output_dim

    def forward(self, input: torch.Tensor, ilens: torch.Tensor):
        """Forward.

        Args:
            input (torch.Tensor): mixed speech [Batch, sample]
            ilens (torch.Tensor): input lengths [Batch]
        """
        # for supporting half-precision training
        if input.dtype in (torch.float16, torch.bfloat16):
            spectrum, flens = self.stft(input.float(), ilens)
            spectrum = spectrum.to(dtype=input.dtype)
        else:
            spectrum, flens = self.stft(input, ilens)
        if is_torch_1_9_plus and self.use_builtin_complex:
            spectrum = torch.complex(spectrum[..., 0], spectrum[..., 1])
        else:
            spectrum = ComplexTensor(spectrum[..., 0], spectrum[..., 1])

        return spectrum, flens

    def _apply_window_func(self, input):
        B = input.shape[0]

        window_func = getattr(torch, f"{self.window}_window")
        window = window_func(self.win_length, dtype=input.dtype, device=input.device)
        n_pad_left = (self.n_fft - window.shape[0]) // 2
        n_pad_right = self.n_fft - window.shape[0] - n_pad_left

        windowed = input * window

        windowed = torch.cat(
            [torch.zeros(B, n_pad_left), windowed, torch.zeros(B, n_pad_right)], 1
        )
        return windowed

    def forward_streaming(self, input: torch.Tensor):
        """Forward.
        Args:
            input (torch.Tensor): mixed speech [Batch, frame_length]
        Return:
            B, 1, F
        """

        assert (
                input.dim() == 2
        ), "forward_streaming only support for single-channel input currently."

        windowed = self._apply_window_func(input)

        feature = (
            torch.fft.rfft(windowed) if self.stft.onesided else torch.fft.fft(windowed)
        )
        feature = feature.unsqueeze(1)
        if not (is_torch_1_9_plus and self.use_builtin_complex):
            feature = ComplexTensor(feature.real, feature.imag)

        return feature

    def streaming_frame(self, audio):
        """streaming_frame. It splits the continuous audio into frame-level
        audio chunks in the streaming *simulation*. It is noted that this
        function takes the entire long audio as input for a streaming simulation.
        You may refer to this function to manage your streaming input
        buffer in a real streaming application.

        Args:
            audio: (B, T)
        Returns:
            chunked: List [(B, frame_size),]
        """

        if self.center:
            pad_len = int(self.win_length // 2)
            signal_dim = audio.dim()
            extended_shape = [1] * (3 - signal_dim) + list(audio.size())
            # the default STFT pad mode is "reflect",
            # which is not configurable in STFT encoder,
            # so, here we just use "reflect mode"
            audio = torch.nn.functional.pad(
                audio.view(extended_shape), [pad_len, pad_len], "reflect"
            )
            audio = audio.view(audio.shape[-signal_dim:])

        _, audio_len = audio.shape

        n_frames = 1 + (audio_len - self.win_length) // self.hop_length
        strides = list(audio.stride())

        shape = list(audio.shape[:-1]) + [self.win_length, n_frames]
        strides = strides + [self.hop_length]

        return audio.as_strided(shape, strides, storage_offset=0).unbind(dim=-1)


class STFTDecoder(nn.Module):
    """STFT decoder for speech enhancement and separation"""

    def __init__(
            self,
            n_fft: int = 512,
            win_length: int = None,
            hop_length: int = 128,
            window="hann",
            center: bool = True,
            normalized: bool = False,
            onesided: bool = True,
    ):
        super().__init__()
        self.stft = Stft(
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            window=window,
            center=center,
            normalized=normalized,
            onesided=onesided,
        )

        self.win_length = win_length if win_length else n_fft
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.window = window
        self.center = center

    def forward(self, input: ComplexTensor, ilens: torch.Tensor):
        """Forward.

        Args:
            input (ComplexTensor): spectrum [Batch, T, (C,) F]
            ilens (torch.Tensor): input lengths [Batch]
        """
        if not isinstance(input, ComplexTensor) and (
                is_torch_1_9_plus and not torch.is_complex(input)
        ):
            raise TypeError("Only support complex tensors for stft decoder")

        bs = input.size(0)
        if input.dim() == 4:
            multi_channel = True
            # input: (Batch, T, C, F) -> (Batch * C, T, F)
            input = input.transpose(1, 2).reshape(-1, input.size(1), input.size(3))
        else:
            multi_channel = False

        # for supporting half-precision training
        if input.dtype in (torch.float16, torch.bfloat16):
            wav, wav_lens = self.stft.inverse(input.float(), ilens)
            wav = wav.to(dtype=input.dtype)
        elif (
                is_torch_complex_tensor(input)
                and hasattr(torch, "complex32")
                and input.dtype == torch.complex32
        ):
            wav, wav_lens = self.stft.inverse(input.cfloat(), ilens)
            wav = wav.to(dtype=input.dtype)
        else:
            wav, wav_lens = self.stft.inverse(input, ilens)

        if multi_channel:
            # wav: (Batch * C, Nsamples) -> (Batch, Nsamples, C)
            wav = wav.reshape(bs, -1, wav.size(1)).transpose(1, 2)

        return wav, wav_lens

    def _get_window_func(self):
        window_func = getattr(torch, f"{self.window}_window")
        window = window_func(self.win_length)
        n_pad_left = (self.n_fft - window.shape[0]) // 2
        n_pad_right = self.n_fft - window.shape[0] - n_pad_left
        return window

    def forward_streaming(self, input_frame: torch.Tensor):
        """Forward.
        Args:
            input (ComplexTensor): spectrum [Batch, 1, F]
            output: wavs [Batch, 1, self.win_length]
        """

        input_frame = input_frame.real + 1j * input_frame.imag
        output_wav = (
            torch.fft.irfft(input_frame)
            if self.stft.onesided
            else torch.fft.ifft(input_frame).real
        )

        output_wav = output_wav.squeeze(1)

        n_pad_left = (self.n_fft - self.win_length) // 2
        output_wav = output_wav[..., n_pad_left: n_pad_left + self.win_length]

        return output_wav * self._get_window_func()

    def streaming_merge(self, chunks, ilens=None):
        """streaming_merge. It merges the frame-level processed audio chunks
        in the streaming *simulation*. It is noted that, in real applications,
        the processed audio should be sent to the output channel frame by frame.
        You may refer to this function to manage your streaming output buffer.

        Args:
            chunks: List [(B, frame_size),]
            ilens: [B]
        Returns:
            merge_audio: [B, T]
        """

        frame_size = self.win_length
        hop_size = self.hop_length

        num_chunks = len(chunks)
        batch_size = chunks[0].shape[0]
        audio_len = int(hop_size * num_chunks + frame_size - hop_size)

        output = torch.zeros((batch_size, audio_len), dtype=chunks[0].dtype).to(
            chunks[0].device
        )

        for i, chunk in enumerate(chunks):
            output[:, i * hop_size: i * hop_size + frame_size] += chunk

        window_sq = self._get_window_func().pow(2)
        window_envelop = torch.zeros((batch_size, audio_len), dtype=chunks[0].dtype).to(
            chunks[0].device
        )
        for i in range(len(chunks)):
            window_envelop[:, i * hop_size: i * hop_size + frame_size] += window_sq
        output = output / window_envelop

        # We need to trim the front padding away if center.
        start = (frame_size // 2) if self.center else 0
        end = -(frame_size // 2) if ilens.max() is None else start + ilens.max()

        return output[..., start:end]


class cross_attention_layer(nn.TransformerEncoderLayer):
    '''
    跨模态层
    '''

    def __init__(self, d_model, nhead, norm_first=True, *args, **kwargs):
        super().__init__(d_model,
                         nhead,
                         norm_first=norm_first,
                         *args,
                         **kwargs)

    def forward(self, x, v, all):
        r"""Pass the input through the encoder layer.

        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """

        # see Fig. 1 of https://arxiv.org/pdf/2002.04745v1.pdf
        if self.norm_first:
            all = all + self._ca_block(self.norm1(x), self.norm1(v), self.norm1(all))
            x = self._ff_block(self.norm2(all))
        else:
            x = self.norm1(x + self._ca_block(x, v, all))
            x = self.norm2(x + self._ff_block(v))

        return x

    def _ca_block(self, x, v, all):
        x = self.self_attn(v, x, all,
                           attn_mask=None,
                           key_padding_mask=None,
                           need_weights=False)[0]
        return self.dropout1(x)

class TFGridNet_M(nn.Module):
    """Offline TFGridNet

    Reference:
    [1] Z.-Q. Wang, S. Cornell, S. Choi, Y. Lee, B.-Y. Kim, and S. Watanabe,
    "TF-GridNet: Integrating Full- and Sub-Band Modeling for Speech Separation",
    in arXiv preprint arXiv:2211.12433, 2022.
    [2] Z.-Q. Wang, S. Cornell, S. Choi, Y. Lee, B.-Y. Kim, and S. Watanabe,
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
        self.n_srcs = n_srcs
        self.n_layers = n_layers
        self.n_imics = n_imics
        assert n_fft % 2 == 0
        n_freqs = n_fft // 2 + 1
        self.ref_channel = ref_channel
        # self.time_enc = nn.Conv1d(in_channels=1, out_channels=50,padding=480,
        #                         kernel_size=256, stride=64, groups=1, bias=False)
        # self.trtime_enc = nn.ConvTranspose1d(in_channels=50, out_channels=1,padding=480,
        #                         kernel_size=256, stride=64)

        self.enc = STFTEncoder(
            n_fft, n_fft, stride, window=window, use_builtin_complex=use_builtin_complex
        )
        self.dec = STFTDecoder(n_fft, n_fft, stride, window=window)

        t_ksize = 3
        ks, padding = (t_ksize, 3), (t_ksize // 2, 1)
        self.conv = nn.Sequential(
            nn.Conv2d(2 * n_imics, emb_dim, ks, padding=padding),
            nn.GroupNorm(1, emb_dim, eps=eps),
        )
        self.conv1 = nn.Sequential(
            nn.Conv2d(2  * n_imics, emb_dim, ks, padding=padding),
            nn.GroupNorm(1, emb_dim, eps=eps),
        )
        self.blocks = nn.ModuleList([])
        self.blocks_refine = nn.ModuleList([])
        for _ in range(n_layers):
            self.blocks.append(
                GridNetBlock_1(
                    emb_dim,
                    emb_ks,
                    emb_hs,
                    n_freqs,
                    lstm_hidden_units,
                    n_head=attn_n_head,
                    approx_qk_dim=attn_approx_qk_dim,
                    activation=activation,
                    eps=eps,
                )
            )
        for _ in range(n_layers):
            self.blocks_refine.append(
                GridNetBlock(
                    emb_dim,
                    emb_ks,
                    emb_hs,
                    n_fft+2,
                    lstm_hidden_units,
                    n_head=attn_n_head,
                    approx_qk_dim=attn_approx_qk_dim,
                    activation=activation,
                    eps=eps,
                    n_layers = n_layers,
                )
            )
        self.deconv = nn.ConvTranspose2d(emb_dim, n_srcs, ks, padding=padding)  # 原来是n_srcs*2
        self.deconv1 = nn.ConvTranspose2d(emb_dim, n_srcs * 2, ks, padding=padding)
        self.linear = nn.Linear(130, 65)
        self.linear2 = nn.Linear(512, 257)
        self.linear3 = nn.Linear(512, 514)
        self.MambaBlock = MambaBlock(seq_len=247, d_model=257, state_size=6, m=2, device=device)
        self.MambaBlock_F = MambaBlock(seq_len=247, d_model=514, state_size=6, m=2, device=device)
    def forward(self, input, mouth, flow):

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
        N = input.shape[1]  # 获取第二个维度的大小，即样本长度
        ilens = torch.full((input.shape[0],), N, dtype=torch.long)  # 创建一个填充了 N 的张量，形状为 [B]
        n_samples = input.shape[1]
        if self.n_imics == 1:
            assert len(input.shape) == 2
            input = input[..., None]  # [B, N, M]
        mix_std_ = torch.std(input, dim=(1, 2), keepdim=True)  # [B, 1, 1]
        input = input / mix_std_  # RMS normalization
        # input = input.permute(0,2,1)
        # input = self.time_enc(input)
        # input = F.relu(input)
        # # input = torch.concat([input,mouth],dim=2)
        # # input =  mouth + input
        # input = input*mouth
        # input = self.trtime_enc(input)
        # input = input.permute(0, 2, 1)

        batch = self.enc(input, ilens)[0]  # [B, T, M, F],F=(fft/2)+1,T=32000/stride +1,2,499,1,65

        batch0 = batch.transpose(1, 2)  # [B, M, T, F]


        batch = torch.cat((batch0.real, batch0.imag), dim=1)  # 2,2,499,65stack之后不是complextensor了

        audio_feat_splits = []
        n_batch, _, n_frames, n_freqs = batch.shape

        mouth = F.interpolate(mouth, size=(247, 512))  # 2,2,499,512
        mouth = mouth.view(-1, 512)
        mouth = self.linear2(mouth)
        mouth = mouth.view(n_batch, _, n_frames,  n_freqs)

        batch = self.MambaBlock(batch, mouth)
        batch_ref = batch
        #
        batch = batch.view(n_batch, _, n_frames, n_freqs)

        assert not torch.any(torch.isnan(batch))
        # batch = self.SSM(batch)
        # batch = self.linear(batch)
        # batch = nn.functional.normalize(batch, p=2, dim=-1)
        # assert not torch.any(torch.isnan(batch))

        batch = self.conv(batch)  # [B, -1, T, F]

        avgpool = nn.AvgPool2d(kernel_size=(247, 1), stride=(1, 1))
        maxpool = nn.MaxPool2d(kernel_size=(247, 1), stride=(1, 1))

        # 应用avgpool和maxpool
        avgpool_result = avgpool(batch)
        maxpool_result = maxpool(batch)
        pool_sum = avgpool_result + maxpool_result
        activated_pool_sum = torch.sigmoid(pool_sum)
        residual_result = batch * activated_pool_sum
        batch = batch + residual_result

        # mouth = self.linear2(mouth)
        # mouth = mouth.view(n_batch, _, n_frames, n_freqs)

        # batch = batch.view(n_batch, _, n_frames, n_freqs)

        flow = F.interpolate(flow, size=(247, 512))
        flow = flow.view(-1, 512)
        flow = self.linear3(flow)
        flow = flow.view(n_batch, _, n_frames, 2 * n_freqs)

        # batch = torch.cat([batch, mouth], dim=3)
        #
        # batch = self.linear3(batch)
        # batch = batch.view(n_batch, _, n_frames, n_freqs)  # 2,2,499,65



        # batch = self.conv(batch)  # [B, -1, T, F]#2,48,499,65
        # mam = self.conv(mam)

        for ii in range(4):
            batch = self.blocks[ii](batch, mouth)  # [B, -1, T, F]2,48,499,65
        # batch = batch[:, :, :, :65]
        batch1 = self.deconv1(batch)  # 2,4,499,65
        batch = self.deconv(batch)  # [B, n_srcs*2, T, F]2,4,499,65/2,2,499,65

        batch_ref = torch.cat([batch_ref, batch], dim=3)

        batch_ref = self.MambaBlock_F(batch_ref, flow)
        batch_ref = self.conv1(batch_ref)
        avgpool_1 = nn.AvgPool2d(kernel_size=(247, 1), stride=(1, 1))
        maxpool_1 = nn.MaxPool2d(kernel_size=(247, 1), stride=(1, 1))

        # 应用avgpool和maxpool
        avgpool_result_f = avgpool_1(batch_ref)
        maxpool_result_f = maxpool_1(batch_ref)
        pool_sum_f = avgpool_result_f + maxpool_result_f
        activated_pool_sum_f = torch.sigmoid(pool_sum_f)
        residual_result_f = batch_ref * activated_pool_sum_f
        batch_ref = batch_ref + residual_result_f

        for iii in range(4):
            batch_refine = self.blocks_refine[iii](batch_ref, flow)  # [B, -1, T, F]

        batch_refine = self.deconv(batch_refine)

        batch = batch1.view(n_batch, _, n_frames, 2 * n_freqs) + batch_refine

        batch = batch.view([n_batch, self.n_srcs, 2, n_frames, n_freqs])
        batch = new_complex_like(batch0, (batch[:, :, 0], batch[:, :, 1]))
        #
        # esti_com = torch.view_as_complex(esti_com)
        batch = self.dec(batch.view(-1, n_frames, n_freqs), ilens)[0]  # [B, n_srcs, -1]

        # batch = self.dec(esti_com.view(-1, n_frames, n_freqs), ilens)[0]

        batch = self.pad2(batch.view([n_batch, self.num_spk, -1]), n_samples)

        batch = batch * mix_std_  # reverse the RMS normalization

        batch = [batch[:, src] for src in range(self.num_spk)]
        batch = torch.stack(batch, dim=1)
        return batch

    @property
    def num_spk(self):
        return self.n_srcs

    @staticmethod
    def pad2(input_tensor, target_len):
        input_tensor = torch.nn.functional.pad(
            input_tensor, (0, target_len - input_tensor.shape[-1])
        )
        return input_tensor
class ChannelMixer(nn.Module):
    "可修改"
    def __init__(self, channel_dim, hidden_dim):
        super(ChannelMixer, self).__init__()
        # 初始化这些参数为正态分布的随机值
        self.time_mix_k_raw = nn.Parameter(torch.randn(channel_dim))
        self.time_mix_r_raw = nn.Parameter(torch.randn(channel_dim))
        self.kw = nn.Linear(channel_dim, hidden_dim, bias=False)  # 假设hidden_dim为内部维度
        self.vw = nn.Linear(hidden_dim, channel_dim, bias=False)
        self.rw = nn.Linear(channel_dim, channel_dim, bias=False)

    # @profile(precision=5,stream=open('/root/data1/ceshi/memory_profiler_ChannelMixer.log','w+'))
    def forward(self, x, state):
        # log_graph_growth(f"inside channel_mixing {i} - start")
        # 使用sigmoid函数确保time_mix_k和time_mix_r的值在0到1之间
        time_mix_k = torch.sigmoid(self.time_mix_k_raw)
        time_mix_r = torch.sigmoid(self.time_mix_r_raw )
        # log_graph_growth(f"inside channel_mixing {i} - after time_mix")
        xk = x * time_mix_k + state * (1 - time_mix_k)
        xr = x * time_mix_r + state * (1 - time_mix_r)
        new_state = state * 0 + x  # 更新状态
        # log_graph_growth(f"inside channel_mixing {i} - after state update")
        r = torch.sigmoid(self.rw(xr))
        k = torch.square(torch.relu(self.kw(xk))) #对张量的每个元素进行平方运算
        mixed_output = r * self.vw(k)

        # log_graph_growth(f"inside channel_mixing {i} - end")
        return mixed_output, new_state

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
        self.intra_rnn = nn.LSTM(
            in_channels, hidden_channels, 1, batch_first=True, bidirectional=True
        )

        self.intra_linear = nn.ConvTranspose1d(
            hidden_channels * 2, emb_dim, emb_ks, stride=emb_hs
        )

        reduction = 5
        channel = 65
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.inter_norm = LayerNormalization4D(emb_dim, eps=eps)
        self.inter_rnn = nn.LSTM(
            in_channels, hidden_channels, 1, batch_first=True, bidirectional=True
        )
        self.inter_linear = nn.ConvTranspose1d(
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

        self.emb_dim = emb_dim
        self.emb_ks = emb_ks
        self.emb_hs = emb_hs
        self.n_head = n_head
        self.linear = nn.Linear(130, 65)
        # i = 0
        # self.intra_gla = GatedLinearAttention(d_model=48, expand_v=6, expand_k=2, num_heads=3,
        #                                       gate_fn='swish', layernorm_eps=1e-5,
        #                                       gate_logit_normalizer=16, gate_logit_multiplier=1,
        #                                       gate_low_rank_dim=16, mode='chunk', chunk_size=16,
        #                                       use_gk=True, use_gv=False, )
        # self.inter_gla = GatedLinearAttention(d_model=48, expand_v=2, expand_k=2, num_heads=6,
        #                                       gate_fn='swish', layernorm_eps=1e-5,
        #                                       gate_logit_normalizer=256, gate_logit_multiplier=1,
        #                                       gate_low_rank_dim=256, mode='chunk', chunk_size=16,
        #                                       use_gk=True, use_gv=False, )
        # self.inter_gla = GatedLinearAttention(d_model=48, expand_v=6, expand_k=2, num_heads=3,
        #                      gate_fn='swish', layernorm_eps=1e-5,
        #                      gate_logit_normalizer=32, gate_logit_multiplier=1,
        #                      gate_low_rank_dim=32, mode='chunk', chunk_size=16,
        #                      use_gk=True, use_gv=False, )
        #
        # self.linear = nn.Linear(130, 65)
        # state = [torch.zeros(65,48,501) for _ in range(5)]
        # self.kw = nn.Parameter(torch.Tensor(emb_dim, emb_dim))
        # self.vw = nn.Parameter(torch.Tensor(emb_dim, emb_dim))
        # self.rw = nn.Parameter(torch.Tensor(emb_dim, emb_dim))
        # # 使用合适的初始化策略初始化权重
        # nn.init.xavier_uniform_(self.kw)
        # nn.init.xavier_uniform_(self.vw)
        # nn.init.xavier_uniform_(self.rw)
        self.channel_mixing = ChannelMixer(channel_dim = 65, hidden_dim=65)

    # @profile(precision=5,stream=open('/root/data1/ceshi/memory_profiler_GridNetBlock.log','w+'))
    def forward(self, x, mouth):
        """GridNetBlock Forward.

        Args:
            x: [B, C, T, Q]
            out: [B, C, T, Q]
        """

        B, C, old_T, old_Q = x.shape
        T = math.ceil((old_T - self.emb_ks) / self.emb_hs) * self.emb_hs + self.emb_ks
        Q = math.ceil((old_Q - self.emb_ks) / self.emb_hs) * self.emb_hs + self.emb_ks
        x = F.pad(x, (0, Q - old_Q, 0, T - old_T))

        # intra RNN
        input_ = x
        intra_rnn = self.intra_norm(input_)  # [B, C, T, Q]
        intra_rnn = (
            intra_rnn.transpose(1, 2).contiguous().view(B * T, C, Q) ## 1002,48,65
        )  # [BT, C, Q]
        # intra_rnn = intra_rnn.transpose(1, 2)
        # intra_rnn_1 = self.intra_gla(torch.flip(intra_rnn, [1]))
        # intra_rnn = self.intra_gla(intra_rnn)
        # intra_rnn = intra_rnn.transpose(1, 2)
        # intra_rnn_1 = intra_rnn_1.transpose(1, 2)
        # intra_rnn = torch.cat([intra_rnn, intra_rnn_1], -1)
        # intra_rnn = self.linear(intra_rnn)

        intra_rnn = F.unfold(
            intra_rnn[..., None], (self.emb_ks, 1), stride=(self.emb_hs, 1)
        )  # [BT, C*emb_ks, -1]
        intra_rnn = intra_rnn.transpose(1, 2)  # [BT, -1, C*emb_ks]
        intra_rnn, _ = self.intra_rnn(intra_rnn)  # [BT, -1, H]
        intra_rnn = intra_rnn.transpose(1, 2)  # [BT, H, -1]
        intra_rnn = self.intra_linear(intra_rnn)  # [BT, C, Q]
        # intra_rnn = intra_rnn.transpose(1, 2)
        # intra_rnn = self.intra_gla(intra_rnn)
        # intra_rnn = intra_rnn.transpose(1, 2)

        intra_rnn = intra_rnn.view([B, T, C, Q])
        intra_rnn = intra_rnn.transpose(1, 2).contiguous()  # [B, C, T, Q]
        intra_rnn = intra_rnn + input_  # [B, C, T, Q]
        # inter RNN
        input_ = intra_rnn

        inter_rnn = self.inter_norm(input_)  # [B, C, T, F]
        inter_rnn = (
            inter_rnn.permute(0, 3, 1, 2).contiguous().view(B * Q, C, T) #65,48,501
        )  # [BF, C, T]
        # inter_rnn = inter_rnn.transpose(1, 2)
        # inter_rnn = inter_rnn.transpose(1, 2)
        inter_rnn = F.unfold(
            inter_rnn[..., None], (self.emb_ks, 1), stride=(self.emb_hs, 1)
        )  # [BF, C*emb_ks, -1]
        inter_rnn = inter_rnn.transpose(1, 2)  # [BF, -1, C*emb_ks]
        inter_rnn, _ = self.inter_rnn(inter_rnn)  # [BF, -1, H]
        inter_rnn = inter_rnn.transpose(1, 2)  # [BF, H, -1]
        inter_rnn = self.inter_linear(inter_rnn)  # [BF, C, T]
        # inter_rnn = inter_rnn.transpose(1, 2)
        # inter_rnn = self.inter_gla(inter_rnn)
        # inter_rnn = inter_rnn.transpose(1, 2)
        inter_rnn = inter_rnn.view([B, Q, C, T])
        inter_rnn = inter_rnn.permute(0, 2, 3, 1).contiguous()  # [B, C, T, Q]
        inter_rnn = inter_rnn + input_  # [B, C, T, Q]

        # attention
        inter_rnn = inter_rnn[..., :old_T, :old_Q]
        # 使用inplace操作更新inter_rnn，避免创建新的张量
        # for i in range(C):
        #     torch.autograd.set_detect_anomaly(True)
        #     output, state[:, i, :, :] = self.channel_mixing(inter_rnn[:, i, :, :], state[:, i, :, :], i)
        #     inter_rnn[:, i, :, :] = output + inter_rnn[:, i, :, :]  # inplace addition

        # new_inter_rnn = inter_rnn.clone()
        # state = torch.zeros_like(inter_rnn)
        #
        # # 使用 ChannelMixer 处理所有通道
        # output, new_state = self.channel_mixing(inter_rnn, state)
        #
        # # 更新 inter_rnn 和状态
        # new_inter_rnn = output
        # state = new_state
        # inter_rnn = new_inter_rnn + inter_rnn
        # new_state = [None] * inter_rnn.size(1)
        # state = [torch.zeros_like(inter_rnn[:, i, :, :]) for i in range(inter_rnn.size(1))]
        # for i in range(C):
        #     output, new_state_i = self.channel_mixing(inter_rnn[:, i, :, :], state, i)
        #     new_inter_rnn[:, i, :, :] = output
        #     new_state[i] = new_state_i.detach()
        #
        # # del new_state_i, output
        # # gc.collect()
        # # torch.cuda.empty_cache()
        # state = new_state
        # inter_rnn = new_inter_rnn + inter_rnn

        inter_rnn = inter_rnn.view(-1, old_Q)
        # inter_rnn = self.linear(inter_rnn)
        inter_rnn = inter_rnn.view(B, C, old_T,-1)
        batch = inter_rnn
        all_Q, all_K, all_V = [], [], []
        for ii in range(self.n_head):
            all_Q.append(self["attn_conv_Q_%d" % ii](batch))  # [B, C, T, Q]
            all_K.append(self["attn_conv_K_%d" % ii](batch))  # [B, C, T, Q]
            all_V.append(self["attn_conv_V_%d" % ii](batch))  # [B, C, T, Q]

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

        batch = V.view([self.n_head, B, emb_dim, old_T, -1])  # [n_head, B, C, T, Q])
        batch = batch.transpose(0, 1)  # [B, n_head, C, T, Q])
        batch = batch.contiguous().view(
            [B, self.n_head * emb_dim, old_T, -1]
        )  # [B, C, T, Q])
        batch = self["attn_concat_proj"](batch)  # [B, C, T, Q])

        out = batch + inter_rnn

        return out
class GridNetBlock_1(nn.Module):
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
    ):
        super().__init__()

        in_channels = emb_dim * emb_ks
        self.conv2d = nn.Sequential(
            nn.Conv2d(2 , 48, (3,3), padding=(1,1)),
            nn.GroupNorm(1, emb_dim, eps=eps),
        )
        self.trconv2d= nn.Sequential(
            nn.ConvTranspose2d(50 , 499, (3,3), padding=(1,1)),
            nn.GroupNorm(1, emb_dim, eps=eps),
        )
        self.intra_norm = LayerNormalization4D(emb_dim, eps=eps)
        self.intra_rnn = nn.LSTM(
            in_channels, hidden_channels, 1, batch_first=True, bidirectional=True
        )
        self.intra_gla = GatedLinearAttention( d_model= 128,expand_v = 2,expand_k= 2,
                                          int = 1,gate_fn= 'swish',layernorm_eps= 1e-5,
                                          gate_logit_normalizer= 32,gate_logit_multiplier = 1,
                                          gate_low_rank_dim = 32,mode = 'chunk',chunk_size = 32,
                                          use_gk = True, use_gv= False, )

        self.intra_linear = nn.ConvTranspose1d(
            in_channels , emb_dim, emb_ks, stride=emb_hs
        )
        self.inter_norm = LayerNormalization4D(emb_dim, eps=eps)

        self.inter_gla=GatedLinearAttention(  d_model= 128,expand_v = 2,expand_k= 2,
                                          int = 1,gate_fn= 'swish',layernorm_eps= 1e-5,
                                          gate_logit_normalizer= 32,gate_logit_multiplier = 1,
                                          gate_low_rank_dim = 32,mode = 'chunk',chunk_size = 32,
                                          use_gk = True, use_gv= False, )

        self.inter_linear = nn.ConvTranspose1d(
            in_channels , emb_dim, emb_ks, stride=emb_hs
        )

        self.emb_dim = emb_dim
        self.emb_ks = emb_ks
        self.emb_hs = emb_hs


    def forward(self, x, mouth):
        """GridNetBlock Forward.

        Args:
            x: [B, C, T, Q]
            out: [B, C, T, Q]
        """
        B, C, old_T, old_Q = x.shape

        T = math.ceil((old_T - self.emb_ks) / self.emb_hs) * self.emb_hs + self.emb_ks
        Q = math.ceil((old_Q - self.emb_ks) / self.emb_hs) * self.emb_hs + self.emb_ks
        x = F.pad(x, (0, Q - old_Q, 0, T - old_T))

        # intra RNN
        input_ = x
        intra_rnn = self.intra_norm(input_)  # [B, C, T, Q]
        intra_rnn = (
            intra_rnn.transpose(1, 2).contiguous().view(B * T, C, Q)
        )  # [BT, C, Q]
        intra_rnn = F.unfold(
            intra_rnn[..., None], (self.emb_ks, 1), stride=(self.emb_hs, 1)
        )  # [BT, C*emb_ks, -1]
        intra_rnn = intra_rnn.transpose(1, 2)  # [BT, -1, C*emb_ks]
        # intra_rnn = intra_rnn.transpose(1, 2)  # [BT, -1, C*emb_ks]
        intra_rnn = self.intra_gla(intra_rnn)  # [BT, -1, H]
        # print('intra_rnn', intra_rnn.shape)
        intra_rnn = intra_rnn.transpose(1, 2)  # [BT, H, -1]

        intra_rnn = self.intra_linear(intra_rnn)  # [BT, C, Q]
        intra_rnn = intra_rnn.view([B, T, C, Q])
        intra_rnn = intra_rnn.transpose(1, 2).contiguous()  # [B, C, T, Q]
        intra_rnn = intra_rnn + input_  # [B, C, T, Q]

        # inter RNN
        input_ = intra_rnn
        inter_rnn = self.inter_norm(input_)  # [B, C, T, F]
        inter_rnn = (
            inter_rnn.permute(0, 3, 1, 2).contiguous().view(B * Q, C, T)
        )  # [BF, C, T]
        inter_rnn = F.unfold(
            inter_rnn[..., None], (self.emb_ks, 1), stride=(self.emb_hs, 1)
        )  # [BF, C*emb_ks, -1]
        inter_rnn = inter_rnn.transpose(1, 2)  # [BF, -1, C*emb_ks]
        # print('inter_rnn', inter_rnn.shape)
        inter_rnn = self.inter_gla(inter_rnn)  # [BF, -1, H]
        inter_rnn = inter_rnn.transpose(1, 2)  # [BF, H, -1]
        inter_rnn = self.inter_linear(inter_rnn)  # [BF, C, T]
        inter_rnn = inter_rnn.view([B, Q, C, T])
        inter_rnn = inter_rnn.permute(0, 2, 3, 1).contiguous()  # [B, C, T, Q]
        inter_rnn = inter_rnn + input_  # [B, C, T, Q]

        inter_rnn = inter_rnn[..., :old_T, :old_Q]
        batch = inter_rnn


        return inter_rnn

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
import objgraph
def log_graph_growth(message=""):
    print(message)
    objgraph.show_growth()
class TF_S6(nn.Module):
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
        self.n_srcs = n_srcs
        self.n_layers = n_layers
        self.n_imics = n_imics
        assert n_fft % 2 == 0
        n_freqs = n_fft // 2 + 1
        self.ref_channel = ref_channel
        self.enc = STFTEncoder(
            n_fft, n_fft, stride, window=window, use_builtin_complex=use_builtin_complex
        )
        self.dec = STFTDecoder(n_fft, n_fft, stride, window=window)
        t_ksize = 3
        ks, padding = (t_ksize, 3), (t_ksize // 2, 1)
        self.conv = nn.Sequential(
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
        self.avgpool = nn.AvgPool2d(kernel_size=(501, 1), stride=(1, 1))
        self.maxpool = nn.MaxPool2d(kernel_size=(501, 1), stride=(1, 1))
        self.linear = nn.Linear(130, 65)
        self.linear1 = nn.Linear(1024, 512)
        self.linear2 = nn.Linear(512, 65)
        self.linear3 = nn.Linear(65, 130)
        # self.S6 = S6(seq_len=499, d_model=130, state_size=5, batch_size=2, m=2, device=device)
        self.MambaBlock = Mamba2(seq_len=501, d_model=65, state_size=128, m=2, device=device)
        # self.SSM = SSM(seq_len=499, d_model=65, state_size=5, batch_size=2, m=48, device=device)
        # self.SSM_3D = SSM_3D(seq_len=499, d_model=65, state_size=2, batch_size=2, device=device, channel=48)

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

        N = input.shape[1]  # 获取第二个维度的大小，即样本长度
        ilens = torch.full((input.shape[0],), N, dtype=torch.long)  # 创建一个填充了 N 的张量，形状为 [B]
        n_samples = input.shape[1]
        if self.n_imics == 1:
            assert len(input.shape) == 2
            input = input[..., None]  # [B, N, M]
        mix_std_ = torch.std(input, dim=(1, 2), keepdim=True)  # [B, 1, 1]
        input = input / mix_std_  # RMS normalization
        batch = self.enc(input, ilens)[0]  # [B, T, M, F],F=(fft/2)+1,T=32000/stride +1
        batch0 = batch.transpose(1, 2)  # [B, M, T, F]
        batch = torch.cat((batch0.real, batch0.imag), dim=1)  # [B, 2*M, T, F]
        batch1 = batch
        audio_feat_splits = []
        n_batch, _, n_frames, n_freqs = batch.shape
        mouth = F.interpolate(mouth, size=(501, 512))  # 2,2,499,512
        mouth = mouth.view(-1, 512)
        mouth = self.linear2(mouth)
        mouth = mouth.view(n_batch, _, n_frames,  n_freqs)
        # batch = self.conv(batch)

        batch = self.MambaBlock(batch, mouth)
        batch = batch + batch1
        #
        batch = batch.view(n_batch, _, n_frames, n_freqs)
        assert not torch.any(torch.isnan(batch))
        # batch = self.SSM(batch)
        # batch = self.linear(batch)
        # batch = nn.functional.normalize(batch, p=2, dim=-1)
        # assert not torch.any(torch.isnan(batch))

        batch = self.conv(batch)  # [B, -1, T, F]

        # 应用avgpool和maxpool
        avgpool_result = self.avgpool(batch)
        maxpool_result = self.maxpool(batch)
        pool_sum = avgpool_result + maxpool_result
        activated_pool_sum = torch.sigmoid(pool_sum)
        residual_result = batch  * activated_pool_sum
        batch = batch + residual_result
        for ii in range(self.n_layers):
            batch = self.blocks[ii](batch, mouth)  # [B, -1, T, F]
        # batch = batch[:, :, :, :65]
        batch = self.deconv(batch)  # [B, n_srcs*2, T, F]
        batch = batch.view([n_batch, self.n_srcs, 2, n_frames, n_freqs])
        batch = new_complex_like(batch0, (batch[:, :, 0], batch[:, :, 1]))

        batch = self.dec(batch.view(-1, n_frames, n_freqs), ilens)[0]  # [B, n_srcs, -1]

        batch = self.pad2(batch.view([n_batch, self.num_spk, -1]), n_samples)

        batch = batch * mix_std_  # reverse the RMS normalization
        # log_graph_growth("1")
        batch = [batch[:, src] for src in range(self.num_spk)]
        # log_graph_growth("2")
        batch = torch.stack(batch, dim=1)
        torch.cuda.empty_cache()
        # log_graph_growth("3")
        return batch

    @property
    def num_spk(self):
        return self.n_srcs

    @staticmethod
    def pad2(input_tensor, target_len):
        input_tensor = torch.nn.functional.pad(
            input_tensor, (0, target_len - input_tensor.shape[-1])
        )
        return input_tensor


class S6(nn.Module):
    def __init__(self, seq_len, d_model, state_size, device, batch_size, m):
        super(S6, self).__init__()
        # Linear transformations for inputs from different modalities
        self.fc1 = nn.Linear(d_model, d_model)
        self.fc2 = nn.Linear(d_model, state_size)
        self.fc3 = nn.Linear(d_model, state_size)
        # 设定一些超参数
        self.seq_len = seq_len
        self.d_model = d_model
        self.state_size = state_size
        self.A = nn.Parameter(F.normalize(torch.ones(d_model, state_size), p=2, dim=-1))
        # 参数初始化
        nn.init.xavier_uniform_(self.A)
        self.B = torch.zeros(batch_size, self.seq_len, m, self.state_size)
        self.C = torch.zeros(batch_size, self.seq_len, m, self.state_size)

        self.D = torch.zeros(batch_size, self.seq_len, m, self.d_model)
        self.F = torch.zeros(batch_size, self.seq_len, m, self.d_model)

        self.delta = torch.zeros(batch_size, self.seq_len, m, self.d_model,device=device)
        self.dA = torch.zeros(batch_size, self.seq_len, m, self.d_model, self.state_size,device=device)
        self.dB = torch.zeros(batch_size, self.seq_len, m, self.d_model, self.state_size,device=device)
        self.dE = torch.zeros(batch_size, self.seq_len, m, self.d_model, self.state_size,device=device)
        # 定义内部参数h和y
        self.h = torch.zeros(batch_size, self.seq_len, m, self.d_model, self.state_size,device=device)
        self.y = torch.zeros(batch_size, self.seq_len, m, self.d_model,device=device)

        self.norm = RMSNorm(d_model)

    def discretization(self):
        # 离散化函数定义介绍在Mamba论文中的28页
        self.dB = torch.einsum("blmd,blmn->blmdn", self.delta, self.B).to(device)
        self.dE = torch.einsum("blmd,blmn->blmdn", self.delta, self.E).to(device)
        # dA = torch.matrix_exp(A * delta)  # matrix_exp() only supports square matrix
        self.dA = torch.exp(torch.einsum("blmd,dn->blmdn", self.delta, self.A)).to(device)
        # print(f"self.dA.shape = {self.dA.shape}")     #print(f"self.dA.requires_grad = {self.dA.requires_grad}")
        return self.dA, self.dB, self.dE

    def forward(self, x, v):
        B, M, L, D = x.shape
        x = x.to(device)
        v = v.to(device)
        S = self.state_size
        x_reshaped = x.view(-1, D)
        v_reshaped = v.view(-1, D)
        x = x.transpose(1, 2).contiguous()
        v = v.transpose(1, 2).contiguous()
        self.E = self.fc2(v_reshaped).view(B, L, M, S)
        self.B = self.fc2(x_reshaped).view(B, L, M, S)
        self.C = self.fc3(x_reshaped).view(B, L, M, S)
        self.D = self.fc1(x_reshaped).view(B, L, M, D)
        self.F = self.fc1(v_reshaped).view(B, L, M, D)
        self.delta = F.softplus(self.fc1(x_reshaped)).view(B, L, M, D)
        # 离散化
        self.discretization()
        global current_batch_size
        current_batch_size = x.shape[0]
        # Assuming u_a and u_b are inputs from two different modalities at time t
        # State equation
        h_new = torch.einsum('blmdn,blmdn->blmdn', self.dA, self.h[:current_batch_size, ...]) + \
                rearrange(x, "b l m d -> b l m d 1") * self.dB + rearrange(v, "b l m d -> b l m d 1") * self.dE
        self.y = torch.einsum('blmn,blmdn->blmd', self.C, h_new)
        # 基于h_new更新h的信息
        global temp_buffer
        temp_buffer = h_new.detach().clone() if not self.h.requires_grad else h_new.clone()

        # x_act = F.silu(self.y)
        # x_residual = F.silu(self.D)
        # v_residual = F.silu(self.F)
        # # y = torch.cat([ x_act * x_residual,x_act * v_residual], dim=3)
        # y = x_act * x_residual + x_act * v_residual
        # y = F.silu(y)
        y = self.y * self.D + self.y * self.F + self.y
        # h_next = F.relu(self.fc_a(u_a) + self.fc_b(u_b) + self.h)  # Simple example of a state transition

        # Observation equation
        # y = self.fc_h(h_next)
        #
        # # Update the state
        # self.h = h_next

        return y

class MambaBlock(nn.Module):
    def __init__(self, seq_len, d_model, state_size, device, m):
        super(MambaBlock, self).__init__()
        self.inp_proj = nn.Linear(d_model, 2 * d_model)
        self.out_proj = nn.Linear(2 * d_model, d_model)

        # 残差连接
        self.D = nn.Linear(d_model, 2 * d_model)
        # 设置偏差属性
        self.out_proj.bias._no_weight_decay = True

        # 初始化偏差
        nn.init.constant_(self.out_proj.bias, 1.0)
        # 初始化S6模块
        self.S6 = S6(seq_len, 2 * d_model, state_size, device, batch_size=1, m=2)

        # 添加1D卷积
        self.conv = nn.Conv2d(m, m, (3, 3), padding=(1, 1))

        # 添加线性层
        self.conv_linear = nn.Linear(2 * d_model, 2 * d_model)

        # 正则化
        self.norm = RMSNorm(d_model)
        # 前向传播

    def forward(self, x, mouth):
        # 参考Mamba论文中的图3
        x = self.norm(x)
        # mouth = self.inp_proj(mouth)
        x_proj = self.inp_proj(x)
        # mouth = self.inp_proj(mouth)
        # h = torch.flip(x_proj,dims=(3,))
        # h_mouth = torch.flip(mouth,dims=(3,))
        # h_conv = self.conv(h)
        # h_conv_act = F.silu(h_conv)
        # h_ssm = self.S6(h_conv_act, h_mouth)
        # h_act = F.silu(h_ssm)  # Swish激活
        #
        # h_2 = torch.flip(x_proj, dims=(2,))
        # h_2_mouth = torch.flip(mouth, dims=(2,))
        # h_2_conv = self.conv(h_2)
        # h_2_conv_act = F.silu(h_2_conv)
        # h_2_ssm = self.S6(h_2_conv_act, h_2_mouth)
        # h_2_act = F.silu(h_2_ssm)  # Swish激活
        #
        # h_23 = torch.flip(x_proj, dims=(2,3))
        # h_23_mouth = torch.flip(mouth, dims=(2,3))
        # h_23_conv = self.conv(h_23)
        # h_23_conv_act = F.silu(h_23_conv)
        # h_23_ssm = self.S6(h_23_conv_act, h_23_mouth)
        # h_23_act = F.silu(h_23_ssm)  # Swish激活

        # 1D卷积操作
        x_conv = self.conv(x_proj)
        x_conv_act = F.silu(x_conv)  # Swish激活

        # 线性操作
        # x_conv_out = self.conv_linear(x_conv_act)  # S6模块操作
        x_ssm = self.SSM(x_conv_act)
        x_act = F.silu(x_ssm)  # Swish激活



        # 残差连接
        x_residual = F.silu(self.D(x))

        x_act = x_act.transpose(1, 2)
        x_combined = x_act * x_residual

        # h_act = h_act.transpose(1, 2)
        # h_combined = h_act * x_residual
        #
        # h_2_act = h_2_act.transpose(1, 2)
        # h_2_combined = h_2_act * x_residual
        #
        # h_23_act = h_23_act.transpose(1, 2)
        # h_23_combined = h_23_act * x_residual

        x_out = self.out_proj(x_combined) #*0.25 + h_combined*0.25 + h_2_combined*0.25 + h_23_combined*0.25

        return x_out


class Mamba2(nn.Module):
    def __init__(self, seq_len, d_model, state_size, device, m):
        super(Mamba2, self).__init__()
        self.inp_proj = nn.Linear(d_model, 2 * d_model)
        self.out_proj = nn.Linear(2 * d_model, d_model)

        # 残差连接
        self.D = nn.Linear(d_model, 2 * d_model)
        # 设置偏差属性
        self.out_proj.bias._no_weight_decay = True

        # 初始化偏差
        nn.init.constant_(self.out_proj.bias, 1.0)
        # 初始化S6模块
        self.SSM = SSM(seq_len, 2 * d_model, state_size, device, batch_size=1, m=2)

        # 添加1D卷积
        self.conv = nn.Conv2d(m, m, (3, 3), padding=(1, 1))

        # 添加线性层
        self.conv_linear = nn.Linear(2 * d_model, 2 * d_model)

        # 正则化
        self.norm = RMSNorm(d_model)
        # 前向传播

    def forward(self, x, mouth):
        # 参考Mamba论文中的图3
        x = self.norm(x)
        mouth = self.norm(mouth)
        # mouth = self.inp_proj(mouth)
        x_proj = self.inp_proj(x)
        mouth_proj = self.inp_proj(mouth)
        # mouth = self.inp_proj(mouth)
        # h = torch.flip(x_proj,dims=(3,))
        # h_mouth = torch.flip(mouth,dims=(3,))
        # h_conv = self.conv(h)
        # h_conv_act = F.silu(h_conv)
        # h_ssm = self.S6(h_conv_act, h_mouth)
        # h_act = F.silu(h_ssm)  # Swish激活
        #
        # h_2 = torch.flip(x_proj, dims=(2,))
        # h_2_mouth = torch.flip(mouth, dims=(2,))
        # h_2_conv = self.conv(h_2)
        # h_2_conv_act = F.silu(h_2_conv)
        # h_2_ssm = self.S6(h_2_conv_act, h_2_mouth)
        # h_2_act = F.silu(h_2_ssm)  # Swish激活
        #
        # h_23 = torch.flip(x_proj, dims=(2,3))
        # h_23_mouth = torch.flip(mouth, dims=(2,3))
        # h_23_conv = self.conv(h_23)
        # h_23_conv_act = F.silu(h_23_conv)
        # h_23_ssm = self.S6(h_23_conv_act, h_23_mouth)
        # h_23_act = F.silu(h_23_ssm)  # Swish激活

        # 1D卷积操作
        x_conv = self.conv(x_proj)
        mouth_conv = self.conv(mouth_proj)
        x_conv_act = F.silu(x_conv)
        mouth_conv_act = F.silu(mouth_conv)  # Swish激活

        # 线性操作
        # x_conv_out = self.conv_linear(x_conv_act)  # S6模块操作
        x_ssm = self.SSM(x_conv_act)
        mouth_ssm = self.SSM(mouth_conv_act)

        x_act = F.silu(x_ssm)  # Swish激活
        mouth_act = F.silu(mouth_ssm)

        # 残差连接
        x_residual = F.silu(self.D(x))


        # x_act = x_act.transpose(1, 2)
        # mouth_act = mouth_act.transpose(1, 2)
        x_combined = x_act * x_residual
        x_mouth = mouth_act * x_residual
        x_combined = x_combined + x_mouth

        # h_act = h_act.transpose(1, 2)
        # h_combined = h_act * x_residual
        #
        # h_2_act = h_2_act.transpose(1, 2)
        # h_2_combined = h_2_act * x_residual
        #
        # h_23_act = h_23_act.transpose(1, 2)
        # h_23_combined = h_23_act * x_residual

        x_out = self.out_proj(x_combined)  # *0.25 + h_combined*0.25 + h_2_combined*0.25 + h_23_combined*0.25

        return x_out




class SSM(nn.Module):
    def __init__(self, seq_len, d_model, state_size, device, batch_size, m):
        super().__init__()
        # Linear transformations for inputs from different modalities
        self.fc1 = nn.Linear(d_model, d_model, device=device)
        self.fc2 = nn.Linear(d_model, state_size, device=device)

        # 设定一些超参数
        self.seq_len = seq_len
        self.d_model = d_model
        self.state_size = state_size

        self.A = nn.Parameter(F.normalize(torch.ones(d_model, state_size, device=device), p=2, dim=-1))
        # 参数初始化
        nn.init.xavier_uniform_(self.A)
        self.B = torch.zeros(batch_size, m, self.seq_len, self.state_size, device=device)
        self.C = torch.zeros(batch_size, m, self.seq_len, self.state_size, device=device)

        self.D = torch.zeros(batch_size, m, self.seq_len, self.d_model, device=device)
        self.delta = torch.zeros(batch_size, m, self.seq_len, self.d_model, device=device)
        self.dA = torch.zeros(batch_size, m, self.seq_len, self.d_model, self.state_size, device=device)
        self.dB = torch.zeros(batch_size, m, self.seq_len, self.d_model, self.state_size, device=device)

        # 定义内部参数h和y
        self.h = torch.ones(batch_size, m, self.seq_len, self.d_model, self.state_size, device=device)
        self.y = torch.zeros(batch_size, m, self.seq_len, self.d_model, device=device)

        self.norm = RMSNorm(d_model, device=device)

    def discretization(self):
        # 离散化函数定义介绍在Mamba论文中的28页
        self.dB = torch.einsum("bmld,bmln->bmldn", self.delta, self.B).to(device)
        self.dA = torch.exp(torch.einsum("bmld,dn->bmldn", self.delta, self.A)).to(device)
        return self.dA, self.dB

    def forward(self, x):
        B, M, L, D = x.shape
        C = 48  # 定义输出通道数
        S = 128
        # x_reshaped = x.reshape(-1, D)
        self.B = self.fc2(x).view(B, M, L, S)

        self.C = self.fc2(x).view(B, M, L, S)  # 修改维度顺序
        #
        # self.D = self.fc1(x).view(B, M, L, D)

        self.delta = F.softplus(self.fc1(x)).view(B, M, L, D)
        self.discretization()
        # self.h = nn.init.xavier_uniform_(self.dB)
        assert not torch.any(torch.isnan(self.dA))
        assert not torch.any(torch.isnan(self.dB))
        assert not torch.any(torch.isnan(self.h))
        assert not torch.any(torch.isnan(x))

        # self.dB = nn.functional.normalize(self.dB, p=2, dim=-1)
        # self.dB = F.silu(self.dB)

        h_new = torch.einsum('bmldn,bmldn->bmldn', self.dA, self.h) + rearrange(x, "b m l d -> b m l d 1") * self.dB
        assert not torch.any(torch.isnan(h_new))
        assert not torch.any(torch.isnan(self.C))

        # h_new = nn.functional.normalize(h_new, p=2, dim=-1)
        # h_new = F.silu(h_new)
        # print('self.C', self.C)
        # print('h_new', h_new)
        self.y = torch.einsum('bmln,bmldn->bmld', self.C, h_new)
        # self.y = F.silu(self.y)
        # self.y = nn.functional.normalize(self.y, p=2, dim=-1)
        # print('self.y', self.y)
        assert not torch.any(torch.isnan(self.y))
        # 基于h_new更新h的信息
        global temp_buffer1
        temp_buffer1 = h_new.detach().clone() if not self.h.requires_grad else h_new.clone()
        # y = torch.cat([torch.einsum("bmld,bmld->bmld", x, self.D), self.y], dim=3)
        # y = torch.einsum("bmld,bmld->bmld", x, self.D) + self.y

        return self.y


def rms_norm(x, weight=None, eps=1e-5, dim=-1):
    square_sum = x.pow(2).sum(dim, keepdim=True)
    rms = torch.sqrt(square_sum.mean(-1, keepdim=True) + eps)
    output = x / rms
    if weight is not None:
        return output * weight
    return output


class RMSNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-5, weight=True, dtype=None, device=None):

        super().__init__()
        self.eps = eps
        if weight:
            self.weight = torch.nn.Parameter(torch.ones(normalized_shape, dtype=dtype, device=device))
        else:
            self.register_parameter("weight", None)

    def forward(self, x):
        return rms_norm(x.float(), self.weight, self.eps, dim=-1).to(dtype=x.dtype)


class SSM_3D(nn.Module):
    def __init__(self, seq_len, d_model, state_size, device, batch_size, channel):
        super().__init__()
        # Linear transformations for inputs from different modalities
        self.fc1 = nn.Linear(d_model, d_model, device=device)
        self.fc2 = nn.Linear(d_model, state_size, device=device)
        self.fc3 = nn.Linear(d_model, channel, device=device)
        # 设定一些超参数
        self.seq_len = seq_len
        self.d_model = d_model
        self.state_size = state_size
        self.output_channels = channel
        self.A = nn.Parameter(F.normalize(torch.ones(d_model, state_size, device=device), p=2, dim=-1))
        # 参数初始化
        nn.init.xavier_uniform_(self.A)
        self.B = torch.zeros(batch_size, self.seq_len, self.state_size, device=device)
        self.C = torch.zeros(batch_size, self.seq_len, self.state_size, device=device)

        self.D = torch.zeros(batch_size, self.seq_len, self.d_model, device=device)
        self.delta = torch.zeros(batch_size, self.seq_len, self.d_model, device=device)
        self.dA = torch.zeros(batch_size, self.seq_len, self.d_model, self.state_size, device=device)
        self.dB = torch.zeros(batch_size, self.seq_len, self.d_model, self.state_size, device=device)

        # 定义内部参数h和y
        self.h = torch.ones(batch_size, self.seq_len, self.d_model, self.state_size, device=device)
        self.y = torch.zeros(batch_size, self.seq_len, self.d_model, device=device)

        self.norm = RMSNorm(d_model, device=device)

    def discretization(self):
        # 离散化函数定义介绍在Mamba论文中的28页
        self.dB = torch.einsum("bld,bln->bldn", self.delta, self.B).to(device)
        self.dE = torch.einsum("bld,bln->bldn", self.delta, self.E).to(device)

        self.dA = torch.exp(torch.einsum("bld,dn->bldn", self.delta, self.A)).to(device)
        return self.dA, self.dB

    def forward(self, x, v):
        B, L, D = x.shape
        C = 48  # 定义输出通道数
        S = self.state_size
        # x_reshaped = x.reshape(-1, D)
        self.B = self.fc2(x).view(B, L, S)

        self.E = self.fc2(v).view(B, L, S)

        self.C = self.fc2(x).view(B, L, S)  # 修改维度顺序

        self.D = self.fc1(x).view(B, L, D)

        self.delta = F.softplus(self.fc1(x)).view(B, L, D)
        self.discretization()
        # self.h = nn.init.xavier_uniform_(self.dB)
        assert not torch.any(torch.isnan(self.dA))
        assert not torch.any(torch.isnan(self.dB))
        assert not torch.any(torch.isnan(self.h))
        assert not torch.any(torch.isnan(x))

        # self.dB = nn.functional.normalize(self.dB, p=2, dim=-1)
        # self.dB = F.silu(self.dB)

        h_new = torch.einsum('bldn,bldn->bldn', self.dA, self.h) + rearrange(x, "b  l d -> b l d 1") * self.dB \
                + rearrange(v, "b l d -> b l d 1") * self.dE

        assert not torch.any(torch.isnan(h_new))
        assert not torch.any(torch.isnan(self.C))

        # h_new = nn.functional.normalize(h_new, p=2, dim=-1)
        # h_new = F.silu(h_new)
        # print('self.C', self.C)
        # print('h_new', h_new)
        self.y = torch.einsum('bln,bldn->bld', self.C, h_new)
        self.y = F.silu(self.y)
        # self.y = nn.functional.normalize(self.y, p=2, dim=-1)
        # print('self.y', self.y)
        assert not torch.any(torch.isnan(self.y))
        # 基于h_new更新h的信息
        global temp_buffer1
        temp_buffer1 = h_new.detach().clone() if not self.h.requires_grad else h_new.clone()
        # y = torch.cat([torch.einsum("bmld,bmld->bmld", x, self.D), self.y], dim=3)
        y = torch.einsum("bld,bld->bld", x, self.D) + self.y

        return y

