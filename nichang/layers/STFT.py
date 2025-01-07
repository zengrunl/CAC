import os

os.environ['PATH'] += '/root/data1/anaconda/envs/nichang/bin/ninja'
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torch.nn.parameter import Parameter
import numpy as np
# from espnet2.enh.decoder.stft_decoder import STFTDecoder
# from espnet2.enh.encoder.stft_encoder import STFTEncoder
# from espnet2.enh.layers.complex_utils import new_complex_like
# from espnet2.enh.separator.abs_separator import AbsSeparator
# from espnet2.torch_utils.get_layer_from_string import get_layer
from packaging.version import parse as V
from torch_complex.tensor import ComplexTensor
# from espnet2.layers.stft import Stft
# from ..layers.resnet_18_1D import ResNet, BasicBlock
# from espnet2.enh.layers.complex_utils import is_torch_complex_tensor
from einops import rearrange
import difflib
import librosa

from mask import make_pad_mask

from memory_profiler import profile
import gc
# from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
# device = torch.device('cuda'if torch.cuda.is_available() else'cpu')
device = torch.device('cuda:0')
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
from abc import ABC, abstractmethod
class InversibleInterface(ABC):
    @abstractmethod
    def inverse(
        self, input: torch.Tensor, input_lengths: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # return output, output_lengths
        raise NotImplementedError
class Stft(torch.nn.Module, InversibleInterface):
    def __init__(
        self,
        n_fft: int = 512,
        win_length: int = None,
        hop_length: int = 128,
        window: Optional[str] = "hann",
        center: bool = True,
        normalized: bool = False,
        onesided: bool = True,
    ):
        # assert check_argument_types()
        super().__init__()
        self.n_fft = n_fft
        if win_length is None:
            self.win_length = n_fft
        else:
            self.win_length = win_length
        self.hop_length = hop_length
        self.center = center
        self.normalized = normalized
        self.onesided = onesided
        if window is not None and not hasattr(torch, f"{window}_window"):
            raise ValueError(f"{window} window is not implemented")
        self.window = window

    def extra_repr(self):
        return (
            f"n_fft={self.n_fft}, "
            f"win_length={self.win_length}, "
            f"hop_length={self.hop_length}, "
            f"center={self.center}, "
            f"normalized={self.normalized}, "
            f"onesided={self.onesided}"
        )

    def forward(
        self, input: torch.Tensor, ilens: torch.Tensor = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """STFT forward function.

        Args:
            input: (Batch, Nsamples) or (Batch, Nsample, Channels)
            ilens: (Batch)
        Returns:
            output: (Batch, Frames, Freq, 2) or (Batch, Frames, Channels, Freq, 2)

        """
        bs = input.size(0)
        if input.dim() == 3:
            multi_channel = True
            # input: (Batch, Nsample, Channels) -> (Batch * Channels, Nsample)
            input = input.transpose(1, 2).reshape(-1, input.size(1))
        else:
            multi_channel = False

        # NOTE(kamo):
        #   The default behaviour of torch.stft is compatible with librosa.stft
        #   about padding and scaling.
        #   Note that it's different from scipy.signal.stft

        # output: (Batch, Freq, Frames, 2=real_imag)
        # or (Batch, Channel, Freq, Frames, 2=real_imag)
        if self.window is not None:
            window_func = getattr(torch, f"{self.window}_window")
            window = window_func(
                self.win_length, dtype=input.dtype, device=input.device
            )
        else:
            window = None

        # For the compatibility of ARM devices, which do not support
        # torch.stft() due to the lack of MKL (on older pytorch versions),
        # there is an alternative replacement implementation with librosa.
        # Note: pytorch >= 1.10.0 now has native support for FFT and STFT
        # on all cpu targets including ARM.
        if input.is_cuda or torch.backends.mkl.is_available() or is_torch_1_10_plus:
            stft_kwargs = dict(
                n_fft=self.n_fft,
                win_length=self.win_length,
                hop_length=self.hop_length,
                center=self.center,
                window=window,
                normalized=self.normalized,
                onesided=self.onesided,
            )
            stft_kwargs["return_complex"] = True
            output = torch.stft(input, **stft_kwargs)
            output = torch.view_as_real(output)
        else:
            if self.training:
                raise NotImplementedError(
                    "stft is implemented with librosa on this device, which does not "
                    "support the training mode."
                )

            # use stft_kwargs to flexibly control different PyTorch versions' kwargs
            # note: librosa does not support a win_length that is < n_ftt
            # but the window can be manually padded (see below).
            stft_kwargs = dict(
                n_fft=self.n_fft,
                win_length=self.n_fft,
                hop_length=self.hop_length,
                center=self.center,
                window=window,
                pad_mode="reflect",
            )

            if window is not None:
                # pad the given window to n_fft
                n_pad_left = (self.n_fft - window.shape[0]) // 2
                n_pad_right = self.n_fft - window.shape[0] - n_pad_left
                stft_kwargs["window"] = torch.cat(
                    [torch.zeros(n_pad_left), window, torch.zeros(n_pad_right)], 0
                ).numpy()
            else:
                win_length = (
                    self.win_length if self.win_length is not None else self.n_fft
                )
                stft_kwargs["window"] = torch.ones(win_length)

            output = []
            # iterate over istances in a batch
            for i, instance in enumerate(input):
                stft = librosa.stft(input[i].numpy(), **stft_kwargs)
                output+=[torch.tensor(np.stack([stft.real, stft.imag], -1))]
            output = torch.stack(output, 0)
            if not self.onesided:
                len_conj = self.n_fft - output.shape[1]
                conj = output[:, 1 : 1 + len_conj].flip(1)
                conj[:, :, :, -1].data *= -1
                output = torch.cat([output, conj], 1)
            if self.normalized:
                output = output * (stft_kwargs["window"].shape[0] ** (-0.5))

        # output: (Batch, Freq, Frames, 2=real_imag)
        # -> (Batch, Frames, Freq, 2=real_imag)
        output = output.transpose(1, 2)
        if multi_channel:
            # output: (Batch * Channel, Frames, Freq, 2=real_imag)
            # -> (Batch, Frame, Channel, Freq, 2=real_imag)
            output = output.view(bs, -1, output.size(1), output.size(2), 2).transpose(
                1, 2
            )

        if ilens is not None:
            if self.center:
                pad = self.n_fft // 2
                ilens = ilens + 2 * pad

            olens = (
                torch.div(ilens - self.n_fft, self.hop_length, rounding_mode="trunc")
                + 1
            )
            output.masked_fill_(make_pad_mask(olens, output, 1), 0.0)
        else:
            olens = None

        return output, olens

    def inverse(
        self, input, ilens: torch.Tensor = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Inverse STFT.

        Args:
            input: Tensor(batch, T, F, 2) or ComplexTensor(batch, T, F)
            ilens: (batch,)
        Returns:
            wavs: (batch, samples)
            ilens: (batch,)
        """
        input = to_complex(input)

        if self.window is not None:
            window_func = getattr(torch, f"{self.window}_window")
            datatype = input.real.dtype
            window = window_func(self.win_length, dtype=datatype, device=input.device)
        else:
            window = None

        input = input.transpose(1, 2)

        wavs = torch.functional.istft(
            input,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=window,
            center=self.center,
            normalized=self.normalized,
            onesided=self.onesided,
            length=ilens.max() if ilens is not None else ilens,
            return_complex=False,
        )

        return wavs, ilens
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
    "d_model": None,
    "d_state": 16,
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
