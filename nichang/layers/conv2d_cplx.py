#!/usr/bin/env python
# coding=utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import (
    Module, Parameter, init,
    Conv2d, ConvTranspose2d, Linear, LSTM, GRU,
    BatchNorm1d, BatchNorm2d,
    PReLU,
)
from torch.nn.functional import (
    avg_pool2d,
    dropout,
    dropout2d,
    interpolate,
    max_pool2d,
    relu,
    sigmoid,
    tanh,
)
from complexPyTorch.complexFunctions import (
    complex_relu,
    complex_max_pool2d,
    complex_avg_pool2d,
    complex_dropout,
    complex_dropout2d,
)
from complexPyTorch.complexLayers import ComplexLinear,ComplexBatchNorm2d


def complex_sigmoid(inp):
    return sigmoid(inp.real).type(torch.complex64) + 1j * sigmoid(inp.imag).type(
        torch.complex64
    )


def complex_tanh(inp):
    return tanh(inp.real).type(torch.complex64) + 1j * tanh(inp.imag).type(
        torch.complex64
    )

EPSILON = torch.finfo(torch.float32).eps

class ComplexConv2d_Encoder(nn.Module):

    def __init__(
                    self,
                    in_channels,
                    out_channels,
                    kernel_size=(1,1),
                    stride=(1,1),
                    padding=(0,0),
                    dilation=(1,1),
                    groups=1,
                ):
        '''
            in_channels: real+imag
            out_channels: real+imag
        '''
        super(ComplexConv2d_Encoder, self).__init__()
        self.real_conv = nn.Conv2d(in_channels, out_channels, kernel_size = kernel_size, stride = stride, padding = padding, dilation = dilation, groups = groups)
        self.imag_conv = nn.Conv2d(in_channels, out_channels, kernel_size = kernel_size, stride = stride, padding = padding, dilation = dilation, groups = groups)

    def forward(self,inputs):
        # inputs : N C F T 2
        inputs_real, inputs_imag = inputs[...,0], inputs[...,1]
        out_real = self.real_conv(inputs_real) - self.imag_conv(inputs_imag)
        out_imag = self.real_conv(inputs_imag) + self.imag_conv(inputs_real)
        out_real = out_real[...,:inputs_real.shape[-1]]
        out_imag = out_imag[...,:inputs_imag.shape[-1]]
        return torch.cat([out_real, out_imag], 1)

class ComplexConv2d_Decoder(nn.Module):

    def __init__(
                    self,
                    in_channels,
                    out_channels,
                    kernel_size=(1,1),
                    stride=(1,1),
                    padding=(0,0),
                    output_padding=(0,0),
                    dilation=(1,1),
                    groups=1,
                ):
        '''
            in_channels: real+imag
            out_channels: real+imag
        '''
        super(ComplexConv2d_Decoder, self).__init__()
        self.real_conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size = kernel_size, stride = stride, padding = padding, output_padding = output_padding, dilation = dilation, groups = groups)
        self.imag_conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size = kernel_size, stride = stride, padding = padding, output_padding = output_padding, dilation = dilation, groups = groups)

    def forward(self,inputs):
        # inputs : N C F T 2
        inputs_real, inputs_imag = inputs[...,0], inputs[...,1]
        out_real = self.real_conv(inputs_real) - self.imag_conv(inputs_imag)
        out_imag = self.real_conv(inputs_imag) + self.imag_conv(inputs_real)
        out_real = out_real[...,:inputs_real.shape[-1]]
        out_imag = out_imag[...,:inputs_imag.shape[-1]]
        return torch.stack([out_real, out_imag], -1)


class ComplexLSTM(Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True,
                 batch_first=False, dropout=0, bidirectional=False):
        super().__init__()
        self.num_layer = num_layers
        self.hidden_size = hidden_size
        self.batch_dim = 0 if batch_first else 1
        self.bidirectional = bidirectional

        self.lstm_re = LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, bias=bias,
                            batch_first=batch_first, dropout=dropout,
                            bidirectional=bidirectional)
        self.lstm_im = LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, bias=bias,
                            batch_first=batch_first, dropout=dropout,
                            bidirectional=bidirectional)

    def forward(self, x):
        real, state_real = self._forward_real(x)
        imaginary, state_imag = self._forward_imaginary(x)

        output = torch.complex(real, imaginary)

        return output, (state_real, state_imag)

    def _forward_real(self, x):
        h_real, h_imag, c_real, c_imag = self._init_state(self._get_batch_size(x), x.is_cuda)
        real_real, (h_real, c_real) = self.lstm_re(x.real, (h_real, c_real))
        imag_imag, (h_imag, c_imag) = self.lstm_im(x.imag, (h_imag, c_imag))
        real = real_real - imag_imag
        return real, ((h_real, c_real), (h_imag, c_imag))

    def _forward_imaginary(self, x):
        h_real, h_imag, c_real, c_imag = self._init_state(self._get_batch_size(x), x.is_cuda)
        imag_real, (h_real, c_real) = self.lstm_re(x.imag, (h_real, c_real))
        real_imag, (h_imag, c_imag) = self.lstm_im(x.real, (h_imag, c_imag))
        imaginary = imag_real + real_imag

        return imaginary, ((h_real, c_real), (h_imag, c_imag))

    def _init_state(self, batch_size, to_gpu=False):
        dim_0 = 2 if self.bidirectional else 1
        dims = (dim_0, batch_size, self.hidden_size)

        h_real, h_imag, c_real, c_imag = [
            torch.zeros(dims) for i in range(4)]

        if to_gpu:
            h_real, h_imag, c_real, c_imag = [
                t.cuda() for t in [h_real, h_imag, c_real, c_imag]]

        return h_real, h_imag, c_real, c_imag

    def _get_batch_size(self, x):
        return x.size(self.batch_dim)
        h_new = (1 + complex_opposite(z)) * n + z * h  # element-wise multiplication

        return h_new


class ComplexReLU(Module):
    @staticmethod
    def forward(inp):
        return complex_relu(inp)


class ComplexSigmoid(Module):
    @staticmethod
    def forward(inp):
        return complex_sigmoid(inp)


class ComplexPReLU(Module):
    def __init__(self):
        super().__init__()
        self.r_prelu = PReLU()
        self.i_prelu = PReLU()

    @staticmethod
    def forward(self, inp):
        return self.r_prelu(inp.real) + 1j * self.i_prelu(inp.imag)


class ComplexTanh(Module):
    @staticmethod
    def forward(inp):
        return complex_tanh(inp)
class ComplexBNGRUCell(Module):
    """
    A BN-GRU cell for complex-valued inputs
    """

    def __init__(self, input_length=10, hidden_length=20):
        super().__init__()
        self.input_length = input_length
        self.hidden_length = hidden_length

        # reset gate components
        self.linear_reset_w1 = ComplexLinear(
            self.input_length, self.hidden_length)
        self.linear_reset_r1 = ComplexLinear(
            self.hidden_length, self.hidden_length)

        self.linear_reset_w2 = ComplexLinear(
            self.input_length, self.hidden_length)
        self.linear_reset_r2 = ComplexLinear(
            self.hidden_length, self.hidden_length)

        # update gate components
        self.linear_gate_w3 = ComplexLinear(
            self.input_length, self.hidden_length)
        self.linear_gate_r3 = ComplexLinear(
            self.hidden_length, self.hidden_length)

        self.activation_gate = ComplexSigmoid()
        self.activation_candidate = ComplexTanh()

        self.bn = ComplexBatchNorm2d(1)

    def reset_gate(self, x, h):
        x_1 = self.linear_reset_w1(x)
        h_1 = self.linear_reset_r1(h)
        # gate update
        reset = self.activation_gate(self.bn(x_1) + self.bn(h_1))
        return reset

    def update_gate(self, x, h):
        x_2 = self.linear_reset_w2(x)
        h_2 = self.linear_reset_r2(h)
        z = self.activation_gate(self.bn(h_2) + self.bn(x_2))
        return z

    def update_component(self, x, h, r):
        x_3 = self.linear_gate_w3(x)
        h_3 = r * self.bn(self.linear_gate_r3(h))  # element-wise multiplication
        gate_update = self.activation_candidate(self.bn(self.bn(x_3) + h_3))
        return gate_update

    def forward(self, x, h):
        # Equation 1. reset gate vector
        r = self.reset_gate(x, h)

        # Equation 2: the update gate - the shared update gate vector z
        z = self.update_gate(x, h)

        # Equation 3: The almost output component
        n = self.update_component(x, h, r)

        # Equation 4: the new hidden state


class ComplexGRU(Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True,
                 batch_first=False, dropout=0, bidirectional=False):
        super().__init__()

        self.gru_re = GRU(input_size=input_size, hidden_size=hidden_size,
                          num_layers=num_layers, bias=bias,
                          batch_first=batch_first, dropout=dropout,
                          bidirectional=bidirectional)
        self.gru_im = GRU(input_size=input_size, hidden_size=hidden_size,
                          num_layers=num_layers, bias=bias,
                          batch_first=batch_first, dropout=dropout,
                          bidirectional=bidirectional)

    def forward(self, x):
        real, state_real = self._forward_real(x)
        imaginary, state_imag = self._forward_imaginary(x)

        output = torch.complex(real, imaginary)
        state = torch.complex(state_real, state_imag)

        return output, state

    def forward(self, x):
        r2r_out = self.gru_re(x.real)[0]
        r2i_out = self.gru_im(x.real)[0]
        i2r_out = self.gru_re(x.imag)[0]
        i2i_out = self.gru_im(x.imag)[0]
        real_out = r2r_out - i2i_out
        imag_out = i2r_out + r2i_out

        return torch.complex(real_out, imag_out), None

    def _forward_real(self, x):
        real_real, h_real = self.gru_re(x.real)
        imag_imag, h_imag = self.gru_im(x.imag)
        real = real_real - imag_imag

        return real, torch.complex(h_real, h_imag)

    def _forward_imaginary(self, x):
        imag_real, h_real = self.gru_re(x.imag)
        real_imag, h_imag = self.gru_im(x.real)
        imaginary = imag_real + real_imag

        return imaginary, torch.complex(h_real, h_imag)
def apply_complex(fr, fi, input, dtype=torch.complex64):
    return (fr(input.real) - fi(input.imag)).type(dtype) \
         + 1j * (fr(input.imag) + fi(input.real)).type(dtype)

class ComplexConvTranspose1d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        output_padding=0,
        groups=1,
        bias=True,
        dilation=1,
        padding_mode="zeros",
    ):
        super(ComplexConvTranspose1d, self).__init__()

        self.conv_tran_r = nn.ConvTranspose1d(in_channels, out_channels, kernel_size, stride, padding,
                                              output_padding, groups, bias, dilation, padding_mode)
        self.conv_tran_i = nn.ConvTranspose1d(in_channels, out_channels, kernel_size, stride, padding,
                                              output_padding, groups, bias, dilation, padding_mode)

    def forward(self, inp):
        return apply_complex(self.conv_tran_r, self.conv_tran_i, inp)