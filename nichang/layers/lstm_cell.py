import torch.nn as nn
import math
import torch
import torch.nn.functional as F


class small_cell(nn.Module):
    def __init__(self, input_size, hidden_size):
        """"Constructor of the class"""
        super(small_cell, self).__init__()
        self.seq = nn.Sequential(nn.Linear(input_size, input_size // 4),
                      nn.ReLU(inplace=True),
                      nn.Linear(input_size // 4, 4 * hidden_size))
    def forward(self,x):
        return self.seq(x)

class LSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size, nlayers, dropout = 0.1):
        """"Constructor of the class"""
        super(LSTMCell, self).__init__()

        self.nlayers = nlayers
        self.dropout = nn.Dropout(p=dropout)

        ih, hh = [], []
        for i in range(nlayers):
            if i==0:
                # ih.append(nn.Linear(input_size, 4 * hidden_size))
                ih.append(small_cell(input_size, hidden_size))
                # hh.append(nn.Linear(hidden_size, 4 * hidden_size))
                hh.append(small_cell(hidden_size, hidden_size))
            else:
                ih.append(nn.Linear(hidden_size, 4 * hidden_size))
                hh.append(nn.Linear(hidden_size, 4 * hidden_size))
        self.w_ih = nn.ModuleList(ih)
        self.w_hh = nn.ModuleList(hh)

    def forward(self, input, hidden):
        """"Defines the forward computation of the LSTMCell"""
        hy, cy = [], []
        for i in range(self.nlayers):
            hx, cx = hidden[0][i], hidden[1][i]
            gates = self.w_ih[i](input) + self.w_hh[i](hx)
            i_gate, f_gate, c_gate, o_gate = gates.chunk(4, 1)
            i_gate = torch.sigmoid(i_gate)
            f_gate = torch.sigmoid(f_gate)
            c_gate = torch.tanh(c_gate)
            o_gate = torch.sigmoid(o_gate)
            ncx = (f_gate * cx) + (i_gate * c_gate)
            # nhx = o_gate * torch.tanh(ncx)
            nhx = o_gate * torch.sigmoid(ncx)
            cy.append(ncx)
            hy.append(nhx)
            input = self.dropout(nhx)

        hy, cy = torch.stack(hy, 0), torch.stack(cy, 0)  # number of layer * batch * hidden
        return hy, cy

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class MultiModalLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size, consistency_size, complement_size):
        """Constructor of the class"""
        super(MultiModalLSTMCell, self).__init__()

        self.hidden_size = hidden_size

        # 音视混合特征处理 (small_cell)
        self.mix_cell = small_cell(input_size, hidden_size)
        self.w_hh = small_cell(input_size, hidden_size)

        # 输入门的线性层，融入音视一致性特征
        self.input_gate = nn.Sequential(
            nn.Linear(hidden_size + consistency_size + consistency_size, hidden_size),
            nn.Sigmoid()
        )

        # 遗忘门的线性层，融入音视互补性特征
        self.forget_gate = nn.Sequential(
            nn.Linear(hidden_size+hidden_size, hidden_size),
            nn.Sigmoid()
        )

        # 输出门的线性层，融入音视一致性特征
        self.output_gate = nn.Sequential(
            nn.Linear(hidden_size + consistency_size, hidden_size),
            nn.Sigmoid()
        )

        # 细胞状态的更新门，主要基于音视混合特征
        self.cell_gate = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh()
        )

        self.dropout = nn.Dropout(p=0.1)

    def forward(self, x_mix, x_consistency, x_complement, hidden):
        hx, cx = hidden

        # 处理音视混合特征
        mix_output = self.mix_cell(x_mix)+ self.w_hh(hx)
        i_gate, f_gate, c_gate, o_gate = mix_output.chunk(4, dim=-1)
        # 输入门 - 融入音视一致性特征
        input_gate = self.input_gate(torch.cat((i_gate, x_consistency,x_complement), dim=-1))

        # 遗忘门 - 融入音视互补性特征
        forget_gate = self.forget_gate(torch.cat((f_gate,x_consistency), dim=-1))

        # 细胞状态更新
        cell_gate = self.cell_gate(c_gate)
        ncx = (forget_gate * cx) + (input_gate * cell_gate)

        # 输出门 - 融入音视一致性特征
        output_gate = self.output_gate(torch.cat((o_gate, x_complement), dim=-1))
        nhx = output_gate * torch.tanh(ncx)
        nhx = self.dropout(nhx)

        return nhx, ncx
