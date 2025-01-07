import torch
import os
import subprocess

from torch import nn

os.environ['TORCH_CUDA_ARCH_LIST'] = '8.6'
os.environ["PATH"] = os.environ["PATH"] + ":/root/data1/anaconda/envs/plain_mamba/bin/"
os.environ["PATH"] = os.environ["PATH"] + ":/usr/local/cuda-11.6/bin"
os.environ["PATH"] = os.environ["PATH"] + ":/usr/local/cuda-11.6/lib64"
# os.environ['TORCH_NVCC_FLAGS'] = '-DNVCC_FLAGS=/root/data1/anaconda/envs/plain_mamba/bin/ninja'
if torch.cuda.is_available():
    os.environ["CUDA_HOME"] = "/usr/local/cuda"
from xlstm import (
    xLSTMBlockStack,
    xLSTMBlockStackConfig,
    mLSTMBlockConfig,
    mLSTMLayerConfig,
    sLSTMBlockConfig,
    sLSTMLayerConfig,
    FeedForwardConfig,
)
cfg = xLSTMBlockStackConfig(
    mlstm_block=mLSTMBlockConfig(
        mlstm=mLSTMLayerConfig(
            conv1d_kernel_size=4, qkv_proj_blocksize=4, num_heads=2
        ),

    ),
    context_length=4000,
    num_blocks=2,
    embedding_dim=256,
)
class mLSTMBlockStack(nn.Module):
    def __init__(self,num_blocks,embedding_dim,conv1d_kernel_size, qkv_proj_blocksize, num_heads,context_length):
        super(mLSTMBlockStack, self).__init__()

        self.num_blocks = num_blocks
        self.embedding_dim = embedding_dim
        self.qkv_proj_blocksize = qkv_proj_blocksize
        self.num_heads = num_heads
        self.conv1d_kernel_size = conv1d_kernel_size
        cfg = xLSTMBlockStackConfig(
            mlstm_block=mLSTMBlockConfig(
                mlstm=mLSTMLayerConfig(
                    conv1d_kernel_size=self.conv1d_kernel_size, qkv_proj_blocksize=self.qkv_proj_blocksize,
                    num_heads=self.num_heads
                ),

            ),
            context_length=context_length,
            num_blocks=self.num_blocks,
            embedding_dim=self.embedding_dim,

        )
        self.xlstm = xLSTMBlockStack(cfg)
    def forward(self, x, H, W):


        x = self.xlstm(x)
        return x

# if __name__ == "__main__":
#     xlstm_stack = xLSTMBlockStack(cfg)