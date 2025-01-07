import torch
# from mmcv.cnn.bricks.transformer import AdaptivePadding
from mmcv.cnn import (build_conv_layer, build_norm_layer)
# from mmcv.runner.base_module import BaseModule
# from mmcv.utils import to_2tuple
from timm.models.layers import to_2tuple, trunc_normal_, DropPath
import torch.nn.functional as F
from functools import partial
from torch import nn
import numpy as np
from xLSTM import mLSTMBlockStack

class ConvPatchEmbed(nn.Module):
    """Image to Patch Embedding.

    We use a conv layer to implement PatchEmbed.

    Args:
        in_channels (int): The num of input channels. Default: 3
        embed_dims (int): The dimensions of embedding. Default: 768
        conv_type (str): The type of convolution
            to generate patch embedding. Default: "Conv2d".
        kernel_size (int): The kernel_size of embedding conv. Default: 16.
        stride (int): The slide stride of embedding conv.
            Default: 16.
        padding (int | tuple | string): The padding length of
            embedding conv. When it is a string, it means the mode
            of adaptive padding, support "same" and "corner" now.
            Default: "corner".
        dilation (int): The dilation rate of embedding conv. Default: 1.
        bias (bool): Bias of embed conv. Default: True.
        norm_cfg (dict, optional): Config dict for normalization layer.
            Default: None.
        input_size (int | tuple | None): The size of input, which will be
            used to calculate the out size. Only works when `dynamic_size`
            is False. Default: None.
        init_cfg (`mmcv.ConfigDict`, optional): The Config for initialization.
            Default: None.
    """

    def __init__(self,
                 input_size =(501,65) ,
                 in_channels=3,
                 embed_dims=768,
                 mid_channels=64,
                 patch_size=64,
                 stride=16,
                 padding='corner',
                 dilation=1,
                 norm_cfg="Layer",
                 ):
        super(ConvPatchEmbed, self).__init__()

        self.embed_dims = embed_dims

        # self.stem = torch.nn.Sequential(
        #     nn.Conv2d(in_channels, mid_channels, kernel_size=(4,2), stride=(4,2), padding=0, bias=False),
        #     nn.BatchNorm2d(mid_channels),
        #     nn.ReLU(True))
        self.proj = nn.Conv2d(in_channels, mid_channels, kernel_size=patch_size, stride=stride, padding=0, bias=False)
        # self.proj_mouth = nn.Conv2d(2, 256, kernel_size=(4, 2), stride=(4, 2), padding=0, bias=False)
        # kernel_size = to_2tuple(patch_size//2)
        # stride = to_2tuple(stride)
        # dilation = to_2tuple(dilation)
        #
        # if isinstance(padding, str):
        #     self.adaptive_padding = AdaptivePadding(
        #         kernel_size=kernel_size,
        #         stride=stride,
        #         dilation=dilation,
        #         padding=padding)
        #     # disable the padding of conv
        #     padding = 0
        # else:
        #     self.adaptive_padding = None
        # padding = to_2tuple(padding)
        self.embed_dim = embed_dims

        self.fc1 = nn.Linear(130, 65)
        if norm_cfg is not None:
            self.norm = build_norm_layer(norm_cfg, embed_dims)[1]
        else:
            self.norm = None
        self.layernorm = nn.LayerNorm(embed_dims)

        # input_size = to_2tuple(499,65)
            # `init_out_size` would be used outside to
            # calculate the num_patches
            # e.g. when `use_abs_pos_embed` outside
        self.img_size = input_size
        h_out, w_out = [(self.img_size[i] + 2 * self.proj.padding[i] -
                         self.proj.dilation[i] *
                         (self.proj.kernel_size[i] - 1) - 1) //
                        self.proj.stride[i] + 1 for i in range(2)]
        self.init_out_size = (h_out, w_out)
        self.num_patches = h_out * w_out


    def forward(self, x):
        """
        Args:
            x (Tensor): Has shape (B, C, H, W). In most case, C is 3.
            B,C,T,Q
        Returns:
            tuple: Contains merged results and its spatial shape.

            - x (Tensor): Has shape (B, out_h * out_w, embed_dims)
            - out_size (tuple[int]): Spatial shape of x, arrange as
              (out_h, out_w).
        """
        B, C, H, W = x.shape
        # mouth = self.fc1(mouth)
        x = self.proj(x)  #扫描分块，分成192块 ，2,192,252,71 ,2,768,7719 ,2,128,125,32

        # mouth = self.proj_mouth(mouth)
        out_size = (x.shape[2], x.shape[3])
        x = x.permute(0, 2, 3, 1)
        x = x.reshape(B, -1, self.embed_dim)
        x = self.layernorm(x)
        # mouth = mouth.permute(0, 2, 3, 1)
        # mouth = mouth.reshape(B, -1, self.embed_dims*2)
        # mouth = self.layernorm(mouth)
        # x_pos_embed = get_2d_sincos_pos_embed(self.embed_dim, out_size,False)
        # x_pos_embed = x_pos_embed[None, :, :]
        # x_pos_embed = x_pos_embed.repeat(B, 0)
        # x_pos_embed = torch.tensor(x_pos_embed).cuda()

        x = x
        return x , out_size , None
def get_1d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w)  # here w goes first
    grid = np.stack(grid, axis=0)
    grid = grid.reshape([1, 1, grid_size])
    pos_embed = get_1d_sincos_pos_embed_from_grid(embed_dim, grid[0])
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed

# --------------------------------------------------------
# 2D sine-cosine position embedding
# References:
# Transformer: https://github.com/tensorflow/models/blob/master/official/nlp/transformer/model_utils.py
# MoCo v3: https://github.com/facebookresearch/moco-v3
# --------------------------------------------------------
def get_2d_sincos_pos_embed(embed_dim, grid_sizes, cls_token=False):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    gH, gW = grid_sizes
    grid_h = np.arange(gH, dtype=np.float32)
    grid_w = np.arange(gW, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)
    grid = grid.reshape([2, 1, gH, gW])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb
def get_sinusoid_encoding_table(n_position, d_hid, cls_token=False):
    ''' Sinusoid position encoding table '''
    # TODO: make it with torch instead of numpy
    def get_position_angle_vec(position):
        return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2]) # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2]) # dim 2i+1

    if cls_token:
        sinusoid_table = np.concatenate([np.zeros([1, d_hid]), sinusoid_table], axis=0)
    return sinusoid_table

def create_2d_relative_bias_trainable_embeddings(n_head,height,width,dim):
    position_embedding = nn.Embedding((2*height-1)*(2*width-1),n_head)
    nn.init.constant_(position_embedding.weight, 0.)

    def get_relative_position_index(height,width):
        coords = torch.stack(torch.meshgrid(torch.arange(height),torch.arange(width)))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords_bias = coords_flatten[:,:, None] - coords_flatten[:, None, :]
        relative_coords_bias[0,:,:]+=height - 1
        relative_coords_bias[1,:,:]+=width - 1
        relative_coords_bias[0,:,:]+=relative_coords_bias[1,:,:].max()+1
        return relative_coords_bias.sum(0)  #height*width,height*width
    relative_position_bias = get_relative_position_index(height,width)
    bias_embedding = position_embedding(torch.flatten(relative_position_bias)).reshape(height*width,height*width, n_head)
    bias_embedding = bias_embedding.permute(2, 0, 1).unsqueeze(0)  #1,n_head,height*width,height*width
    return bias_embedding




class DeformablePatchTransformer(nn.Module):
    def __init__(self,embed_dims=[64, 128, 128],num_heads=[1, 2, 4], mlp_ratios=[2, 3, 4], drop_rate=0, F4=False, patch_embeds=[]):

        super().__init__()
        in_channels = [48, 64 ,128 ]
        input_size = (501,65)
        self.F4 = F4
        # for i in range(3):
        #     conv_patch = ConvPatchEmbed(input_size=input_size,
        #                        in_channels=in_channels[i],
        #                        mid_channels=embed_dims[i],
        #                        embed_dims=embed_dims[i],
        #                        patch_size=[4,1],
        #                        stride=[4,1],
        #                        padding='corner',
        #                        dilation=1,
        #                        norm_cfg=None,
        #                        )
        #     patch_embeds.append(conv_patch)
        #     input_size = conv_patch.init_out_size

        # patch_embed
        # (self.patch_embed1,
        #  self.patch_embed2,
        #  self.patch_embed3) = patch_embeds
        self.patch_embed1 = ConvPatchEmbed(input_size=input_size,
                                           in_channels=in_channels[0],
                                           mid_channels=embed_dims[0],
                                           embed_dims=embed_dims[0],
                                           patch_size=[4,1],
                                           stride=[4,1],
                                           padding='corner',
                                           dilation=1,
                                           norm_cfg=None,
                                           )
        self.patch_embed2 = ConvPatchEmbed(input_size= self.patch_embed1.init_out_size,
                                           in_channels=in_channels[1],
                                           mid_channels=embed_dims[1],
                                           embed_dims=embed_dims[1],
                                           patch_size=[4, 1],
                                           stride=[4, 1],
                                           padding='corner',
                                           dilation=1,
                                           norm_cfg=None,
                                           )
        self.patch_embed3 = ConvPatchEmbed(input_size=self.patch_embed2.init_out_size,
                                           in_channels=in_channels[2],
                                           mid_channels=embed_dims[2],
                                           embed_dims=embed_dims[2],
                                           patch_size=[4, 1],
                                           stride=[4, 1],
                                           padding='corner',
                                           dilation=1,
                                           norm_cfg=None,
                                           )

        # pos_embed
        self.pos_embed1 = nn.Parameter(torch.zeros(1, self.patch_embed1.num_patches, embed_dims[0]))
        self.pos_drop1 = nn.Dropout(p=drop_rate)
        self.pos_embed2 = nn.Parameter(torch.zeros(1, self.patch_embed2.num_patches, embed_dims[1]))
        self.pos_drop2 = nn.Dropout(p=drop_rate)
        self.pos_embed3 = nn.Parameter(torch.zeros(1, self.patch_embed3.num_patches, embed_dims[2]))
        self.pos_drop3 = nn.Dropout(p=drop_rate)

        # transformer encoder
        self.u1 = mLSTMBlockStack(num_blocks=1,embedding_dim=embed_dims[0],context_length = self.patch_embed1.num_patches
                                  ,conv1d_kernel_size=4, qkv_proj_blocksize= mlp_ratios[0], num_heads=num_heads[0])
        self.u2 = mLSTMBlockStack(num_blocks=1, embedding_dim=embed_dims[1],context_length = self.patch_embed2.num_patches
                                  , conv1d_kernel_size=4, qkv_proj_blocksize=mlp_ratios[1], num_heads=num_heads[1])
        self.u3 = mLSTMBlockStack(num_blocks=1, embedding_dim=embed_dims[2],context_length = self.patch_embed3.num_patches
                                  , conv1d_kernel_size=2, qkv_proj_blocksize=mlp_ratios[2], num_heads=num_heads[2])

        # self.up4 = nn.ConvTranspose2d(embed_dims[3], embed_dims[2], kernel_size=[5, 6], stride=[2, 1], padding=[1, 1])
        self.up3 = nn.ConvTranspose2d(embed_dims[2], embed_dims[1], kernel_size=[7,1], stride=[4,1], padding=[0,0])
        self.up2 = nn.ConvTranspose2d(embed_dims[1], embed_dims[0], kernel_size=[5,1], stride=[4,1], padding=[0,0])
        self.up1 = nn.ConvTranspose2d(embed_dims[0], in_channels[0], kernel_size=[5,1], stride=[4,1], padding=[0,0])
    def _get_pos_embed(self, pos_embed, patch_embed, H, W):
        if H * W == patch_embed.num_patches:
            return pos_embed
        else:
            return F.interpolate(
                pos_embed.reshape(1, patch_embed.H, patch_embed.W, -1).permute(0, 3, 1, 2),
                size=(H, W), mode="bilinear").reshape(1, -1, H * W).permute(0, 2, 1)
    def forward_features(self, x):
        outs = [None, None, None]
        # print(
        #     f"Before training one epoch: CUDA Memory Allocated: {torch.cuda.memory_allocated()} Memory Reserved: {torch.cuda.memory_reserved()}")
        B = x.shape[0]

        # stage 1
        x, (H, W) = self.patch_embed1(x)
        pos_embed1 = self._get_pos_embed(self.pos_embed1, self.patch_embed1, H, W)
        x = x + pos_embed1
        x = self.pos_drop1(x)
        x = self.u1(x, H, W)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs[0]=x

        # stage 2
        x, (H, W) = self.patch_embed2(x)
        pos_embed2 = self._get_pos_embed(self.pos_embed2, self.patch_embed2, H, W)
        x = x + pos_embed2
        x = self.pos_drop2(x)
        x = self.u2(x, H, W)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs[1]=x

        # stage 3
        x, (H, W) = self.patch_embed3(x)
        pos_embed3 = self._get_pos_embed(self.pos_embed3, self.patch_embed3, H, W)
        x = x + pos_embed3
        x = self.pos_drop3(x)
        x = self.u3(x, H, W)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs[2]=x
        del x


        up_x = self.up3(outs[2])
        up_x = self.up2(up_x + outs[1])
        up_x = self.up1(up_x + outs[0])  # Skip connection

        return up_x

    def forward(self, x):
        x = self.forward_features(x)


        return x

# class dpt_tiny(DeformablePatchTransformer):
#     def __init__(self, **kwargs):
#         # patch_embed
#         embed_dims=[64, 128, 256, 512]
#         T = 499
#         Q = 65
#         img_size = 224
#         Depatch = [False, True, True, True]
#         patch_embeds=[]
#         for i in range(4):
#             inchans = embed_dims[i-1] if i>0 else 3
#             in_size = img_size // 2**(i+1) if i>0 else img_size
#             patch_size = 2 if i > 0 else 4
#             patch_embeds.append(
#                 ConvPatchEmbed(in_channels=48,
#                                     mid_channels=embed_dims[i],
#                                     embed_dims=embed_dims[i],
#                                     patch_size=16,
#                                     stride=4,
#                                     padding='corner',
#                                     dilation=1,
#                                     norm_cfg=None,
#                                     input_size=None,))
#
#         super(dpt_tiny, self).__init__(
#             patch_size=4, embed_dims=[64, 128, 256, 512], num_heads=[1, 2, 2, 4], mlp_ratios=[4, 4, 4, 4],
#             qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[2, 2, 2, 2],
#             sr_ratios=[8, 4, 2, 1], drop_rate=0.0, drop_path_rate=0.1, patch_embeds=patch_embeds)
