import torch
from torch import nn, Tensor
# from zeta.nn import MambaBlock
# from zeta.nn import FeedForward
# from zeta.nn import MultiQueryAttention


import torch.nn.functional as F
# from mamba_transformer.blocks import LinearAttention
from einops import einsum, rearrange, repeat
from torch import Tensor, nn
import math

from typing import Callable

from abc import abstractmethod

import torch.nn as nn

class LinearAttention(nn.Module):
    def __init__(self, dim, *, heads=4, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = heads * dim_head
        self.heads = heads
        self.scale = dim_head**-0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim), nn.Dropout(dropout)
        )

    def forward(self, x, mask=None):
        h = self.heads
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(
            lambda t: rearrange(t, "b n (h d) -> (b h) n d", h=h),
            (q, k, v),
        )

        q = q * self.scale
        q, k = q.softmax(dim=-1), k.softmax(dim=-2)

        if exists(mask):
            k.masked_fill_(mask, 0.0)

        context = einsum("b n d, b n e -> b d e", q, k)
        out = einsum("b d e, b n d -> b n e", context, v)
        out = rearrange(out, " (b h) n d -> b n (h d)", h=h)
        return self.to_out(out)
class BaseAttention(nn.Module):
    @abstractmethod
    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, x, context=None, mask=None):
        pass

class GLU(nn.Module):
    """
    GLU (Gated Linear Unit) module.

    Args:
        dim_in (int): Input dimension.
        dim_out (int): Output dimension.
        activation (Callable[[Tensor], Tensor]): Activation function to be applied to the gate.
        mult_bias (bool, optional): Whether to multiply the bias term. Defaults to False.
    """

    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        activation: Callable[[Tensor], Tensor],
        mult_bias: bool = False,
    ):
        super().__init__()
        self.act = activation
        self.proj = nn.Linear(dim_in, dim_out * 2)
        self.mult_bias = nn.Parameter(torch.ones(dim_out)) if mult_bias else 1.0

    def forward(self, x: Tensor) -> Tensor:
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * self.act(gate) * self.mult_bias

class ReluSquared(nn.Module):
    def forward(self, x):
        return F.relu(x) ** 2

def exists(val):
    return val is not None

def default(val, default_val):
    return default_val if val is None else val
def init_zero_(layer):
    nn.init.constant_(layer.weight, 0.0)
    if exists(layer.bias):
        nn.init.constant_(layer.bias, 0.0)
class FeedForward(nn.Module):
    """
    Feedforward neural network with LayerNorms and GELU activations

    Args:
        dim (int): Input dimension
        hidden_dim (int): Hidden dimension
        dropout (float): Dropout probability

    Usage:
    >>> model = FeedForward(768, 2048, 0.1)
    >>> x = torch.randn(1, 768)
    >>> model(x).shape

    """

    def __init__(
        self,
        dim: int,
        dim_out: int = None,
        mult=4,
        glu=False,
        glu_mult_bias=False,
        swish=False,
        relu_squared=False,
        post_act_ln=False,
        dropout: float = 0.0,
        no_bias=False,
        zero_init_output=False,
    ):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = default(dim_out, dim)

        if relu_squared:
            activation = ReluSquared()
        elif swish:
            activation = nn.SiLU()
        else:
            activation = nn.GELU()

        if glu:
            project_in = GLU(
                dim, inner_dim, activation, mult_bias=glu_mult_bias
            )
        else:
            project_in = nn.Sequential(
                nn.Linear(dim, inner_dim, bias=not no_bias), activation
            )

        if post_act_ln:
            self.ff = nn.Sequential(
                project_in,
                nn.LayerNorm(inner_dim),
                nn.Dropout(dropout),
                nn.Linear(inner_dim, dim_out, bias=not no_bias),
            )
        else:
            self.ff = nn.Sequential(
                project_in,
                nn.Dropout(dropout),
                nn.Linear(inner_dim, dim_out, bias=not no_bias),
            )

        # init last linear layer to 0
        if zero_init_output:
            init_zero_(self.ff[-1])

    def forward(self, x):
        """
        Forward pass of the feedforward network

        Args:
            x (torch.Tensor): Input tensor

        Returns:
            torch.Tensor: Output tensor
        """
        return self.ff(x)
class MambaBlock(nn.Module):
    """
    Initialize a single Mamba block.

    Args:
        dim (int): The input dimension.
        dim_inner (Optional[int]): The inner dimension. If not provided, it is set to dim * expand.
        depth (int): The depth of the Mamba block.
        d_state (int): The state dimension. Default is 16.
        expand (int): The expansion factor. Default is 2.
        dt_rank (Union[int, str]): The rank of the temporal difference (Δ) tensor. Default is "auto".
        d_conv (int): The dimension of the convolutional kernel. Default is 4.
        conv_bias (bool): Whether to include bias in the convolutional layer. Default is True.
        bias (bool): Whether to include bias in the linear layers. Default is False.

    Examples:
        >>> import torch
        >>> from zeta.nn.modules.simple_mamba import MambaBlock
        >>> block = MambaBlock(dim=64, depth=1)
        >>> x = torch.randn(1, 10, 64)
        >>> y = block(x)
        >>> y.shape
        torch.Size([1, 10, 64])
    """

    def __init__(
        self,
        dim: int = None,
        depth: int = 5,
        d_state: int = 16,
        expand: int = 2,
        d_conv: int = 4,
        conv_bias: bool = True,
        bias: bool = False,
    ):
        """A single Mamba block, as described in Figure 3 in Section 3.4 in the Mamba paper [1]."""
        super(MambaBlock,self).__init__()
        self.dim = dim
        self.depth = depth
        self.d_state = d_state
        self.expand = expand
        self.d_conv = d_conv
        self.conv_bias = conv_bias
        self.bias = bias

        # If dt_rank is not provided, set it to ceil(dim / d_state)
        dt_rank = math.ceil(self.dim / 16)
        self.dt_rank = dt_rank

        # If dim_inner is not provided, set it to dim * expand
        dim_inner = dim * expand
        self.dim_inner = dim_inner

        # If dim_inner is not provided, set it to dim * expand
        self.in_proj = nn.Linear(dim, dim_inner * 2, bias=bias)

        self.conv1d = nn.Conv1d(
            in_channels=dim_inner,
            out_channels=dim_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=dim_inner,
            padding=d_conv - 1,
        )

        # x_proj takes in `x` and outputs the input-specific Δ, B, C
        self.x_proj = nn.Linear(
            dim_inner, dt_rank + self.d_state * 2, bias=False
        )

        # dt_proj projects Δ from dt_rank to d_in
        self.dt_proj = nn.Linear(dt_rank, dim_inner, bias=True)

        A = repeat(torch.arange(1, self.d_state + 1), "n -> d n", d=dim_inner)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(dim_inner))
        self.out_proj = nn.Linear(dim_inner, dim, bias=bias)

    def forward(self, x: Tensor):
        """Mamba block forward. This looks the same as Figure 3 in Section 3.4 in the Mamba paper [1].

        Args:
            x: shape (b, l, d)    (See Glossary at top for definitions of b, l, d_in, n...)

        Returns:
            output: shape (b, l, d)


        Official Implementation:
            class Mamba, https://github.com/state-spaces/mamba/blob/main/mamba_ssm/modules/mamba_simple.py#L119
            mamba_inner_ref(), https://github.com/state-spaces/mamba/blob/main/mamba_ssm/ops/selective_scan_interface.py#L311

        """
        (b, l, d) = x.shape

        x_and_res = self.in_proj(x)  # shape (b, l, 2 * d_in)
        x_and_res = rearrange(x_and_res, "b l x -> b x l")
        (x, res) = x_and_res.split(
            split_size=[self.dim_inner, self.dim_inner], dim=1
        )

        x = self.conv1d(x)[:, :, :l]
        x = F.silu(x)

        y = self.ssm(x)

        y = y * F.silu(res)

        output = self.out_proj(rearrange(y, "b dim l -> b l dim"))

        return output

    def ssm(self, x: Tensor):
        """Runs the SSM. See:
            - Algorithm 2 in Section 3.2 in the Mamba paper [1]
            - run_SSM(A, B, C, u) in The Annotated S4 [2]

        Args:
            x: shape (b, d_in, l)    (See Glossary at top for definitions of b, l, d_in, n...)

        Returns:
            output: shape (b, d_in, l)

        Official Implementation:
            mamba_inner_ref(), https://github.com/state-spaces/mamba/blob/main/mamba_ssm/ops/selective_scan_interface.py#L311

        """
        (d_in, n) = self.A_log.shape

        # Compute ∆ A B C D, the state space parameters.
        #     A, D are input independent
        #     ∆, B, C are input-dependent (this is a key difference between Mamba and the linear time invariant S4)

        A = -torch.exp(self.A_log.float())  # shape (d_in, n)
        D = self.D.float()

        x_dbl = rearrange(x, "b d l -> b l d")
        x_dbl = self.x_proj(x_dbl)  # (b, l, dt_rank + 2*n)

        (delta, B, C) = x_dbl.split(
            split_size=[self.dt_rank, n, n], dim=-1
        )  # delta: (b, l, dt_rank). B, C: (b, l, n)
        delta = F.softplus(self.dt_proj(delta))  # (b, l, d_in)

        y = self.selective_scan(
            x, delta, A, B, C, D
        )  # This is similar to run_SSM(A, B, C, u) in The Annotated S4 [2]

        return y

    def selective_scan(self, u, delta, A, B, C, D):
        """Does selective scan algorithm. See:
            - Section 2 State Space Models in the Mamba paper [1]
            - Algorithm 2 in Section 3.2 in the Mamba paper [1]
            - run_SSM(A, B, C, u) in The Annotated S4 [2]

        This is the classic discrete state space formula:
            x(t + 1) = Ax(t) + Bu(t)
            y(t)     = Cx(t) + Du(t)
        except B and C (and the step size delta, which is used for discretization) are dependent on the input x(t).

        Args:
            u: shape (b, d_in, l)    (See Glossary at top for definitions of b, l, d_in, n...)
            delta: shape (b, l, d_in)
            A: shape (d_in, n)
            B: shape (b, l, n)
            C: shape (b, l, n)
            D: shape (d_in,)

        Returns:
            output: shape (b, d_in, l)

        Official Implementation:
            selective_scan_ref(), https://github.com/state-spaces/mamba/blob/main/mamba_ssm/ops/selective_scan_interface.py#L86
            Note: I refactored some parts out of `selective_scan_ref` out, so the functionality doesn't match exactly.

        """
        (b, d_in, l) = u.shape
        # print('u',u.shape)
        n = A.shape[1]

        # Discretize continuous parameters (Δ, A, B)  (see Section 2 Equation 4 in the Mamba paper [1])
        # Note that B is parameterized directly
        # print('A',A.shape)
        deltaA = torch.exp(einsum(delta, A, "b l d_in, d_in n -> b d_in l n"))
        deltaB_u = einsum(
            delta, B, u, "b l d_in, b l n, b d_in l -> b d_in l n"
        )
        # Perform selective scan (see scan_SSM() in The Annotated S4 [2])
        x = torch.zeros((b, d_in,l, n)).cuda()
        ys = []

        x = deltaA * x + deltaB_u
        y = einsum(x, C, "b d_in l n , b l n -> b d_in l")
        # ys.append(y)
        # y = torch.stack(ys, dim=2)  # (b d_in l)

        if D is not None:
            y = y + u * rearrange(D, "d_in -> d_in 1")

        return y
def _cast_if_autocast_enabled(tensor):
    if torch.is_autocast_enabled():
        if tensor.device.type == "cuda":
            dtype = torch.get_autocast_gpu_dtype()
        elif tensor.device.type == "cpu":
            dtype = torch.get_autocast_cpu_dtype()
        else:
            raise NotImplementedError()
        return tensor.to(dtype=dtype)
    return tensor
def rms_norm(x, weight=None, eps=1e-5):
    output = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    if weight is not None:
        return output * weight
    return output
class RMSNorm(nn.Module):
    def __init__(
        self,
        normalized_shape,
        eps=1e-5,
        weight=True,
        dtype=None,
        device=None,
    ):
        super().__init__()
        self.eps = eps
        if weight:
            self.weight = torch.nn.Parameter(
                torch.ones(normalized_shape, dtype=dtype, device=device)
            )
        else:
            self.register_parameter("weight", None)

    def forward(self, x):
        return rms_norm(x.float(), self.weight, self.eps).to(dtype=x.dtype)
class LPRMSNorm(RMSNorm):
    def __init__(
        self,
        normalized_shape,
        eps=1e-5,
        weight=True,
        dtype=None,
        device=None,
    ):
        super().__init__(
            normalized_shape=normalized_shape,
            eps=eps,
            weight=weight,
            dtype=dtype,
            device=device,
        )

    def forward(self, x):
        downcast_x = _cast_if_autocast_enabled(x)
        downcast_weight = (
            _cast_if_autocast_enabled(self.weight)
            if self.weight is not None
            else self.weight
        )
        with torch.autocast(enabled=False, device_type=x.device_type):
            return rms_norm(downcast_x, downcast_weight, self.eps).to(
                dtype=x.dtype
            )
def scaled_multihead_dot_product_attention(
    query,
    key,
    value,
    heads,
    past_key_value=None,
    softmax_scale=None,
    bias=None,
    key_padding_mask=None,
    causal=False,
    dropout=0.0,
    training=False,
    needs_weights=False,
    multiquery=False,
):
    q = rearrange(query, "b s (h d) -> b h s d", h=heads)
    kv_heads = 1 if multiquery else heads
    k = rearrange(key, "b s (h d) -> b h d s", h=kv_heads)
    v = rearrange(value, "b s (h d) -> b h s d", h=kv_heads)

    if past_key_value is not None:
        # attn_impl: flash & triton use kernels which expect input shape [b, s, h, d_head].
        # kv_cache is therefore stored using that shape.
        # attn_impl: torch stores the kv_cache in the ordering which is most advantageous
        # for its attn computation ie
        # keys are stored as tensors with shape [b, h, d_head, s] and
        # values are stored as tensors with shape [b, h, s, d_head]
        if len(past_key_value) != 0:
            k = torch.cat([past_key_value[0], k], dim=3)
            v = torch.cat([past_key_value[1], v], dim=2)

        past_key_value = (k, v)

    b, _, s_q, d = q.shape
    s_k = k.size(-1)

    if softmax_scale is None:
        softmax_scale = 1 / math.sqrt(d)

    attn_weight = q.matmul(k) * softmax_scale

    if bias is not None:
        # clamp to 0 necessary for torch 2.0 compile()
        _s_q = max(0, bias.size(2) - s_q)
        _s_k = max(0, bias.size(3) - s_k)
        bias = bias[:, :, _s_q:, _s_k:]

        if (bias.size(-1) != 1 and bias.size(-1) != s_k) or (
            bias.size(-2) != 1 and bias.size(-2) != s_q
        ):
            raise RuntimeError(
                f"bias (shape: {bias.shape}) is expected to broadcast to shape:"
                f" {attn_weight.shape}."
            )
        attn_weight = attn_weight + bias

    min_val = torch.finfo(q.dtype).min

    if key_padding_mask is not None:
        # if bias is not None:
            # warnings.warn(
            #     "Propogating key_padding_mask to the attention module "
            #     + "and applying it within the attention module can cause "
            #     + "unneccessary computation/memory usage. Consider integrating "
            #     + "into bias once and passing that to each attention "
            #     + "module instead."
            # )
        attn_weight = attn_weight.masked_fill(
            ~key_padding_mask.view((b, 1, 1, s_k)), min_val
        )

    if causal and (not q.size(2) == 1):
        s = max(s_q, s_k)
        causal_mask = attn_weight.new_ones(s, s, dtype=torch.float32)
        causal_mask = causal_mask.tril()
        causal_mask = causal_mask.to(torch.bool)
        causal_mask = ~causal_mask
        causal_mask = causal_mask[-s_q:, -s_k:]
        attn_weight = attn_weight.masked_fill(
            causal_mask.view(1, 1, s_q, s_k), min_val
        )

    attn_weight = torch.softmax(attn_weight, dim=-1)

    if dropout:
        attn_weight = torch.nn.functional.dropout(
            attn_weight, p=dropout, training=training, inplace=True
        )

    out = attn_weight.to(v.dtype).matmul(v)
    out = rearrange(out, "b h s d -> b s (h d)")

    if needs_weights:
        return out, attn_weight, past_key_value
    return out, None, past_key_value


def check_valid_inputs(*tensors, valid_dtypes=[torch.float16, torch.bfloat16]):
    for tensor in tensors:
        if tensor.dtype not in valid_dtypes:
            raise TypeError(f"{tensor.dtype=} must be in {valid_dtypes=}.")
        if not tensor.is_cuda:
            raise TypeError(f"Inputs must be cuda tensors ({tensor.is_cuda=}).")

def _reset_causal(
    num_query_tokens: int, num_key_tokens: int, original_causal: bool
):
    # disable causal when it is not needed
    # necessary for flash & triton for generation with kv_cache
    if original_causal and num_query_tokens != num_key_tokens:
        if num_query_tokens != 1:
            raise NotImplementedError(
                "MPT does not support query and key with different number of"
                " tokens, unless number of query tokens is 1."
            )
        else:
            return False
    return original_causal
def attn_bias_shape(
    attn_impl, heads, seq_len, alibi, prefix_lm, causal, use_sequence_id
):
    if attn_impl == "flash":
        return None
    elif attn_impl in ["torch", "triton"]:
        if alibi:
            if (prefix_lm or not causal) or use_sequence_id:
                return (1, heads, seq_len, seq_len)
            return (1, heads, 1, seq_len)
        elif prefix_lm or use_sequence_id:
            return (1, 1, seq_len, seq_len)
        return None
    else:
        raise ValueError(f"{attn_impl=} is an invalid setting.")

class MultiQueryAttention(BaseAttention):
    """Multi-Query self attention.

    Using torch or triton attention implemetation enables user to also use
    additive bias.

    Look for documentation
    """

    def __init__(
        self,
        d_model: int,
        heads: int,
        attn_impl: str = "torch",
        clip_qkv = None,
        qk_ln: bool = False,
        softmax_scale = None,
        attn_pdrop: float = 0.0,
        norm_type: str = "low_precision_layernorm",
        fc_type: str = "torch",
        verbose: int = 0,
        device = None,
    ):
        super().__init__()

        self.attn_impl = attn_impl
        self.clip_qkv = clip_qkv
        self.qk_ln = qk_ln

        self.d_model = d_model
        self.heads = heads
        self.head_dim = d_model // heads
        self.softmax_scale = softmax_scale
        if self.softmax_scale is None:
            self.softmax_scale = 1 / math.sqrt(self.head_dim)
        self.attn_dropout = attn_pdrop

        fc_kwargs = {}
        if fc_type != "te":
            fc_kwargs["device"] = device
        # - vchiley
        self.Wqkv = nn.Linear(
            d_model,
            d_model + 2 * self.head_dim,
            **fc_kwargs,
        )
        # for param init fn; enables shape based init of fused layers
        fuse_splits = (d_model, d_model + self.head_dim)
        self.Wqkv._fused = (0, fuse_splits)  # type: ignore

        if self.qk_ln:
            norm_class = LPRMSNorm[norm_type.lower()]
            self.q_ln = norm_class(d_model, device=device)
            self.k_ln = norm_class(self.head_dim, device=device)

        # if self.attn_impl == "flash":
        #     self.attn_fn = flash_attn_fn
        # elif self.attn_impl == "triton":
        #     self.attn_fn = triton_flash_attn_fn
        #     if verbose:
        #         warnings.warn(
        #             "While `attn_impl: triton` can be faster than `attn_impl:"
        #             " flash` "
        #             + "it uses more memory. When training larger models"
        #             " this can"
        #             " trigger "
        #             + "alloc retries which hurts performance. If"
        #             " encountered, we"
        #             " recommend "
        #             + "using `attn_impl: flash` if your model does not use"
        #             " `alibi` or `prefix_lm`."
        #         )
        if self.attn_impl == "torch":
            self.attn_fn = scaled_multihead_dot_product_attention
        #     if torch.cuda.is_available() and verbose:
        #         warnings.warn(
        #             "Using `attn_impl: torch`. If your model does not use"
        #             " `alibi` or "
        #             + "`prefix_lm` we recommend using `attn_impl: flash`"
        #             " otherwise "
        #             + "we recommend using `attn_impl: triton`."
        #         )
        # else:
        #     raise ValueError(f"{attn_impl=} is an invalid setting.")

        self.out_proj =  nn.Linear(
            self.d_model,
            self.d_model,
            **fc_kwargs,
        )
        self.out_proj._is_residual = True  # type: ignore

    def forward(
        self,
        x,
        past_key_value=None,
        bias=None,
        mask=None,
        causal=True,
        needs_weights=False,
    ):
        qkv = self.Wqkv(x)

        if self.clip_qkv:
            qkv = qkv.clamp(min=-self.clip_qkv, max=self.clip_qkv)

        query, key, value = qkv.split(
            [self.d_model, self.head_dim, self.head_dim], dim=2
        )

        key_padding_mask = mask

        if self.qk_ln:
            # Applying layernorm to qk
            dtype = query.dtype
            query = self.q_ln(query).to(dtype)
            key = self.k_ln(key).to(dtype)

        context, attn_weights, past_key_value = self.attn_fn(
            query,
            key,
            value,
            self.heads,
            past_key_value=past_key_value,
            softmax_scale=self.softmax_scale,
            bias=bias,
            key_padding_mask=key_padding_mask,
            causal=causal,
            dropout=self.attn_dropout,
            training=self.training,
            needs_weights=needs_weights,
            multiquery=True,
        )

        return self.out_proj(context), attn_weights, past_key_value



class RMSNorm(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.scale = dim ** (-0.5)
        self.g = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        return F.normalize(x, dim=-1) * self.scale * self.g


class TransformerBlock(nn.Module):
    """
    TransformerBlock is a module that represents a single block of the Multi-Query Transformer.
    It consists of a multi-query attention layer, a feed-forward network, and layer normalization.

    Args:
        dim (int): The input and output dimension of the block.
        heads (int): The number of attention heads.
        dim_head (int): The dimension of each attention head.
        dropout (float, optional): The dropout probability. Defaults to 0.1.
        ff_mult (int, optional): The multiplier for the feed-forward network dimension. Defaults to 4.

    Attributes:
        dim (int): The input and output dimension of the block.
        heads (int): The number of attention heads.
        dim_head (int): The dimension of each attention head.
        dropout (float): The dropout probability.
        ff_mult (int): The multiplier for the feed-forward network dimension.
        attn (MultiQueryAttention): The multi-query attention layer.
        ffn (FeedForward): The feed-forward network.
        norm (nn.LayerNorm): The layer normalization.

    Methods:
        forward(x: Tensor) -> Tensor:
            Performs a forward pass of the TransformerBlock.

    """

    def __init__(
        self,
        dim: int,
        heads: int,
        dim_head: int,
        dropout: float = 0.1,
        ff_mult: int = 4,
        use_linear_attn: bool = False,
        * args,
        **kwargs,
    ):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.dim_head = dim_head
        self.dropout = dropout
        self.ff_mult = ff_mult
        self.use_linear_attn = use_linear_attn

        self.attn = MultiQueryAttention(dim, heads, *args, **kwargs)

        # Linear Attention
        # self.linear_attn = LinearAttention(
        #     dim=dim, heads=heads, dim_head=dim_head, dropout=dropout
        # )

        self.ffn = FeedForward(dim, dim, ff_mult, *args, **kwargs)

        # Normalization
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: Tensor) -> Tensor:
        """
        Performs a forward pass of the TransformerBlock.

        Args:
            x (Tensor): The input tensor.

        Returns:
            Tensor: The output tensor.

        """
        if self.use_linear_attn:
            x = self.linear_attn(x)
            x = self.norm(x)
            x = self.ffn(x)
        else:
            x, _, _ = self.attn(x)
            x = self.norm(x)
            x = self.ffn(x)

        return x


class MambaTransformerblock(nn.Module):
    """
    MambaTransformerblock is a module that represents a block in the Mamba Transformer model.

    Args:
        dim (int): The input dimension of the block.
        heads (int): The number of attention heads in the block.
        depth (int): The number of layers in the block.
        dim_head (int): The dimension of each attention head.
        dropout (float, optional): The dropout rate. Defaults to 0.1.
        ff_mult (int, optional): The multiplier for the feed-forward network dimension. Defaults to 4.
        d_state (int, optional): The dimension of the state. Defaults to None.
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Attributes:
        dim (int): The input dimension of the block.
        depth (int): The number of layers in the block.
        dim_head (int): The dimension of each attention head.
        d_state (int): The dimension of the state.
        dropout (float): The dropout rate.
        ff_mult (int): The multiplier for the feed-forward network dimension.
        mamba_blocks (nn.ModuleList): List of MambaBlock instances.
        transformer_blocks (nn.ModuleList): List of TransformerBlock instances.
        ffn_blocks (nn.ModuleList): List of FeedForward instances.
        norm (nn.LayerNorm): Layer normalization module.

    Examples:
        import torch
        from mt import MambaTransformerblock

        x = torch.randn(1, 10, 512)
        model = MambaTransformerblock(
            dim=512,
            heads=8,
            depth=4,
            dim_head=64,
            d_state=512,
            dropout=0.1,
            ff_mult=4
        )
        print(model(x).shape)


    """

    def __init__(
        self,
        dim: int,
        heads: int,
        depth: int,
        dim_head: int,
        dropout: float = 0.1,
        ff_mult: int = 4,
        d_state: int = None,
        transformer_depth: int = 1,
        mamba_depth: int = 1,
        use_linear_attn: bool = False,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.dim = dim
        self.depth = depth
        self.dim_head = dim_head
        # self.d_state = d_state
        self.dropout = dropout
        self.ff_mult = ff_mult
        # self.d_state = d_state
        self.transformer_depth = transformer_depth
        self.mamba_depth = mamba_depth

        # Mamba, Transformer, and ffn blocks
        self.mamba_blocks = nn.ModuleList([
            MambaBlock(dim, mamba_depth, d_state, *args, **kwargs)
            for _ in range(mamba_depth)
        ])
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(
                dim,
                heads,
                dim_head,
                dropout,
                ff_mult,
                use_linear_attn,
                *args,
                **kwargs,
            ) for _ in range(transformer_depth)
        ])

        self.ffn_blocks = nn.ModuleList([
            FeedForward(dim, dim, ff_mult, *args, **kwargs)
            for _ in range(depth)
        ])

        # Layernorm
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: Tensor) -> Tensor:
        for mamba, attn, ffn in zip(
            self.mamba_blocks,
            self.transformer_blocks,
            self.ffn_blocks,
        ):
            x = self.norm(x)
            # print('x',x.shape)
            x = mamba(x) + x
            x = self.norm(x)
            x = attn(x) + x
            x = self.norm(x)
            x = ffn(x) + x

        return x


class MambaTransformer(nn.Module):
    """
    MambaTransformer is a PyTorch module that implements the Mamba Transformer model.

    Args:
        num_tokens (int): The number of tokens in the input vocabulary.
        dim (int): The dimensionality of the token embeddings and model hidden states.
        heads (int): The number of attention heads.
        depth (int): The number of transformer blocks.
        dim_head (int): The dimensionality of each attention head.
        dropout (float, optional): The dropout rate. Defaults to 0.1.
        ff_mult (int, optional): The multiplier for the feed-forward network dimension. Defaults to 4.
        d_state (int, optional): The dimensionality of the state embeddings. Defaults to None.
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Examples:
        >>> import torch
        >>> from mt import MambaTransformer
        >>> x = torch.randint(0, 100, (1, 10))
        >>> model = MambaTransformer(
        ...     num_tokens=100,
        ...     dim=512,
        ...     heads=8,
        ...     depth=4,
        ...     dim_head=64,
        ...     d_state=512,
        ...     dropout=0.1,
        ...     ff_mult=4
        ... )
        >>> print(model(x).shape)
        torch.Size([1, 10, 100])
    """

    def __init__(
        self,
        num_tokens: int,
        dim: int,
        heads: int,
        depth: int,
        dim_head: int,
        dropout: float = 0.1,
        ff_mult: int = 4,
        d_state: int = None,
        return_embeddings: bool = False,
        transformer_depth: int = 1,
        mamba_depth: int = 1,
        use_linear_attn=False,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.dim = dim
        self.depth = depth
        self.dim_head = dim_head
        self.d_state = d_state
        self.dropout = dropout
        self.ff_mult = ff_mult
        self.d_state = d_state
        self.return_embeddings = return_embeddings
        self.transformer_depth = transformer_depth
        self.mamba_depth = mamba_depth

        self.emb = nn.Embedding(num_tokens, dim)
        self.mt_block = MambaTransformerblock(
            dim,
            heads,
            depth,
            dim_head,
            dropout,
            ff_mult,
            d_state,
            return_embeddings,
            transformer_depth,
            mamba_depth,
            use_linear_attn,
            *args,
            **kwargs,
        )
        self.to_logits = nn.Sequential(
            RMSNorm(dim), nn.Linear(dim, num_tokens)
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the MambaTransformer model.

        Args:
            x (Tensor): Input tensor of shape (batch_size, sequence_length).

        Returns:
            Tensor: Output tensor of shape (batch_size, sequence_length, num_tokens).
        """
        x = self.emb(x)
        x = self.mt_block(x)

        if self.return_embeddings:
            return x

        else:
            return self.to_logits(x)
