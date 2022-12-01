from einops import rearrange, reduce
from functools import partial
import torch.nn.functional as F
import torch.nn as nn
import torch
import math

"""
1. p2 weight
2. linear attention
3. weight balance conv
4. clip norm
5. concat
"""

class WeightStandardizedConv2d(nn.Conv2d):
    """
    https://arxiv.org/abs/1903.10520
    weight standardization purportedly works synergistically with group normalization
    """
    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3

        weight = self.weight
        mean = reduce(weight, 'o ... -> o 1 1 1', 'mean')
        var = reduce(weight, 'o ... -> o 1 1 1', partial(torch.var, unbiased = False))
        normalized_weight = (weight - mean) * (var + eps).rsqrt()

        return F.conv2d(x, normalized_weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

class LayerNorm(nn.Module):
    def __init__(self, dim) -> None:
        super().__init__()
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))

    def forward(self, x):
        var = torch.var(x, dim = 1, unbiased=False, keepdim=True)
        mean = torch.mean(x, dim = 1, keepdim=True)
        return (x - mean) * (var + 1e-5).rsqrt() * self.g

class Resample(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor) -> None:
        super().__init__()
        self.main = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.scale_factor = scale_factor

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale_factor)
        return self.main(x)


class SinusoidalPositionalEmbedding(nn.Module):
    def __init__(self, dim) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        emb = math.log(10000) / (self.dim // 2 - 1)
        emb = torch.exp(torch.arange(self.dim // 2).to(device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
        return emb

class Attention(nn.Module):
    def __init__(self, dim, heads = 4, dim_head = 32) -> None:
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        self.to_qkv = nn.Conv2d(dim, dim_head * heads * 3, kernel_size=1, stride=1, bias=False)
        self.to_out = nn.Conv2d(dim_head * heads, dim, kernel_size=1, stride=1)
        self.norm = LayerNorm(dim)

    def forward(self, x):
        b, c, h, w = x.shape
        out = self.norm(x)
        qkv = self.to_qkv(out).chunk(3, dim=1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h c (x y)', h=self.heads), qkv)
        q = q * self.scale        
        attn = torch.einsum('b h d i, b h d j -> b h i j', q, k).softmax(-1)
        out = torch.einsum('b h i j, b h d j -> b h i d', attn, v)
        out = rearrange(out, 'b h (x y) d -> b (h d) x y', x=h, y=w)
        out = self.to_out(out)
        return out + x

class LinearAttention(nn.Module):
    def __init__(self, dim, heads = 4, dim_head = 32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)

        self.to_out = nn.Sequential(
            nn.Conv2d(hidden_dim, dim, 1),
            LayerNorm(dim)
        )

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h c (x y)', h = self.heads), qkv)

        q = q.softmax(dim = -2)
        k = k.softmax(dim = -1)

        q = q * self.scale
        v = v / (h * w)

        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)

        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
        out = rearrange(out, 'b h c (x y) -> b (h c) x y', h = self.heads, x = h, y = w)
        return self.to_out(out)

class Block(nn.Module):
    def __init__(self, in_channels, out_channels, groups=8) -> None:
        super().__init__()
        self.proj = nn.Conv2d(in_channels, out_channels, 3, 1, 1)
        # self.proj = WeightStandardizedConv2d(in_channels, out_channels, 3, 1, 1)
        self.norm = nn.GroupNorm(groups, out_channels)
        self.act  = nn.SiLU()

    def forward(self, x, scale_shift = None):
        x = self.norm(self.proj(x))
        if scale_shift is not None:
            scale, shift = scale_shift
            x  = x * scale + shift
        return self.act(x)

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_channel, groups=8) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_channel, out_features=out_channels*2),
        )
        self.block1 = Block(in_channels, out_channels, groups=groups)
        self.block2 = Block(out_channels, out_channels, groups=groups)        
        self.final  = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x, time_emb = None):
        scale_shift = None
        if time_emb is not None:
            time_emb = self.mlp(time_emb)
            time_emb = rearrange(time_emb, 'b c -> b c 1 1')
            scale_shift = time_emb.chunk(2, dim=1)
        h = self.block1(x, scale_shift=scale_shift)
        h = self.block2(h)
        return h + self.final(x)
