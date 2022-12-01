import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from functools import partial

def extract_into_tensor(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

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

class Block(nn.Module):
    def __init__(self, in_channels, out_channels, groups=8) -> None:
        super().__init__()
        self.proj = nn.Conv2d(in_channels, out_channels, 3, 1, 1)
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

class UNet(nn.Module):
    def __init__(self, in_channels, out_channels, dim, dim_mults=(1, 2, 4, 8)) -> None:
        super().__init__()

        time_dim = dim * 4
        dims = [dim * m for m in dim_mults]
        in_out = list(zip(dims[:-1], dims[1:]))

        self.time_mlp = nn.Sequential(
            SinusoidalPositionalEmbedding(dim),
            nn.Linear(dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
        )
        self.init_conv = nn.Conv2d(in_channels, dim, 7, padding=3)
        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (len(in_out) - 1)
            self.downs.append(nn.ModuleList([
                ResBlock(dim_in, dim_in, time_dim),
                ResBlock(dim_in, dim_in, time_dim),
                Attention(dim_in),
                Resample(dim_in, dim_out, scale_factor=0.5) if not is_last else nn.Conv2d(dim_in, dim_out, 3, padding=1),
            ]))
        
        self.mid_block1 = ResBlock(dim_out, dim_out, time_dim)
        self.mid_attn = Attention(dim_out)
        self.mid_block2 = ResBlock(dim_out, dim_out, time_dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind == (len(in_out) - 1)
            self.ups.append(nn.ModuleList([
                ResBlock(dim_in + dim_out, dim_out, time_dim),
                ResBlock(dim_in + dim_out, dim_out, time_dim),
                Attention(dim_out),
                Resample(dim_out, dim_in, scale_factor=2) if not is_last else nn.Conv2d(dim_out, dim_in, 3, padding=1),
            ]))

        self.final_block = ResBlock(dim_in, dim, time_dim)
        self.final_conv  = nn.Conv2d(dim, out_channels, 1)

    def forward(self, x, t):
        t = self.time_mlp(t)
        x = self.init_conv(x)
        r = x.clone()
        h = []
        for block1, block2, attn, downsample in self.downs:
            x = block1(x, t)
            h.append(x)
            x = block2(x, t)
            x = attn(x)
            h.append(x)
            x = downsample(x)

        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)

        for block1, block2, attn, upsample in self.ups:
            x = torch.cat([x, h.pop()], dim=1)
            x = block1(x)
            x = torch.cat([x, h.pop()], dim=1)
            x = block2(x)
            x = attn(x)
            x = upsample(x)
        x += r
        x = self.final_block(x)
        return self.final_conv(x)