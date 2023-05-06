import math
import torch
import torch.nn as nn
import torch.nn.functional as F

def extract_into_tensor(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

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
        half_dim = self.dim // 2
        emb = math.log(10000) / half_dim 
        emb = torch.exp(torch.arange(half_dim).to(device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
        return emb

class AttentionBlock(nn.Module):
    def __init__(self, in_channels, num_groups=32):
        super().__init__()
        
        self.in_channels = in_channels
        self.norm = nn.GroupNorm(num_groups, in_channels)
        self.to_qkv = nn.Conv2d(in_channels, in_channels * 3, 1)
        self.to_out = nn.Conv2d(in_channels, in_channels, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        q, k, v = torch.split(self.to_qkv(self.norm(x)), self.in_channels, dim=1)

        q = q.permute(0, 2, 3, 1).view(b, h * w, c)
        k = k.view(b, c, h * w)
        v = v.permute(0, 2, 3, 1).view(b, h * w, c)

        dot_products = torch.bmm(q, k) * (c ** (-0.5))
        assert dot_products.shape == (b, h * w, h * w)

        attention = torch.softmax(dot_products, dim=-1)
        out = torch.bmm(attention, v)
        assert out.shape == (b, h * w, c)
        out = out.view(b, h, w, c).permute(0, 3, 1, 2)

        return self.to_out(out) + x

class Block(nn.Module):
    def __init__(self, in_channels, out_channels, groups=32) -> None:
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
            time_emb = time_emb[:, :, None, None]
            scale_shift = time_emb.chunk(2, dim=1)
        h = self.block1(x, scale_shift=scale_shift)
        h = self.block2(h)
        return h + self.final(x)

class UNet(nn.Module):
    def __init__(self, in_channels, out_channels, dim, dim_mults=(1, 2, 4, 8), 
                    mid_channels: int = None, attention_mid_only: bool = False) -> None:
        super().__init__()
        self.attention_mid_only = attention_mid_only

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
                nn.Identity() if self.attention_mid_only else AttentionBlock(dim_in),
                Resample(dim_in, dim_out, scale_factor=0.5) if not is_last else nn.Conv2d(dim_in, dim_out, 3, padding=1),
            ]))

        mid_channels = mid_channels if mid_channels else dim_out
        self.mid_block1 = ResBlock(dim_out, dim_out, time_dim)
        self.mid_attn = AttentionBlock(dim_out)
        self.mid_block2 = ResBlock(dim_out, dim_out, time_dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind == (len(in_out) - 1)
            self.ups.append(nn.ModuleList([
                ResBlock(dim_in + dim_out, dim_out, time_dim),
                ResBlock(dim_in + dim_out, dim_out, time_dim),
                nn.Identity() if self.attention_mid_only else AttentionBlock(dim_out),
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

class Encoder(nn.Module):
    def __init__(self, in_channels, out_channels, dim, dim_mults=(1, 2, 4, 8), 
                    attention_mid_only: bool = False) -> None:
        super().__init__()
        self.attention_mid_only = attention_mid_only

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

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (len(in_out) - 1)
            self.downs.append(nn.ModuleList([
                ResBlock(dim_in, dim_in, time_dim),
                ResBlock(dim_in, dim_in, time_dim),
                nn.Identity() if self.attention_mid_only else AttentionBlock(dim_in),
                Resample(dim_in, dim_out, scale_factor=0.5) if not is_last else nn.Conv2d(dim_in, dim_out, 3, padding=1),
            ]))

        self.mid_block = ResBlock(dim_out, dim_out, time_dim)
        self.mid_attn = AttentionBlock(dim_out)

        self.final_block = ResBlock(dim_out, dim, time_dim)
        self.final_conv  = nn.Conv2d(dim, out_channels, 1)

    def forward(self, x, t):
        if t is not None:
            t = self.time_mlp(t)
        x = self.init_conv(x)
        r = x.clone()
        for block1, block2, attn, downsample in self.downs:
            x = block1(x, t)
            x = block2(x, t)
            x = attn(x)
            x = downsample(x)

        x = self.mid_block(x, t)
        x = self.mid_attn(x)
        x = self.final_block(x)
        x = self.final_conv(x)

        return x

class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels, dim, dim_mults=(1, 2, 4, 8), 
                    attention_mid_only: bool = False) -> None:
        super().__init__()
        self.attention_mid_only = attention_mid_only

        time_dim = dim * 4
        dims = [dim * m for m in dim_mults]
        in_out = list(zip(dims[:-1], dims[1:]))

        self.time_mlp = nn.Sequential(
            SinusoidalPositionalEmbedding(dim),
            nn.Linear(dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
        )
        self.ups = nn.ModuleList([])

        self.mid_block = ResBlock(in_channels, dim_mults[-1] * dim, time_dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind == (len(in_out) - 1)
            self.ups.append(nn.ModuleList([
                ResBlock(dim_out, dim_out, time_dim),
                ResBlock(dim_out, dim_out, time_dim),
                nn.Identity() if self.attention_mid_only else AttentionBlock(dim_out),
                Resample(dim_out, dim_in, scale_factor=2) if not is_last else nn.Conv2d(dim_out, dim_in, 3, padding=1),
            ]))

        self.final_block = ResBlock(dim_in, dim, time_dim)
        self.final_conv  = nn.Conv2d(dim, out_channels, 1)

    def forward(self, x, t):
        if t is not None:
            t = self.time_mlp(t)
        x = self.mid_block(x, t)
        for block1, block2, attn, upsample in self.ups:
            x = block1(x)
            x = block2(x)
            x = attn(x)
            x = upsample(x)

        x = self.final_block(x)
        return self.final_conv(x)