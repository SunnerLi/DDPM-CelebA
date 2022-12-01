from module import SinusoidalPositionalEmbedding, Resample, ResBlock, Attention, LinearAttention
from torchvision import datasets, transforms
from einops import rearrange, reduce
import torch.nn.functional as F
import torch.nn as nn
import logging
import torch
import fire
import sys
import os

def extract_into_tensor(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

class UNet(nn.Module):
    def __init__(self, in_channels, out_channels, dim, dim_mults=(1, 2, 4, 8)) -> None:
        super().__init__()

        time_dim = dim * 4
        dims = [dim * m for m in dim_mults]
        in_out = list(zip(dims[:-1], dims[1:]))

        self.time_mlp = nn.Sequential(
            SinusoidalPositionalEmbedding(time_dim),
            nn.Linear(time_dim, time_dim),
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
        # self.final_block = ResBlock(2 * dim_in, dim, time_dim)
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
        # x = torch.cat([x, r], dim=1)

        x = self.final_block(x)
        return self.final_conv(x)

def main(
    data_folder : str = "~/data", output_folder : str = './cifar10-ddpm-recover',                           # Data & IO
    batch_size : int = 64, device : str = 'cuda',                                                   # Training basic
    img_size : int = 32, img_channels : int = 3,   
    iter_log : int = 100, iter_save : int = 10000, iter_train : int = 100000,                       # Time stage
    timesteps : int = 1000,                                                                         # Diffusion model hyper-parameters
    ):

    ### Check output path
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    logging.basicConfig(filename=os.path.join(output_folder, 'log.txt'), level=logging.DEBUG)
    logger = logging.getLogger()
    
    ### Loading dataset
    dataloader = torch.utils.data.DataLoader(
        dataset=datasets.CIFAR10(
            root=data_folder, download=True, transform=transforms.Compose([
                transforms.Resize(img_size),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])
        ), batch_size=batch_size, shuffle=True, num_workers=2, drop_last=True,
    )

    ### Define model and optimizer
    model = UNet(in_channels=img_channels, out_channels=img_channels, dim=64).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=1e-4)

    ################################ Define Diffusion model hyper-parameters ################################
    ### Define beta (noise STD) in linear form
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    betas = torch.linspace(beta_start, beta_end, steps=timesteps, dtype=torch.float64, device=device)   # \beta_t

    ### Basic notations
    alphas = 1. - betas                                                                                 # \alpha_t
    alphas_cumprod = torch.cumprod(alphas, dim=0)                                                       # \bar_\alpha_t
    alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.)                                  # \bar_\alpha_{t-1}

    ### Calculation for diffusion q(x_t | x_{t-1})
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)                                                    # \sqrt_{\bar_\alpha_t}
    sqrt_one_minus_alpha_cumprod = torch.sqrt(1. - alphas_cumprod)                                      # \sqrt_{1 - \bar_\alpha_t}
    sqrt_recip_alphas_cumprod = torch.sqrt(1. / alphas_cumprod)                                         # 1 / \sqrt_{\bar_\alpha_t}
    sqrt_recipm1_alphas_cumprod = torch.sqrt(1. / alphas_cumprod - 1)

    ### Calculation for gaussian posterior q(x_{t-1} | x_t, x_0)
    posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)                     # Refer to telta beta definition in DDPM paper page 3
    posterior_log_variance_clipped = torch.log(posterior_variance.clamp(min=1e-20))
    posterior_mean_coef1 = (torch.sqrt(alphas_cumprod_prev) * betas) / (1. - alphas_cumprod)            # Refer to eq. 7 in DDPM paper
    posterior_mean_coef2 = (torch.sqrt(alphas) * (1. - alphas_cumprod_prev)) / (1. - alphas_cumprod)    # Refer to eq. 7 in DDPM paper

    p2_loss_weight_gamma = 1.0
    p2_loss_weight_k = 1
    p2_loss_weight = (p2_loss_weight_k + alphas_cumprod / (1 - alphas_cumprod)) ** -p2_loss_weight_gamma
    ##########################################################################################################

    ### Training loop
    curr_iter = 1
    while curr_iter < iter_train + 1:
        for x_start, target in dataloader:
            x_start = x_start.float().to(device)

            ### Random sample time step & noise
            t = torch.randint(0, timesteps, (batch_size,), device=device).long()
            noise = torch.randn_like(x_start, device=device).float()

            ### forward process to sample q(x_t | x_0) in simplified form. Please Refer to Alg. 1 in DDPM paper
            x = extract_into_tensor(sqrt_alphas_cumprod, t, x_start.shape) * x_start + \
                    extract_into_tensor(sqrt_one_minus_alpha_cumprod, t, x_start.shape) * noise
            x = x.float().to(device)

            ### feed input into model and compute loss (eq. 14 in DDPM paper)
            model_output = model(x, t)
            # loss = F.mse_loss(model_output, noise)
            loss = F.mse_loss(model_output, noise, reduction = 'none')
            loss = reduce(loss, 'b ... -> b (...)', 'mean')
            loss = loss * extract_into_tensor(p2_loss_weight, t, loss.shape)
            loss = loss.mean()

            ### Update model parameters
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.)
            optim.step()
            optim.zero_grad()

            ### Log & save
            if curr_iter % iter_log == 0:
                logger.info("[{:>6} / {:>6}]  Loss: {:.6f}".format(curr_iter, iter_train, loss.item()))
            if curr_iter % iter_save == 0:
                torch.save(model.state_dict(), os.path.join(output_folder, '{}.pth'.format(str(curr_iter).zfill(6))))
            if curr_iter > iter_train + 1:
                break
            curr_iter += 1
    logger.info("Done.")

if __name__ == "__main__":
    fire.Fire(main)