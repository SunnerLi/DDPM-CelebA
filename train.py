import os
import fire
import torch
import random
import logging
import numpy as np
import torch.nn.functional as F
from unet import UNet, extract_into_tensor
from torchvision import datasets, transforms

def main(
    data_folder : str = "~/data", output_folder : str = './cifar10-diffusion',      # Data & IO
    batch_size : int = 32, device : str = 'cuda', seed : int = 0,                   # Training basic
    img_size : int = 32, img_channels : int = 3,                                    # Image shape
    iter_log : int = 500, iter_save : int = 100000, iter_train : int = 800000,      # Time stage
    timesteps : int = 1000,                                                         # Diffusion model hyper-parameters
    ):

    ### For reproduce result
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

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
        ), batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True,
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
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)                                     # \sqrt_{1 - \bar_\alpha_t}
    sqrt_recip_alphas_cumprod = torch.sqrt(1. / alphas_cumprod)                                         # 1 / \sqrt_{\bar_\alpha_t}
    sqrt_recipm1_alphas_cumprod = torch.sqrt(1. / alphas_cumprod - 1)

    ### Calculation for gaussian posterior q(x_{t-1} | x_t, x_0)
    posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)                     # Refer to telta beta definition in DDPM paper page 3
    posterior_log_variance_clipped = torch.log(posterior_variance.clamp(min=1e-20))
    posterior_mean_coef1 = (torch.sqrt(alphas_cumprod_prev) * betas) / (1. - alphas_cumprod)            # Refer to eq. 7 in DDPM paper
    posterior_mean_coef2 = (torch.sqrt(alphas) * (1. - alphas_cumprod_prev)) / (1. - alphas_cumprod)    # Refer to eq. 7 in DDPM paper
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
                    extract_into_tensor(sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
            x = x.float().to(device)

            ### feed input into model and compute loss (eq. 14 in DDPM paper)
            model_output = model(x, t)
            loss = F.mse_loss(model_output, noise)

            ### Update model parameters
            loss.backward()
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