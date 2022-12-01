import os
import fire
import torch
import random
import numpy as np
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from torchvision.utils import save_image
from unet import UNet, extract_into_tensor

@torch.no_grad()
def main(
    model_path : str = './cifar10-ddpm/800000.pth', output_folder : str = './cifar10-ddpm/sample',  # Data & IO
    batch_size : int = 64, device : str = 'cuda', num_sample : int = 64, seed : int = 0,            # Inference basic
    img_size : int = 32, img_channels : int = 3,                                                    # Image shape
    timesteps : int = 1000, sampling_timesteps : int = 50, sample_fn : str = 'ddim',                # Diffusion model hyper-parameters
    grid_form : bool = True,                                                                        # Save format
    ):

    ### For reproduce result
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    ### Check output path
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    model = UNet(in_channels=img_channels, out_channels=img_channels, dim=64).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

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
    ##########################################################################################################

    ### Sampling loop
    ctr = 0
    bar = tqdm(total=num_sample)
    x_shape=[batch_size, img_channels, img_size, img_size]
    while ctr < num_sample:
        imgs = torch.randn(x_shape, device=device, dtype=torch.float)
        if sample_fn == 'ddpm':
            for t in reversed(range(0, timesteps)):
                imgs = imgs.float().clamp_(-1., 1.)
                T = torch.full((x_shape[0],), t, device=device, dtype=torch.long)
                pred_noise = model(imgs, T)

                ### Estimate x_0 as requirement of gaussian posterior. 
                ###    x_t = \sqrt_{\cumprod_{\alpha}} * x_start + \sqrt_{1 - \cumprod_{\alpha}} * \epsilon             (Reparameterizing eq. 4 in DDPM paper page 3 below)
                ### => x_start = (1 / \sqrt_{\cumprod_{\alpha}}) * x_t - \sqrt_{1 / \cumprod_{\alpha} - 1} * \epsilon   (Transposing and divide dominate term)
                pred_x_start = extract_into_tensor(sqrt_recip_alphas_cumprod, T, x_shape) * imgs - \
                                extract_into_tensor(sqrt_recipm1_alphas_cumprod, T, x_shape) * pred_noise
                pred_x_start.clamp_(-1., 1.)

                ### Estimate gaussian posterior q(x_{t-1} | x_t, x_0) = N(x_{t-1}; \mu(x_t, x_0), \beta*I) as shown in eq. 6 in DDPM paper
                posterior_mean = extract_into_tensor(posterior_mean_coef1, T, x_shape) * pred_x_start + \
                                    extract_into_tensor(posterior_mean_coef2, T, x_shape) * imgs                           # Refer to eq. 7 in DDPM paper
                posterior_var  = extract_into_tensor(posterior_variance, T, x_shape)                                    # Refer to eq. 7 in DDPM paper
                posterior_log_var_clipped = extract_into_tensor(posterior_log_variance_clipped, T, x_shape)
                noise = torch.randn_like(img, device=device) if t > 0 else 0.
                imgs = posterior_mean + (0.5 * posterior_log_var_clipped).exp() * noise
        elif sample_fn == 'ddim':
            ts = torch.linspace(-1, timesteps - 1, sampling_timesteps + 1)
            ts = list(reversed(ts.int().tolist()))
            ts = list(zip(ts[:-1], ts[1:]))
            for t, t_prev in ts:
                imgs = imgs.float().clamp_(-1., 1.)
                T = torch.full((x_shape[0],), t, device=device, dtype=torch.long)
                pred_noise = model(imgs, T)

                ### Estimate x_0 as requirement of gaussian posterior. 
                ###    x_t = \sqrt_{\cumprod_{\alpha}} * x_start + \sqrt_{1 - \cumprod_{\alpha}} * \epsilon             (Reparameterizing eq. 4 in DDPM paper page 3 below)
                ### => x_start = (1 / \sqrt_{\cumprod_{\alpha}}) * x_t - \sqrt_{1 / \cumprod_{\alpha} - 1} * \epsilon   (Transposing and divide dominate term)
                pred_x_start = extract_into_tensor(sqrt_recip_alphas_cumprod, T, x_shape) * imgs - \
                                extract_into_tensor(sqrt_recipm1_alphas_cumprod, T, x_shape) * pred_noise
                pred_x_start.clamp_(-1., 1.)

                if t_prev < 0:
                    imgs = pred_x_start
                    continue

                ### Generate sample via eq. 12 in DDIM paper page 5 above
                alpha = alphas_cumprod[t]
                alpha_prev = alphas_cumprod[t_prev]
                sigma = ((1 - alpha_prev) / (1 - alpha) * (1 - alpha / alpha_prev)).sqrt()
                noise = torch.randn_like(imgs, device=device)
                imgs = alpha_prev.sqrt() * pred_x_start + (1 - alpha_prev - sigma ** 2).sqrt() * pred_noise + sigma * noise
        else:
            raise NotImplementedError()

        imgs = (imgs / 2 + 0.5)

        if grid_form:
            save_image(imgs, os.path.join(output_folder, '{}.png'.format(str(ctr).zfill(6))))
            ctr += imgs.shape[0]
            bar.update(n=imgs.shape[0])
        else:
            imgs = imgs.permute(0, 2, 3, 1).cpu().numpy()
            for img in imgs:
                img = Image.fromarray((img * 255).astype(np.uint8))
                img.save(os.path.join(output_folder, '{}.png'.format(str(ctr).zfill(6))))
                ctr += 1
                bar.update(n=1)

if __name__ == '__main__':
    fire.Fire(main)