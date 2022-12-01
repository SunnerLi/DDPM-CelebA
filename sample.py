from train import UNet, extract_into_tensor
from torchvision.utils import save_image
from PIL import Image
from tqdm import tqdm
import torch.nn.functional as F
import numpy as np
import torch
import fire
import os

@torch.no_grad()
def ddpm_sample(x_shape, timesteps, model, device):
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

    x = torch.randn(x_shape, device=device, dtype=torch.float)
    for t in reversed(range(0, timesteps)):
        x = x.float().clamp_(-1., 1.)
        T = torch.full((x.shape[0],), t, device=device, dtype=torch.long)
        pred_noise = model(x, T)

        ### Estimate x_0 as requirement of gaussian posterior. 
        ###    x_t = \sqrt_{\cumprod_{\alpha}} * x_start + \sqrt_{1 - \cumprod_{\alpha}} * \epsilon             (Reparameterizing eq. 4 in DDPM paper page 3 below)
        ### => x_start = (1 / \sqrt_{\cumprod_{\alpha}}) * x_t - \sqrt_{1 / \cumprod_{\alpha} - 1} * \epsilon   (Transposing and divide dominate term)
        pred_x_start = extract_into_tensor(sqrt_recip_alphas_cumprod, T, x_shape) * x - \
                        extract_into_tensor(sqrt_recipm1_alphas_cumprod, T, x_shape) * pred_noise
        pred_x_start.clamp_(-1., 1.)

        ### Estimate gaussian posterior q(x_{t-1} | x_t, x_0) = N(x_{t-1}; \mu(x_t, x_0), \beta*I) as shown in eq. 6 in DDPM paper
        posterior_mean = extract_into_tensor(posterior_mean_coef1, T, x_shape) * pred_x_start + \
                            extract_into_tensor(posterior_mean_coef2, T, x_shape) * x                           # Refer to eq. 7 in DDPM paper
        posterior_var  = extract_into_tensor(posterior_variance, T, x_shape)                                    # Refer to eq. 7 in DDPM paper
        posterior_log_var_clipped = extract_into_tensor(posterior_log_variance_clipped, T, x_shape)
        noise = torch.randn_like(x, device=device) if t > 0 else 0.
        x = posterior_mean + (0.5 * posterior_log_var_clipped).exp() * noise
    return x

@torch.no_grad()
def ddim_sample(x_shape, timesteps, sampling_timesteps, model, device):
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

    ts = torch.linspace(-1, timesteps - 1, sampling_timesteps + 1)
    ts = list(reversed(ts.int().tolist()))
    ts = list(zip(ts[:-1], ts[1:]))
    x = torch.randn(x_shape, device=device, dtype=torch.float)
    for t, t_prev in ts:
        x = x.float().clamp_(-1., 1.)
        T = torch.full((x.shape[0],), t, device=device, dtype=torch.long)
        pred_noise = model(x, T)

        ### Estimate x_0 as requirement of gaussian posterior. 
        ###    x_t = \sqrt_{\cumprod_{\alpha}} * x_start + \sqrt_{1 - \cumprod_{\alpha}} * \epsilon             (Reparameterizing eq. 4 in DDPM paper page 3 below)
        ### => x_start = (1 / \sqrt_{\cumprod_{\alpha}}) * x_t - \sqrt_{1 / \cumprod_{\alpha} - 1} * \epsilon   (Transposing and divide dominate term)
        pred_x_start = extract_into_tensor(sqrt_recip_alphas_cumprod, T, x_shape) * x - \
                        extract_into_tensor(sqrt_recipm1_alphas_cumprod, T, x_shape) * pred_noise
        pred_x_start.clamp_(-1., 1.)

        if t_prev < 0:
            x = pred_x_start
            continue

        ### Generate sample via eq. 12 in DDIM paper page 5 above
        alpha = alphas_cumprod[t]
        alpha_prev = alphas_cumprod[t_prev]
        sigma = ((1 - alpha_prev) / (1 - alpha) * (1 - alpha / alpha_prev)).sqrt()
        noise = torch.randn_like(x, device=device)
        x = alpha_prev.sqrt() * pred_x_start + (1 - alpha_prev - sigma ** 2).sqrt() * pred_noise + sigma * noise
    return x


@torch.no_grad()
def ddim_sample_official(x_shape, timesteps, model, device):
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

    batch, sampling_timesteps, eta = x_shape[0], 1000, 1.0

    times = torch.linspace(-1, timesteps - 1, steps=sampling_timesteps + 1)   # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
    times = list(reversed(times.int().tolist()))
    time_pairs = list(zip(times[:-1], times[1:])) # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]

    img = torch.randn(x_shape, device = device)

    x_start = None

    for time, time_next in tqdm(time_pairs, desc = 'sampling loop time step'):
        time_cond = torch.full((batch,), time, device=device, dtype=torch.long)
        img = img.float().clamp_(-1., 1.)
        pred_noise = model(img, time_cond)

        ### Estimate x_0 as requirement of gaussian posterior. 
        ###    x_t = \sqrt_{\cumprod_{\alpha}} * x_start + \sqrt_{1 - \cumprod_{\alpha}} * \epsilon             (Reparameterizing eq. 4 in DDPM paper page 3 below)
        ### => x_start = (1 / \sqrt_{\cumprod_{\alpha}}) * x_t - \sqrt_{1 / \cumprod_{\alpha} - 1} * \epsilon   (Transposing and divide dominate term)
        x_start = extract_into_tensor(sqrt_recip_alphas_cumprod, time_cond, x_shape) * img - \
                        extract_into_tensor(sqrt_recipm1_alphas_cumprod, time_cond, x_shape) * pred_noise
        x_start.clamp_(-1., 1.)

        if time_next < 0:
            img = x_start
            continue

        alpha = alphas_cumprod[time]
        alpha_next = alphas_cumprod[time_next]

        sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
        c = (1 - alpha_next - sigma ** 2).sqrt()

        noise = torch.randn_like(img)

        img = x_start * alpha_next.sqrt() + \
                c * pred_noise + \
                sigma * noise

    return img

def main(
    model_path : str, output_folder : str = './cifar10-ddpm/sample',        # Data & IO
    batch_size : int = 64, device : str = 'cuda', num_sample : int = 64,    # Inference basic
    img_size : int = 32, img_channels : int = 3,
    timesteps : int = 1000, sample_fn : str = 'ddim',                       # Diffusion model hyper-parameters
    grid_form : bool = True,                                               # Save format
    ):
    sample_fn = {
        'ddpm': ddpm_sample,
        'ddim': ddim_sample,
    }[sample_fn]

    ### Check output path
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    model = UNet(in_channels=img_channels, out_channels=img_channels, dim=64).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    ### Sampling loop
    ctr = 0
    bar = tqdm(total=num_sample)
    while ctr < num_sample:
        imgs = sample_fn(x_shape=[batch_size, img_channels, img_size, img_size], timesteps=timesteps, sampling_timesteps=1000, model=model, device=device)
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