import os
import fire
import torch
import random
import numpy as np
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from torchvision.utils import save_image
from model import UNet, Encoder, Decoder, extract_into_tensor

@torch.no_grad()
def main(
    model_path: str = './celeba-ddpm-ddimm/0000.ckpt',                                  # Input
    output_folder : str = './celeba-ddpm-ddimm/sample',                                 # Output
    data_size: int = 128, data_channels: int = 3, cond_channels: int = None,            # Data metadata
    batch_size: int = 32, device: str = 'cuda', seed: int = 0, num_sample : int = 64,   # Inference basic
    timesteps : int = 1000, sampling_timesteps : int = 50, 
    sample_fn : str = 'ddim', ddim_sampling_eta : float = 1.0,                          # Diffusion model hyper-parameters
    grid_form : bool = True,                                                            # Save format
):

    ### For reproduce result
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    ### Check output path
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    model_df = UNet(in_channels=data_channels+cond_channels if cond_channels else data_channels, out_channels=data_channels, dim=64).to(device)
    model_en = Encoder(in_channels=data_channels, out_channels=data_channels*2, dim=64, attention_mid_only=True).to(device)
    model_de = Decoder(in_channels=data_channels, out_channels=data_channels, dim=64, attention_mid_only=True).to(device)
    state_dict = torch.load(model_path)
    model_df.load_state_dict(state_dict['df'])
    model_en.load_state_dict(state_dict['en'])
    model_de.load_state_dict(state_dict['de'])
    model_df.eval()
    model_en.eval()
    model_de.eval()
    if cond_channels:
        model_cd = Encoder(in_channels=cond_channels, out_channels=cond_channels, dim=64, attention_mid_only=True).to(device)
        model_cd.load_state_dict(state_dict['cd'])
        model_cd.eval()

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

    ### Sampling loop
    ctr = 0
    bar = tqdm(total=num_sample)
    x_shape=[batch_size, data_channels+cond_channels if cond_channels else data_channels, data_size // 4, data_size // 4]
    while ctr < num_sample:
        x = torch.randn(x_shape, device=device, dtype=torch.float)
        if sample_fn == 'ddpm':
            for t in reversed(range(0, timesteps)):
                x = x.float()
                T = torch.full((x_shape[0],), t, device=device, dtype=torch.long)
                pred_noise = model_df(x, T)

                ### Estimate x_0 as requirement of gaussian posterior. 
                ###    x_t = \sqrt_{\cumprod_{\alpha}} * x_start + \sqrt_{1 - \cumprod_{\alpha}} * \epsilon             (Reparameterizing eq. 4 in DDPM paper page 3 below)
                ### => x_start = (1 / \sqrt_{\cumprod_{\alpha}}) * x_t - \sqrt_{1 / \cumprod_{\alpha} - 1} * \epsilon   (Transposing and divide dominate term)
                pred_x_start = extract_into_tensor(sqrt_recip_alphas_cumprod, T, x_shape) * x - \
                                extract_into_tensor(sqrt_recipm1_alphas_cumprod, T, x_shape) * pred_noise
                pred_x_start.clamp_(-1., 1.)

                ### Estimate gaussian posterior q(x_{t-1} | x_t, x_0) = N(x_{t-1}; \mu(x_t, x_0), \beta*I) as shown in eq. 6 in DDPM paper
                posterior_mean = extract_into_tensor(posterior_mean_coef1, T, x_shape) * pred_x_start + \
                                    extract_into_tensor(posterior_mean_coef2, T, x_shape) * x                        # Refer to eq. 7 in DDPM paper
                posterior_var  = extract_into_tensor(posterior_variance, T, x_shape)                                    # Refer to eq. 7 in DDPM paper
                posterior_log_var_clipped = extract_into_tensor(posterior_log_variance_clipped, T, x_shape)
                noise = torch.randn_like(x, device=device) if t > 0 else 0.
                x = posterior_mean + (0.5 * posterior_log_var_clipped).exp() * noise
            x = x.float()
            imgs = model_de(x, t=None)
        elif sample_fn == 'ddim':
            ts = torch.linspace(-1, timesteps - 1, sampling_timesteps + 1)
            ts = list(reversed(ts.int().tolist()))
            ts = list(zip(ts[:-1], ts[1:]))
            for t, t_prev in ts:
                x = x.float()
                T = torch.full((x_shape[0],), t, device=device, dtype=torch.long)
                pred_noise = model_df(x, T)

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
                sigma = ddim_sampling_eta * ((1 - alpha_prev) / (1 - alpha) * (1 - alpha / alpha_prev)).sqrt()
                noise = torch.randn_like(x, device=device)
                x = alpha_prev.sqrt() * pred_x_start + (1 - alpha_prev - sigma ** 2).sqrt() * pred_noise + sigma * noise
            x = x.float()
            imgs = model_de(x, t=None)
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