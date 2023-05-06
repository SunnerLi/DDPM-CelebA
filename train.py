import os
import fire
import torch
import logging
import itertools
import lightning as L
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from pathlib import Path
from torchvision import transforms
from torchvision.datasets import DatasetFolder
from model import UNet, Encoder, Decoder, extract_into_tensor
from dataset import CustomParallelDataset

torch.set_float32_matmul_precision('medium')

data_read_fn = lambda path: Image.open(path)
cond_read_fn = lambda path: Image.open(path)

def main(
    output_folder: str = './celeba-ddpm-ddimm/800000.pth',                  # IO
    data_folder: str = "~/data/celeba", cond_folder: str = None,            # Data (please assign absolute path)
    data_size: int = 128, data_channels: int = 3, cond_channels: int = 3,   # Data metadata
    batch_size: int = 32, device: str = 'cuda', seed: int = 0,              # Training basic
    save_every_n_epochs: int = 1, train_epochs: int = 2,                    # Time
    kl_loss_weight: float = 0.0001,                                         # Loss weight (Stable diffusion use 1.0)
    timesteps : int = 1000,                                                 # Diffusion model hyper-parameters
):
    L.seed_everything(seed=seed)
    os.makedirs(output_folder, exist_ok=True)
    logging.basicConfig(filename=Path(output_folder) / 'log.txt', level=logging.DEBUG)
    logger = logging.getLogger()

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

    ############################################ Stage 1 Training ############################################
    ### Loading dataset
    dataset = CustomParallelDataset(
        root_dict={'x': [data_folder]},
        loader_dict={'x': data_read_fn},
        extensions_dict={'x': ['jpg']},
        transform=transforms.Compose([
            transforms.CenterCrop(data_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]),
        cache=False, 
        trim=None,
    )
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)

    ### Define auto-encoder model and optimizer
    model_en = Encoder(in_channels=data_channels, out_channels=data_channels*2, dim=64, attention_mid_only=True)
    model_de = Decoder(in_channels=data_channels, out_channels=data_channels, dim=64, attention_mid_only=True)
    optim_en = torch.optim.Adam(model_en.parameters(), lr=1e-4)
    optim_de = torch.optim.Adam(model_de.parameters(), lr=1e-4)

    ### Set up fabric for stage-1 training
    fabric = L.Fabric(accelerator=device)
    fabric.launch()
    model_en, optim_en = fabric.setup(model_en, optim_en)
    model_de, optim_de = fabric.setup(model_de, optim_de)
    dataloader = fabric.setup_dataloaders(dataloader)

    ### Training loop (1st stage)
    for epoch in range(train_epochs):
        bar = tqdm(dataloader)
        for batch_idx, batch in enumerate(bar):
            input = batch['x']
            optim_en.zero_grad()
            optim_de.zero_grad()

            ### foreward
            middle = model_en(input, t=None)
            mean, logvar = torch.chunk(middle, chunks=2, dim=1)
            logvar = torch.clamp(logvar, -30.0, 20.0)
            middle_ = mean + torch.exp(0.5 * logvar) * torch.randn(mean.shape).to(mean.device)
            output = model_de(middle_, t=None)

            ### compute KL-divergence loss & reconstruction loss
            loss_kl = (0.5 * torch.sum(torch.pow(mean, 2) + torch.exp(logvar) - 1.0 - logvar)) / middle.shape[0]
            loss_rec = F.mse_loss(output, input, reduction="mean")
            loss_sum = loss_rec + loss_kl * kl_loss_weight

            ### backward
            fabric.backward(loss_sum)
            optim_en.step()
            optim_de.step()

            ### Print loss information
            log = "[Stage1] {:>3} / {:>4} | L_kl: {:.4f}  L_rec: {:.4f}  L_sum: {:.4f}".format(epoch, batch_idx, loss_kl, loss_rec, loss_sum)
            bar.set_description(log)
            if batch_idx % 100: logger.info(log)

    ############################################ Stage 2 Training ############################################
    ### Freeze auto-encoder and do not train in stage-2 training
    for param in itertools.chain(model_en.parameters(), model_de.parameters()):
        param.requires_grad = False

    ### Load conditional dataset if needed and form dataset for stage-2 training
    if cond_folder:
        dataset = CustomParallelDataset(
            root_dict={'x': [data_folder], 'c': [cond_folder]},
            loader_dict={'x': data_read_fn, 'c': cond_read_fn},
            extensions_dict={'x': ['jpg'], 'c': ['jpg']},
            transform=transforms.Compose([
                transforms.CenterCrop(data_size),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]),
            cache=False, 
            trim=None,
        )
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)

    ### Define diffusion model and optimizer
    model_cd = Encoder(in_channels=cond_channels, out_channels=cond_channels, dim=64, attention_mid_only=True)
    model_df = UNet(in_channels=data_channels+cond_channels if cond_folder else data_channels, out_channels=data_channels, dim=64)
    optim_cd = torch.optim.Adam(model_cd.parameters(), lr=1e-4)
    optim_df = torch.optim.Adam(model_df.parameters(), lr=1e-4)

    ### Set up fabric for stage-2 training
    model_cd, optim_cd = fabric.setup(model_cd, optim_cd)
    model_df, optim_df = fabric.setup(model_df, optim_df)
    dataloader = fabric.setup_dataloaders(dataloader)

    ### Training loop (2nd)
    for epoch in range(train_epochs):
        bar = tqdm(dataloader)
        for batch_idx, batch in enumerate(bar):
            optim_cd.zero_grad()
            optim_df.zero_grad()

            ### Form input
            x_start, _ = torch.chunk(model_en.forward(batch['x'], t=None), chunks=2, dim=1)

            ### Random sample time step & noise
            t = torch.randint(0, timesteps, (batch_size,), device=device).long()
            noise = torch.randn_like(x_start, device=device).float()

            ### forward process to sample q(x_t | x_0) in simplified form. Please Refer to Alg. 1 in DDPM paper
            x = extract_into_tensor(sqrt_alphas_cumprod, t, x_start.shape) * x_start + \
                    extract_into_tensor(sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
            x = x.float().to(device)
            xc = torch.cat([x, model_cd.forward(batch['c'], t=None)], dim=1) if cond_folder else x

            ### feed input into model and compute loss (eq. 14 in DDPM paper)
            model_output = model_df(xc, t)
            loss = F.mse_loss(model_output, noise)

            ### Update model parameters
            fabric.backward(loss)
            optim_cd.step()
            optim_df.step()

            ### Print loss information
            log = "[Stage2] {:>3} / {:>4} | L: {:.4f}".format(epoch, batch_idx, loss)
            bar.set_description(log)
            if batch_idx % 100: logger.info(log)

        if (epoch+1) % save_every_n_epochs == 0: 
            fabric.save(
                path=Path(output_folder) / f'{str(epoch).zfill(4)}.ckpt', 
                state={
                    'en': model_en.state_dict(), 'de': model_de.state_dict(),
                    'df': model_df.state_dict(), 'cd': model_cd.state_dict(),}
            )

if __name__ == "__main__":
    fire.Fire(main)