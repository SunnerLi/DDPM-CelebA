# DDPM-CelebA

Train & sampling diffusion models on CelebA

Usage
---
```cmd
### Train
$ python train.py --output_folder checkpoint/celeba-ddpm-ddim
### Sampling with DDPM
$ python sample.py --model_path checkpoint/celeba-ddpm-ddim/800000.pth --sample_fn ddpm 
### Sampling with DDIM
$ python sample.py --model_path checkpoint/celeba-ddpm-ddim/800000.pth --sample_fn ddim --sampling_timesteps 50 --ddim_sampling_eta 0.0 
```

<!-- Result
---

(CIFAR-10)
|   Source   | DDIM(eta=1.0, S=50) | DDIM(eta=0.0, S=50) |
|------------|---------------------|---------------------|
|    Paper   |         8.01        |         4.67        |
|  Reproduce |        24.67        |        17.52        | -->


Reference
---
[1](https://github.com/lucidrains/denoising-diffusion-pytorch), [2](https://github.com/CompVis/latent-diffusion) and [3](https://github.com/abarankab/DDPM)