# DDPM-CIFAR10

Train & sampling denoising diffusion probabilistic models on CIFAR-10.

Usage
---
```cmd
### Train
$ python train.py --output_folder checkpoint/cifar10-diffusion
### Sampling with DDPM
$ python sample.py --model_path checkpoint/cifar10-diffusion/800000.pth --sample_fn ddpm 
### Sampling with DDIM
$ python sample.py --model_path checkpoint/cifar10-diffusion/800000.pth --sample_fn ddim --sampling_timesteps 50 --ddim_sampling_eta 0.0 
```

Result
---

|   Source   | DDIM(eta=1.0, S=50) | DDIM(eta=0.0, S=50) |
|------------|---------------------|---------------------|
|    Paper   |         8.01        |         4.67        |
|  Reproduce |        24.67        |        17.52        |


Reference
---
[1](https://github.com/lucidrains/denoising-diffusion-pytorch), [2](https://github.com/CompVis/latent-diffusion) and [3](https://github.com/abarankab/DDPM)