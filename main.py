from rin_pytorch import GaussianDiffusion, RIN, VideoTrainer
from dataset import WebVidFast
import wandb

use_wandb = True

if use_wandb:
    run = wandb.init(
        # Set the project where this run will be logged
         project="RIN-Cond", name='16x336x336')

model = RIN(
    dim = 768,                  # model dimensions
    image_size = (16, 336, 336),           # image size
    patch_size = (2, 8, 8),             # patch size
    depth = 6,                  # depth
    num_latents = 512,          # number of latents. they used 256 in the paper
    dim_latent = 1024,           # can be greater than the image dimension (dim) for greater capacity
    patches_self_attn=False,
    latent_self_attn_depth = 4, # number of latent self attention blocks per recurrent step, K in the paper
)



diffusion = GaussianDiffusion(
    model,
    timesteps = 400,
    train_prob_self_cond = 0.9,  # how often to self condition on latents
    scale = 1.                   # this will be set to < 1. for more noising and leads to better convergence when training on higher resolution images (512, 1024) - input noised images will be auto variance normalized
)

dataset = WebVidFast("/home/bingliang/data/WebVid2.5M/videos",
            "/home/bingliang/data/WebVid2.5M/subset_new_info.json",
                 frame_size=16, overfitting_test=False, return_caption=True, image_size=336)


trainer = VideoTrainer(
    diffusion,
    dataset,
    num_samples = 16,
    train_batch_size = 32,
    gradient_accumulate_every = 1,
    train_lr = 1e-4,
    save_and_sample_every = 5000,
    train_num_steps = 1000000,         # total training steps
    ema_decay = 0.995,                # exponential moving average decay
    use_wandb = use_wandb
)

trainer.train()

