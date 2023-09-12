from rin_pytorch import GaussianDiffusion, RIN, VideoTrainer
from dataset import WebVidFast
import wandb

use_wandb = False

image_size = [32, 128, 128]
patch_size = (2, 8, 8)
split_factor = 8
image_size[0] = image_size[0] // split_factor
cond = {
    'language': None,
    'first_frame': (768, image_size[1:], patch_size[1:]),
    #'entire_video': (768, image_size, patch_size)
}


model = RIN(
    dim=1024,  # model dimensions
    image_size=image_size,  # image size
    patch_size=patch_size,  # patch size
    depth=6,  # depth
    num_latents=512,  # number of latents. they used 256 in the paper
    dim_latent=768,  # can be greater than the image dimension (dim) for greater capacity
    patches_self_attn=False,
    latent_self_attn_depth=4,  # number of latent self attention blocks per recurrent step, K in the paper
    condition_types=cond,
    frame_split_factor=split_factor
)


diffusion = GaussianDiffusion(
    model,
    timesteps = 400,
    train_prob_self_cond = 0.9,  # how often to self condition on latents
    scale = 1.                   # this will be set to < 1. for more noising and leads to better convergence when training on higher resolution images (512, 1024) - input noised images will be auto variance normalized
)

dataset = WebVidFast("/home/bingliang/data/WebVid2.5M/videos",
                         "/home/bingliang/data/WebVid2.5M/meta_full.json",
                         image_size=128, frame_size=32, return_caption=True)


trainer = VideoTrainer(
    diffusion,
    dataset,
    num_samples = 16,
    train_batch_size = 6,
    gradient_accumulate_every = 1,
    train_lr = 1e-4,
    save_and_sample_every = 5000,
    train_num_steps = 1000000,         # total training steps
    ema_decay = 0.995,                # exponential moving average decay
    use_wandb = use_wandb
)

trainer.train()

