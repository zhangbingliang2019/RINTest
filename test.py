from rin_pytorch import GaussianDiffusion, RIN, VideoTrainer
from dataset import WebVidFast
import wandb
import torch

model = RIN(
    dim=768,  # model dimensions
    image_size=(16, 336, 336),  # image size
    patch_size=(2, 8, 8),  # patch size
    depth=6,  # depth
    num_latents=512,  # number of latents. they used 256 in the paper
    dim_latent=1024,  # can be greater than the image dimension (dim) for greater capacity
    patches_self_attn=False,
    latent_self_attn_depth=4,  # number of latent self attention blocks per recurrent step, K in the paper
).cuda()

step = 10
bs = 4
data = [torch.randn(bs, 3, 16, 336, 336).cuda() for i in range(step)]
time = [torch.randint(0, 10, size=(bs,)).cuda() for i in range(step)]
cond = [torch.randn(bs, 77, 768).cuda() for i in range(step)]
import time

start = time.time()
for x in data:
    out = model(data, time, cond)
end = time.time()
print((end - start) / step)
print((end -start) / step)
