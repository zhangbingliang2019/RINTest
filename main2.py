from rin_pytorch import RIN, VideoTrainer
import torch

image_size = [16, 128, 128]
patch_size = (2, 8, 8)
split_factor = 4
image_size[0] = image_size[0] // split_factor
cond = {
    'language': None,
    'first_frame': (768, image_size[1:], patch_size[1:]),
    'entire_video': (768, image_size, patch_size)
}


model = RIN(
    dim=768,  # model dimensions
    image_size=image_size,  # image size
    patch_size=patch_size,  # patch size
    depth=2,  # depth
    num_latents=512,  # number of latents. they used 256 in the paper
    dim_latent=1024,  # can be greater than the image dimension (dim) for greater capacity
    patches_self_attn=False,
    latent_self_attn_depth=1,  # number of latent self attention blocks per recurrent step, K in the paper
    condition_types=cond,
    frame_split_factor=split_factor
)

data = torch.randn(2, 3, 16, 128, 128)
caption = ['a girl', 'a boy']
time = torch.tensor([1, 3]*4)
new_data, cond = model.prepare_data_and_cond(data, caption, device='cpu')
out = model.forward(new_data, time, cond)
print(new_data.shape)
for k, v in cond.items():
    print(k)
    print(v)
print(out.shape)
