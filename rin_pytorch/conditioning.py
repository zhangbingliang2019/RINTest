import torch
import torch.nn as nn
import clip
import math
from itertools import repeat
import collections.abc
from enum import Enum
from transformers import CLIPTokenizer, CLIPTextModel
from torch.nn import Sequential
from einops.layers.torch import Rearrange
from timm.models.vision_transformer import PatchEmbed
import numpy as np

def get_3d_sincos_pos_embed(embed_dim, frame_grid_size, grid_size):
    grid_f = np.arange(frame_grid_size, dtype=np.float32)
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_f, grid_h, grid_w)
    grid = np.stack(grid, axis=0)
    grid = grid.transpose(0, 2, 1, 3)

    pos_embed = get_3d_sincos_pos_embed_from_grid(embed_dim, grid)
    return pos_embed


def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_1d_sincos_pos_embed(embed_dim, grid_size):
    grid = np.arange(grid_size, dtype=np.float32)
    pos_embed = get_1d_sincos_pos_embed_from_grid(embed_dim, grid)
    return pos_embed


def get_3d_sincos_pos_embed_from_grid(embed_dim, grid):
    spacial_embed_dim = embed_dim // 6 * 2
    # use half of dimensions to encode grid_h
    emb_f = get_1d_sincos_pos_embed_from_grid(embed_dim - spacial_embed_dim * 2, grid[0])
    emb_h = get_1d_sincos_pos_embed_from_grid(spacial_embed_dim, grid[1])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(spacial_embed_dim, grid[2])  # (H*W, D/2)

    emb = np.concatenate([emb_f, emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000 ** omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


class FrozenCLIPEmbedder(nn.Module):
    """(B, ) captions -> (B, 77, 768) """

    def __init__(self, version="openai/clip-vit-large-patch14", device='cuda', max_length=77):
        super().__init__()
        self.tokenizer = CLIPTokenizer.from_pretrained(version)
        self.transformer = CLIPTextModel.from_pretrained(version)
        self.max_length = max_length
        self.device = device
        self.freeze()

    def freeze(self):
        self.transformer = self.transformer.eval()
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, text, device):
        with torch.no_grad():
            batch_encoding = self.tokenizer(text, truncation=True, max_length=self.max_length, return_length=True,
                                            return_overflowing_tokens=False, padding="max_length", return_tensors="pt")
            tokens = batch_encoding["input_ids"].to(device)
            outputs = self.transformer(input_ids=tokens)

            z = outputs.last_hidden_state
        return z

    def encode(self, text):
        return self(text)


class ImageEmbedder(nn.Module):
    """
        (B, C, H, W) -> (B, L, D)
    """

    def __init__(self, dim, image_size, patch_size, in_channels=3):
        super().__init__()
        assert patch_size[0] == patch_size[1]
        self.embedder = PatchEmbed(image_size, patch_size[0], in_channels, dim, norm_layer=nn.LayerNorm)

        num_patches = (image_size[0] // patch_size[0]) * (image_size[1] // patch_size[1])
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, dim), requires_grad=False)
        pos_embed = get_2d_sincos_pos_embed(dim, image_size[0] // patch_size[0])
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

    def forward(self, image):
        return self.embedder(image) + self.pos_embed


class VideoEmbedder(nn.Module):
    """
        (B, C, F, H, W) -> (B, L, D)
    """
    def __init__(self, dim, image_size, patch_size, in_channels=3):
        super().__init__()
        self.proj = nn.Conv3d(in_channels, dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(dim)

        num_patches = (image_size[0] // patch_size[0]) * (image_size[1] // patch_size[1]) * (image_size[2] // patch_size[2])
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, dim), requires_grad=False)
        pos_embed = get_3d_sincos_pos_embed(dim, image_size[0] // patch_size[0], image_size[1] // patch_size[1])
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)  # NCFHW -> NLC
        x = self.norm(x)
        return x + self.pos_embed


def get_embedder(k, v):
    if k == 'language':
        return FrozenCLIPEmbedder(), 768
    elif k == 'first_frame':
        return ImageEmbedder(*v), v[0]
    elif k == 'entire_video':
        return VideoEmbedder(*v), v[0]
    else:
        raise NotImplementedError

def prepare_cond(k, data, caption, device, frame_split_factor):
    if k == 'language':
        if frame_split_factor is None:
            return caption, device
        new_caption = []
        for text in caption:
            new_caption += [text] * frame_split_factor
        return new_caption, device
    elif k == 'first_frame':
        if frame_split_factor is None:
            return data[:, :, 0]
        frame_size = data.shape[2]
        indices = list(range(0, frame_size, frame_size//frame_split_factor))
        return data[:, :, indices].permute(0, 2, 1, 3, 4).flatten(0, 1)
    elif k == 'entire_video':
        if frame_split_factor is None:
            return data
        b, c, f, h, w = data.shape
        new_data = data.reshape(b, c, frame_split_factor, -1, h, w)
        new_data = new_data.permute(0, 2, 1, 3, 4, 5).flatten(0, 1)
        return new_data
    else:
        raise NotImplementedError


if __name__ == '__main__':
    a = VideoEmbedder(768, (16, 256,256), patch_size=(2, 4,4)).cuda()
    b = torch.randn((4, 3, 16, 256, 256)).cuda()
    x = a(b)
    print(x.shape)
