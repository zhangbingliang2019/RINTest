import math
from pathlib import Path
from random import random
from functools import partial
from multiprocessing import cpu_count
from language import FrozenCLIPEmbedder

import torch
from torch import nn, einsum
from torch.special import expm1
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from torch.optim import Adam
from torchvision import transforms as T, utils

from beartype import beartype

from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange

from rin_pytorch.attend import Attend

from PIL import Image
from tqdm.auto import tqdm
from ema_pytorch import EMA

from accelerate import Accelerator, DistributedDataParallelKwargs
from rin_pytorch.rin_pytorch import GaussianDiffusion

# helpers functions

def exists(x):
    return x is not None


def identity(x):
    return x


def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d


def divisible_by(numer, denom):
    return (numer % denom) == 0


def safe_div(numer, denom, eps=1e-10):
    return numer / denom.clamp(min=eps)


def cycle(dl):
    while True:
        for data in dl:
            yield data


def has_int_squareroot(num):
    num_sqrt = math.sqrt(num)
    return int(num_sqrt) == num_sqrt


def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr


def convert_image_to(img_type, image):
    if image.mode != img_type:
        return image.convert(img_type)
    return image


def Sequential(*mods):
    return nn.Sequential(*filter(exists, mods))



# trainer class

@beartype
class VideoTrainer(object):
    def __init__(
            self,
            diffusion_model: GaussianDiffusion,
            dataset,
            *,
            train_batch_size=16,
            gradient_accumulate_every=1,
            train_lr=1e-4,
            train_num_steps=100000,
            max_grad_norm=1.,
            ema_update_every=10,
            ema_decay=0.995,
            betas=(0.9, 0.99),
            save_and_sample_every=1000,
            num_samples=25,
            results_folder='./results',
            amp=True,
            mixed_precision_type='fp16',
            split_batches=True,
            convert_image_to=None
    ):
        super().__init__()

        self.accelerator = Accelerator(
            split_batches=split_batches,
            mixed_precision=mixed_precision_type if amp else 'no',
            kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)]
        )

        self.model = diffusion_model
        self.text_encoder = FrozenCLIPEmbedder()

        assert has_int_squareroot(num_samples), 'number of samples must have an integer square root'
        self.num_samples = num_samples
        self.save_and_sample_every = save_and_sample_every

        self.batch_size = train_batch_size
        self.gradient_accumulate_every = gradient_accumulate_every
        self.max_grad_norm = max_grad_norm

        self.train_num_steps = train_num_steps
        self.image_size = diffusion_model.image_size

        # dataset and dataloader

        dl = DataLoader(dataset, batch_size=train_batch_size, shuffle=True, pin_memory=True, num_workers=8)

        dl = self.accelerator.prepare(dl)
        self.dl = cycle(dl)

        # optimizer

        self.opt = Adam(diffusion_model.parameters(), lr=train_lr, betas=betas)

        # for logging results in a folder periodically

        self.results_folder = Path(results_folder)

        if self.accelerator.is_local_main_process:
            self.results_folder.mkdir(exist_ok=True)

        if self.accelerator.is_main_process:
            self.ema = EMA(diffusion_model, beta=ema_decay, update_every=ema_update_every)

        # step counter state

        self.step = 0

        # prepare model, dataloader, optimizer with accelerator

        self.model, self.opt, self.text_encoder = self.accelerator.prepare(self.model, self.opt, self.text_encoder)

    def save(self, milestone):
        if not self.accelerator.is_local_main_process:
            return

        data = {
            'step': self.step + 1,
            'model': self.accelerator.get_state_dict(self.model),
            'opt': self.opt.state_dict(),
            'ema': self.ema.state_dict(),
            'scaler': self.accelerator.scaler.state_dict() if exists(self.accelerator.scaler) else None
        }

        torch.save(data, str(self.results_folder / f'model-{milestone}.pt'))

    def load(self, milestone):
        data = torch.load(str(self.results_folder / f'model-{milestone}.pt'))

        model = self.accelerator.unwrap_model(self.model)
        model.load_state_dict(data['model'])

        self.step = data['step']
        self.opt.load_state_dict(data['opt'])

        if self.accelerator.is_main_process:
            self.ema.load_state_dict(data['ema'])

        if exists(self.accelerator.scaler) and exists(data['scaler']):
            self.accelerator.scaler.load_state_dict(data['scaler'])

    def train(self):
        accelerator = self.accelerator
        device = accelerator.device
        self.text_encoder.device = device

        with tqdm(initial=self.step, total=self.train_num_steps, disable=not accelerator.is_main_process) as pbar:

            while self.step < self.train_num_steps:

                total_loss = 0.

                for _ in range(self.gradient_accumulate_every):
                    data, caption = next(self.dl)
                    data = data.to(device)
                    # obtain language embedding
                    cond = self.text_encoder(caption)

                    with accelerator.autocast():
                        loss = self.model(data, cond)
                        loss = loss / self.gradient_accumulate_every
                        total_loss += loss.item()

                    accelerator.backward(loss)

                pbar.set_description(f'loss: {total_loss:.4f}')

                accelerator.wait_for_everyone()
                accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

                self.opt.step()
                self.opt.zero_grad()

                accelerator.wait_for_everyone()

                # save milestone on every local main process, sample only on global main process

                if accelerator.is_local_main_process:
                    milestone = self.step // self.save_and_sample_every
                    save_and_sample = self.step != 0 and self.step % self.save_and_sample_every == 0

                    if accelerator.is_main_process:
                        self.ema.to(device)
                        self.ema.update()

                        if save_and_sample:
                            self.ema.ema_model.eval()

                            with torch.no_grad():
                                batches = num_to_groups(self.num_samples, self.batch_size)
                                all_video_list = list(map(lambda n: self.ema.ema_model.sample(batch_size=n), batches))

                            all_video = torch.cat(all_video_list, dim=0)
                            all_video = all_video.permute(0, 2, 1, 3, 4).flatten(0, 1)
                            utils.save_image(all_video, str(self.results_folder / f'sample-{milestone}.png'),
                                             nrow=int(math.sqrt(all_video.shape[0])))

                    if save_and_sample:
                        self.save(milestone)

                self.step += 1
                pbar.update(1)

        accelerator.print('training complete')
