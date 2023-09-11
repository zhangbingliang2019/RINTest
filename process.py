import cv2
from torch.utils.data import Dataset
import torch
import json
from PIL import Image
import numpy as np
import copy
import tqdm
from language import FrozenCLIPTextEmbedder, FrozenCLIPEmbedder
import torch.multiprocessing as mp
import torch


def extend(json_path, out_path, rank, total_rank = 8):
    with torch.no_grad():
        print('I am in')
        torch.set_default_device(rank)
        print(rank)
        batch_size = 64
        clip_embedder = FrozenCLIPEmbedder().to(rank)
        #clip_text_embedder = FrozenCLIPTextEmbedder(device=rank).to(rank)
        info = json.load(open(json_path, 'r'))
        total = len(info)
        start = (total // total_rank + 1) * rank
        end = min(start + (total // total_rank + 1), total)
        new_info = []
        for t in tqdm.trange(start, end, batch_size):
            last = min(total, t+batch_size)
            captions = [info[i]['caption'] for i in range(t, last)]
            full = clip_embedder.forward(captions, rank)   # (B, L, D)
            #single = clip_text_embedder.forward(captions)   # (B, D)
            full_lists = full.chunk(batch_size)
            #single_lists = single.chunk(batch_size)
            for i in range(t, last):
                item = copy.copy(info[i])
                #item['caption_single_feature'] = single_lists[i-t].detach().cpu().numpy()
                item['caption_full_feature'] = full_lists[i-t].detach().cpu().numpy()
                new_info.append(item)
        np.save(out_path, np.array(new_info))

def merge():
    full = []
    for r in range(8):
        print(r)
        sub = np.load('/home/bingliang/data/WebVid2.5M/subset_new_info_cond-{}.npy'.format(r), allow_pickle=True).tolist()
        full += sub
    np.save('/home/bingliang/data/WebVid2.5M/subset_new_info_cond.npy', np.array(full))

def main():
    rank = range(8)
    proc = []
    mp.set_start_method('spawn')
    for r in rank:
        p = mp.Process(target=extend, args=('/home/bingliang/data/WebVid2.5M/subset_new_info.json',
                                     '/home/bingliang/data/WebVid2.5M/subset_new_info_cond-{}.npy'.format(r),
                                     r, len(rank)))
        p.start()
        proc.append(p)
    for p in proc:
        p.join()

if __name__ == '__main__':
    merge()