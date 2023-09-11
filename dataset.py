from torchvision.io import read_video
from torch.utils.data import Dataset
import torch
import torch.nn.functional as F
import json
import numpy as np
from decord import VideoLoader, VideoReader
from decord import cpu, gpu
import decord
from torch.utils.data import DataLoader


# Video Data Augmentation
def center_crop(t_video, image_size):
    # t_video: (F, C, H, W)
    scale = image_size / min(*t_video.shape[2:])
    resized_t_video = F.interpolate(t_video, scale_factor=scale, mode="bilinear")

    crop_x = (resized_t_video.size(2) - image_size) // 2
    crop_y = (resized_t_video.size(3) - image_size) // 2
    return resized_t_video[:, :, crop_x:crop_x + image_size, crop_y:crop_y + image_size]


def frame_sampling(t_video, frame_size):
    # t_video: (F, C, H, W)
    sampled_frame = (torch.linspace(0, t_video.size(0), frame_size + 1)[:frame_size] + 0.5).to(torch.int)
    return t_video[sampled_frame]


class WebVid(Dataset):
    def __init__(self, video_root, json_path, image_size=336, frame_size=16,
                 overfitting_test=False, return_caption=False):
        super().__init__()
        # ['videoid', 'name', 'page_idx', 'page_dir', 'duration', 'contentUrl']
        self.info = json.load(open(json_path, 'r'))  # list of dict: {video_id, caption, page_dir},
        self.video_paths = [video_root + '/{}/{}.mp4'.format(item['page_dir'], item['video_id']) for item in self.info]
        self.video_root = video_root
        self.frame_size = frame_size
        self.image_size = image_size
        self.return_caption = return_caption
        if overfitting_test:
            self.info = [self.info[i] for i in range(16)]

        # # filter out extra data
        # for i in tqdm.trange(meta.shape[0]):
        #     feature = meta.iloc[i]
        #     # print(feature)
        #     # print(np.asarray(feature))
        #     if isinstance(feature.iloc[3], str) and int(feature.iloc[3].split('_')[-1]) <= group_limit:
        #         self.info.append((feature.iloc[0], feature.iloc[1], feature.iloc[3]))

    def load_video(self, path):
        vr = VideoReader(path, ctx=cpu(0))
        frames_num = len(vr)
        sampled_frame = np.linspace(0, frames_num, self.frame_size, endpoint=False).astype(int)
        frames = vr.get_batch(sampled_frame)
        return frames

    def load_video_by_loader(self, path):
        vl = VideoLoader([path], ctx=[cpu(0)], shape=(self.frame_size, self.image_size, self.image_size, 3),
                         interval=1, skip=5, shuffle=0)
        return vl.next()[0]

    def __getitem__(self, item):
        """
            (0, 1) range images (C, F, H, W)
        """
        video_id, caption, page_dir = self.info[item].values()
        video_path = self.video_root + '/{}/{}.mp4'.format(page_dir, video_id)

        frames = self.load_video_by_loader(video_path)
        # normalize
        if not self.return_caption:
            return torch.from_numpy(frames.asnumpy()).permute(3, 0, 1, 2) / 255
        return torch.from_numpy(frames.asnumpy()).permute(3, 0, 1, 2) / 255, caption

    def __len__(self):
        return len(self.info)


class WebVidFast(Dataset):
    """
        all videos are of resolution (336, 596)
    """

    def __init__(self, video_root, json_path, image_size=336, frame_size=16,
                 overfitting_test=False, return_caption=False):
        super().__init__()
        self.info = json.load(open(json_path, 'r'))  # list of dict: {video_id, caption, page_dir},
        self.video_root = video_root
        self.frame_size = frame_size
        self.image_size = image_size
        self.return_caption = return_caption
        decord.bridge.set_bridge('torch')
        if overfitting_test:
            self.info = [self.info[i] for i in range(16)]
        self.width = int(self.image_size / 336.0 * 596.0)
        self.start = (self.width - self.image_size) // 2

    def load_video(self, path):
        vr = VideoReader(path, ctx=cpu(0))
        frames_num = len(vr)
        sampled_frame = np.linspace(0, frames_num, self.frame_size, endpoint=False).astype(int)
        frames = vr.get_batch(sampled_frame)
        return frames

    def load_video_by_loader(self, path):
        vl = VideoLoader([path], ctx=[cpu(0)], shape=(self.frame_size, self.image_size, self.width, 3),
                         interval=1, skip=5, shuffle=0)
        return self.center_crop(vl.next()[0])

    def center_crop(self, video):
        # (B, H, W, C)
        return video[:, :, self.start:self.start + self.image_size]

    def __getitem__(self, item):
        """
            (0, 1) range images (C, F, H, W)
        """
        video_id, caption, page_dir = self.info[item].values()
        video_path = self.video_root + '/{}/{}.mp4'.format(page_dir, video_id)

        frames = self.load_video_by_loader(video_path)
        # normalize
        if not self.return_caption:
            return frames.permute(3, 0, 1, 2) / 255.0
        return frames.permute(3, 0, 1, 2) / 255, caption

    def __len__(self):
        return len(self.info)


if __name__ == '__main__':
    import tqdm

    dataset = WebVidFast("/home/bingliang/data/WebVid2.5M/videos",
                         "/home/bingliang/data/WebVid2.5M/meta_full.json",
                         image_size=336, frame_size=16, return_caption=False)
    # import torch
    # from torchvision.utils import save_image
    # frame = []
    # frames = [dataset[i][0].permute(1,0,2,3) for i in range(4)]
    # captions = [dataset[i][1] for i in range(4)]
    # save_image(torch.cat(frames), 'dataset.png', nrow=8)
    # print(captions)
    # a = dataset[0]
    # print(type(a))
    # print(a.shape)
    # shape = set()
    # min_frames = 10000000
    # for i in tqdm.trange(len(dataset)):
    #     video = dataset[i]

    bs = 64
    dl = DataLoader(dataset, batch_size=bs, shuffle=False, pin_memory=False, num_workers=0, drop_last=True)

    import time

    for i in tqdm.trange(6):
        if i == 1:
            start = time.time()
        s = next(iter(dl))
        # print(s.shape)
        # break
    end = time.time()
    print(end - start)
    print((end-start)/320 * 100)
# (336, 596)
