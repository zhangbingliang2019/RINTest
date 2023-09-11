import pandas as pd
from decord import VideoLoader, VideoReader
from decord import cpu, gpu
import os
import json
import tqdm
import multiprocessing as mp

def get_video_path(page_dir, video_id, video_root='/home/bingliang/data/WebVid2.5M/videos'):
    return video_root + '/{}/{}.mp4'.format(page_dir, video_id)

def check_video_path(video_path):
    try:
        if os.path.exists(video_path):
            vr = VideoReader(video_path, ctx=cpu(0))
            if len(vr) > 60 and vr[0].shape == (336, 596, 3):
                return True
        return False
    except:
        return False

def process(r, total_rank = 8):
    print('here')
    print(r)
    csv_path = '/home/bingliang/data/WebVid2.5M/results_2M_train.csv'
    #out_path = '/home/bingliang/data/WebVid2.5M/meta_{}M.json'
    meta = pd.read_csv(csv_path)
    info = []
    start = (len(meta)//total_rank + 1) * r
    end = min(start + (len(meta)//total_rank + 1), len(meta))
    for i in tqdm.trange(start, end):
        item = meta.iloc[i]
        video_path = get_video_path(item['page_dir'], item['videoid'])
        if check_video_path(video_path):
            info.append({'video_id': str(item['videoid']), 'caption': str(item['name']), 'page_dir': str(item['page_dir'])})
            if len(info) % 500_000 == 0:
                #json.dump(info, open(out_path.format(len(info)//1000000), 'w'))
                print('Saved!')

    json.dump(info, open('/home/bingliang/data/WebVid2.5M/meta_full_{}.json'.format(r), 'w'))

def main():
    proc = []
    for i in range(32):
        p = mp.Process(target=process, args=(i, 32))
        p.start()
        proc.append(p)
    for p in proc:
        p.join()


def merge():
    full = []
    for r in range(32):
        j = json.load(open('/home/bingliang/data/WebVid2.5M/meta_full_{}.json'.format(r), 'r'))
        full += j
    json.dump(full, open('/home/bingliang/data/WebVid2.5M/meta_full.json'))


if __name__ == '__main__':
    main()