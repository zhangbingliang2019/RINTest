import zipfile
import os
import tqdm

# input_path = '/home/bingliang/data/WebVid2.5M/videos'
#
# out_path = '/home/bingliang/data/WebVid2.5M/videos_pack.zip'
#
# zipf = zipfile.ZipFile(out_path, 'w')
# pre_len = len(os.path.dirname(input_path))
# for parent, dirnames, filenames in tqdm.tqdm(os.walk(input_path)):
#     print(filenames)
#     for filename in filenames:
#         pathfile = os.path.join(parent, filename)
#         arcname = pathfile[pre_len:].strip(os.path.sep)     #相对路径
#         zipf.write(pathfile, arcname)
# zipf.close()


# importing the zipfile module
from zipfile import ZipFile

# loading the temp.zip and creating a zip object
with ZipFile('/home/bingliang/data/WebVid2.5M/videos_pack.zip', 'r') as zObject:
    # Extracting all the members of the zip
    # into a specific location.
    zObject.extractall(
        path="/home/bingliang/data/WebVid2.5M")
