import imageio
from tqdm import tqdm
import cv2

vpath = '/home/khanh/Downloads/64_20_poses.mp4'
gpath = './teaser.gif'

owriter = imageio.get_writer(gpath, fps=24)
vreader = imageio.get_reader(vpath)
for frm_idx, img in tqdm(enumerate(vreader)):
    if frm_idx % 5 == 0:
        img = cv2.resize(img, dsize=(0, 0), fx=0.3, fy=0.3)
        owriter.append_data(img)
owriter.close()
