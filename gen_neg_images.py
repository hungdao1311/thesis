import glob
import os
import sys

from pre_processing_data import resize_img
from PIL import Image


CROP_SIZE = 400
DES_IMG_SIZE = 32
DISPLACEMENT = 32

REQUIRED_DIR = str(sys.argv[1])
count = 249238
stored_path = 'D:\Project\Python\data_set\\additional_neg_set'
for filename in glob.glob(os.path.join(REQUIRED_DIR, '*.jpg')):
    im = Image.open(filename).convert('LA')
    width, height = im.size
    if CROP_SIZE > width or CROP_SIZE > height:
        im.save(os.path.join(stored_path, str(count) + '.png'), 'png')
        continue
    curr_pos_x = 0
    curr_pos_y = 0
    while curr_pos_x + CROP_SIZE <= width:
        curr_pos_y = 0
        while curr_pos_y + CROP_SIZE <= height:
            gen_im = im.crop((curr_pos_x, curr_pos_y, curr_pos_x + CROP_SIZE, curr_pos_y + CROP_SIZE))
            gen_im = resize_img(gen_im, DES_IMG_SIZE)
            gen_im.save(os.path.join(stored_path, str(count) + '.png'), 'png')
            curr_pos_y += DISPLACEMENT
            count += 1
        curr_pos_x += DISPLACEMENT
print(count)