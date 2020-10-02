import collections
import glob
import os
import random
import sys
import utils
import math
import cv2
import imutils
import numpy as np
from PIL import Image
from skimage.feature import hog
def main() :
    image = Image.open('/Users/hungdao/Documents/Study/LVTN/graduate-thesis/tmp/1111.png')
    fd = hog(image, orientations=9, pixels_per_cell=(32, 32),
                        cells_per_block=(2, 2), visualize=False, feature_vector=False, multichannel=None)
    print(fd.shape)
def get_ori_coor(x, y, scaleX, scaleY):
    return int(x*scaleX), int(y*scaleY)

image = cv2.imread('/Users/hungdao/Pictures/rubick.JPG')
x = 150
y = 100
win_size = 128
ori_clone = image.copy()
resized_img = imutils.resize(image, width=int(image.shape[1]/1.25))
scaleX = 1/(resized_img.shape[1]/image.shape[1])
scaleY = 1/(resized_img.shape[0]/image.shape[0])
ori_x, ori_y = get_ori_coor(x, y, scaleX, scaleY)
ori_win_size = int(win_size*1.25)
cv2.rectangle(resized_img, (x,y), (x + win_size, y + win_size), (0, 255, 0), 2)
cv2.rectangle(ori_clone, (ori_x,ori_y), (ori_x + ori_win_size, ori_y + ori_win_size), (0, 255, 0), 2)
cv2.imshow("new win", resized_img)
cv2.imshow("ori win", ori_clone)
resized_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)
new_img = utils.to_image(resized_img).save(os.path.join('/Users/hungdao/Pictures/', 'test' + '.jpg'), 'jpeg')
cv2.waitKey()
 
