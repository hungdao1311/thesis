# USAGE
# python new_sliding.py --image imageApp/obama.jpg
import datetime
import os

import predict
import pre_processing_data
import utils
from helpers import pyramid, reverse_window
from helpers import sliding_window
import argparse
import time
import cv2

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to the image")
args = vars(ap.parse_args())

stored_path = 'D:\Workspace\Thesis\Data\ori_true_res'
result = 'D:\Workspace\Thesis\graduate-thesis\\final_output'

# load the image and define the window width and height
image = cv2.imread(args["image"])
(winW, winH) = (128, 128)

start_time = datetime.datetime.now()

# loop over the image pyramid
count = 0
level = 0
scale = 1.5
lst_res = list()
ori_clone = image.copy()
for resized_image in pyramid(image, scale):
    # loop over the sliding window for each layer of the pyramid
    for (x, y, window) in sliding_window(resized_image, stepSize=32, windowSize=(winW, winH)):
        # if the window does not meet our desired window size, ignore it
        if window.shape[0] != winH or window.shape[1] != winW:
            continue

        # THIS IS WHERE YOU WOULD PROCESS YOUR WINDOW, SUCH AS APPLYING A
        # MACHINE LEARNING CLASSIFIER TO CLASSIFY THE CONTENTS OF THE
        # WINDOW

        curWindow = (x, y, x + winW, y + winH)
        subImage = utils.to_image(resized_image).crop(curWindow)
        normalized_img = pre_processing_data.process_single_file(subImage)

        if predict.detect_single_img(normalized_img) > 0:
            # Get original window
            ori_x, ori_y, new_winW, new_winH = reverse_window(x, y, resized_image.shape[1], resized_image.shape[0], scale**level, image.shape[1], image.shape[0])
            ori_window = (ori_x, ori_y, ori_x + new_winW, ori_y + new_winH)
            ori_subImage = utils.to_image(image).crop(ori_window)
            ori_subImage.save(os.path.join(stored_path, str(count) + '_level' + str(level) + '.png'), 'png')
            count += 1

            # cv2.rectangle(resized_image, (x, y), (x + winW, y + winH), (0, 255, 0), 2)
            cv2.rectangle(ori_clone, (ori_x, ori_y), (ori_x + new_winW, ori_y + new_winH), (0, 255, 0), 2)
            # cv2.imshow("new win", resized_image)



        # since we do not have a classifier, we'll just draw the window
        # clone = resized_image.copy()
        # cv2.rectangle(clone, (x, y), (x + winW, y + winH), (0, 255, 0), 2)
        # cv2.imshow("Window", clone)
        # cv2.waitKey(1)
        # time.sleep(0.025)
    level += 1

print(f'There are {count} people')
utils.to_image(ori_clone).save(os.path.join(result, 'obama.png'), 'png')

elapsed_time = datetime.datetime.now() - start_time
print("Time elapsed --- %s seconds ---" % elapsed_time)
