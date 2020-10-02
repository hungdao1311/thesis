# USAGE
# python cnn.py --image D:\Workspace\Thesis\graduate-thesis\imageApp\meet5.jpg
import datetime
import os

from PIL import Image
import tensorflow as tf
import numpy as np
import predict
import pre_processing_data
import utils
from helpers import pyramid, reverse_window, run_NMS
from helpers import sliding_window
from final_predict import predict_tensorflow
import argparse
import time
import cv2

# Check if the server/ instance is having GPU/ CPU from python code
import sys

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to the image")
args = vars(ap.parse_args())

stored_path = 'D:\Workspace\Thesis\Data\improve_true_res'
result = 'D:\Workspace\Thesis\graduate-thesis\\final_output'
overlapDir = 'D:\Workspace\Thesis\graduate-thesis\overlapping_output'
model_path = './nam_model_joint_12layers_res8.sav'
tensoflow_path = os.path.abspath('D:\Workspace\Thesis\graduate-thesis\cnn\saved_model\my_model')

# load the image and define the window width and height
image = cv2.imread(args["image"], cv2.IMREAD_COLOR)
# Get image name for output
imageName = args["image"].split("\\")[1]
(winW, winH) = (100, 100)

start_time = datetime.datetime.now()

# loop over the image pyramid
count = 0
level = 0
scale = 1.1
lst_img = list()
lst_imgDetail = list()
ori_clone = image.copy()
overlapImg = image.copy()

# for overlapping
lstRect = list()
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

        lst_img.append(normalized_img)
        imgDetail = (x, y, level, resized_image)
        lst_imgDetail.append(imgDetail)

        # since we do not have a classifier, we'll just draw the window
        # clone = resized_image.copy()
        # cv2.rectangle(clone, (x, y), (x + winW, y + winH), (0, 255, 0), 2)
        # cv2.imshow("Window", clone)
        # cv2.waitKey(1)
        # time.sleep(0.025)

    level += 1
# Predict all window
images = list()
for img in lst_img:
    images.append(tf.keras.preprocessing.image.img_to_array(img))
images = np.array(images)
lst_indexPositive, positive_scores = predict_tensorflow(images, tensoflow_path)

for i in lst_indexPositive:
    # print(lst_imgDetail[i])
    subX, subY, subLevel, subImg = lst_imgDetail[i]
    ori_x, ori_y, new_winW, new_winH = reverse_window(subX, subY, subImg.shape[1], subImg.shape[0],
                                                      scale ** subLevel, image.shape[1], image.shape[0], winW, winH)
    # Get positive image and save it
    ori_window = (ori_x, ori_y, ori_x + new_winW, ori_y + new_winH)
    ori_subImage = utils.to_image(image).crop(ori_window)
    # ori_subImage.save(os.path.join(stored_path, str(count) + '_level' + str(subLevel) + '.png'), 'png')
    count += 1
    # Draw rectangle on output image
    cv2.rectangle(ori_clone, (ori_x, ori_y), (ori_x + new_winW, ori_y + new_winH), (0, 255, 0), 2)

    lstRect.append(ori_window)

# print(f'There are {count} people')
# RGB_img = cv2.cvtColor(ori_clone, cv2.COLOR_BGR2RGB)
# utils.to_image(ori_clone).save(os.path.join(result, imageName), 'png')

overlappedLst = run_NMS(lstRect, positive_scores, 0.3)
print(lstRect)
for i in overlappedLst:
    print(i)
    x1, y1, x2, y2 = lstRect[i]
    cv2.rectangle(overlapImg, (x1, y1), (x2, y2), (0, 255, 0), 2)

# for (x, y, width, height) in overlappedLst:
#     cv2.rectangle(overlapImg, (x, y), (x + width, y + height), (0, 255, 0), 2)

cv2.imshow("before overlapping", ori_clone)
cv2.imshow("after overlapping", overlapImg)

cv2.waitKey()

elapsed_time = datetime.datetime.now() - start_time
print("Time elapsed --- %s seconds ---" % elapsed_time)
