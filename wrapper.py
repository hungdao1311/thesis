# USAGE
# python sliding_improvement.py --image imageApp/obama.jpg
import datetime
import os

import predict
import pre_processing_data
import utils
from helpers import pyramid, reverse_window, run_NMS
from helpers import sliding_window
from final_predict import predict_multi, time_now
import argparse
import cv2

model_path = 'D:\Workspace\Thesis\graduate-thesis\model_joint_13layers.sav'
stored_path = 'D:\Workspace\Thesis\graduate-thesis\\final_output'
(winW, winH) = (128, 128)
stepSize = 12
scale = 1.25

# for overlapping
lstRect = list()


def scan_image(path):
    # load the image and define the window width and height
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    # loop over the image pyramid
    level = 0
    lst_img = list()
    lst_imgDetail = list()
    ori_clone = image.copy()
    overlapImg = image.copy()
    for resized_image in pyramid(image, scale):
        # loop over the sliding window for each layer of the pyramid
        for (x, y, window) in sliding_window(resized_image, stepSize=stepSize, windowSize=(winW, winH)):
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

        level += 1

    # Predict all window
    lst_indexPositive, positive_scores = predict_multi(lst_img, model_path)
    time_now("fusing")
    for i in lst_indexPositive:
        subX, subY, subLevel, subImg = lst_imgDetail[i]
        ori_x, ori_y, new_winW, new_winH = reverse_window(subX, subY, subImg.shape[1], subImg.shape[0],
                                                          scale ** subLevel, image.shape[1], image.shape[0], winW, winH)
        # Get positive image and save it
        ori_window = (ori_x, ori_y, ori_x + new_winW, ori_y + new_winH)
        # Draw rectangle on output image
        cv2.rectangle(ori_clone, (ori_x, ori_y), (ori_x + new_winW, ori_y + new_winH), (0, 255, 0), 2)

        lstRect.append(ori_window)
    overlappedLst = run_NMS(lstRect, positive_scores, 0.05)
    time_now("draw rect")
    for i in overlappedLst:
        x1, y1, x2, y2 = lstRect[i]
        cv2.rectangle(overlapImg, (x1, y1), (x2, y2), (0, 255, 0), 2)

    result_path = os.path.join(stored_path, 'result.png')
    not_overlap = os.path.join(stored_path, 'be4.png')
    utils.to_image(ori_clone).save(not_overlap, 'png')
    utils.to_image(overlapImg).save(result_path, 'png')
    return result_path, len(overlappedLst)
