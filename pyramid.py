# USAGE
# python pyramid.py --image imageApp/obama.jpg

import predict
import pre_processing_data
# import imutils
from helpers import pyramid
from helpers import sliding_window
import argparse
import time
import cv2

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to the image")
ap.add_argument("-s", "--scale", type=float, default=1.5, help="scale factor size")
args = vars(ap.parse_args())

# load the image and define the window width and height
image = cv2.imread(args["image"])
# (winW, winH) = (128, 128)

for (i, resized) in enumerate(pyramid(image, scale=args["scale"])):
    # show the resized image
    cv2.imshow("Layer {}".format(i + 1), resized)
    cv2.waitKey(0)
