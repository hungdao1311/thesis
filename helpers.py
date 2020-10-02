# import the necessary packages
import imutils
import numpy as np


def pyramid(image, scale, minSize=(256, 256)):
# def pyramid(image, scale, minSize=(512, 512)):
    # yield the original image
    yield image

    # keep looping over the pyramid
    while True:
        # compute the new dimensions of the image and resize it
        w = int(image.shape[1] / scale)
        image = imutils.resize(image, width=w)

        # if the resized image does not meet the supplied minimum
        # size, then stop constructing the pyramid
        if image.shape[0] < minSize[1] or image.shape[1] < minSize[0]:
            break

        # yield the next image in the pyramid
        yield image


def sliding_window(image, stepSize, windowSize):
    # slide a window across the image
    for y in range(0, image.shape[0], stepSize):
        for x in range(0, image.shape[1], stepSize):
            # yield the current window
            yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])


def reverse_window(x, y, sub_imgW, sub_imgH, scale, oriW, oriH, winW=128, winH=128):
    if scale == 1:
        return int(x), int(y), int(winW), int(winH)
    else:
        scaleX = 1 / (sub_imgW / oriW)
        scaleY = 1 / (sub_imgH / oriH)

        newX = x * scaleX
        newY = y * scaleY
        return int(newX), int(newY), int(scale * winW), int(scale * winH)


def run_NMS(rects, scores, thresh):
    #####################
    # refer to https://github.com/rbgirshick/fast-rcnn/blob/master/lib/utils/nms.py
    x1 = np.array([i[0] for i in rects])
    y1 = np.array([i[1] for i in rects])
    x2 = np.array([i[2] for i in rects])
    y2 = np.array([i[3] for i in rects])

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep
