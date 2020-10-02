import cv2
import numpy as np
import random
from PIL import Image


def to_float_array(img: Image.Image) -> np.ndarray:
    return np.array(img).astype(np.float32) / 255.


def to_image(values: np.ndarray) -> Image.Image:
    values = cv2.cvtColor(values, cv2.COLOR_RGB2BGR)
    img = Image.fromarray(values)
    return img

def shuffle_data(samples, samples_labeled, samples_weight, samples_weight_ada, sam_ori_idx):
    tmp = list(zip(samples, samples_labeled, samples_weight, samples_weight_ada, sam_ori_idx))
    random.shuffle(tmp)
    samples_shuffled, samples_labeled_shuffled, samples_weight_shuffled, samples_weight_ada_shuffled, sam_ori_idx = zip(*tmp)
    samples_shuffled = np.array(samples_shuffled)
    samples_labeled_shuffled = np.array(samples_labeled_shuffled)
    samples_weight_shuffled = np.array(samples_weight_shuffled)
    samples_weight_ada_shuffled = np.array(samples_weight_ada_shuffled)
    return samples_shuffled, samples_labeled_shuffled, samples_weight_shuffled, samples_weight_ada_shuffled, sam_ori_idx


def write_data(file_output, data):
    f = open(file_output, 'w+')
    f.write(str(data)+'\n')
    f.close()


if __name__ == '__main__':
    samples = np.load('samples2000.npz')['arr_0']
    labels = np.loadtxt('labels2000.txt')
    num_samples = len(samples)
    weights = np.array([1 / num_samples] * num_samples)
    shuffle_data(samples, labels, weights)