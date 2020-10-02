import collections
import datetime
import glob
import math
import os
import random
import sys
import time
import logging

import utils
import itertools
import Layer_Roi
import pickle
import importlib

import numpy as np

from sklearn.svm import SVC
from PIL import Image
from skimage.feature import hog
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from numba import jit
from enum import Enum

# from thundersvm import SVC

NUM_FEATURES = 30
CELL_PER_BLOCK = 2
PIXEL_PER_CELL = 4
IMAGE_SIZE = 32
ROI_MIN_SIZE = 3

DEFAULT_NEG_DIR = 'D:\Workspace\Thesis\graduate-thesis\\negative_sample'
DEFAULT_NEG_HOG_FEATURE_DIR = 'D:\Workspace\Thesis\Data\\negative_roi_hog_3'

NEG_IMAGES = glob.glob(os.path.join(DEFAULT_NEG_DIR, '*.png'))

logging.shutdown()
importlib.reload(logging)

logging.basicConfig(filename='train_roi' + str(ROI_MIN_SIZE),
                    filemode='w',
                    format='%(asctime)s, %(levelname)s %(message)s',
                    datefmt='%H:%M:%S',
                    level=logging.INFO)

logger = logging.getLogger('train_roi')

TrainStatus = Enum('TrainStatus', 'Running_normally Running_after_reset')


class LayerInfo:
    def __init__(self, kwargs):
        self.__dict__.update(kwargs)


def clf_on_strong_clf(samples, labels, weak_clf_ensemble, threshold, sam_idx=None):
    samples_hypothesis = run_strong_classifier_multi(samples, weak_clf_ensemble, threshold)
    return get_clf_result(samples, labels, samples_hypothesis, threshold)


@jit
def get_clf_result(samples, labels, samples_hypothesis, threshold):
    num_samples = len(samples)
    false_pos = 0
    true_pos = 0
    false_neg = 0
    true_neg = 0
    count = 0
    for i in range(num_samples):
        count += 1
        # sample_pred = run_strong_classifier(samples[i], weak_clf_ensemble, threshold)
        sample_pred = 1 if samples_hypothesis[i] >= threshold else 0
        if sample_pred == 1:
            if sample_pred != labels[i]:
                false_pos += 1
            else:
                true_pos += 1
        else:
            if sample_pred == labels[i]:
                true_neg += 1
            else:
                false_neg += 1
    return false_pos, true_pos, false_neg, true_neg, count


def run_strong_classifier_multi(samples: np.ndarray, weak_classifiers, threshold):
    # Loop through all our layer: if sample return TRUE, move to next layer, otherwise return FALSE
    # See https://viblo.asia/p/haar-cascade-la-gi-luan-ve-mot-ky-thuat-chuyen-dung-de-nhan-biet-cac-khuon-mat-trong-anh-E375zamdlGW
    # for more information on cascade classifier
    # here is just how we calculate output for ONE layer
    sum_hypotheses = 0

    for c in weak_classifiers:
        a = np.array([sample[c['feature']].reshape((1, 36)) for sample in samples]).reshape((len(samples), 36))
        sum_hypotheses += c['alpha'] * c['clf'].predict(a)
    return sum_hypotheses


def create_combination_blocks():
    num_block = int(IMAGE_SIZE / PIXEL_PER_CELL) - CELL_PER_BLOCK + 1
    block_coordinates = list()
    for i in range(num_block):
        for j in range(num_block):
            block_coordinates.append((i, j))
    return list(itertools.combinations(block_coordinates, 2))


def create_single_hog(image: Image) -> np.ndarray:
    """
    Extract hog feature
    :param image: image sample need to be extracted
    :type image: Image
    :return fd: joint hog feature vector
    """
    fd, hog_image = hog(image, orientations=9, pixels_per_cell=(4, 4),
                        cells_per_block=(2, 2), visualize=True, multichannel=True)
    fd = fd.reshape((49, 36))
    return fd


def create_joint_hog(image: Image, combination_blocks) -> np.ndarray:
    """
    Extract joint hog feature
    :param image: image sample need to be extracted
    :type image: Image
    :return np.array(joint_hog): numpy array of joint hog feature
    """
    fd = create_single_hog(image)
    joint_hog = list()
    for coor in combination_blocks:
        joint_hog.append(np.array([fd[coor[0][0] + coor[0][1] * 7], fd[coor[1][0] + coor[1][1] * 7]]))
    return np.array(joint_hog)


def create_roi_hog(image: Image):
    scales = [(1, 1), (2, 1), (1, 2)]
    roi_hog = np.empty((0, 36))
    for scale in scales:
        for cell_size in range(ROI_MIN_SIZE, (int((IMAGE_SIZE / 2) / max(scale)) + 1)):
            fd = hog(image, orientations=9, pixels_per_cell=(cell_size * scale[0], cell_size * scale[1]),
                                cells_per_block=(2, 2), visualize=False, multichannel=True, feature_vector=False)
            fd = fd.reshape((fd.shape[0] * fd.shape[1], fd.shape[2] * fd.shape[3] * fd.shape[4]))
            roi_hog = np.append(roi_hog, fd, 0)
    return np.array(roi_hog)


def explore_weak_clf(samples, labels, num_pos, num_neg, weights, weights_ada, train_param, sam_ori_index):
    """
    :param samples: líst of samples
    :param labels:
    :param weights:
    :return selected_clf, min_err: the classifier is added to strong classifier and its err value
    """
    # selected_feature = random.sample(range(0, len(samples[0]) - 1), NUM_FEATURES)
    logger.info(f'Start exploring weak classifier on {len(samples[0])} blocks')
    selected_feature = random.sample(range(0, len(samples[0])), NUM_FEATURES)
    min_err = sys.maxsize
    selected_clf = None
    feature_idx = None
    num_samples = len(samples)

    # samples, labels, weights, weights_ada, sam_ori_index = utils.shuffle_data(samples, labels, weights, weights_ada,
    #                                                                           sam_ori_index)
    count = 1
    for idx in selected_feature:
        samples_feature = list()
        for i in range(len(samples)):
            samples_feature.append(samples[i][idx])
        # l = [feature for feature in samples[idx][selected_feature[idx]] for idx in range(len(samples))]
        samples_feature = np.array(samples_feature)
        samples_feature = samples_feature.reshape((num_samples, 36))
        labels = labels.reshape((num_samples,))
        weights = weights.reshape((num_samples,))
        weights_ada = weights_ada.reshape((num_samples,))

        # split data
        # data = train_test_split(samples_feature, labels, weights, test_size=0.3)
        # samples_feature_train = data[0]
        # samples_feature_test = data[1]
        # labels_train = data[2]
        # labels_test = data[3]
        # weights_train = data[4]
        # weights_test = data[5]

        # svm_clf = svm_generator(samples_feature, labels)
        # clf_err = 1 - svm_clf.score(samples_feature, labels, weights)
        svm_clf = svm_generator(samples_feature, labels, num_pos, num_neg, weights_ada, train_param)
        # clf_err = 1 - svm_clf.score(samples_feature_test, labels_test, weights_test)
        clf_err = 0
        # logger.info(f'sum weight {weights_ada.sum()}')
        preds = svm_clf.predict(samples_feature)
        clf_err += get_err_weak_clf(samples_feature, labels, preds, weights_ada)

        # logger.info(f'{count}.err: {clf_err}, index: {idx}')
        logger.info("%s.err: %s, index: %s" % (count, clf_err, idx))

        if clf_err < min_err:
            min_err = clf_err
            selected_clf = svm_clf
            feature_idx = idx
        count += 1
    return selected_clf, min_err, feature_idx


@jit
def get_err_weak_clf(samples_feature, labels, preds, weights_ada):
    clf_err = 0
    for index in range(len(samples_feature)):
        # pred = svm_clf.predict(samples_feature[index].reshape((1, 36)))[0]
        if preds[index] == 1:
            pred = 1
        else:
            pred = 0
        # pred = pred if pred == 1 else 0
        clf_err += weights_ada[index] * abs(pred - labels[index])
    return clf_err



def run_weak_classifier(x: np.ndarray, c) -> int:
    """This is where we execute the weak classifier (could be changed depends on how we use scikit-learn)"""
    x = x.reshape((1, 36))
    return 1 if c.predict(x)[0] == 1 else 0



def initial_weight(labels, num_pos, num_neg):
    num_samples = len(labels)
    weights = list()
    for i in range(num_samples):
        if labels[i] == 1:
            weight = 1 / (2 * num_pos)
        else:
            weight = 1 / (2 * num_neg)
        weights.append(weight)
    weights = np.array(weights)  # initial sample weight
    return weights



def predict_neg_set(weak_cls_ensemble, threshold, start, end, neg_samples_path=DEFAULT_NEG_HOG_FEATURE_DIR):
    logger.info(f'Predict neg set - expected start: {start} - expected end: {end}')
    if not neg_samples_path:
        neg_samples_path = str(sys.argv[2])
    list_file = glob.glob(os.path.join(neg_samples_path, '*.npz'))
    if start > len(list_file):
        start = 0
    if end > len(list_file):
        end = len(list_file)
    logger.info(f'actual start: {start} - actual end: {end}')
    logger.info(f"Start predicting using strong clf on {end - start} negative samples")
    false_pos = 0
    true_neg = 0

    for index in range(start, end):
        sample_path = os.path.join(neg_samples_path, str(index) + '.npz')
        sample = np.load(sample_path)['arr_0']
        sample_pred = run_strong_classifier(sample, weak_cls_ensemble, threshold)
        if sample_pred == 1:
            false_pos += 1
        else:
            true_neg += 1

    return false_pos, true_neg



def svm_generator(samples: np.array, labels: np.array, num_pos, num_neg, weights, train_param):
    """
    Generate SVM classifier
    :param samples: list of sample feature used to classify
    :param labels: list of label
    :return svm_clf: SVM classifier
    """

    svm_clf = SVC(kernel='linear', C=train_param[0])
    svm_clf = svm_clf.fit(samples, labels)
    # if num_pos / num_neg > 50:
    #     svm_clf = svm.SVC(C=train_param[3])
    # elif num_pos / num_neg > 20:
    #     svm_clf = svm.SVC(C=train_param[2])
    # elif num_pos / num_neg > 2:
    #     svm_clf = svm.SVC(C=train_param[1])
    # else:
    #     svm_clf = svm.SVC(C=train_param[0])
    return svm_clf


# This is the strong classifier
# params: @x: sample, @strong_classifier: list of layers (each contains a list of weak classifiers)
# How we get the output of the classifier is described in the example bellow

def run_strong_classifier(x: np.ndarray, weak_classifiers, threshold):
    # Loop through all our layer: if sample return TRUE, move to next layer, otherwise return FALSE
    # See https://viblo.asia/p/haar-cascade-la-gi-luan-ve-mot-ky-thuat-chuyen-dung-de-nhan-biet-cac-khuon-mat-trong-anh-E375zamdlGW
    # for more information on cascade classifier
    # here is just how we calculate output for ONE layer
    sum_hypotheses = 0.
    sum_alphas = 0.

    for c in weak_classifiers:
        sum_hypotheses += c['alpha'] * run_weak_classifier(x[c['feature']], c['clf'])
        sum_alphas += c['alpha']
    return 1 if (sum_hypotheses >= threshold) else 0



def run_cascaded_detector(x: np.ndarray, layers: list):
    for layer in layers:
        pred = layer.predict(x)
        if pred < 1:
            return pred
    return 1


# Build the strong classifier as described in our document, input params are pos/neg samples and desired model accuracy

def build_strong_classifier(samples_file, labels_file, train_param, train_status=TrainStatus.Running_normally):
    F_target = 0.00001
    if train_status == TrainStatus.Running_normally:
        num_layer = 0
        D_layer = 1
        F_layer = 1
        layers = list()
        samples = np.load(samples_file + '.npz')['arr_0']
        labels = np.loadtxt(labels_file)
        num_pos = len(samples) / 2
        num_neg = num_pos
        ori_num_samples = len(samples)
        start_next_neg_count = int(ori_num_samples/2)
        previous_neg_count = 0
        curr_neg_count = int(ori_num_samples/2)
        real_img_index = np.array(list(range(int(ori_num_samples / 2))))
    else:
        layers = load_model('curr_model.sav')
        num_layer = len(layers)
        samples = np.load(samples_file + '_curr.npz')['arr_0']
        labels = np.loadtxt(labels_file)
        ori_num_samples = len(samples) - 1
        saved_param = load_param()
        logger.info(f'Load successfully {saved_param}')
        if num_layer == samples[-1][0][0] and num_layer == samples[-1]:
            samples = samples[:-1]
            labels = samples[:-1]
            start_next_neg_count = saved_param.start_next_neg_count
            real_img_index = saved_param.real_img_index
            print(len(real_img_index))
        else:
            samples = samples[:int(ori_num_samples/2)]
            labels = labels[:int(ori_num_samples/2)]
            negs, neg_labels, start_next_neg_count, real_img_index = get_neg_samples(0, int(ori_num_samples / 2), layers)
            if negs and neg_labels.size:
                samples = np.concatenate([samples, negs])
                labels = np.concatenate([labels, neg_labels])
        num_pos = len(samples) / 2
        num_neg = num_pos
        D_layer = saved_param.D_layer
        F_layer = saved_param.F_layer

    while F_layer > F_target:
        num_layer += 1
        logger.info("Load data completed")
        num_samples = len(samples)
        logger.info(f"Staring training on {num_samples} records on layer {num_layer}")
        logger.info(f'weights - pos: {1 / (2 * num_pos)}, neg: {1 / (2 * num_neg)}')
        weights = initial_weight(labels, num_pos, num_neg)
        weights_ada = weights.copy()
        f_max = train_param['f_max']
        weak_cls_ensemble = list()
        fi = 1
        d_min = 0.995
        d = D_layer
        sum_alphas = 0
        threshold = 0
        sam_ori_idx = list(range(num_samples))
        while fi > f_max:
            err = 1
            while err > 0.5:
                weak_clf, err, feature_idx = explore_weak_clf(samples, labels, num_pos, num_neg, weights, weights_ada,
                                                              train_param['C'], sam_ori_idx)
            if err == 0:
                err = sys.float_info.epsilon
            alpha = np.log((1 - err) / err)
            weak_cls_ensemble.append({'clf': weak_clf, 'alpha': alpha, 'feature': feature_idx})
            curr_wae_clf_index = len(weak_cls_ensemble)
            logger.info(f'End exploring weak classifier  {curr_wae_clf_index}')
            logger.info(f'Running weak classifier {curr_wae_clf_index} with error {err} and alpha {alpha}')
            logger.info(f'Samples length: {len(samples)}')
            sum_alphas += alpha
            threshold = 0.5 * sum_alphas
            logger.info(f'alpha {alpha} --- threshold {threshold}')
            wrong_samples = list()
            for i in range(num_samples):
                sample_pred = run_weak_classifier(samples[i][feature_idx], weak_clf)
                if labels[i] == 0 and sample_pred != labels[i]:
                    wrong_samples.append(i)
                weights_ada[i] = weights_ada[i] * math.pow(err / (1 - err), (1 - abs(sample_pred - labels[i])))
            weights_ada = weights_ada / weights_ada.sum()  # normalize sample weight

            real_wrong_img = list()
            for idx in wrong_samples:
                real_wrong_img += [real_img_index[idx - int(num_samples / 2)]]
            real_wrong_img = np.array([real_wrong_img])
            logger.info(f'Wrong images {real_wrong_img}')

            num_samples = len(samples)
            logger.info(
                f"Run strong classifier on {int(ori_num_samples / 2 + curr_neg_count)} samples with {len(weak_cls_ensemble)} weak clf")

            # Running strong classifier on whole dataset
            false_pos, true_pos, false_neg, true_neg, _ = clf_on_strong_clf(samples, labels, weak_cls_ensemble,
                                                                            threshold,
                                                                            sam_ori_idx)
            logger.info(f'False pos: {false_pos} --- False neg: {false_neg}')
            logger.info(f"True neg samples: {true_neg} --- True pos: {true_pos}")
            d = true_pos / (true_pos + false_neg)
            fi = false_pos / (false_pos + true_neg)
            logger.info(f'fi: {fi}')
            logger.info(f'd: {d}')

            # Adjust threshold
            upper_bound = sum_alphas
            lower_bound = 0
            while not np.isclose((upper_bound - lower_bound), 0):
                # if d >= d_min:
                #     break
                if len(weak_cls_ensemble) < 2:
                    break
                if d < d_min:
                    upper_bound = threshold
                    threshold = (lower_bound + threshold) / 2
                else:
                    lower_bound = threshold
                    threshold = (upper_bound + threshold) / 2
                # Running strong classifier on whole dataset
                false_pos, true_pos, false_neg, true_neg, _ = clf_on_strong_clf(samples, labels, weak_cls_ensemble,
                                                                             threshold)
                d = true_pos / (true_pos + false_neg)
                fi = false_pos / (false_pos + true_neg)

            if d < d_min:
                if np.isclose(threshold, 0):
                    threshold = 0
                logger.info(f'd before: {d} --- fi before: {fi}')
                d = 1
                fi = 1
                false_neg = true_neg = 0
                false_pos = true_pos = len(samples)

            logger.info(f'final threshold: {threshold}')
            logger.info(f'False pos after adjusted threshold {false_pos}')
            logger.info(f"True neg samples after adjusted threshold {true_neg}")
            logger.info(f'fi after adjusted threshold : {fi}')
            logger.info(f'd after adjusted threshold: {d}')

        F_layer = F_layer * fi
        D_layer = D_layer * d
        logger.info(f'F_layer: {F_layer} --- D_layer: {D_layer}')
        layers.append(Layer_Roi.Layer(threshold, weak_cls_ensemble))
        file_name = 'model_roi_' + str(ROI_MIN_SIZE) + '_curr_' + str(datetime.datetime.now().date()) + '.sav'
        save_model(layers, file_name)
        # run strong classifier on neg set, only keep a sample that is detected false
        neg_sample_idx = list()
        for i in range(len(samples)):
            if labels[i] == 0:
                neg_sample_idx.append(i)

        logger.info(len(neg_sample_idx))
        neg_sample_removed = list()
        num_missed_neg = 0
        for idx in neg_sample_idx:
            sample_pred = run_cascaded_detector(samples[idx], layers)
            if sample_pred == labels[idx]:
                neg_sample_removed.append(idx)
            else:
                num_missed_neg += 1

        logger.info(f"Deleted {len(neg_sample_removed)} negative samples, remain {num_missed_neg} negative samples")
        samples = np.delete(samples, neg_sample_removed, axis=0)
        labels = np.delete(labels, neg_sample_removed)
        real_img_index = np.delete(real_img_index, np.array(neg_sample_removed) - int(ori_num_samples/2))

        new_negs, new_neg_labels, start_next_neg_count, real_add_img_index = get_neg_samples(start_next_neg_count,
                                                                         int(ori_num_samples / 2 - num_missed_neg),
                                                                         layers)

        if new_negs and new_neg_labels.size:
            samples = np.concatenate([samples, new_negs])
            labels = np.concatenate([labels, new_neg_labels])
            real_img_index = np.concatenate([real_img_index, real_add_img_index])
        samples_file = 'samples_roi_' + str(ROI_MIN_SIZE) + '_curr'
        labels_file = 'labels_roi_' + str(ROI_MIN_SIZE) + '_curr.txt'
        if ROI_MIN_SIZE == 3:
            shape = (1, 357, 36)
        else:
            shape = (1, 204, 36)
        np.savez_compressed(samples_file, np.concatenate([samples, np.full(shape, num_layer)]))
        np.savetxt(labels_file, np.concatenate([labels, np.array([num_layer])]), fmt='%f')
        saved_param = {
            'F_layer': F_layer,
            'D_layer': D_layer,
            'start_next_neg_count': start_next_neg_count
        }
        save_param(saved_param)
        num_neg = len(samples) - ori_num_samples / 2
        logger.info(f'num negs: {num_neg}')

    logger.info("Ending training")
    return layers


def main():
    pos_sample_path = 'D:\Workspace\Thesis\graduate-thesis\positive_sample'
    neg_sample_path = 'D:\Workspace\Thesis\graduate-thesis\\negative_sample'
    labels = np.array([])
    samples = list(np.array([]))

    logger.info("Start extracting Roi Hog")
    # Pos train set
    pos_list = glob.glob(os.path.join(pos_sample_path, '*.png'))
    for index in range(len(pos_list)):
        file_name_without_ext = os.path.splitext(os.path.basename(str(index)))[0]
        img_path = os.path.join(os.getcwd(), pos_sample_path, file_name_without_ext + '.png')
        image = Image.open(img_path)
        # logger.info(f'Converting file {img_path} to numpy')
        samples.append(create_roi_hog(image))
        labels = np.append(labels, 1)

    pos_length = len(pos_list)
    logger.info(f'Pos sample length: {pos_length}')

    # Neg train set
    neg_list = glob.glob(os.path.join(neg_sample_path, '*.png'))
    for index in range(len(neg_list))[0:pos_length]:
        file_name_without_ext = os.path.splitext(os.path.basename(str(index)))[0]
        img_path = os.path.join(os.getcwd(), neg_sample_path, file_name_without_ext + '.png')
        image = Image.open(img_path)
        # logger.info(f'Converting file {img_path} to numpy')
        samples.append(create_roi_hog(image))
        labels = np.append(labels, 0)

    neg_length = len(neg_list)
    logger.info(f'Neg sample length: {neg_length}')

    samples_file = 'samples_roi_' + str(ROI_MIN_SIZE)
    labels_file = 'labels_roi_' + str(ROI_MIN_SIZE) + '.txt'
    np.savez_compressed(samples_file, samples)
    np.savetxt(labels_file, labels, fmt='%f')
    logger.info("Saved complete")
    return samples_file, labels_file
    # build_strong_classifier(samples, labels)


def store_neg_image(neg_samples_path=DEFAULT_NEG_DIR, stored_path=DEFAULT_NEG_HOG_FEATURE_DIR, start=0, end=0):
    if not neg_samples_path:
        neg_samples_path = str(sys.argv[2])
    list_file = glob.glob(os.path.join(neg_samples_path, '*.png'))
    if end == 0 or end > len(list_file):
        end = len(glob.glob(os.path.join(neg_samples_path, '*.png')))
    if start > len(list_file):
        start = 0
    if not os.path.exists(stored_path):
        os.mkdir(stored_path)
    logger.info("Start extracting Roi Hog for negative samples")
    count = 3680
    for index in range(start, end):
        img_path = os.path.join(os.getcwd(), neg_samples_path, str(index) + '.png')
        image = Image.open(img_path)
        hog_extracted = create_roi_hog(image)
        stored_file = os.path.join(stored_path, str(count))
        np.savez_compressed(stored_file, hog_extracted)
        count += 1

    logger.info(f'Neg samples length: {int(end - start)}')
    logger.info('Saved completed')


def get_neg_samples(start, num_sample_required, layers, end=None, neg_samples_path=DEFAULT_NEG_HOG_FEATURE_DIR):
    logger.info('Start getting additional neg samples')
    logger.info(f'expected start: {start} - expected end: {end}')
    if not neg_samples_path:
        neg_samples_path = str(sys.argv[2])
    file_count = len(glob.glob(os.path.join(neg_samples_path, '*.npz')))
    if start > file_count:
        start = file_count
    if not end or end > file_count:
        end = file_count

    logger.info(f'actual start: {start} - actual end: {end}')
    neg_samples = list(np.array([]))
    neg_labels = np.array([])
    new_img_idx = list()
    for index in range(start, end):
        sample_path = os.path.join(neg_samples_path, str(index) + '.npz')
        sample = np.load(sample_path)['arr_0']
        pred = run_cascaded_detector(sample, layers)
        if pred != 0:
            neg_samples.append(sample)
            neg_labels = np.append(neg_labels, 0)
            new_img_idx.append(index)
        if len(neg_samples) == num_sample_required:
            end = index + 1
            break
    logger.info(f'Pick {len(neg_samples)} difficult samples from {int(end - start)} negative samples')
    return neg_samples, neg_labels, end, new_img_idx


def train(num_training_times, train_param):
    start = 1
    for index in range(num_training_times):
        # start_time = time.time()
        # logger.info(f"{index} --- Start time --- {datetime.datetime.now()} ---")
        # run main if samples are not generated before
        samples_file = 'samples_roi_' + str(ROI_MIN_SIZE)
        labels_file = 'labels_roi_' + str(ROI_MIN_SIZE) + '.txt'
        # samples_file = 'samples_roi_3'
        # labels_file = 'labels_roi_3.txt'
        # samples_file, labels_file = main()
        model = build_strong_classifier(samples_file, labels_file, train_param[index % num_training_times], TrainStatus.Running_normally)
        file_name = 'model_roi_' + str(ROI_MIN_SIZE) + '_' + str(index + start) + '_' + str(
            train_param[index % num_training_times]['f_max']) + '_' + str(
            datetime.datetime.now().date()) + '.sav'
        logger.info("Start save model")
        pickle.dump(model, open(file_name, 'wb'))
        logger.info("Save completed")
        # elapsed_time = time.time() - start_time
        # logger.info("Time elapsed --- %s seconds ---" % elapsed_time)
        # logger.info(f"--- End time --- {datetime.datetime.now()} ---")
        # loaded_model = pickle.load(open(filename, 'rb'))


def save_model(model, stored_file):
    model_save_name = stored_file
    pickle.dump(model, open(model_save_name, 'wb'))
    logger.info(f'Saved successfully model {model_save_name}')


def load_model(model):
    model_save_name = model
    model = pickle.load(open(model_save_name, 'rb'))
    logger.info(f'Load successfully model')
    return model


def save_param(saved_param):
    with open('saved_param.pkl', 'wb') as output:
        layer_info = LayerInfo(saved_param)
        pickle.dump(layer_info, output, pickle.HIGHEST_PROTOCOL)
    logger.info(f'Saved successfully param {saved_param}')


def load_param():
    with open('saved_param.pkl', 'wb') as input_param:
        return pickle.load(input_param)



if __name__ == '__main__':
    # image = Image.open('./neg_test_set/dao_body.png')
    # hog_feature = create_roi_hog(image)
    # print(hog_feature)
    train_param = [
        {'f_max': 0.65, 'C': [1]},
    ]
    train(1, train_param)
    # store_neg_image()
    # store_neg_image(neg_samples_path='D:\Workspace\Thesis\Data\\additional_neg_set')
    # main()
    # a = LayerInfo({'a':1, 'b': 3})
