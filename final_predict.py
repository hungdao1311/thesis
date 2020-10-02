import glob
import os
import pickle
import time
from collections import defaultdict
from datetime import datetime
import numpy as np
# import tensorflow as tf
from functools import partial
# sess = tf.Session()

from PIL import Image
from skimage.feature import hog
from numba import jit
from multiprocessing import Pool
from train import create_combination_blocks

COMBINATION_BLOCK = create_combination_blocks()


def time_now(str):
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print(str)
    print(" at Time =", current_time)


def create_single_hog(image: Image) -> np.ndarray:
    """
    Extract hog feature
    :param image: image sample need to be extracted
    :type image: Image
    :return single_hog: joint hog feature vector
    """
    single_hog = hog(image, orientations=9, pixels_per_cell=(4, 4),
                     cells_per_block=(2, 2), visualize=False, multichannel=True)
    single_hog = single_hog.reshape((49, 36))
    return single_hog


def create_joint_hog_with_spec_feature(features_index, single_hog, joint_hogs):
    coor = COMBINATION_BLOCK[features_index]
    joint_hogs[features_index] = np.concatenate(
        (single_hog[:, coor[0][0] + coor[0][1] * 7, :], single_hog[:, coor[1][0] + coor[1][1] * 7, :]), axis=1)


def calculate_threshold_func(list_preds, alphas):
    res = np.full((len(list_preds[0]),), 0.)
    for count in range(len(alphas)):
        res += alphas[count] * list_preds[count]
    return res


def remove_neg(samples_joint_hog, remove_idx):
    for feature in samples_joint_hog.keys():
        samples_joint_hog[feature] = np.delete(samples_joint_hog[feature], remove_idx, axis=0)
    return samples_joint_hog


def predict_multi(images, model_path):
    """
    Using for predict the large number of samples
    :param images: list of images need to be predicted
    :param model_path: path of model used to predict
    :return: list index of positive images
    """
    time_now("load model")
    model = pickle.load(open(model_path, 'rb'))
    images_hog = list()
    print(len(images))
    time_now("single hog")
    with Pool(6) as p:
        images_hog = p.map(create_single_hog, images)
    # for image in images:
    #     images_hog.append(create_single_hog(image))
    images_hog = np.array(images_hog)
    features_idx = list()
    time_now("joint")
    for layer in model:
        features_idx += [clf['feature'] for clf in layer.weak_clf_ensemble]
    features_idx = list(dict.fromkeys(features_idx))  # remove duplicate item
    images_joint_hog = defaultdict(list)
    for index in features_idx:
        create_joint_hog_with_spec_feature(index, images_hog, images_joint_hog)

    time_now("predict")
    index_map = np.array(range(len(images)))
    count = 0
    for layer in model:
        time_now("layer " + str(count))
        preds = list()
        alphas = list()
        for weak_clf in layer.weak_clf_ensemble:
            pred = weak_clf['clf'].predict(images_joint_hog[weak_clf['feature']].reshape(-1, 72))
            preds.append(pred)
            alphas.append(weak_clf['alpha'])
        res = calculate_threshold_func(np.array(preds), np.array(alphas))
        time_now("np_where")
        remove_idx = np.where(res < layer.threshold)
        time_now("np delete")
        res = np.delete(res, remove_idx)
        index_map = np.delete(index_map, remove_idx)
        time_now("remove neg")
        images_joint_hog = remove_neg(images_joint_hog, remove_idx)
        count = count + 1
    return index_map, res


def predict_tensorflow(images, model_path):
    time_now("Start predict")
    new_model = tf.keras.models.load_model(model_path)
    time_now("load model successfully")
    index_map = np.array(range(len(images)))
    time_now("load image successfully")
    scores = new_model.predict(images)
    res = np.argmax(scores, axis=1)
    remove_idx = np.where(res == 0)
    index_map = np.delete(index_map, remove_idx, axis=0)
    scores = np.delete(scores, remove_idx, axis=0)
    scores = np.array([i[1] for i in scores])
    time_now("End predict")
    # print(index_map)
    return index_map, scores


if __name__ == '__main__':
    # model = 'nam_model_joint_11layers_res14.sav'
    # sample_path = '/home/thaophan/Thesis/samples_test'
    # images = [Image.open('./neg_test_set/dao_body.png'), Image.open('./neg_test_set/dao_body_flipped.png'),
    #           Image.open('./pos_test_set/ngoc_bae1.png')]
    # for filename in glob.glob(os.path.join(sample_path, '*.png')):
    #     file_name_without_ext = os.path.splitext(os.path.basename(filename))[0]
    #     img_path = os.path.join(os.getcwd(), sample_path, file_name_without_ext + '.png')
    #     images += [Image.open(img_path)]
    #
    # pos_img, scores = predict_multi(images, model)
    # print(len(pos_img))
    # print(pos_img)
    start_time = time.time()
    tensoflow_path = os.path.abspath('D:\Workspace\Thesis\graduate-thesis\cnn\saved_model\my_model')
    # print(f"--- Start time --- {datetime.datetime.now()} ---")
    # multi_predict = MultiProcess(5)
    # multi_predict(predict_multiple, ('./pos_test_set', './neg_test_set', 'model6000_Fmax0.05_fmax0.1_09162020.sav'))
    pos_samples_dir = './FINAL_POS_TEST_SET'
    neg_samples_dir = './FINAL_NEG_TEST_SET'
    images = list()
    for filename in glob.glob(os.path.join(pos_samples_dir, '*.png')):
    # for img in lst_img:
        images.append(tf.keras.preprocessing.image.img_to_array(Image.open(filename)))
    images = np.array(images)
    res, _ = predict_tensorflow(images, tensoflow_path)
    print(f"Detection rate: {res.size/len(images)}")
    neg_images = list()
    for filename in glob.glob(os.path.join(neg_samples_dir, '*.png')):
    # for img in lst_img:
        neg_images.append(tf.keras.preprocessing.image.img_to_array(Image.open(filename)))
    neg_images = np.array(neg_images)
    res, _ = predict_tensorflow(neg_images, tensoflow_path)
    print(f"False pos rate: {res.size/len(neg_images)}")
