import datetime
import glob
import os
import pickle
import time

import numpy as np
import tensorflow as tf

from PIL import Image

from train import create_combination_blocks, create_joint_hog
from train_roi import create_roi_hog
from final_predict import predict_tensorflow

COMBINATION_BLOCK = create_combination_blocks()


def predict_multiple(pos_sample_path, neg_sample_path, model_path):
    # img_path = '/home/administrator/graduate-thesis-develop/graduate-thesis-develop/negative_sample/3005.png'  # change to your dir
    # img_path = '/home/administrator/graduate-thesis-develop/graduate-thesis-develop/positive_sample/1700.png'  # change to your dir

    model = pickle.load(open(model_path, 'rb'))
    # model = tf.keras.models.load_model(model_path)
    num_layer = len(model)
    print(f'model has {num_layer} layer')
    for i in range(num_layer):
        print(f'{i}. {len(model[i].weak_clf_ensemble)}')
    true_pos = 0
    false_pos = 0
    true_neg = 0
    false_neg = 0
    # Pos train set
    for filename in glob.glob(os.path.join(pos_sample_path, '*.png')):
        file_name_without_ext = os.path.splitext(os.path.basename(filename))[0]
        img_path = os.path.join(os.getcwd(), pos_sample_path, file_name_without_ext + '.png')
        image = Image.open(img_path)
        hog_feature = create_joint_hog(image, COMBINATION_BLOCK)
        # hog_feature = create_roi_hog(image)
        expected = 1
        # print('expected result: is head&shoulder')
        curr_layer = 0
        for layer in model:
            curr_layer += 1
            if layer.predict(hog_feature) != expected:
                # print('not head&shoulder')
                false_neg += 1
                # print(f'Wrong image {img_path}')
                # os.remove(img_path)
                break
            else:
                if curr_layer == num_layer:
                    true_pos += 1

        # print('predict: is head&shoulder')

    for filename in glob.glob(os.path.join(neg_sample_path, '*.png')):
        file_name_without_ext = os.path.splitext(os.path.basename(filename))[0]
        img_path = os.path.join(os.getcwd(), neg_sample_path, file_name_without_ext + '.png')
        image = Image.open(img_path)
        hog_feature = create_joint_hog(image, COMBINATION_BLOCK)
        expected = -1
        # print('expected result: is head&shoulder')
        curr_layer = 0
        for layer in model:
            curr_layer += 1
            if layer.predict(hog_feature) < 1:
                true_neg += 1
                break
            elif curr_layer == num_layer:
                false_pos += 1
                # print(f'Wrong image {img_path}')
                # os.remove(img_path)

        # print('predict: is head&shoulder')
    print("Accuracy: ", float((true_neg + true_pos) / (true_neg + true_pos + false_pos + false_neg)))
    print("False Positive Rate", float(false_pos / (false_pos + true_neg)))
    print("False Negative Rate", float(false_neg / (false_neg + true_pos)))


def predict_single_img(img_path, expected_res):
    model_path = 'model.sav'  # change to your dir
    model = pickle.load(open(model_path, 'rb'))
    image = Image.open(img_path)
    hog_feature = create_joint_hog(image, COMBINATION_BLOCK)
    # print(hog_feature.size*hog_feature.itemsize*9660)
    expected_str = 'is head&shoulder' if expected_res == 1 else 'not head&shoulder'
    print('Predict image: ', img_path)
    print('Expected result: ', expected_str)
    curr_layer = 0
    for layer in model:
        curr_layer += 1
        print(layer.weak_clf_ensemble)
        if layer.predict(hog_feature) < 1:
            print('Predict: not head&shoulder')
            break
        elif curr_layer == len(model):
            print('Predict: is head&shoulder')


def detect_single_img(image):
    model_path = 'nam_model_joint_10layers.sav'  # change to your dir
    model = pickle.load(open(model_path, 'rb'))
    hog_feature = create_joint_hog(image, COMBINATION_BLOCK)
    # print(hog_feature.size*hog_feature.itemsize*9660)
    curr_layer = 0
    for layer in model:
        curr_layer += 1
        print(layer.weak_clf_ensemble)
        if layer.predict(hog_feature) < 1:
            return 0
        elif curr_layer == len(model):
            return 1


if __name__ == '__main__':
    start_time = time.time()
    print(f"--- Start time --- {datetime.datetime.now()} ---")
    # multi_predict = MultiProcess(5)
    # multi_predict(predict_multiple, ('./pos_test_set', './neg_test_set', 'model6000_Fmax0.05_fmax0.1_09162020.sav'))
    pos_samples_dir = './FINAL_POS_TEST_SET'
    neg_samples_dir = './FINAL_NEG_TEST_SET'
    model = 'model_joint_13layers.sav'
    # model = 'model_roi_3_curr_2020-07-11.sav'
    # model = 'D:\Workspace\Thesis\graduate-thesis\cnn\saved_model\my_model'
    # predict_multiple(pos_samples_dir, neg_samples_dir, model)
    predict_multiple(pos_samples_dir, neg_samples_dir, model)
    # predict_single_img('./neg_test_set/dao_body.png', -1)
    # predict_single_img(
    #     './neg_test_set/dao_body_flipped.png', -1)
    elapsed_time = time.time() - start_time
    print("Time elapsed --- %s seconds ---" % elapsed_time)
    print(f"--- End time --- {datetime.datetime.now()} ---")
