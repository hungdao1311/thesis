import glob
import os
import sys
from xml.etree import ElementTree
from PIL import Image, ImageOps

SIZE_32 = 32
SIZE_64 = 64
FOLDER_NAME = 'positive_sample'


def extract_boxes(file_name):
    # load and parse the file
    tree = ElementTree.parse(file_name)

    # get the root of the document
    root = tree.getroot()

    boxes = list()
    # extract each bounding box
    for box in root.findall('.//bndbox'):
        x_min = int(box.find('xmin').text)
        y_min = int(box.find('ymin').text)
        x_max = int(box.find('xmax').text)
        y_max = int(box.find('ymax').text)
        coors = (x_min, y_min, x_max, y_max)
        boxes.append(coors)
    width = int(root.find('.//size/width').text)
    height = int(root.find('.//size/height').text)
    return width, height, boxes


# def resize_img()

def write_data(file_output, data):
    f = open(file_output, 'w+')
    n = len(data)
    for i in range(n):
        f.write(str(data[i]) + '\n')
    f.close()
    return True


def extract_label_box(image_path, box_coordinates):
    """
    Extract label box from input image
    :param: image_path:
    :param: box_coordinates: list of coordinates of label box
    :return: list of images which have been extracted
    """
    im_list = list()
    im = Image.open(image_path).convert('LA')
    print(image_path)
    for coordinate in box_coordinates:
        im_list += [im.crop(coordinate)]
    return im_list


def resize_img(img, des_size):
    old_size = img.size  # old_size[0] is in (width, height) format
    ratio = float(des_size) / max(old_size)
    new_size = tuple([int(x * ratio) for x in old_size])

    img = img.resize(new_size, Image.ANTIALIAS)
    # create a new image and paste the resized on it

    new_img = Image.new("RGB", (des_size, des_size))
    new_img.paste(img, ((des_size - new_size[0]) // 2,
                        (des_size - new_size[1]) // 2))
    return new_img


def flip_img(img):
    return img.transpose(Image.FLIP_LEFT_RIGHT)


def process_single_file(image):
    resized_img_32 = resize_img(image, SIZE_32)
    return resized_img_32

def main():
    img_dir_path = str(sys.argv[1])
    annotation_path = str(sys.argv[2])
    if not os.path.exists(FOLDER_NAME):
        os.mkdir(FOLDER_NAME)
    # print(glob.glob(os.path.join(annotation_path, '*.xml')))
    # print(glob.glob(os.path.join(annotation_path, '*.txt')))
    count = 2239
    for filename in glob.glob(os.path.join(annotation_path, '*.xml')):
        file_path = os.path.join(os.getcwd(), filename)
        file_name_without_ext = os.path.splitext(os.path.basename(filename))[0]
        print(f'Parsing file {file_path}')
        _, _, obj_coordinates = extract_boxes(file_path)
        img_path = os.path.join(os.getcwd(), img_dir_path, file_name_without_ext + '.jpg')
        obj_imgs = extract_label_box(img_path, obj_coordinates)
        stored_path = os.path.join(os.getcwd(), FOLDER_NAME)
        print("des_path", stored_path)
        if not os.path.exists(stored_path):
            os.mkdir(stored_path)

        for obj_img in obj_imgs:
            resized_img_32 = resize_img(obj_img, SIZE_32)
            resized_img_32.save(os.path.join(stored_path, str(count)) + '.png', 'png')
            flipped_img = flip_img(resized_img_32)
            flipped_img.save(os.path.join(stored_path, str(count) + '_flipped') + '.png', 'png')
            count += 1

        # main()
        stored_path = '/home/thaophan/Thesis/graduate-thesis/pos_test_set'
        obj_img = Image.open('ngoc_bae2.png')
        resized_img_32 = resize_img(obj_img, SIZE_32)
        resized_img_32.save(os.path.join(stored_path, 'ngoc_bae2') + '.png', 'png')
        flipped_img = flip_img(resized_img_32)
        flipped_img.save(os.path.join(stored_path, 'ngoc_bae2' + '_flipped') + '.png', 'png')


if __name__ == '__main__':
    main()
