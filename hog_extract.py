import glob
import os

import sys
import utils

from PIL import Image
from skimage.feature import hog
from skimage import data, exposure

# FOLDER_NAME = 'pos_hog'
FOLDER_NAME = 'neg_hog'

# Read single file
# image = data.astronaut()
# image_path = str(sys.argv[1])
# image = Image.open(image_path)

img_dir_path = str(sys.argv[1])
stored_path = os.path.join(os.getcwd(), FOLDER_NAME)
print("des_path", stored_path)
if not os.path.exists(stored_path):
    os.mkdir(stored_path)

for filename in glob.glob(os.path.join(img_dir_path, '*.png')):
    file_path = os.path.join(os.getcwd(), filename)
    file_name_without_ext = os.path.splitext(os.path.basename(filename))[0]
    print(f'Generating hog feature from {file_path}')
    img_path = os.path.join(os.getcwd(), img_dir_path, file_name_without_ext + '.png')
    image = Image.open(img_path)

    fd, hog_image = hog(image, orientations=9, pixels_per_cell=(4, 4),
                        cells_per_block=(2, 2), visualize=True, multichannel=True)
    # Rescale histogram for better display
    hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))

    res = utils.to_image(hog_image_rescaled)
    # res.save(file_name_without_ext + ".png")
    res.save(os.path.join(stored_path, file_name_without_ext) + '.png', 'png')
    
    





