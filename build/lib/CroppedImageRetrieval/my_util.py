import cv2
import os

import numpy as np
import pandas as pd
from PIL import Image

"""
* SHOW_IMAGE 
* input:
    * name: 
    * image: 
    * height: 
"""
def show_image(name, image, height=1000):
    s = image.shape
    h = s[0]
    w = s[1]
    cv2.namedWindow(name, 0)
    cv2.resizeWindow(name, int(height / h * w), height)
    cv2.imshow(name, image)


"""
* FIND_FILES 
* input: 
    * image_file_list: [] 
    * path: String 
"""
def find_files(image_file_list, path, suffix=None):
    if suffix is None:
        suffix = [".jpg", ".png", ".jpeg"]
    file_list = os.listdir(path)
    for f in file_list:
        cur_path = path + f
        if os.path.isdir(cur_path):
            cur_path += '/'
            find_files(image_file_list, cur_path)
        else:
            for suf in suffix:
                if cur_path.endswith(suf):
                    image_file_list.append(cur_path)

"""
* FIND_FILES 
* input: 
    * image_file_list: [] 
    * path: String 
"""
def find_filenames(image_file_list, path, suffix=None):
    if suffix is None:
        suffix = [".jpg", ".png", ".jpeg"]
    file_list = os.listdir(path)
    for f in file_list:
        cur_path = path + f
        if os.path.isdir(cur_path):
            cur_path += '/'
            find_files(image_file_list, cur_path)
        else:
            for suf in suffix:
                if cur_path.endswith(suf):
                    image_file_list.append(f)


"""
* SAVE_MATRIX_TO_PICKLE 
* input:
    * matrix: DataFrame 
    * dir_path: String 
    * file_name: String 
"""
def save_matrix_to_pickle(matrix, dir_path, file_name):
    if os.path.exists(dir_path) is False:
        os.mkdir(dir_path)
    file_path = os.path.join(dir_path, file_name + ".pkl")
    print("Save Matrix to", file_path)
    matrix.to_pickle(file_path)


"""
* LOAD_MATRIX_FROM_PICKLE 
* input:    
    * file_path: String 
* return:
    * matrix: DataFrame 
"""
def load_matrix_from_pickle(file_path):
    # print("Load Matrix from", file_path)
    matrix = pd.read_pickle(file_path)
    return matrix


"""
* FILE_EXISTS: check if the file exists for file
"""
def file_exists(dir_path, file_name, suffix=".pkl"):
    if os.path.exists(dir_path) is False:
        os.mkdir(dir_path)
    file_path = os.path.join(dir_path, file_name + suffix)

    if os.path.exists(file_path):
        print("File exists", file_path)
        return True
    else:
        return False


"""
* PROCESS_IMAGE_CHANNELS: change the image format to 3 channel
"""
def process_image_channels(image):
    image = Image.fromarray(image)
    # process the 4 channels .png
    if image.mode == 'RGBA':
        r, g, b, a = image.split()
        image = Image.merge("RGB", (r, g, b))    # process the 1 channel image
    elif image.mode != 'RGB':
        image = image.convert("RGB")
    image = np.array(image)
    return image

