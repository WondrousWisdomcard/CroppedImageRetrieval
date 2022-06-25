import argparse

import cv2
import numpy as np
import os

from matplotlib import pyplot as plt
from my_util import *


"""
* ROTATE_IMAGE rotate the image at an angle
* input: 
    * image: <Image> 
    * degree: <Float>
    * show: <Boolean>[default: False] show the rotated image visually
"""
def rotate_image(image, degree, show=False):
    image_h, image_w = image.shape[:2]
    center = (image_w // 2, image_h // 2)
    rotate_matrix = cv2.getRotationMatrix2D(center, degree, 1.0)
    image_rotated = cv2.warpAffine(image, rotate_matrix, (image_w, image_h))
    if show:
        cv2.imshow("Rotate", image_rotated)
        cv2.waitKey(0)
    return image_rotated


"""
* RAND_CROP_IMAGE crop one image randomly
* input: 
    * image: <Image>
    * num: <Integer>[default: 128] the number of crop image return
    * rotate: <Boolean>[default: False] allow image to rotate randomly
* return:
    * crop_images: <List of Image>
"""
def rand_crop_image(image, num=128, rotate=True):
    crop_images = []
    h, w = image.shape[:2]
    print("Origin Center (", h // 2, ", ", w // 2, ") Height: ", h, ", Width: ", w)

    for i in range(num):
        # Gaussian distribution selects the center point
        c_i_x = int(h / 2 + h * np.random.randn() / 12)
        c_i_y = int(w / 2 + w * np.random.randn() / 12)

        # Poisson distribution selects length and width
        h_i = 0
        w_i = 0
        while h_i <= 0 or w_i <= 0 or h_i / w_i > 2 or w_i / h_i > 2:
            h_i = int(h - (np.random.poisson(lam=3) + np.random.normal(0, 1)) / 10 * h)
            w_i = int(w - (np.random.poisson(lam=3) + np.random.normal(0, 1)) / 10 * w)
        if h_i > h:
            h_i = h
        if w_i > w:
            w_i = w

        # Determine the top left corner of the clipping image
        x = c_i_x - h_i // 2
        y = c_i_y - w_i // 2
        if x < 0:
            x = 0
        if y < 0:
            y = 0

        # Mix gaussian distribution to select rotation Angle
        deg = 0
        if rotate:
            s = 20
            r = np.random.choice(5, p=[0.4, 0.15, 0.15, 0.15, 0.15])
            if r == 0:
                deg = np.random.normal(180, s)
            elif r == 1:
                deg = np.random.normal(0, s)
            elif r == 2:
                deg = np.random.normal(90, s)
            elif r == 3:
                deg = np.random.normal(270, s)
            elif r == 4:
                deg = np.random.normal(360, s)
            deg -= 180

        if rotate and np.random.choice(2) == 1:
            rotate_img = rotate_image(image, deg)
            crop_image = rotate_img[x: x + h_i, y: y + w_i]
        else:
            crop_image = image[x: x + h_i, y: y + w_i]
        crop_images.append(crop_image)
        print("Center (", c_i_x, c_i_y, "), Height: ", h_i, ", Width: ", w_i, ", Degree: ", deg)
    return crop_images


"""
* RAND_CROP_IMAGE_V2 crop one image randomly at a series of different distribution
* input: 
    * image: <Image>
    * num: <Integer>[default: 128] the number of crop image return
    * rotate: <Boolean>[default: False] allow image to rotate randomly
* return:
    * crop_images: <List of Image>
"""
def rand_crop_image_v2(image, num=128, rotate=True):
    crop_images = []
    h, w = image.shape[:2]
    print("Origin Center (", h // 2, ", ", w // 2, ") Height: ", h, ", Width: ", w)

    for i in range(num):
        # Gaussian distribution selects the center point
        c_i_x = int(h / 2 + h * np.random.randn() / 20)
        c_i_y = int(w / 2 + w * np.random.randn() / 20)

        # Poisson distribution selects length and width
        h_i = 0
        w_i = 0
        while h_i <= 0 or w_i <= 0 or h_i / w_i > 2 or w_i / h_i > 2:
            h_i = int(h - (np.random.poisson(lam=3) + np.random.normal(0, 1)) / 20 * h)
            w_i = int(w - (np.random.poisson(lam=3) + np.random.normal(0, 1)) / 20 * w)
        if h_i > h:
            h_i = h
        if w_i > w:
            w_i = w

        # Determine the top left corner of the clipping image
        x = c_i_x - h_i // 2
        y = c_i_y - w_i // 2
        if x < 0:
            x = 0
        if y < 0:
            y = 0

        # Mix gaussian distribution to select rotation Angle
        deg = 0
        if rotate:
            r = np.random.choice(4, p=[0.7, 0.1, 0.1, 0.1])
            if r == 0:
                deg = 0
            elif r == 1:
                deg = 90
            elif r == 2:
                deg = -90
            elif r == 3:
                deg = 180

        if rotate and np.random.choice(2) == 1:
            rotate_img = rotate_image(image, deg)
            crop_image = rotate_img[x: x + h_i, y: y + w_i]
        else:
            crop_image = image[x: x + h_i, y: y + w_i]
        crop_images.append(crop_image)
        print("Center (", c_i_x, c_i_y, "), Height: ", h_i, ", Width: ", w_i, ", Degree: ", deg)
    return crop_images

# (Distribution Visualized) Clipping center: Gaussian curve
def gaussian_visual():
    x = []
    for i in range(100000):
        x.append(1000 / 2 + 1000 * np.random.randn() / 20)
    a = plt.hist(x, bins=1000, range=[0, 1000], color='g', alpha=0.5)
    plt.plot(a[1][0: 1000], a[0], 'r')
    plt.grid()
    plt.show()


# (Distribution Visualized) Cutting length and width: Poisson curve
def poisson_visual():
    x = []
    for i in range(100000):
        h = 1000
        p = (h - (np.random.poisson(lam=3) + np.random.normal(0, 1)) / 20 * h)
        x.append(p)
    a = plt.hist(x, bins=1000, range=[0, 1000], color='g', alpha=0.5)
    plt.plot(a[1][0: 1000], a[0], 'r')
    plt.grid()
    plt.show()


# (Distribution Visualized) Clipping Angle: Mixed Gaussian curve
def multi_gaussian_visual():
    x = []
    for i in range(100000):
        s = 20
        r = np.random.choice(5, p=[0.4, 0.15, 0.15, 0.15, 0.15])
        if r == 0:
            x.append(np.random.normal(180, 10) - 180)
        elif r == 1:
            x.append(np.random.normal(0, s) - 180)
        elif r == 2:
            x.append(np.random.normal(90, s) - 180)
        elif r == 3:
            x.append(np.random.normal(270, s) - 180)
        elif r == 4:
            x.append(np.random.normal(360, s) - 180)

        # r = np.random.choice(5, p=[0.7, 0.1, 0.1, 0.05, 0.05])
        # if r == 0:
        #     x.append(0)
        # elif r == 1:
        #     x.append(90)
        # elif r == 2:
        #     x.append(-90)
        # elif r == 3:
        #     x.append(180)
        # elif r == 4:
        #     x.append(-180)

    a = plt.hist(x, bins=360, range=[-180, +180], color='g', alpha=0.5)
    plt.plot(a[1][0: 360], a[0], 'r')
    plt.grid()
    plt.show()


if __name__ == "__main__":
    # Args
    parser = argparse.ArgumentParser(description='Image Cropping')
    parser.add_argument('--data_path', help='The existed folder path that stores origin image')
    parser.add_argument('--crop_path', help='The existed folder path that will store crop image')
    parser.add_argument('--info_path', help='The existed folder path that will store image index info')
    parser.add_argument('--num', type=int, default=128, help='The num of cropping per image')
    parser.add_argument('--ver', default='v1', choices=["v1", "v2"], help='The num of cropping per image')
    args = parser.parse_args()
    data_path = args.data_path
    crop_path = args.crop_path
    info_path = args.info_path
    num = args.num
    ver = args.ver

    if info_path is None:
        raise Exception("Args: --info_path is needed")
    elif data_path is None:
        raise Exception("Args: --data_path is needed")
    elif crop_path is None:
        raise Exception("Args: --crop_path is needed")

    if not data_path.endswith("/"):
        data_path += "/"
    if not crop_path.endswith("/"):
        crop_path += "/"
    if not info_path.endswith("/"):
        info_path += "/"

    if ver == "v2":
        print("Image Cropping(V2) Running")

    image_paths = []
    find_files(image_paths, data_path)

    # Generate Class-OriginImage Mapping information
    with open(info_path + "info.csv", "w", encoding="utf-8") as f:
        s = ""
        for i in range(len(image_paths)):
            s += str(i) + "," + image_paths[i] + "\n"
        f.write(s)
        f.close()

    # Generate Crop Image File
    path_list = []
    for i in range(len(image_paths)):
        path_list.append(str(i) + "/")
    for i in range(len(image_paths)):
        print("\nImage", i)

        file_path = crop_path
        if not os.path.exists(file_path):
            os.mkdir(file_path)
        file_path += path_list[i]
        if not os.path.exists(file_path):
            os.mkdir(file_path)

        image = cv2.imdecode(np.fromfile(image_paths[i], dtype=np.uint8), -1)
        if ver == "v2":
            crop_images = rand_crop_image_v2(image, num=num)  # Invoice Only
        else:
            crop_images = rand_crop_image(image, num=num)

        for i, crop_image in enumerate(crop_images):
            cv2.imwrite(file_path + "{:04d}.jpg".format(i), crop_image)
