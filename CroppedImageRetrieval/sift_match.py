# REQUIREMENT:
# Visit https://blog.csdn.net/MOZHOUH/article/details/123924038 for help
# python                3.7
# opencv-contrib-python 3.4.2.17
# opencv-python         3.4.2.17
# pandas                1.3.5

import math
import numpy as np
import cv2
import time
import random
import pandas as pd
from matplotlib import pyplot as plt
from CroppedImageRetrieval.my_util import *


"""
* COMPUTE_VAR 
"""
def compute_var(pts_ori, pts_crop, use3sigma):
    div_dis = []
    for i in range(len(pts_ori)):
        for j in range(i + 1, len(pts_ori)):
            ori_pi, crop_pi = pts_ori[i], pts_crop[i]
            ori_pj, crop_pj = pts_ori[j], pts_crop[j]
            # if ori_pi == ori_pj or crop_pi == crop_pj:
            #     continue
            ori_dis = math.sqrt((ori_pi[0] - ori_pj[0]) ** 2 + (ori_pi[1] - ori_pj[1]) ** 2)
            crop_dis = math.sqrt((crop_pi[0] - crop_pj[0]) ** 2 + (crop_pi[1] - crop_pj[1]) ** 2)
            div_dis.append((ori_dis + 1) / (crop_dis + 1))
    if len(div_dis) != 0:
        if not use3sigma:
            # Compute the variance directly
            var_dis = np.var(div_dis)
            return var_dis
        else:
            # Use the 3-sigma principle of normal distribution to remove the outlier values
            mu = np.mean(div_dis)
            sigma = np.std(div_dis)
            div_dis_denoise = []
            for v in div_dis:
                if mu - 3 * sigma < v < mu + 3 * sigma:
                    div_dis_denoise.append(v)
            if len(div_dis_denoise) <= 1:
                return -1
            var_dis = np.var(div_dis_denoise)
            return var_dis
    else:
        return -1


"""
* SIFT_MATCH a Demo of the cropped image matching algorithm
* input:
  * ori_image_path: The file path to the original image
  * crop_image_path: The file path to the cropped image
  * show: [0, 1, 2] show=1 for print output, show=2 for visual output
  * ver: The adopted feature extraction algorithm ["SIFT", "ORB", "SURF"]
  * ori_nfeatures: The number of feature points extracted from the original image
  * crop_nfeatures: The number of feature points extracted from the cropped image
  * ori_nkeeps: The final number of feature points retained by random selection from the original image
  * crop_nkeeps: The final number of feature points retained by random selection from the cropped image
  * npairs: the number of matching pairs
* output:
  * var_dis: The variance of the ratio of feature point distances between two images
"""
def sift_match(ori_image_path, crop_image_path, show=1, ver="SURF", ori_nfeatures=1500, crop_nfeatures=500,
               ori_nkeeps=1500, crop_nkeeps=500, npairs=20, use3sigma=False):
    # Read image
    start = time.time()
    ori_image = cv2.imdecode(np.fromfile(ori_image_path, dtype=np.uint8), -1)
    crop_image = cv2.imdecode(np.fromfile(crop_image_path, dtype=np.uint8), -1)
    # Use zooming for visualization
    ori_h, ori_w = ori_image.shape[0], ori_image.shape[1]
    crop_h, crop_w = crop_image.shape[0], crop_image.shape[1]
    crop_image = cv2.resize(crop_image, (int(ori_h / crop_h * crop_w), ori_h))
    end = time.time()
    if show == 1:
        print("Read Image Cost: %.2f s" % (end - start))

    # Extracting image features
    start = time.time()
    if ver == "SURF":  # SURF
        print("[SURF]")
        # print("- Hint: SURF dont have nfeatures")
        surf = cv2.xfeatures2d.SURF_create()
        ori_kp, ori_des = surf.detectAndCompute(ori_image, None)
        surf = cv2.xfeatures2d.SURF_create()
        crop_kp, crop_des = surf.detectAndCompute(crop_image, None)
    elif ver == "ORB":  # ORB
        print("[ORB]")
        orb = cv2.ORB_create(nfeatures=ori_nfeatures)
        ori_kp, ori_des = orb.detectAndCompute(ori_image, None)
        orb = cv2.ORB_create(nfeatures=crop_nfeatures)
        crop_kp, crop_des = orb.detectAndCompute(crop_image, None)
    else:  # SIFT
        print("[SIFT]")
        sift = cv2.xfeatures2d.SIFT_create(nfeatures=ori_nfeatures)
        ori_kp, ori_des = sift.detectAndCompute(ori_image, None)
        sift = cv2.xfeatures2d.SIFT_create(nfeatures=crop_nfeatures)
        crop_kp, crop_des = sift.detectAndCompute(crop_image, None)

    ori_des = ori_des.astype(np.float32)
    crop_des = crop_des.astype(np.float32)
    ori_kp, ori_des = ori_kp[:ori_nfeatures], ori_des[:ori_nfeatures]
    crop_kp, crop_des = crop_kp[:crop_nfeatures], crop_des[:crop_nfeatures]
    if len(ori_kp) < ori_nfeatures:
        print("For Original Image, the actual number of detected features is smaller than nfeatures", len(ori_kp))
        ori_nfeatures = len(ori_kp)
    if len(crop_kp) < crop_nfeatures:
        print("For Cropped Image, the actual number of detected features is smaller than nfeatures", len(crop_kp))
        crop_nfeatures = len(crop_kp)

    # Randomly select $nkeeps$ features from $nfeatures$ features
    if 0 < ori_nkeeps < ori_nfeatures:
        ran_idx = sorted(random.sample(range(ori_nfeatures), ori_nkeeps))
        ori_kp = np.array([ori_kp[i] for i in ran_idx])
        ori_des = np.array([ori_des[i] for i in ran_idx])
    if 0 < crop_nkeeps < crop_nfeatures:
        ran_idx = sorted(random.sample(range(crop_nfeatures), crop_nkeeps))
        crop_kp = np.array([crop_kp[i] for i in ran_idx])
        crop_des = np.array([crop_des[i] for i in ran_idx])
    end = time.time()

    if show >= 1:
        print(ver, "Detect and Compute Cost: %.2f s" % (end - start))

    # FLANN Match
    start = time.time()
    fbm = cv2.FlannBasedMatcher()
    matches = fbm.match(ori_des, crop_des)
    matches = sorted(matches, key=lambda x: x.distance)
    end = time.time()
    if show >= 1:
        print(ver, "Match Cost: %.2f s" % (end - start))

    # Compute variance of distance ratio
    start = time.time()
    ori_pts = []
    crop_pts = []
    for i in range(npairs):
        ori_idx, crop_idx = matches[i].queryIdx, matches[i].trainIdx
        ori_pts.append(ori_kp[ori_idx].pt)
        crop_pts.append(crop_kp[crop_idx].pt)

    var_dis = compute_var(ori_pts, crop_pts, use3sigma=use3sigma)

    end = time.time()
    if show >= 1:
        print("Compute Var-dis: %.2f s" % (end - start))

    # Output
    if show >= 1:  # Print the match result
        print("Number of Keypoint:", len(ori_kp))
        print("Number of Match Pair:", len(matches))
        print("Var of dis:", var_dis)
    if show == 2:  # Visual the match
        ori_image = process_image_channels(ori_image)
        crop_image = process_image_channels(crop_image)

        ori_img = cv2.drawKeypoints(ori_image, ori_kp[:npairs], ori_image)
        crop_img = cv2.drawKeypoints(crop_image, crop_kp[:npairs], crop_image)
        for p, pc in zip(ori_pts, crop_pts):
            p = (int(p[0]), int(p[1]))
            pc = (int(pc[0]), int(pc[1]))
            for p2, pc2 in zip(ori_pts, crop_pts):
                p2 = (int(p2[0]), int(p2[1]))
                pc2 = (int(pc2[0]), int(pc2[1]))
                if p != p2 and pc != pc2:
                    color = (np.random.randint(1, 255), np.random.randint(1, 255), np.random.randint(1, 255))
                    cv2.line(ori_img, p, p2, color, thickness=2)
                    cv2.line(crop_img, pc, pc2, color, thickness=2)

        image_match = cv2.drawMatches(ori_img, ori_kp, crop_img, crop_kp, matches[:npairs], crop_img, flags=2, matchColor=(0, 0, 0))
        pic = image_match[:, :, ::-1]
        plt.imshow(pic)
        plt.xticks([])
        plt.yticks([])
        plt.show()
    return var_dis


"""
* PREPROCESS
* input:
  * path_dir: The folder path to the image
  * target_dir: The folder path to save the feature file
  * ver: The adopted feature extraction algorithm ["SIFT", "ORB", "SURF"]
  * ori_nfeatures: The number of feature points extracted from the original image
  * ori_nkeeps: The final number of feature points retained by random selection from the original image
  * fcheck: check if the feature file exists, if the file already exists, the feature file is no longer generated
"""
def preprocess(path_dir, target_dir, ver="SURF", ori_nfeatures=1500, ori_nkeeps=1500, fcheck=True):
    path_list = []
    name_list = []
    find_files(path_list, path_dir)
    find_filenames(name_list, path_dir)

    # Select the type of feature extraction algorithm
    if ver == "SURF":
        print("-SURF-")
        # print("[Warn] SURF dont have nfeatures")
        fea = cv2.xfeatures2d.SURF_create()
    elif ver == "ORB":
        print("-ORB-")
        fea = cv2.ORB_create(nfeatures=ori_nfeatures)
    else:
        print("-SIFT-")
        fea = cv2.xfeatures2d.SIFT_create(nfeatures=ori_nfeatures)

    print("Start SIFT: ", len(path_list), "Images")
    o_nf = ori_nfeatures
    for i in range(len(path_list)):
        image_path = path_list[i]
        image_name = name_list[i]
        print(i, image_path)

        # check if the feature files already exists
        if fcheck:
            if file_exists(target_dir, "pts_" + image_name) and file_exists(target_dir, "des_" + image_name):
                continue

        # load image and detect image feature
        image = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), -1)
        kp, des = fea.detectAndCompute(image, None)
        des = des.astype(np.float32)

        # Select first $crop_nfeatures$ feature of original image
        if len(kp) < o_nf:
            print("[Warn] Actual number of detected features is", len(kp))
            o_nf = len(kp)
        if o_nf <= 0:
            print("Select all detected features", len(kp))
            o_nf = len(kp)
        else:
            kp, des = kp[:o_nf], des[:o_nf]

        pts = [[p.pt[0], p.pt[1]] for p in kp]
        df_pts = pd.DataFrame(pts)
        df_des = pd.DataFrame(des)

        # Select random $crop_nkeeps$ feature of original image
        print("Total", o_nf, "KeyPoints, Keep KeyPoints", ori_nkeeps)
        if 0 < ori_nkeeps < o_nf:
            drop_idx = sorted(random.sample(range(o_nf), o_nf - ori_nkeeps))
            df_pts.drop(drop_idx, inplace=True)
            df_des.drop(drop_idx, inplace=True)
        # print(df_des.head())
        # print(df_pts.head())

        # Save file, suppose image_name is "dog.jpg", we save its feature files as:
        # the feature points file: "pts_dog.jpg.pkl"
        # the descriptors file: "des_dog.jpg.pkl"
        save_matrix_to_pickle(df_pts, target_dir, "pts_" + image_name)
        save_matrix_to_pickle(df_des, target_dir, "des_" + image_name)


"""
* LOAD_FEATURE 
* input: 
    * feature_dir: Path to the folder where the feature file resides
* return:
    * images_feature: {} Dictionary, key: file name, value: tuple of feature points and descriptors
"""
def load_feature(feature_dir):
    # Suppose image_name is "dog.jpg", we save its feature files as:
    # the feature points file: "pts_dog.jpg.pkl"
    # the descriptors file: "des_dog.jpg.pkl"

    path_list = []
    find_files(path_list, feature_dir, suffix=[".pkl"])

    name_list = []
    des_plist = []
    pts_plist = []
    for path_name in path_list:
        if path_name.startswith(feature_dir + "des_"):
            des_plist.append(path_name)
        elif path_name.startswith(feature_dir + "pts_"):
            pts_plist.append(path_name)
            name_list.append(path_name[len(feature_dir) + 4:-4])
    # print(des_plist)
    # print(pts_plist)
    # print(name_list)

    images_feature = {}
    for i in range(len(pts_plist)):
        df_pts = load_matrix_from_pickle(pts_plist[i])
        df_des = load_matrix_from_pickle(des_plist[i])
        # print(df_pts.head())
        # print(df_des.head())
        pts = [(p[0], p[1]) for p in df_pts.values]
        images_feature[name_list[i]] = (pts, df_des.values)
    return images_feature


"""
* SIFT_RETRIEVAL
* input:
  * crop_path: The file path to the cropped image
  * images_feature
  * ver: The adopted feature extraction algorithm ["SIFT", "ORB", "SURF"]
  * crop_nfeatures: The number of feature points extracted from the cropped image
  * crop_nkeeps: The final number of feature points retained by random selection from the cropped image
  * npairs: The number of matching pairs
* output:
  * res: List of tuple (imageName, variance), save the variance between cropped image and each original image
"""
def sift_retrieval(crop_path, images_feature, ver="SURF", crop_nfeatures=1500, crop_nkeeps=1500, npairs=20, use3sigma=False):

    s = time.time()
    # Read cropped image
    crop_image = cv2.imdecode(np.fromfile(crop_path, dtype=np.uint8), -1)

    # Select feature extraction algorithm
    if ver == "SURF":  # SURF
        print("-SURF-")
        # print("[Warn] SURF dont have nfeatures")
        fea = cv2.xfeatures2d.SURF_create()
    elif ver == "ORB":  # ORB
        print("-ORB-")
        fea = cv2.ORB_create(nfeatures=crop_nfeatures)
    else:  # SIFT
        print("-SIFT-")
        fea = cv2.xfeatures2d.SIFT_create(nfeatures=crop_nfeatures)

    # Feature extract for cropped image
    kp_crop, des_crop = fea.detectAndCompute(crop_image, None)
    des_crop = des_crop.astype(np.float32)

    # Select first $crop_nfeatures$ feature of cropped image
    if crop_nfeatures <= 0:
        print("Consider all detected features. (crop_nfeatures <= 0)")
        crop_nfeatures = len(kp_crop)
    if len(kp_crop) < crop_nfeatures:
        print("[Warn] For Cropped Image, the actual number of detected features is smaller than nfeatures", len(kp_crop))
        crop_nfeatures = len(kp_crop)
    kp_crop, des_crop = kp_crop[:crop_nfeatures], des_crop[:crop_nfeatures]

    # Select random $crop_nkeeps$ feature of cropped image
    if 0 < crop_nkeeps < crop_nfeatures:
        ran_idx = sorted(random.sample(range(crop_nfeatures), crop_nkeeps))
        kp_crop = np.array([kp_crop[i] for i in ran_idx])
        des_crop = np.array([des_crop[i] for i in ran_idx])

    print("Number of feature point:", len(kp_crop))
    e = time.time()
    t_extract = e - s

    res = []  # List of tuple (imageName, variance), save the variance between cropped image and each original image
    t_match = 0
    t_dis = 0
    fbm = cv2.FlannBasedMatcher()
    match_num = npairs
    # Go through each picture and calculate the variance
    for image_name in images_feature.keys():
        s = time.time()
        # Read the features of the original image
        kp_pts_ori, des_ori = images_feature[image_name]
        # Feature point matching
        matches = fbm.match(des_crop, des_ori)
        matches = sorted(matches, key=lambda x: x.distance)
        e = time.time()
        t_match += (e - s)

        if match_num > len(matches):
            print("[Warn] Actual Match", len(matches), " is less than npairs", npairs)
            match_num = len(matches)

        s = time.time()
        # Get the points in match
        pts_crop = [kp_crop[matches[i].queryIdx].pt for i in range(match_num)]
        pts_ori = [kp_pts_ori[matches[i].trainIdx] for i in range(match_num)]
        # Calculate distances in cropped image and original image separately, then compute the ratio of distance,
        # lastly compute the variance
        var_dis = compute_var(pts_ori, pts_crop, use3sigma=use3sigma)
        if var_dis != -1:
            res.append((image_name, var_dis))
        else:
            print("[Warn] Detect 'NaN':", image_name)
        e = time.time()
        t_dis += (e - s)

    print("Time cost on \n- feature extract and process for cropped image", t_extract)
    print("- FBM matching", t_match)
    print("- variance computing", t_dis)

    res = sorted(res, key=lambda x: x[1])
    return res