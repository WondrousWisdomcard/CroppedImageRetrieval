import argparse
import sys
import cv2
import os
import time
import numpy as np


"""
* SHOW_IMAGE Display Image
* input:
    * name: Name of Image
    * image: Images
    * height: Display height
"""
def show_image(name, image, height=1000):
    s = image.shape
    h = s[0]
    w = s[1]
    cv2.namedWindow(name, 0)
    cv2.resizeWindow(name, int(height / h * w), height)
    cv2.imshow(name, image)


"""
* FIND_FILES Extracts the relative paths of all images in a folder
* input: 
    * image_file_list: [] Save a list of file paths
    * path: String Folder path
"""
def find_files(image_file_list, path):
    file_list = os.listdir(path)
    for f in file_list:
        cur_path = path + f
        if os.path.isdir(cur_path):
            cur_path += '/'
            find_files(image_file_list, cur_path)
        else:
            if cur_path.endswith(".jpg") or cur_path.endswith(".png") or cur_path.endswith(".jpeg"):
                image_file_list.append(cur_path)


"""
* ROTATE_IMAGE Rotate the image at an Angle degree and the image size will change
* input:
    * image: Image matrix
    * degree: Degree
* return:
    * image_rotated: The rotated image will change in size (unified into a square image with 0.7 times the length of the original short edge)
"""
def rotate_image(image, degree):
    if degree == 0:  # The mirror image
        return np.fliplr(image)

    image_h, image_w = image.shape[:2]
    center = (image_w // 2, image_h // 2)
    if image_h < image_w:
        short = image_h
    else:
        short = image_w
    # h = w = \approx \frac{\sqrt{2}}{2} * short
    rotate_h = int(short * 0.7)
    rotate_w = int(short * 0.7)
    rotate_matrix = cv2.getRotationMatrix2D(center, degree, 1.0)
    image_rotated = cv2.warpAffine(image, rotate_matrix, (image_w, image_h))
    image_rotated = image_rotated[(image_h // 2 - rotate_h // 2): (image_h // 2 + rotate_h // 2),
                    (image_w // 2 - rotate_w // 2): (image_w // 2 + rotate_w // 2)]
    # cv2.imshow("Rotate", image_rotated)
    # cv2.waitKey(0)
    return image_rotated


"""
* TEMPLATE_MATCH Template matching
* input:
    * imageGray: Grayscale matrix 
    * templateGray: Grayscale matrix
* return:
    * maxVal: The correlation coefficient of the two graphs
"""
def template_match(imageGray, templateGray):
    result = cv2.matchTemplate(imageGray, templateGray, cv2.TM_CCOEFF_NORMED)
    (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(result)
    return maxVal


"""
* RETRIEVAL_TEMPLATE_MATCH 残缺图像匹配
* input: 
    * template_path: The path of the clipping diagram
    * image_file_list: [] A list of paths to store gallery images
    * resize_rate: The compression scale of the original image and the cropped image, the length and width of the 
                   compressed image are 1 / resize_rate times of the original
    * rotate: Whether to allow template rotation to match
    * size: Whether to allow the template to scale to match
    * threshold: The threshold of the correlation coefficient of template matching, which will not be flipped or 
                 scaled in the next step of the template
* return:
    * best_match_info: {} The best five results are Key: matching result image path, Value: correlation coefficient 
                       with clipping graph (the graph is the credibility of the original clipping graph)
    * time_cost: Algorithm Time (seconds)
"""
def retrieval_template_match(template_path, image_file_list, resize_rate=1, rotate=True, size=True, threshold=0.9,
                             show=True):
    # Template matching
    start = time.time()
    template = cv2.imdecode(np.fromfile(template_path, dtype=np.uint8), -1)

    templateGray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    template_h, template_w = templateGray.shape
    # Compress the template
    if resize_rate != 1:
        templateGray = cv2.resize(templateGray, (int(template_w / resize_rate), int(template_h / resize_rate)))
        template_h, template_w = templateGray.shape

    match_info = {}  # Store matching results Key: image file name, Value: max val
    count = 0  # Calculate the number of original graphs whose size is smaller than that of the template. Only such
    # graphs can the template matching algorithm be applied
    get_it = False  # The flag has found the original graph whose reliability is greater than the threshold

    for image_path in image_file_list:
        # Grayscale processing and compression of the original image
        image = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), -1)
        image_h, image_w, c = image.shape

        # Determine whether the size of the original drawing is smaller than the template, and then apply the template
        # matching algorithm
        if int(image_h / resize_rate) >= template_h and int(image_w / resize_rate) >= template_w:
            # matchTemplate requires that the template size be strictly smaller than the image size
            imageGray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            if resize_rate != 1:
                imageGray = cv2.resize(imageGray, (int(image_w / resize_rate), int(image_h / resize_rate)))
                image_h, image_w = imageGray.shape
            count += 1
            s = template_match(imageGray, templateGray)
            match_info[image_path] = s
            if s > threshold:
                get_it = True
        else:
            match_info[image_path] = 0

    # Consider the rotation of the template
    if rotate:
        # The order of the rotation Angle of the template, 0 generation here refers to the template image
        rotate_choice = [90, -90, 45, -45, 0, 180, 135, -135]
        for rotate_drgree in rotate_choice:
            if get_it:
                break

            # Generate flipped template: grayscale, rotate, compress (rotate will make the template image smaller)
            if show:
                print("Rotate Template Stage:", rotate_drgree)
            template = cv2.imdecode(np.fromfile(template_path, dtype=np.uint8), -1)
            templateGray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
            templateRotate = rotate_image(templateGray, rotate_drgree)
            template_h, template_w = templateRotate.shape
            if resize_rate != 1:
                templateRotate = cv2.resize(templateRotate,
                                            (int(template_w / resize_rate), int(template_h / resize_rate)))
                template_h, template_w = templateRotate.shape

            # Match the rotated template with all the files in the original image folder
            for image_path in image_file_list:
                image = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), -1)
                image_h, image_w, c = image.shape

                # Determine whether the size of the original drawing is smaller than the template, and then apply the
                # template matching algorithm
                if int(image_h / resize_rate) >= template_h and int(image_w / resize_rate) >= template_w:
                    imageGray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    if resize_rate != 1:
                        imageGray = cv2.resize(imageGray, (int(image_w / resize_rate), int(image_h / resize_rate)))
                    if match_info[image_path] == 0:
                        count += 1
                    s = template_match(imageGray, templateRotate)
                    # Update the result if the rotated template can be more reliable than the original
                    if s > match_info[image_path]:
                        match_info[image_path] = s
                    # If the original image whose reliability is greater than threshold is found, no further
                    # flipping is performed
                    if s > threshold:
                        get_it = True

    # Consider the scaling of the template
    if size:
        template_resize_choice = [0.5, 2]
        for template_resize in template_resize_choice:
            if get_it:
                break

            # Generate scaled template: grayscale, scale, compress
            if show:
                print("Resize Template Stage:", template_resize)
            template = cv2.imdecode(np.fromfile(template_path, dtype=np.uint8), -1)
            templateGray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
            templateResize = cv2.resize(templateGray,
                                        (int(template_w / resize_rate), int(template_h / resize_rate)))
            template_h, template_w = templateResize.shape
            if resize_rate != 1:
                templateResize = cv2.resize(templateResize,
                                            (int(template_w / template_resize), int(template_h / template_resize)))
                template_h, template_w = templateResize.shape

            # Rematch the scaled template with all the files in the original image folder
            for image_path in image_file_list:
                image = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), -1)
                image_h, image_w, c = image.shape

                # Determine whether the size of the original drawing is smaller than the template, and then apply the
                # template matching algorithm
                if int(image_h / resize_rate) >= template_h and int(image_w / resize_rate) >= template_w:
                    imageGray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    if resize_rate != 1:
                        imageGray = cv2.resize(imageGray, (int(image_w / resize_rate), int(image_h / resize_rate)))
                    if match_info[image_path] == 0:
                        count += 1
                    s = template_match(imageGray, templateResize)
                    # Update the result if the scaled template can be more reliable than the original
                    if s > match_info[image_path]:
                        match_info[image_path] = s
                    # If the original image whose reliability is greater than threshold is found, the search is not
                    # continued
                    if s > threshold:
                        get_it = True

    # Get retrieval result Top 5
    time_cost = time.time() - start
    best_match_info = sorted(match_info.items(), key=lambda x: -x[1])[:5]
    if show:
        print("[INFO] Time cost: %.2f" % time_cost)
        print("[INFO] Size satisfied: ", count, "/", len(image_file_list), "Images")
        print("[INFO] Top 5 Match: ", best_match_info)

        show_image("Template", template, 400)
        i = 1
        for image_path, v in best_match_info:
            image_best = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), -1)
            show_image("Result" + str(i), image_best, 600)
            i += 1
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        print("If template.py doesn't stopped, try 'ctrl + z'")

    return best_match_info, time_cost


def test_acc(template_dir, image_dir):
    acc_top5 = 0
    acc = 0
    time_cost = 0
    same_image_num = 0

    template_file_list = []
    template_file_name_list = []
    def find_template_files(path):
        file_list = os.listdir(path)
        for f in file_list:
            cur_path = path + f
            if os.path.isdir(cur_path):
                cur_path += '/'
                find_template_files(cur_path)
            else:
                if cur_path.endswith(".jpg") or cur_path.endswith(".png") or cur_path.endswith(".jpeg"):
                    template_file_list.append(cur_path)
                    template_file_name_list.append(f)
    find_template_files(template_dir)

    count = 0

    origin_list = []
    find_files(origin_list, image_dir)

    for template_path, template_file_name in zip(template_file_list, template_file_name_list):
        if (image_dir + template_file_name) not in origin_list:
            continue

        count += 1
        print("\n[", count, "] Now Test File:", template_path)

        res, tc = retrieval_template_match(template_path, origin_list, resize_rate=4, rotate=True, size=True,
                                           threshold=0.80, show=False)
        time_cost += tc
        best_file_path, best_sim = res[0]
        second_best_file_path, second_best_sim = res[1]

        if best_file_path == image_dir + template_file_name or (second_best_file_path == image_dir + template_file_name
                                                                and abs(best_sim - second_best_sim) < 0.0001):
            print(image_dir + template_file_name, "Match Top1 Successfully")
            acc += 1
            acc_top5 += 1
            if abs(best_sim - second_best_sim) < 0.0001:
                print("Detect 2 Same Image:", best_file_path, "and", second_best_file_path)
                same_image_num += 1
        else:
            print(image_dir + template_file_name, "Match Top1 Failed")
            print(res)
            for file_path, sim in res:
                if file_path == image_dir + template_file_name:
                    acc_top5 += 1

    acc /= count
    acc_top5 /= count
    time_cost /= count
    print("Acc:", acc, "Top5 Acc:", acc_top5, "Average Times Cost:", time_cost)
    return acc, acc_top5, time_cost


def my_acc_demo():
    with open('./Wire.txt', 'w', encoding='utf-8') as f:
        sys.stdout = f

        template_dir = "./Data/WireCrop/"
        image_dir = "./Data/Wire/"
        test_acc(template_dir, image_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Template Matching')
    parser.add_argument('--template_img', help="The path of template image, like './home/template/1.png'")
    parser.add_argument('--retrieval_dir', help="The folder path of retrieval image library, like './home/image/'")
    parser.add_argument('--compress_rate', type=int, default=4,
                        help="The compress rate of image for its height and width (default: 4)")
    parser.add_argument('--threshold', type=float, default=0.9,
                        help="The threshold of template rotate and resize, template stop to rotate / resize when getting similarity larger than threshold ([0.0 - 1.0] default: 0.9)")
    parser.add_argument('--show', type=bool, default=True, help="Visual and print the matching result")
    args = parser.parse_args()
    template_img = args.template_img
    retrieval_dir = args.retrieval_dir
    compress_rate = args.compress_rate
    threshold = args.threshold
    show = args.show

    if template_img is None:
        raise Exception("Args: --template_img is needed")
    elif retrieval_dir is None:
        raise Exception("Args: --retrieval_dir is needed")

    if not retrieval_dir.endswith("/"):
        retrieval_dir += "/"

    retrieval_imgs = []
    find_files(retrieval_imgs, retrieval_dir)
    retrieval_template_match(template_img, retrieval_imgs, resize_rate=compress_rate, rotate=True, size=True,
                             threshold=threshold, show=show)

