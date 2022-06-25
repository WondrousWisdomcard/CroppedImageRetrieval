import cv2
import os

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
    file_list = sorted(file_list)

    for f in file_list:
        cur_path = path + f
        if os.path.isdir(cur_path):
            cur_path += '/'
            find_files(image_file_list, cur_path)
        else:
            if cur_path.endswith(".jpg") or cur_path.endswith(".png") or cur_path.endswith(".jpeg"):
                image_file_list.append(cur_path)
