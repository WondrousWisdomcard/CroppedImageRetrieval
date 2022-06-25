import csv
import os
import sys
import time

import torch
import torchvision
import torch.nn as nn
import argparse

from data_utils import *


"""
* IMAGE_TEST Crop image retrieval
* input:
    * model: CNN model that have been trained
    * image_path: <String> the path of crop image
    * info: <{} Key: class, Value: origin image file>
    * device: <torch.device>
* return:
    * res: <String> Top1 retrieval result
    * top5: <List of String> Top5 retrieval result
"""
def image_test(model, image_path, info, device, ver="v1"):
    model.eval()
    if ver == "v2":
        test_loader = image_to_data_loader(image_path, hw=(600, 600))
    else:
        test_loader = image_to_data_loader(image_path)
    for step, (x, y) in enumerate(test_loader):
        x, y = x.to(device), y.to(device)
        with torch.no_grad():
            y_ = model(x)
        pred = y_.max(-1, keepdim=True)[1]
        pred5 = y_.sort()[1][0][-5:]

        top5 = []
        for i in pred5:
            top5.append(info[int(i)])
        top5.reverse()
        res = info[int(pred[0][0])]
        return res, top5


"""
* TEST_ACC test a series of images on image retrieval task
* input:
    * template_dir: Crop Image folder (crop image and origin must have same filename)
    * image_dir: Origin Image folder
    * model: CNN model that havel been trained
    * image_path: <String> the path of crop image
    * info: <{} Key: class, Value: origin image file>
    * device: <torch.device>
* return:
    * acc:  <Float> Top1 retrieval accuracy
    * acc5: <Float> Top5 retrieval accuracy
    * time_cost: Total time spent
"""
def test_acc(template_dir, image_dir, model, info, device, ver="v1"):
    acc = 0
    acc5 = 0
    time_cost = 0

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

        s = time.time()
        res, top5_res = image_test(model, template_path, info, device, ver)
        time_cost += time.time() - s

        l = len(template_file_name)
        for i in range(5):
            top5_res[i] = top5_res[i][-l:]
        if res[-l:] == template_file_name:
            print(template_file_name, "Match Top1 Successfully")
            acc += 1
            acc5 += 1
        elif template_file_name in top5_res:
            print(template_file_name, "Match Top5 Successfully")
            acc5 += 1
        else:
            print(template_file_name, "Match Fail")
            print(res)

    acc /= count
    acc5 /= count
    time_cost /= count
    print("\nAcc:", acc, "Top5 Acc", acc5, "Average Times Cost:", time_cost)
    return acc, acc5, time_cost


"""
* MY_ACC_DEMO test a series of images on image retrieval task and store the result
* input:
    * class_num: <Integer> the num of origin image(class)
    * model_path: <String> the path of CNN model that havel been trained
    * info_path: <String> the path of Class-OriginImage mapping file
    * template_dir: <String> Crop Image folder (crop image and origin must have same filename)
    * image_dir: <String> Origin Image folder
    * output_file: <String> The file that store result from stdout
"""
def my_acc_demo(class_num, model_path, info_path, template_dir, image_dir, output_file, ver="v1"):

    # Check Device
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device: ", DEVICE)

    # Load/Create ResNet Model
    model = torchvision.models.resnet18(pretrained=True).to(DEVICE)
    if ver == "v2":
        model.fc = nn.Linear(86528, class_num).to(DEVICE)
    else:
        model.fc = nn.Linear(512, class_num).to(DEVICE)
        
    if os.path.exists(model_path):
        print("Load pretrained model from", model_path)
        model.load_state_dict(torch.load(model_path, map_location=torch.device(DEVICE)))
    else:
        print("Don't find pretrained model from", model_path)
        return

    # Load image file mapping file
    info = {}
    if not os.path.exists(info_path):
        print("Could not find", info_path)

    with open(info_path, encoding="utf-8") as f:
        f_csv = csv.reader(f)
        for r in f_csv:
            info[int(r[0])] = r[1]
        f.close()

    # Code for Testing Acc
    stdout = sys.stdout
    with open(output_file, 'w', encoding='utf-8') as f:
        sys.stdout = f
        test_acc(template_dir, image_dir, model, info, DEVICE, ver=ver)
    sys.stdout = stdout


if __name__ == "__main__":
    # # Test Acc Demo (Linux + GPU)
    # my_acc_demo(114, "/home/zhengyw/Model/Invoice/resnet.pth", "/home/zhengyw/Info/Invoice/info.csv",
    #             "/home/zhengyw/Data/InvoiceCut/", "/home/zhengyw/Data/Invoice/", "/home/zhengyw/Invoice_ResNet.txt")
    # my_acc_demo(190, "/home/zhengyw/Model/Risk/resnet.pth", "/home/zhengyw/Info/Risk/info.csv",
    #             "/home/zhengyw/Data/RiskCut/", "/home/zhengyw/Data/Risk/", "/home/zhengyw/Risk_ResNet.txt")
    # my_acc_demo(172, "/home/zhengyw/Model/Tag/resnet.pth", "/home/zhengyw/Info/Tag/info.csv",
    #             "/home/zhengyw/Data/TagCut/", "/home/zhengyw/Data/Tag/", "/home/zhengyw/Tag_ResNet.txt")
    # my_acc_demo(178, "/home/zhengyw/Model/Wire/resnet.pth", "/home/zhengyw/Info/Wire/info.csv",
    #             "/home/zhengyw/Data/WireCut/", "/home/zhengyw/Data/Wire/", "/home/zhengyw/Wire_ResNet.txt")
    # # V2 May Don't Work Well
    # my_acc_demo(114, "/home/zhengyw/Model/InvoiceV2/resnet.pth", "/home/zhengyw/Info/InvoiceV2/info.csv",
    #             "/home/zhengyw/Data/InvoiceCut/", "/home/zhengyw/Data/Invoice/", "/home/zhengyw/InvoiceV2_ResNet.txt",
    #             ver="v2")

    # Set Args
    parser = argparse.ArgumentParser(description='Training Model')
    parser.add_argument('--class_num', type=int, help='The class num of this classification task.')
    parser.add_argument('--model_path', help='The model path that stores training model')
    parser.add_argument('--info_path', help='The csv file path that stores class info')
    parser.add_argument('--image_path', help='The path of input image')
    parser.add_argument('--ver', default='v1', choices=["v1", "v2"], help='The num of cropping per image')

    args = parser.parse_args()
    class_num = args.class_num
    model_path = args.model_path
    info_path = args.info_path
    image_path = args.image_path
    ver = args.ver

    if class_num is None:
        raise Exception("Args: --class_num is needed")
    elif info_path is None:
        raise Exception("Args: --info_path is needed")
    elif model_path is None:
        raise Exception("Args: --model_path is needed")
    elif image_path is None:
        raise Exception("Args: --image_path is needed")

    if ver == "v2":
        print("Testing(V2) Running")

    # Check Device
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device: ", DEVICE)

    # Load/Create ResNet Model
    model = torchvision.models.resnet18(pretrained=True).to(DEVICE)

    if ver == "v2":
        model.fc = nn.Linear(86528, class_num).to(DEVICE)
    else:
        model.fc = nn.Linear(512, class_num).to(DEVICE)

    if os.path.exists(model_path):
        print("Load pretrained model from", model_path)
        model.load_state_dict(torch.load(model_path, map_location=torch.device(DEVICE)))
    else:
        raise Exception("Don't find pretrained model from" + model_path)

    # Load image file mapping file
    info = {}
    with open(info_path, encoding="utf-8") as f:
        f_csv = csv.reader(f)
        for r in f_csv:
            info[int(r[0])] = r[1]
        f.close()

    # Image Retrieval
    print("Cutting:", image_path)
    res, top5_res = image_test(model, image_path, info, DEVICE, ver=ver)
    print("Top1 Retrieval Result:", res)
    print("Top5 Retrieval Result:")
    for i, s in enumerate(top5_res):
        print(i, s)
