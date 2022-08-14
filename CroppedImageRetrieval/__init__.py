from CroppedImageRetrieval.template_match import *
from CroppedImageRetrieval.my_util import *
from CroppedImageRetrieval.data_utils import *
from CroppedImageRetrieval.crop import *
from CroppedImageRetrieval.cnn import *
from CroppedImageRetrieval.cnn_test import *

from CroppedImageRetrieval.sift_match import *


"""
* TEMPLATE_RETRIEVAL
* input:
    * template_img: The path of the clipping diagram
    * retrieval_dir: The folder path of origin images
    * compress_rate: The compression scale of the original image and the cropped image, the length and width of the
                   compressed image are 1 / resize_rate times of the original
    * rotate: Whether to allow template rotation to match
    * zoom: Whether to allow the template to scale to match
    * threshold: The threshold of the correlation coefficient of template matching, which will not be flipped or
                 scaled in the next step of the template
    * show: show image
* return:
    * best_match_info: [] The best five results are Key: matching result image path, Value: correlation coefficient
                       with clipping graph (the graph is the credibility of the original clipping graph)
"""
def template_retrieval(template_img, retrieval_dir, compress_rate=4, threshold=0.8, show=True, rotate=True, zoom=True):
    if not retrieval_dir.endswith("/"):
        retrieval_dir += "/"
    retrieval_imgs = []
    find_files(retrieval_imgs, retrieval_dir)
    best_match_info, time_cost = retrieval_template_match(template_img, retrieval_imgs, resize_rate=compress_rate, rotate=rotate, size=zoom,
                             threshold=threshold, show=show)
    return best_match_info


"""
* IMAGES_CROPPING
* input:
    * data_path: The existed folder path that stores origin image
    * crop_path: The existed folder path that will store crop image 
    * info_path: The existed folder path that will store image index info 
    * num: The num of cropping per image (default: 128)
"""
def images_cropping(data_path, crop_path, info_path, num=128):

    if not data_path.endswith("/"):
        data_path += "/"
    if not crop_path.endswith("/"):
        crop_path += "/"
    if not info_path.endswith("/"):
        info_path += "/"

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
        crop_images = rand_crop_image(image, num=num)

        for i, crop_image in enumerate(crop_images):
            cv2.imwrite(file_path + "{:04d}.jpg".format(i), crop_image)


"""
* MODEL_TRAIN
* input:
    * data_path: The folder path that stores training data
    * model_path: The model path that stores training model
    * class_num: The class num of this classification task
    * epoch: Training Epoch (Default: 5)
"""
def model_train(data_path, model_path, class_num, epoch=5):

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device: ", DEVICE)

    # Load Data
    if not data_path.endswith("/"):
        data_path += "/"
    path_list = []
    for i in range(class_num):
        path_list.append(str(i) + "/")

    train_loader, test_loader = load_data(data_path, path_list, num=0)

    # Load/Create ResNet Model
    print("[ResNet18]")
    model = torchvision.models.resnet18(pretrained=True).to(DEVICE)
    model.fc = nn.Linear(512, class_num).to(DEVICE)
    if os.path.exists(model_path):
        print("Load pretrained model from", model_path)
        model.load_state_dict(torch.load(model_path, map_location=torch.device(DEVICE)))

    print(model, "\n")
    optimizer = optim.Adam(model.parameters())

    # Train
    best_acc = 0.0
    t = time.time()
    for e in range(1, epoch + 1):
        train(model, train_loader, optimizer, e, DEVICE)
        print("\n")

        t_acc = test(model, train_loader, DEVICE)
        acc = test(model, test_loader, DEVICE)
        if best_acc < acc:
            best_acc = acc
            torch.save(model.state_dict(), model_path)
            print("Model Update")

        print("Train Acc: {:.4f}".format(t_acc))
        print("Test Acc: {:.4f}, Best Test Acc: {:.4f}\n".format(acc, best_acc))
    print("Total Cost: %.3f s" % (time.time() - t))


"""
* MODEL_TEST
* input:
    * class_num: The class num of this classification task 
    * model_path: The model path that stores training model
    * info_path: The csv file path that stores class info
    * image_path: The path of input image
* return:
    * top5_res: [] of top 5 result
"""
def model_test(model_path, info_path, image_path, class_num):
    
    # Check Device
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device: ", DEVICE)

    # Load/Create ResNet Model
    model = torchvision.models.resnet18(pretrained=True).to(DEVICE)
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
    res, top5_res = image_test(model, image_path, info, DEVICE)
    print("Top1 Retrieval Result:", res)
    print("Top5 Retrieval Result:")
    for i, s in enumerate(top5_res):
        print(i, s)

    return top5_res