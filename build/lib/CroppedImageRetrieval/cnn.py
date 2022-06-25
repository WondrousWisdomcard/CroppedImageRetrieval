import os
import time

import torch
import torchvision
from torch import optim
import torch.nn as nn
import argparse

from CroppedImageRetrieval.data_utils import *



# Training cnn model
def train(model, train_loader, optimizer, epoch, device, verbose=True):
    losses = []
    model.train()
    loss_func = nn.CrossEntropyLoss()
    for step, (x, y) in enumerate(train_loader):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        y_ = model(x)
        loss = loss_func(y_, y)
        loss.backward()
        optimizer.step()

        if verbose and ((step + 1) % 10 == 0):
            print('Train Epoch:{} [{:0>5d}/{} ({:0>2.0f}%)]\tLoss:{:.6f}'.format(
                epoch, step * len(x), len(train_loader.dataset),
                       100. * step / len(train_loader), loss.item()
            ))
            losses.append(loss.item())
    return losses


# Testing cnn model
def test(model, test_loader, device, verbose=False):
    model.eval()
    loss_func = nn.CrossEntropyLoss(reduction='sum')
    test_loss = 0.0
    acc = 0

    for step, (x, y) in enumerate(test_loader):
        x, y = x.to(device), y.to(device)
        with torch.no_grad():
            y_ = model(x)
        test_loss += loss_func(y_, y)
        pred = y_.max(-1, keepdim=True)[1]
        acc += pred.eq(y.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    if verbose:
        print('Test Average loss:{:.4f}, Accuracy:{:0>2.0f}% ({:0>5d}/{})'.format(
            test_loss, 100 * acc / len(test_loader.dataset), acc, len(test_loader.dataset)
        ))

    return acc / len(test_loader.dataset)


# if __name__ == "__main__":
#     # Set Args
#     parser = argparse.ArgumentParser(description='Training Model')
#     parser.add_argument('--class_num', type=int, help='The class num of this classification task.')
#     parser.add_argument('--epoch', type=int, default=5, help='Training Epoch (Default: 5).')
#     parser.add_argument('--data_path', help='The folder path that stores training data')
#     parser.add_argument('--model_path', help='The model path that stores training model')
#     parser.add_argument('--ver', default='v1', choices=["v1", "v2"], help='The num of cropping per image')

#     args = parser.parse_args()
#     class_num = args.class_num
#     epoch = args.epoch
#     data_path = args.data_path
#     model_path = args.model_path
#     ver = args.ver

#     if class_num is None:
#         raise Exception("Args: --class_num is needed")
#     elif data_path is None:
#         raise Exception("Args: --data_path is needed")
#     elif model_path is None:
#         raise Exception("Args: --model_path is needed")

#     if ver == "v2":
#         print("Training(V2) Running")

#     DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     print("Device: ", DEVICE)

#     # Load Data
#     if not data_path.endswith("/"):
#         data_path += "/"
#     path_list = []
#     for i in range(class_num):
#         path_list.append(str(i) + "/")

#     if ver == "v2":
#         train_loader, test_loader = load_data(data_path, path_list, num=100, hw=(600, 600))
#     else:
#         train_loader, test_loader = load_data(data_path, path_list, num=100)

#     # Load/Create ResNet Model
#     print("[ResNet18]")
#     model = torchvision.models.resnet18(pretrained=True).to(DEVICE)

#     if ver == "v2":
#         model.fc = nn.Linear(86528, class_num).to(DEVICE)
#     else:
#         model.fc = nn.Linear(512, class_num).to(DEVICE)
#     if os.path.exists(model_path):
#         print("Load pretrained model from", model_path)
#         model.load_state_dict(torch.load(model_path, map_location=torch.device(DEVICE)))

#     print(model, "\n")
#     optimizer = optim.Adam(model.parameters())

#     # Train
#     best_acc = 0.0
#     t = time.time()
#     for e in range(1, epoch + 1):
#         train(model, train_loader, optimizer, e, DEVICE)
#         print("\n")

#         t_acc = test(model, train_loader, DEVICE)
#         acc = test(model, test_loader, DEVICE)
#         if best_acc < acc:
#             best_acc = acc
#             torch.save(model.state_dict(), model_path)
#             print("Model Update")

#         print("Train Acc: {:.4f}".format(t_acc))
#         print("Test Acc: {:.4f}, Best Test Acc: {:.4f}\n".format(acc, best_acc))
#     print("Total Cost: %.3f s" % (time.time() - t))

