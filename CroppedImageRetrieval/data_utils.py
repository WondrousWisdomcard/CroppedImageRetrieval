from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from my_util import find_files


"""
* INIT_PROCESS
* input:
    * dir_path: <String> folder that stores a kind of image(class)
    * label: <Integer> the class of this image
* return:
    * data: <List of Tuple> Tuple(image_path, label)
"""
def init_process(dir_path, label):
    paths = []
    find_files(paths, dir_path)
    data = []
    for path in paths:
        data.append([path, label])
    return data

"""
* MY_LOADER
* input: 
    * path: <String> path of image
* return:
    * Image: <Plt.Image>
"""
def my_loader(path):
    return Image.open(path).convert('RGB')


class MyDataset(Dataset):
    def __init__(self, data, transform, loader):
        self.data = data
        self.transform = transform
        self.loader = loader

    def __getitem__(self, item):
        img, label = self.data[item]
        img = self.loader(img)
        img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.data)


"""
* LOAD_DATA load image file data to DataLoader
* input: 
    * d_path: <String> folder that store (crop) image data
    * path_list: <List of String> the sub dirs of d_path
    * num: <Integer> first num images as training data, the rest of them as testing data
    * hw: <Tuple of Integer> the unified size for compression
* return:
    * train_data: <DataLoader> DataLoader for training
    * test_data: <DataLoader> DataLoader for testing
"""
def load_data(d_path, path_list, num=100, hw=(224, 224)):
    transform = transforms.Compose([
        transforms.Resize(hw),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])
    data_list = []
    for i, path in enumerate(path_list):
        data = init_process(d_path + path, i)
        data_list.append(data)

    if len(data_list[0]) <= num:
        print("Data from one class is less than", num)
        num = int(len(data_list[0]) * 0.8)
        print("We set train 80%, and test 20%, change num to", num)

    train_data = data_list[0][:num]
    test_data = data_list[0][num:]
    for i in range(1, len(data_list)):
        train_data += data_list[i][:num]
        test_data += data_list[i][num:]


    train = MyDataset(train_data, transform=transform, loader=my_loader)
    test = MyDataset(test_data, transform=transform, loader=my_loader)
    train_data = DataLoader(dataset=train, batch_size=16, shuffle=True, num_workers=0, pin_memory=True)
    test_data = DataLoader(dataset=test, batch_size=1, shuffle=True, num_workers=0, pin_memory=True)

    return train_data, test_data


"""
* IMAGE_TO_DATA_LOADER load one image file data to DataLoader
* input: 
    * image_path: <String> the path of image
    * hw: <Tuple of Integer> the unified size for compression
* return:
    * test_data: <DataLoader> DataLoader for testing (only store one data)
"""
def image_to_data_loader(image_path, hw=(224, 224)):
    transform = transforms.Compose([
        transforms.Resize(hw),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])
    test_data = [(image_path, 0)]
    test = MyDataset(test_data, transform=transform, loader=my_loader)
    test_data = DataLoader(dataset=test, batch_size=1, shuffle=True, num_workers=0, pin_memory=True)

    return test_data

