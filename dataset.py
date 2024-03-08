import glob
import random
import os
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import torch
from natsort import natsorted
import tifffile as tif
import numpy as np
import cv2
## 如果输入的数据集是灰度图像，将图片转化为rgb图像(本次采用的facades不需要这个)
class BWDataset(torch.utils.data.Dataset):
    """
    dataset of image files of the form
       stuff<number>_trans.pt
       stuff<number>_density.pt
    """

    def __init__(self, data_dir,size):
        self.input_dir = os.path.join(data_dir, 'input/')
        self.label_dir = os.path.join(data_dir, 'label/')
        self.transform = transforms.Compose([transforms.ToPILImage(),
                                             transforms.CenterCrop(size),transforms.ToTensor(),
                                             transforms.Normalize(mean=0.5, std=0.5)])
        labels_path_list = os.listdir(os.path.join(data_dir, 'label/'))
        images_path_list = os.listdir(os.path.join(data_dir, 'input/'))


        labels_path_list.sort(key=lambda f: (''.join(filter(str.isdigit, f))))
        images_path_list.sort(key=lambda f: (''.join(filter(str.isdigit, f))))
        self.noise = 10 / 255.0 * np.random.randn(len(images_path_list),181,217,1)###噪声水平
        self.lst_data = images_path_list
        self.lst_label = labels_path_list


    def __getitem__(self, index):
        data = cv2.imread(os.path.join(self.input_dir, self.lst_data[index]),0)
        label = cv2.imread(os.path.join(self.label_dir, self.lst_label[index]),0)
        if data.dtype or label.dtype== np.uint8:
            data = data / 255.0
            label = label /255.0

        if data.ndim or label.ndim== 2:
            data = np.expand_dims(data, axis=2)
            label = np.expand_dims(label, axis=2)

        if data.shape[0] > data.shape[1]:
            data = data.transpose((1, 0, 2))

        data = data+ self.noise[index]
        data=data.astype(np.float32)
        label=label.astype(np.float32)
        data = self.transform(data)
        label = self.transform(label)
        daset = {'label': label, 'input': data}

        return daset

    def __len__(self):
        return len(self.lst_data)
class BWTestset(torch.utils.data.Dataset):
    """
    dataset of image files of the form
       stuff<number>_trans.pt
       stuff<number>_density.pt
    """

    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.transform = transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize(mean=0.5, std=0.5)])
        images_path_list = os.listdir(data_dir)

        images_path_list.sort(key=lambda f: (''.join(filter(str.isdigit, f))))

        self.lst_data = images_path_list
        # self.noise = 20 / 255.0 * np.random.randn(len(self.images_path_list), 181,181,1)

    def __getitem__(self, index):
        data = cv2.imread(os.path.join(self. data_dir, self.lst_data[index]),0)
        if data.dtype== np.uint8:
            data = data / 255.0

        if data.ndim == 2:
            data = np.expand_dims(data, axis=2)

        if data.shape[0] > data.shape[1]:
            data = data.transpose((1, 0, 2))

        data=data.astype(np.float32)
        data = self.transform(data)
        daset = {'input': data}

        return daset

    def __len__(self):
        return len(self.lst_data)

class MyDataset(torch.utils.data.Dataset):
    """
    dataset of image files of the form
       stuff<number>_trans.pt
       stuff<number>_density.pt
    """

    def __init__(self, data_dir):
        self.input_dir = os.path.join(data_dir, 'input/')
        self.label_dir = os.path.join(data_dir, 'label/')
        self.transform = transforms.Compose([transforms.ToPILImage(),
                                             transforms.ToTensor(),
                                             transforms.Normalize(mean=0.5, std=0.5)])
        labels_path_list = os.listdir(os.path.join(data_dir, 'label/'))
        images_path_list = os.listdir(os.path.join(data_dir, 'input/'))


        labels_path_list.sort(key=lambda f: (''.join(filter(str.isdigit, f))))
        images_path_list.sort(key=lambda f: (''.join(filter(str.isdigit, f))))
        self.lst_data = images_path_list
        self.lst_label = labels_path_list


    def __getitem__(self, index):
        data = cv2.imread(os.path.join(self.input_dir, self.lst_data[index]),0)
        label = cv2.imread(os.path.join(self.label_dir, self.lst_label[index]),0)
        if data.dtype or label.dtype== np.uint8:
            data = data / 255.0
            label = label /255.0

        data=data.astype(np.float32)
        label=label.astype(np.float32)
        data = self.transform(data)
        label = self.transform(label)
        daset = {'label': label, 'input': data}

        return daset

    def __len__(self):
        return len(self.lst_data)
class MyTestset(torch.utils.data.Dataset):
    """
    dataset of image files of the form
       stuff<number>_trans.pt
       stuff<number>_density.pt
    """

    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.transform = transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize(mean=0.5, std=0.5)])
        images_path_list = os.listdir(data_dir)

        images_path_list = natsorted(images_path_list)



        self.lst_data = images_path_list
        # self.noise = 20 / 255.0 * np.random.randn(len(self.images_path_list), 181,181,1)

    def __getitem__(self, index):
        data = cv2.imread(os.path.join(self. data_dir, self.lst_data[index]),0)
        if data.dtype== np.uint8:
            data = data / 255.0


        data=data.astype(np.float32)
        data = self.transform(data)
        daset = {'input': data}

        return daset

    def __len__(self):
        return len(self.lst_data)

class KDataset(torch.utils.data.Dataset):
    """
    dataset of image files of the form
       stuff<number>_trans.pt
       stuff<number>_density.pt
    """

    def __init__(self, data_dir):
        self.input_dir = os.path.join(data_dir, 'input/')
        self.label_dir = os.path.join(data_dir, 'label/')
        self.transform = transforms.Compose([
                                             transforms.ToTensor(),
                                             transforms.Normalize(mean=0.5, std=0.5)])
        labels_path_list = os.listdir(os.path.join(data_dir, 'label/'))
        images_path_list = os.listdir(os.path.join(data_dir, 'input/'))


        labels_path_list.sort(key=lambda f: (''.join(filter(str.isdigit, f))))
        images_path_list.sort(key=lambda f: (''.join(filter(str.isdigit, f))))
        self.lst_data = images_path_list
        self.lst_label = labels_path_list


    def __getitem__(self, index):
        data = tif.imread(os.path.join(self.input_dir, self.lst_data[index]))
        label = tif.imread(os.path.join(self.label_dir, self.lst_label[index]))
        if data.dtype or label.dtype== np.uint8:
            data = data / 255.0
            label = label /255.0
        # data = data.transpose((2, 0, 1))
        # label = label.transpose((2, 0, 1))
        data=data.astype(np.float32)
        label=label.astype(np.float32)
        data = self.transform(data)
        label = self.transform(label)
        daset = {'label': label, 'input': data}

        return daset

    def __len__(self):
        return len(self.lst_data)
class KTestset(torch.utils.data.Dataset):
    """
    dataset of image files of the form
       stuff<number>_trans.pt
       stuff<number>_density.pt
    """

    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.transform = transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize(mean=0.5, std=0.5)])
        images_path_list = os.listdir(data_dir)

        images_path_list.sort(key=lambda f: (''.join(filter(str.isdigit, f))))

        self.lst_data = images_path_list
        # self.noise = 20 / 255.0 * np.random.randn(len(self.images_path_list), 181,181,1)

    def __getitem__(self, index):
        data = tif.imread(os.path.join(self. data_dir, self.lst_data[index]))
        if data.dtype== np.uint8:
            data = data / 255.0
        data = data.transpose((2, 0, 1))
        data=data.astype(np.float32)
        data = self.transform(data)
        daset = {'input': data}

        return daset

    def __len__(self):
        return len(self.lst_data)
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = MyTestset(data_dir='./datasets/kspace/test')
    trainset = MyDataset(data_dir='./datasets/kspace/train')
    data = cv2.imread('./datasets/kspace/train/input/image_1.png',0)
    if data.ndim == 2:
        data = np.expand_dims(data, axis=2)
    transform = transforms.Compose([ transforms.ToTensor(),
                                    transforms.Normalize(mean=0.5, std=0.5)])
    data =transform(data)
    x=2