import torch
from scipy.io import loadmat
import random
import os
import tifffile
from utils import *
import scipy.io as sio
import numpy as np
import argparse
def del_file(path):
    ls = os.listdir(path)
    for i in ls:
        c_path = os.path.join(path, i)
        if os.path.isdir(c_path):
            del_file(c_path)
        else:
            os.remove(c_path)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
row=4
col=5
Nx=128
Ny=128
Nz=row*col
yz=Ny*Nz


train_list = list(range(0, yz))  # rest
test_list = list(range(0, yz))


train_list.sort()

test_list.sort()
print(train_list)
print(test_list)
datapa0=r".\datasets\firdenoise"
datapa=[datapa0+r"\train",datapa0+r"\test"]

for dir in datapa:
    if os.path.exists(dir):
        del_file(dir)

for dir in datapa:
    if os.path.exists(dir):
        pass
    else:
        os.makedirs(dir)

file_train_data = r'.\trainmat\ksp4.mat'
file_train_label = r'.\trainmat\ksp2.mat'

file_test_data = r'.\trainmat\ksp3.mat'
file_test_label = r'.\trainmat\ksp1.mat'

#加载训练数据和label
train_data_dict = loadmat(file_train_data)
train_label_dict = loadmat(file_train_label)

# print(type(train_label_dict2))
# print(train_label_dict2.keys())
#加载试验数据及其label
test_data_dict = loadmat(file_test_data)
test_label_dict = loadmat(file_test_label)

#取出训练数据并转为张量
train_data = train_data_dict["ksp4"]
train_data = torch.from_numpy(train_data)
train_data = train_data.permute(2, 1, 0, 3)  # 3072 test 128 test
train_data = (train_data.float())
# tenshow(train_data)

#取出训练label并转为张量
train_label = train_label_dict["ksp2"]
train_label = torch.from_numpy(train_label)
train_label = train_label.permute(2, 1, 0, 3)
train_label = (train_label.float())
# tenshow(train_label)
#取出test数据及其label并转为张量
test_data = test_data_dict["ksp3"]
test_data = torch.from_numpy(test_data)
test_data = test_data.permute(2, 1, 0, 3)
test_data = (test_data.float())
# tenshow(test_data)

test_label = test_label_dict["ksp1"]
test_label = torch.from_numpy(test_label)
test_label = test_label.permute(2, 1, 0, 3)
test_label = (test_label.float())
# tenshow(test_label)
## train data

for i in range(0, len(train_list)):
    index = train_list[i]
    train_tmp = train_data[index]#将取出训练数据入暂存变量tmp
    train_tmp = train_tmp.clone().detach()

    label_tmp = train_label[index, :, :,0].clone().detach()#将取出训练label入暂存变量tmp
    label_tmp = label_tmp.unsqueeze(-1)
    label_tmp=torch.cat([label_tmp, label_tmp], dim=2)

    tmp = {'k-space': train_tmp, 'label': label_tmp}
    torch.save(tmp,f'{datapa[0]}\\'+ str(i) + '.pth')

dout = torch.empty((len(train_list), 2, Nx, 2))
## test data
for i in range(0, len(test_list)):
    index = test_list[i]
    test_tmp = test_data[index].clone().detach()
    test_tmp = test_tmp

    test_label_tmp = test_label[index, :, :, 0].clone().detach()
    test_label_tmp = test_label_tmp.unsqueeze(-1)
    test_label_tmp=torch.cat([test_label_tmp, test_label_tmp], dim=2)
    dout[i] = test_label_tmp
    tmp = {'k-space': test_tmp, 'label': test_label_tmp}
    torch.save(tmp,f'{datapa[1]}\\'+ str(i) + '.pth')
fig=tenshow(dout,row,col,Nx,Ny)
plt.show()
wflag = 0
# fig = cv2.normalize(fig, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
diri=f"./results/first_denoising/initial/"
if os.path.exists(diri):
    pass
else:
    os.makedirs(diri)
lenf = fig.shape[-1]
for i in range(lenf):
    cv2.imwrite(diri+f"image_{i + 1}.png",fig[..., i])