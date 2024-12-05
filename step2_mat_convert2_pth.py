import torch
from scipy.io import loadmat
import random
import cv2
import tifffile
from utils import *
from seting import Nx,Ny,Nc,Nz,row,col
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
yz=Ny*Nz


train_list = list(range(0, yz))  # rest
test_list = list(range(0, yz))
train_list.sort()
test_list.sort()
# 计算验证集的大小
val_size = int(len(train_list) * 0.15)

# 从训练集中随机选择验证集的索引
val_list = random.sample(train_list, val_size)
print(train_list)
print(val_list)
print(test_list)
datapa0=r".\datasets\firdenoise"
datapa=[datapa0+r"\train",datapa0+r"\val",datapa0+r"\test"]

for dir in datapa:
    if os.path.exists(dir):
        del_file(dir)

for dir in datapa:
    if os.path.exists(dir):
        pass
    else:
        os.makedirs(dir)
sub=""
file_train_data = r'trainmat/'+sub+'ksp4.mat'
file_train_label = r'trainmat/'+sub+'ksp2.mat'

file_test_data = r'trainmat/'+sub+'ksp3.mat'
file_test_label = r'trainmat/'+sub+'ksp1.mat'

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
train_data = train_data.permute(2, 1, 0, 3) #(NyNz,RI,Nx,Nc)
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
# tenshow(test_label,row,col,Nx,Ny)
# plt.show()
## train data
vali=0
traini=0
for i in range(0, len(train_list)):
    index = train_list[i]
    train_tmp = train_data[index]#将取出训练数据入暂存变量tmp
    train_tmp = train_tmp.clone().detach()

    label_tmp = train_label[index, :, :,:].clone().detach()#将取出训练label入暂存变量tmp
    label_tmp = label_tmp
    tmp = {'k-space': train_tmp, 'label': label_tmp}
    # label_tmp=torch.cat([label_tmp, label_tmp], dim=2)
    if i in val_list:
        folder = datapa[1]  # 验证集文件夹
        torch.save(tmp, f'{folder}\\' + str(vali) + '.pth')
        vali+=1
    else:
        folder = datapa[0]  # 训练集文件夹
        torch.save(tmp, f'{folder}\\' + str(traini) + '.pth')
        traini+=1

dout = torch.empty((len(train_list), 2, Nx, Nc))
## test data
for i in range(0, len(test_list)):
    index = test_list[i]
    test_tmp = test_data[index].clone().detach()
    test_tmp = test_tmp

    test_label_tmp = test_label[index, :, :, :].clone().detach()
    test_label_tmp = test_label_tmp
    # test_label_tmp=torch.cat([test_label_tmp, test_label_tmp], dim=2)
    dout[i] = test_label_tmp
    tmp = {'k-space': test_tmp, 'label': test_label_tmp}
    torch.save(tmp,f'{datapa[2]}\\'+ str(i) + '.pth')
fig=mulc_tenshow(dout,row,col,Nx,Ny,0)
