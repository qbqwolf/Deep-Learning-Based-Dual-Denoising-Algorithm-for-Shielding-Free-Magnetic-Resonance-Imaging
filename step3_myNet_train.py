import argparse
import os
import numpy as np
import math
from torch.utils.tensorboard import SummaryWriter
import itertools
import datetime
import time
import torchvision.transforms as transforms
from tqdm import tqdm
from torchvision.utils import save_image, make_grid
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.autograd import Variable
from modelfile.modelres import Net1
from utils import *
import torch.nn as nn
import torch.nn.functional as F
import torch
from PIL import Image
from lossfuc import *
import torch



# move the model to cuda device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def abs_tensor(tensor):
    x= tensor.clone()
    return torch.abs(x)

def del_file(path):
    ls = os.listdir(path)
    for i in ls:
        c_path = os.path.join(path, i)
        if os.path.isdir(c_path):
            del_file(c_path)
        else:
            os.remove(c_path)

datapa=['./logs']

## 超参数配置
parser = argparse.ArgumentParser()
parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from")
parser.add_argument("--n_epochs", type=int, default=6, help="number of epochs of training")
parser.add_argument("--dataset_name", type=str, default="firdenoise",
                    help="name of the dataset")  ## ../input/facades-dataset
parser.add_argument("--batch_size", type=int, default=12, help="size of the batches")
parser.add_argument("--lr", type=float, default=3e-4, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--decay_epoch", type=int, default=0, help="epoch from which to start lr decay")
parser.add_argument("--n_cpu", type=int, default=2, help="number of cpu threads to use during batch generation")
parser.add_argument("--img_height", type=int, default=128, help="size of image height")
parser.add_argument("--img_width", type=int, default=1, help="size of image width")
parser.add_argument("--channels", type=int, default=2, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=100, help="interval between saving generator outputs")
parser.add_argument("--checkpoint_interval", type=int, default=-1, help="interval between saving model checkpoints")
opt = parser.parse_args()


## 创建文件夹
os.makedirs("images/%s" % opt.dataset_name, exist_ok=True)
os.makedirs("checkpoints/%s" % opt.dataset_name, exist_ok=True)

## input_shape:(3, 256, 256)
input_shape = (opt.channels, opt.img_height, opt.img_width)

## 创建生成器，判别器对象
#model = RIDNET(feats=64, reduction=16, rgb_range=255)
# model = Inception_resnet_CNN()
model = Net1()
print(model)
## 损失函数
## MES 二分类的交叉熵
## L1loss 相比于L2 Loss保边缘
mse = torch.nn.MSELoss()
l1 = torch.nn.L1Loss()
resloss = ResBlockLoss()
## 如果有显卡，都在cuda模式中运行
if torch.cuda.is_available():
    model.cuda()
    mse.cuda()
    l1.cuda()

## 定义优化函数,优化函数的学习率为0.0003
optimizer = torch.optim.Adam(params=model.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
# optimizer = torch.optim.SGD(params=model.parameters(), lr=opt.lr , momentum=0.1, dampening=0.5, weight_decay=0.000, nesterov=False)
## 学习率更行进程
lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
    optimizer, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step
)

## 先前生成的样本的缓冲区
fake_out_buffer = ReplayBuffer()

datapath=f'datasets/{opt.dataset_name}/'

class prepareData_train(Dataset):
    def __init__(self, train_or_test):
        self.files = os.listdir(datapath + train_or_test)
        self.train_or_test = train_or_test

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = torch.load(datapath + self.train_or_test + '/' + self.files[idx])
        return data['k-space'], data['label']

## Training data loader
trainset = prepareData_train('train')
dataloader = torch.utils.data.DataLoader(trainset, batch_size=opt.batch_size, shuffle=True, num_workers=0)
## Test data loader

validationset = prepareData_train('test')
val_dataloader = torch.utils.data.DataLoader(validationset, batch_size=1,shuffle=True, num_workers=0)

'''
Compatible with tensorflow backend
'''



def train():
    # ----------
    #  Training
    # ----------
    prev_time = time.time()  ## 开始时间
    losses = []
    for epoch in range(opt.epoch, opt.n_epochs):  ## for epoch in (0, 50)
        loop=tqdm(dataloader, leave=True)
        for i, batch in enumerate(
                loop):
            loop.set_description('Epoch {}/{} - Training batch {}'.format(epoch + 1, opt.n_epochs, i))
            loop.update(1)
            ## 读取数据集中的真图片
            ## 将tensor变成Variable放入计算图中，tensor变成variable之后才能进行反向传播求梯度
            torch.autograd.set_detect_anomaly(True)
            input = batch[0].cuda() # inputs = data[0].reshape(-1,test,Nx,test).to(device)
            label = batch[1].cuda()[:,:,:,0].unsqueeze(-1) # torch.Size([16, test, 448, 1]) 关闭了RF的rcv线圈的batch_size（对应16个pth文件）、实部虚部、Nx、Nc

            output=model(input)
            # Total loss
            # loss = mse(output,label)
            loss = mse(output,label)
            # if epoch>0:
            #     loss = torch.clamp(loss, max=100)
            optimizer.zero_grad()  ## 在反向传播之前，先将梯度归0
            loss.backward()  ## 将误差反向传播
            # nn.utils.clip_grad_norm_(model.parameters(), max_norm=20, norm_type=2)
            optimizer.step()  ## 更新参数

            loop.set_postfix(loss=loss.item())
            losses+=[loss.item()]
            ## ----------------------
            ##  打印日志Log Progress
            ## ----------------------
            ## 确定剩下的大约时间  假设当前 epoch = 5， i = 100
            # batches_done = epoch * len(dataloader) + i  ## 已经训练了多长时间 5 * 400 + 100 次
            # batches_left = opt.n_epochs * len(dataloader) - batches_done  ## 还剩下 50 * 400 - 2100 次
            # time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))  ## 还需要的时间 time_left = 剩下的次数 * 每次的时间
            # prev_time = time.time()
            writer.add_scalar('training_loss', loss.item(), epoch * len(dataloader) + i)

        # 更新学习率
        # lr_scheduler.step()
        torch.save(model.state_dict(), "checkpoints/%s/myNet_%d.pth" % (opt.dataset_name, epoch + 1))

    ## 训练结束后，保存模型
    total_time = datetime.timedelta(seconds=time.time() - prev_time)
    writer.close()
    # torch.save(model.state_dict(), "save/%s/myNet_%d.pth" % (opt.dataset_name, epoch+1))
    print("save my model finished !!")
    print("Total training time:", total_time)
    # np.save(f'./checkpoints/{opt.dataset_name}/rescnn', losses)
    # np.save(f'./checkpoints/{opt.dataset_name}/rescnn_time', total_time.total_seconds())
    # plt.plot(losses, label=f'residual CNN,train time:{total_time.total_seconds():.2f}s')
    np.save(f'./checkpoints/{opt.dataset_name}/cnn', losses)
    np.save(f'./checkpoints/{opt.dataset_name}/cnn_time', total_time.total_seconds())
    plt.plot(losses, label=f'CNN,train time:{total_time.total_seconds():.2f}s')
    plt.xlabel("Batch")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.legend(loc='upper right')  # 显示图例
    plt.savefig(f'./checkpoints/{opt.dataset_name}/trainloss.png')
    plt.show()

## 函数的起始
if __name__ == '__main__':
    flag=1
    for dir in datapa:
        if os.path.exists(dir) and flag:
            del_file(dir)
    writer = SummaryWriter('./logs')
    train()
    #tensorboard --logdir logs
