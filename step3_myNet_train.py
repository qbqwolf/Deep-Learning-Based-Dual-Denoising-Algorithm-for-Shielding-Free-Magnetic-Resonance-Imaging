import argparse
from torch.utils.tensorboard import SummaryWriter
import datetime
import time
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from modelfile.modelres import Net1
from utils import *
from lossfuc import *
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
modelname = 'rescnn'
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
parser.add_argument("--dataset_name", type=str, default="firdenoise",help="name of the dataset")  ## ../input/facades-dataset
parser.add_argument("--batch_size", type=int, default=8, help="size of the batches")
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
input_shape = (opt.channels, opt.img_height, opt.img_width)

model = Net1()
print(model)
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
lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
    optimizer, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)
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
## val data loader
valset = prepareData_train('val')
val_dataloader = torch.utils.data.DataLoader(valset, batch_size=opt.batch_size, shuffle=True, num_workers=0)
'''
Compatible with tensorflow backend
'''
def train():
    prev_time = time.time()  ## 开始时间
    losses = []
    val_losses = []
    for epoch in range(opt.epoch, opt.n_epochs):  ## for epoch in (0, 50)
        model.train()
        loop=tqdm(dataloader, leave=True)
        for i, batch in enumerate(loop):
            loop.set_description('Epoch {}/{} - Training batch {}'.format(epoch + 1, opt.n_epochs, i))
            loop.update(1)
            ## 读取数据集中的真图片
            torch.autograd.set_detect_anomaly(True)
            input = batch[0].cuda() # inputs = data[0].reshape(-1,test,Nx,test).to(device)
            label = batch[1].cuda()[:,:,:,0].unsqueeze(-1)

            output=model(input)
            loss = mse(output,label)
            optimizer.zero_grad()  ## 在反向传播之前，先将梯度归0
            loss.backward()  ## 将误差反向传播
            optimizer.step()  ## 更新参数
            # average_value = output.squeeze()[:, 0,:].mean()
            loop.set_postfix(loss=loss.item())
            losses+=[loss.item()]
            writer.add_scalar('training_loss', loss.item(), epoch * len(dataloader) + i)
        model.eval()  # 设置为评估模式
        print('\n testing...')
        val_loop = tqdm(val_dataloader, leave=True)
        with torch.no_grad():
            for i, val_batch in enumerate(val_loop):
                val_loop.set_description('Epoch {}/{} - Validation batch {}'.format(epoch + 1, opt.n_epochs, i))
                val_loop.update(1)
                val_input = val_batch[0].cuda()
                val_label = val_batch[1].cuda()[:, :, :, 0].unsqueeze(-1)
                val_output = model(val_input)
                val_loss = mse(val_output, val_label)
                val_loop.set_postfix(loss=val_loss.item())
                val_losses += [val_loss.item()]
                writer.add_scalar('validation_loss', val_loss.item(),  epoch * len(val_dataloader) + i)
        # 更新学习率
        # lr_scheduler.step()
        torch.save(model.state_dict(), "checkpoints/%s/myNet_%d.pth" % (opt.dataset_name, epoch + 1))
    ## 训练结束后，保存模型
    total_time = datetime.timedelta(seconds=time.time() - prev_time)
    writer.close()
    # torch.save(model.state_dict(), "save/%s/myNet_%d.pth" % (opt.dataset_name, epoch+1))
    print("save my model finished !!")
    print("Total training time:", total_time)

    # np.save(f'./checkpoints/{opt.dataset_name}/rescnn_time', total_time.total_seconds())
    # plt.plot(losses, label=f'residual CNN,train time:{total_time.total_seconds():.2f}s')
    np.save(f'./checkpoints/{opt.dataset_name}/{modelname}', losses)
    np.save(f'./checkpoints/{opt.dataset_name}/{modelname}_val', val_losses)
    np.save(f'./checkpoints/{opt.dataset_name}/{modelname}_time', total_time.total_seconds())
    # 保存训练损失和验证损失到文件


    # 绘制训练损失曲线
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(losses, label=f'{modelname} Training Loss')
    plt.xlabel("Batch")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.legend(loc='upper right')

    # 绘制验证损失曲线
    plt.subplot(1, 2, 2)
    plt.plot(val_losses, label=f'{modelname} Validation Loss')
    plt.xlabel("Batch")
    plt.ylabel("Loss")
    plt.title("Validation Loss")
    plt.legend(loc='upper right')

    # 保存图像
    plt.savefig(f'./checkpoints/{opt.dataset_name}/train_val_loss.png')
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
