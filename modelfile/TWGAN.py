import torch.nn as nn
import torch
from torch.nn import init
from thop import profile
from torchsummary import summary
from torch.optim import lr_scheduler
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.nn.utils import spectral_norm
import os
# artifacts learning
def del_file(path):
    ls = os.listdir(path)
    for i in ls:
        c_path = os.path.join(path, i)
        if os.path.isdir(c_path):
            del_file(c_path)
        else:
            os.remove(c_path)
def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.

    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); test. initialize the network weights
    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        gain (float)       -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,test

    Return an initialized network.
    """
    if gpu_ids:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
    init_weights(net, init_type, init_gain=init_gain)
    return net
class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, stride=1, padding=padding)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)

        if in_channels != out_channels:
            self.conv_identity = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if identity.shape != out.shape:
            identity = self.conv_identity(identity)

        out += identity
        out = self.relu2(out)

        return out

class downResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(downResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, stride=1, padding=padding)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)

        if in_channels != out_channels:
            self.conv_identity = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0)
        self.olayer = nn.Conv2d(out_channels, out_channels, 4, stride=2, padding=1)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if identity.shape != out.shape:
            identity = self.conv_identity(identity)

        out += identity
        out = self.olayer(out)
        out = self.relu2(out)

        return out

class RDBSN(nn.Module):
    def __init__(self, nChannels, nDenselayer, growthRate):
        super(RDBSN, self).__init__()
        nChannels_ = nChannels
        modules = []
        for i in range(nDenselayer):
            modules.append(make_dense_sn(nChannels_, growthRate))
            nChannels_ += growthRate
        self.dense_layers = nn.Sequential(*modules)
        self.conv_1x1 = nn.Conv2d(nChannels_, nChannels, kernel_size=1, padding=0, bias=False)

    def forward(self, x):
        out = self.dense_layers(x)
        out = self.conv_1x1(out)
        out = out + x
        return out


class make_dense_sn(nn.Module):
    def __init__(self, nChannels, growthRate, kernel_size=3):
        super(make_dense_sn, self).__init__()
        self.conv = spectral_norm(nn.Conv2d(nChannels, growthRate, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False))

    def forward(self, x):
        out = F.relu(self.conv(x))
        out = torch.cat((x, out), 1)
        return out

class Net1(nn.Module):
    def __init__(self,nch_in, nch_out):
        super(Net1, self).__init__()
        self.features = nn.Sequential(
            ResBlock(nch_in, 128, kernel_size=11, stride=1, padding=5),
            ResBlock(128, 64, kernel_size=9, stride=1, padding=4),
            ResBlock(64, 32, kernel_size=5, stride=1, padding=2),
            ResBlock(32, 32, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(32, nch_out, kernel_size=1, stride=1, padding=0)
        )
        self.tanh = nn.Tanh()
    def forward(self, x):
        x = self.features(x)
        x=self.tanh(x)
        return x


class Net2(nn.Module):
    def __init__(self,nch_in, nch_out):
        super(Net2, self).__init__()
        self.features = nn.Sequential(
            ResBlock(nch_in, 128, kernel_size=11, stride=1, padding=5),
            ResBlock(128, 64, kernel_size=9, stride=1, padding=4),
            ResBlock(64, 32, kernel_size=5, stride=1, padding=2),
            ResBlock(32, 32, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(32, nch_out, kernel_size=1, stride=1, padding=0)
        )
        self.tanh = nn.Tanh()
    def forward(self, x1,x2):
        xin =torch.cat([x1, x2], dim=1)
        x = self.features(xin)
        x = self.tanh(x)
        return x

class Discriminator(nn.Module):
    def __init__(self,size=128):
        super(Discriminator, self).__init__()
        def discriminator_block(in_filters, out_filters, bn=True):
            block = [nn.Conv2d(in_filters, out_filters, 3, 2, 1), nn.LeakyReLU(0.2, inplace=True), nn.Dropout2d(0.25)]#Conv卷积，Relu激活，Dropout将部分神经元失活，进而防止过拟合
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))    #如果bn这个参数为True，那么就需要在block块里面添加上BatchNorm的归一化函数
            return block

        self.model = nn.Sequential(
            *discriminator_block(1, 16, bn=False),
            *discriminator_block(16, 32),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
        )

        # The height and width of downsampled image
        ds_size = size // 2 ** 4
        self.adv_layer = nn.Sequential(nn.Linear(128 * ds_size ** 2, 1)) #先进行线性变换，再进行激活函数激活
                          #上一句中 128是指model中最后一个判别模块的最后一个参数决定的，ds_size由model模块对单张图片的卷积效果决定的，而2次方是整个模型是选取的长宽一致的图片
    def forward(self, img):
        out = self.model(img)
        out = out.view(out.shape[0], -1)    #将处理之后的数据维度变成batch * N的维度形式
        validity = self.adv_layer(out)      #第92行定义

        return validity

if __name__ == '__main__':

    datapa = ['./logs/netG1', './logs/Dis', './logs/netG2', './logs/netG3']
    for dir in datapa:
        if os.path.exists(dir):
            pass
        else:
            os.makedirs(dir)
    flag = 1
    for dir in datapa:
        if os.path.exists(dir) and flag:
            del_file(dir)
    netG1 = Net1(1, 1)
    netG2 = Net1(1, 1)
    netG3 = Net2(2, 1)
    Dis = Discriminator(size=128)
    data1 = torch.randn(2, 1, 128, 128)
    data2 = torch.randn(2, 1, 128, 128)
    output = netG1(data1)
    print(output.size())
    # if tf.gfile.Exists(logdir):
    #     tf.gfile.DeleteRecursively(logdir)
    writer = SummaryWriter("logs/netG1")
    writer.add_graph(netG1, data1)
    writer.close()

    writer = SummaryWriter("logs/Dis")
    writer.add_graph(Dis, data1)
    writer.close()

    writer = SummaryWriter("logs/netG2")
    writer.add_graph(netG2, data2)
    writer.close()

    writer = SummaryWriter("logs/netG3")
    dummy_input = (data1, data2)
    writer.add_graph(netG3, dummy_input)
    writer.close()
# tensorboard --logdir logs