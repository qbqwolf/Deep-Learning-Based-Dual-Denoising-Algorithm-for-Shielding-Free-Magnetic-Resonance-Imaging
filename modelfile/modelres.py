import torch.nn as nn
import torch
from thop import profile
import torch.nn.functional as F
from torchsummary import summary
# artifacts learning
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

class Net1(nn.Module):
    def __init__(self):
        super(Net1, self).__init__()
        self.features = nn.Sequential(
            ResBlock(2, 128, kernel_size=11, stride=1, padding=5),
            ResBlock(128, 64, kernel_size=9, stride=1, padding=4),
            ResBlock(64, 32, kernel_size=5, stride=1, padding=2),
            ResBlock(32, 32, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(32, 2, kernel_size=(1, 4), stride=1, padding=0)
        )

    def forward(self, x):
        x = self.features(x)
        return x

class Discriminator(nn.Module):
    def __init__(self,size=128):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, bn=True):
            block = [nn.Conv2d(in_filters, out_filters, (3,1), (2,1), (1,0)), nn.LeakyReLU(0.2, inplace=True), nn.Dropout2d(0.25)]#Conv卷积，Relu激活，Dropout将部分神经元失活，进而防止过拟合
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))    #如果bn这个参数为True，那么就需要在block块里面添加上BatchNorm的归一化函数
            return block
        self.model = nn.Sequential(
            *discriminator_block(2, 16, bn=False),
            *discriminator_block(16, 32),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
        )

        # The height and width of downsampled image
        ds_size = size // 2 ** 4
        self.adv_layer = nn.Sequential(nn.Linear(128 * ds_size, 1)) #先进行线性变换，再进行激活函数激活
                          #上一句中 128是指model中最后一个判别模块的最后一个参数决定的，ds_size由model模块对单张图片的卷积效果决定的，而2次方是整个模型是选取的长宽一致的图片
    def forward(self, img):
        # o1 = self.model[0](img)
        # o2 = self.model[1](o1)
        # o3 = self.model[test](o2)
        # o4 = self.model[3](o3)
        out = self.model(img)
        out = out.view(out.shape[0], -1)    #将处理之后的数据维度变成batch * N的维度形式
        validity = self.adv_layer(out)      #第92行定义
        return validity

if __name__ == '__main__':
    model = Net1()
    data = torch.randn(1, 2, 128, 2)
    output = model(data)
    print(output.size())
    flops, params = profile(model, (data,))
    # print('flops: ', flops, 'params: ', params)
    print('flops: %.2f M, params: %.2f M' % (flops / 1000000.0, params / 1000000.0))
    # summary(model.cuda(),input_size=(test, 128, test))
