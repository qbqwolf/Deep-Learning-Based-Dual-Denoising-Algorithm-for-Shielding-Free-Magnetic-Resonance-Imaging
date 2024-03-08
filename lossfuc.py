import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
import torchvision
from torchvision import models



class VGGLoss(nn.Module):
    def __init__(self, device, n_layers=5):
        super().__init__()

        feature_layers = (2, 7, 12, 21, 30)
        self.weights = (1.0, 1.0, 1.0, 1.0, 1.0)

        vgg = torchvision.models.vgg19(weights=models.VGG19_Weights.DEFAULT).features

        self.layers = nn.ModuleList()
        prev_layer = 0
        for next_layer in feature_layers[:n_layers]:
            layers = nn.Sequential()
            for layer in range(prev_layer, next_layer):
                layers.add_module(str(layer), vgg[layer])
            self.layers.append(layers.to(device))
            prev_layer = next_layer

        for param in self.parameters():
            param.requires_grad = False

        # self.criterion = nn.L1Loss().to(device)
        self.criterion = nn.MSELoss().to(device)

    def forward(self, source, target):
        loss = 0
        source = source.repeat(1, 3, 1, 1)
        target = target.repeat(1 ,3, 1, 1)
        for layer, weight in zip(self.layers, self.weights):
            source = layer(source)
            with torch.no_grad():
                target = layer(target)
            loss += weight * self.criterion(source, target)

        return loss
class VGGLoss2(nn.Module):
    def __init__(self, device, n_layers=5):
        super().__init__()

        feature_layers = (2, 7, 12, 21, 30)
        self.weights = (1.0, 1.0, 1.0, 1.0, 1.0)

        self.conv = nn.Conv2d(2,3,1,stride=1,padding=0).to(device)
        vgg = torchvision.models.vgg19(weights=models.VGG19_Weights.DEFAULT).features

        self.layers = nn.ModuleList()
        prev_layer = 0
        for next_layer in feature_layers[:n_layers]:
            layers = nn.Sequential()
            for layer in range(prev_layer, next_layer):
                layers.add_module(str(layer), vgg[layer])
            self.layers.append(layers.to(device))
            prev_layer = next_layer

        for param in self.parameters():
            param.requires_grad = False

        # self.criterion = nn.L1Loss().to(device)
        self.criterion = nn.MSELoss().to(device)

    def forward(self, source, target):
        loss = 0
        source = self.conv(source)
        target = self.conv(target)
        for layer, weight in zip(self.layers, self.weights):
            source = layer(source)
            with torch.no_grad():
                target = layer(target)
            loss += weight * self.criterion(source, target)

        return loss
def cal_gp(D, real_imgs, fake_imgs, device):  # 定义函数，计算梯度惩罚项gp
    r = torch.rand(size=(real_imgs.shape[0], 1, 1, 1), device=device)  # 真假样本的采样比例r，batch size个随机数，服从区间[0,1)的均匀分布

    x = (r * real_imgs + (1 - r) * fake_imgs).requires_grad_(True)  # 输入样本x，由真假样本按照比例产生，需要计算梯度
    d = D(x)  # 判别网络D对输入样本x的判别结果D(x)
    fake = torch.ones_like(d).to(device)  # 定义与d形状相同的张量，代表梯度计算时每一个元素的权重
    g = torch.autograd.grad(  # 进行梯度计算
        outputs=d,  # 计算梯度的函数d，即D(x)
        inputs=x,  # 计算梯度的变量x
        grad_outputs=fake,  # 梯度计算权重
        create_graph=True,  # 创建计算图
        retain_graph=True  # 保留计算图
    )[0]  # 返回元组的第一个元素为梯度计算结果
    gp = ((g.norm(2, dim=1) - 1) ** 2).mean()  # (||grad(D(x))||test-1)^test 的均值
    return gp  # 返回梯度惩罚项gp
class ResBlockLoss(nn.Module):
    def __init__(self):
        super(ResBlockLoss, self).__init__()
        self.criterion = nn.MSELoss()

    def forward(self, input, label,model):
        resblock_output = model.features[0](input)
        resblock_label = model.features[0](label)
        loss = self.criterion(resblock_output, resblock_label)
        return loss

if __name__ == '__main__':

    # 创建一个虚拟设备，如果可用的话使用GPU，否则使用CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 实例化 VGGLoss
    vgg_loss = VGGLoss(device=device, n_layers=5)

    # 加载测试图像
    image = torch.randn(2, 1, 128, 128).to(device)

    # 创建目标图像（例如，可以使用相同的测试图像）
    target_image = torch.randn(2, 1, 128, 128).to(device)

    # 计算损失
    loss = vgg_loss(image, target_image)

    # 打印损失值
    print(f"VGG Loss: {loss.item()}")
