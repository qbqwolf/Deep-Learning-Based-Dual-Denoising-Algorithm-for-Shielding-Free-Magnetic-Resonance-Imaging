import argparse
import matplotlib.pyplot as plt
from scipy.io import loadmat
from utils import *
from dataset import *
from modelfile.modelres import Net1
from torch.utils.data import DataLoader
import sys
from seting import Nx,Ny,Nc,Nz,row,col
import cv2
def del_file(path):
    ls = os.listdir(path)
    for i in ls:
        c_path = os.path.join(path, i)
        if os.path.isdir(c_path):
            del_file(c_path)
        else:
            os.remove(c_path)
def test():
    ## 超参数设置
    parser = argparse.ArgumentParser()
    parser.add_argument('--batchSize', type=int, default=1, help='size of the batches')
    parser.add_argument('--dataroot', type=str, default='./datasets/firdenoise/', help='root directory of the dataset')
    parser.add_argument('--channels', type=int, default=2, help='number of channels of input data')
    parser.add_argument('--n_residual_blocks', type=int, default=9, help='number of channels of output data')
    parser.add_argument('--size', type=int, default=128, help='size of the data (squared assumed)')
    parser.add_argument('--cuda', type=bool, default=True, help='use GPU computation')
    parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')
    parser.add_argument('--model_data', type=str, default='./checkpoints/firdenoise/', help='model checkpoint file')
    parser.add_argument("--n_epochs", type=int, default=2, help="number of epochs of training")
    opt = parser.parse_args()
    # print(opt)
    l1 = torch.nn.L1Loss()
    mse = torch.nn.MSELoss()
    model = Net1()
    ## 使用cuda
    if opt.cuda:
        model.cuda()
        mse.cuda()
        l1.cuda()
    datapath = opt.dataroot

    class prepareData_train(Dataset):
        def __init__(self, train_or_test):
            self.files = os.listdir(datapath + train_or_test)
            self.train_or_test = train_or_test

        def __len__(self):
            return len(self.files)

        def __getitem__(self, idx):
            data = torch.load(datapath + self.train_or_test + '/' + self.files[idx])
            return data['k-space'], data['label']
    class prepareData_test(Dataset):
        def __init__(self, train_or_test):
            self.files = os.listdir(datapath + train_or_test)
            self.files.sort(key=lambda x: int(x[:-4]))
            self.train_or_test = train_or_test

        def __len__(self):
            return len(self.files)

        def __getitem__(self, idx):
            data = torch.load(datapath + self.train_or_test + '/' + self.files[idx])
            return data['k-space'], data['label']

    trainset = prepareData_train('train')
    testset = prepareData_test('test')
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=1, shuffle=True, num_workers=0)
    dataloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False, num_workers=0)
    #################################
    ##           test开始          ##
    #################################
    '''如果文件路径不存在, 则创建一个 (存放测试输出的图片)'''

    #################################
    ##           test开始          ##baji
    #################################
    filename = os.listdir(datapath + 'test/')
    length = len(filename)
    dout = torch.empty((length, Nc, opt.size, 2))
    dlab = torch.empty((length, Nc, opt.size, 2))
    dlabn = torch.empty((length, Nc, opt.size, 2))
    for epoch in range(opt.n_epochs-1, opt.n_epochs):
        ## 载入训练模型参数
        model.load_state_dict(torch.load(opt.model_data+"myNet_"+str(epoch+1)+".pth"))
        print("载入第"+str(epoch+1)+"周期模型")
        ## 设置为测试模式
        model.eval()
        for i, (batch,tbatch) in enumerate(zip(dataloader,trainloader),0):
            ## 输入数据
            real_A = batch[0].cuda()#input
            real_B = batch[1].cuda()[:, :, :, 0].unsqueeze(-1)#label
            rfnoise = tbatch[1].cuda()[:,:,:,0].unsqueeze(-1)#trainset label
            ## 输出预测的EMI
            fake_B = model(real_A).data
            loss = mse(fake_B, real_B)
            ## 保存图片
            print('processing (%04d)-th image--loss %04d' % (i, loss))
            output = fake_B.cuda().data
            labelo = real_B.cuda().data
            rfno = rfnoise.cuda().data
            dout[i] = output
            dlab[i] = labelo
            dlabn[i] = rfno
        nout = dlabn-dout##residual noise
        mean_nout = nout.mean(dim=(0,2), keepdim=True)
        nout = nout - mean_nout
    resinoise = tenshow(nout,row,col,Nz,Nx,Ny)
    labelpath=savepa[0]
    inputpath=savepa[1]
    label_files = natsorted(os.listdir(labelpath))
    total_images = len(label_files)  # 计算图像文件的数量
    num = 10
    center =  resinoise.shape[-1] // 2
    start_idx = center - num // 2
    end_idx = start_idx + num
    resnoi = resinoise[...,start_idx:end_idx]
    st=0
    ed=total_images//2
    for i in range(st,ed):
        label_path = os.path.join(labelpath, label_files[i])  # 获取图像的完整路径
        label = cv2.imread(label_path, -1)
        noise_index = i % num
        input = norm(label+ 2.5*resnoi[..., noise_index] )
        save_path = os.path.join(inputpath, f"image_{st + 1}.png")
        cv2.imwrite(save_path, input, [cv2.IMWRITE_PNG_COMPRESSION, 0])
        st += 1
    print("测试完成")
if __name__ == '__main__':
    savepa0='./datasets/secondary_denoising/T1/'
    savepa=[savepa0+'train/label/',savepa0+'train/input/',savepa0+'val/label/',savepa0+'val/input/',savepa0+'test/']
    for dir in savepa:
        if os.path.exists(dir):
            pass
        else:
            os.makedirs(dir)
    test()
