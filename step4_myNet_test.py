import argparse
from utils import *
from dataset import *
from modelfile.modelres import Net1
from torch.utils.data import DataLoader
from seting import Nx,Ny,Nc,Nz,row,col
import torch

def add_noise(signal):
    signal_max = torch.max(signal)
    normalized_signal = signal / signal_max
    noise = torch.randn_like(signal)*0.02
    noisy_normalized = normalized_signal+noise
    # figin = tenshow(noisy_normalized, row, col, Nx, Ny)
    # plt.show()
    noisy_signal = noisy_normalized *signal_max
    return noisy_signal


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
    parser.add_argument('--batchSize', type=int, default=2, help='size of the batches')
    parser.add_argument('--dataroot', type=str, default='./datasets/firdenoise/', help='root directory of the dataset')
    parser.add_argument('--channels', type=int, default=2, help='number of channels of input data')
    parser.add_argument('--n_residual_blocks', type=int, default=9, help='number of channels of output data')
    parser.add_argument('--size', type=int, default=128, help='size of the data (squared assumed)')
    parser.add_argument('--cuda', type=bool, default=True, help='use GPU computation')
    parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')
    parser.add_argument('--outchannel', type=int, default=1, help='number of channels of out data')
    parser.add_argument('--num_display', type=int, default=20, help='number of channels of out data')
    parser.add_argument('--model_data', type=str, default='./checkpoints/firdenoise/', help='model checkpoint file')
    parser.add_argument("--n_epochs", type=int, default=6, help="number of epochs of training")
    opt = parser.parse_args()
    # print(opt)
    l1 = torch.nn.L1Loss()
    mse = torch.nn.MSELoss()
    #################################
    ##          test准备工作        ##
    #################################

    ## input_shape:(3, 256, 256)
    input_shape = (opt.channels, opt.size, 2)
    ## 创建生成器，判别器对象
    # model = RIDNET(feats=64, reduction=16, rgb_range=255)
    # model = Inception_resnet_CNN()
    model = Net1()
    ## 使用cuda
    if opt.cuda:
        model.cuda()
        mse.cuda()
        l1.cuda()


    ## 创建一个tensor数组
    Tensor = torch.cuda.FloatTensor if opt.cuda else torch.Tensor

    datapath = opt.dataroot

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

    testset = prepareData_test('test')
    dataloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False, num_workers=0)
    #################################
    ##           test开始          ##baji
    #################################
    savepath = './results/first_denoising/results/'
    os.makedirs(savepath, exist_ok=True)
    filename = os.listdir(datapath + 'test/')
    length = len(filename)
    dout = torch.empty((length, 2, 128, Nc))
    dlab = torch.empty((length, 2, 128, Nc))
    for epoch in range(opt.n_epochs-1, opt.n_epochs):
        ## 载入训练模型参数
        model.load_state_dict(torch.load(opt.model_data+"myNet_"+str(epoch+1)+".pth"))
        print("载入第"+str(epoch+1)+"周期模型")
        ## 设置为测试模式
        model.eval()
        imgshow=np.zeros((Nx,Ny, row*col,2))
        for i, batch in enumerate(dataloader, 0):
            ## 输入数据 real
            real_A = batch[0].cuda()
            real_B = batch[1].cuda()[:, :, :, opt.outchannel-1].unsqueeze(-1)
            fake_B = model(real_A).data
            loss = mse(fake_B, real_B)
            ## 保存图片
            print('processing (%04d)-th image--loss %04d' % (i, loss))
            output = fake_B.cuda().data
            labelo = real_B.cuda().data
            dout[i] = output
            dlab[i] = labelo
        ##去均值
        mean_dout = dout.mean(dim=(0,2), keepdim=True)
        dout = dout - mean_dout
        out = dlab - dout
        figin = tenshow(dlab,row,col,Nz,Nx,Ny)
        # nfigin= tenshow(ndlab,row,col,Nx,Ny)
        fig = tenshow(out,row,col,Nz,Nx,Ny)
        # plt.show()
        plt.close("all")
        lenf=fig.shape[-1]
        id=0
        for i in range(lenf):
            if epoch == opt.n_epochs - 1 :
                # flipped_fig = np.flipud(fig[..., i])
                # flipped_figin = np.flipud(figin[..., i])
                imgshow[:, :, id,0] = figin[..., i]
                # imgshow[:, :, id,1] = nfigin[..., i]
                imgshow[:, :, id,1] = fig[..., i]
                id += 1
        num=opt.num_display
        center = figin.shape[-1] // 2
        start_idx = center - num//2
        end_idx = center + num//2
        indices = list(range(start_idx, end_idx))
        id = 0
        for i in indices:
                cv2.imwrite(savepa[0]+f"image_{id}.png", imgshow[:, :, i,0],
                            [cv2.IMWRITE_PNG_COMPRESSION, 0])
                cv2.imwrite(savepa[1]+f"image_{id}.png",imgshow[:, :, i,1],
                            [cv2.IMWRITE_PNG_COMPRESSION, 0])
                cv2.imwrite(savepa[2]+f"image_{id}.png",imgshow[:, :, i,0],
                            [cv2.IMWRITE_PNG_COMPRESSION, 0])
                cv2.imwrite(savepa[3]+f"image_{id}.png", imgshow[:, :, i,1],
                            [cv2.IMWRITE_PNG_COMPRESSION, 0])
                id += 1
        figcontr = figshow(imgshow,num)
        figcontr.savefig(savepa[3]+'denoise.png')
        plt.show()
        print("测试完成")
if __name__ == '__main__':
    savepa1='./results/first_denoising/'
    savepa2='./datasets/secondary_denoising/N2V/'
    savepa=[savepa2+'initial/',savepa2+'test/',savepa1+'initial/',savepa1+'results/']
    flag=1
    for dir in savepa:
        if os.path.exists(dir):
            pass
        else:
            os.makedirs(dir)
    for dir in savepa:
        if os.path.exists(dir) and flag:
            del_file(dir)
    test()
