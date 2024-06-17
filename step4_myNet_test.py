import argparse
from utils import *
from dataset import *
from modelfile.modelres import Net1
from torch.utils.data import DataLoader
Nx=128
Ny=128
Nc=2
row=4
col=5
Nz=row*col
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
    parser.add_argument('--model_data', type=str, default='./checkpoints/firdenoise/1ch/', help='model checkpoint file')
    parser.add_argument("--n_epochs", type=int, default=7, help="number of epochs of training")
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
    dout = torch.empty((length, Nc, 128, 2))
    dlab = torch.empty((length, Nc, 128, 2))
    for epoch in range(opt.n_epochs-1, opt.n_epochs):
        ## 载入训练模型参数
        model.load_state_dict(torch.load(opt.model_data+"myNet_"+str(epoch+1)+".pth"))
        print("载入第"+str(epoch+1)+"周期模型")
        ## 设置为测试模式
        model.eval()
        inputimg = np.zeros((128, 128, row*col))
        outimg = np.zeros((128, 128, row*col))
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

        out = dlab - dout
        _ = tenshow(dout,row,col,Nx,Ny)
        figin = tenshow(dlab,row,col,Nx,Ny)
        fig = tenshow(out,row,col,Nx,Ny)
        plt.close("all")
        lenf=fig.shape[-1]
        id=0
        for i in range(lenf):
            if epoch == opt.n_epochs - 1 :
                # flipped_fig = np.flipud(fig[..., i])
                # flipped_figin = np.flipud(figin[..., i])
                inputimg[:, :, id] = figin[..., i]
                outimg[:, :, id] = fig[..., i]
                # cv2.imwrite(f"./results/first_denoising/results/image_{id + 13}.png", fig[..., i],[cv2.IMWRITE_PNG_COMPRESSION, 0])
                # cv2.imwrite(savepa0+pname+f"image_{id + 11}.png", fig[..., i],[cv2.IMWRITE_PNG_COMPRESSION, 0])
                id += 1
        figcontr = figshow(inputimg, outimg)
        # figcontr.savefig('./results/first_denoising/results/denoise'+f'{opt.outchannel}'+'ch.png')
        plt.show()
        print("测试完成")
if __name__ == '__main__':
    savepa0='./datasets/T1phatom/'
    pname='test/'
    savepa=[savepa0+'train/',savepa0+'val/',savepa0+pname]
    flag=0
    for dir in savepa:
        if os.path.exists(dir):
            pass
        else:
            os.makedirs(dir)
    for dir in savepa:
        if os.path.exists(dir) and flag:
            del_file(dir)
    test()
