# from dataset_n2v import *
from dataset import *
from modelfile.TWGAN import *
from lossfuc import *
from utils import *
import torch
import torch.nn as nn
import sys
sys.path.append('D:\pythonspace\My_ultra_low_field\modelfile')
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt


def del_file(path):
    ls = os.listdir(path)
    for i in ls:
        c_path = os.path.join(path, i)
        if os.path.isdir(c_path):
            del_file(c_path)
        else:
            os.remove(c_path)
class TVLoss(nn.Module):
    def __init__(self,TVLoss_weight=1):
        super(TVLoss,self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self,x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self._tensor_size(x[:,:,1:,:])
        count_w = self._tensor_size(x[:,:,:,1:])
        h_tv = torch.abs(x[:,:,1:,:]-x[:,:,:h_x-1,:]).sum()
        w_tv = torch.abs(x[:,:,:,1:]-x[:,:,:,:w_x-1]).sum()
        return self.TVLoss_weight*2*(h_tv/count_h+w_tv/count_w)/batch_size

    def _tensor_size(self,t):
        return t.size()[1]*t.size()[2]*t.size()[3]
##
class Train:
    def __init__(self, args):
        self.mode = args.mode
        self.train_continue = args.train_continue

        self.scope = args.scope
        self.norm = args.norm
        self.size_window = args.size_window
        self.ratio = args.ratio

        self.dir_checkpoint = args.dir_checkpoint
        self.dir_log = args.dir_log

        self.name_data = args.name_data
        self.dir_data = args.dir_data
        self.dir_result = args.dir_result
        self.dir_result_test = args.dir_result_test

        self.st_epoch = args.st_epoch
        self.num_epoch = args.num_epoch
        self.test_epoch = args.test_epoch
        self.batch_size = args.batch_size

        self.lr_G = args.lr_G

        self.optim = args.optim
        self.beta1 = args.beta1

        self.ny_in = args.ny_in
        self.nx_in = args.nx_in
        self.nch_in = args.nch_in

        self.ny_out = args.ny_out
        self.nx_out = args.nx_out
        self.nch_out = args.nch_out

        self.nch_ker = args.nch_ker

        self.data_type = args.data_type

        self.num_freq_disp = args.num_freq_disp
        self.num_freq_save = args.num_freq_save

        self.gpu_ids = args.gpu_ids

        if self.gpu_ids and torch.cuda.is_available():
            self.device = torch.device("cuda:%d" % self.gpu_ids[0])
            torch.cuda.set_device(self.gpu_ids[0])
        else:
            self.device = torch.device("cpu")

    def save(self, dir_chck, net, optim,name, epoch):
        if not os.path.exists(dir_chck):
            os.makedirs(dir_chck)

        torch.save({f'net{name}': net.state_dict(),
                    f'optim{name}': optim.state_dict()},
                   f'%s/{name}_epoch%04d.pth' % (dir_chck, epoch))

    def load(self, dir_chck, net,name, optim=[], epoch=[], mode='train'):

        if not os.path.exists(dir_chck) or not os.listdir(dir_chck):
            epoch = 0
            if mode == 'train':
                return net, optim, epoch
            elif mode == 'test':
                return net, epoch

        if not epoch:
            ckpt = os.listdir(dir_chck)
            ckpt.sort()
            epoch = int(ckpt[-1].split('epoch')[1].split('.pth')[0])

        dict_net = torch.load('%s/%s_epoch%04d.pth' % (dir_chck,name, epoch))

        print('Loaded %dth network' % epoch)

        if mode == 'train':
            net.load_state_dict(dict_net[f'net{name}'])
            optim.load_state_dict(dict_net[f'optim{name}'])

            return net, optim, epoch

        elif mode == 'test':
            net.load_state_dict(dict_net[f'net{name}'])

            return net, epoch

    def train(self):
        mode = self.mode

        train_continue = self.train_continue
        num_epoch = self.num_epoch

        lr_G = self.lr_G

        batch_size = self.batch_size
        device = self.device

        gpu_ids = self.gpu_ids

        nch_in = self.nch_in
        nch_out = self.nch_out
        nch_ker = self.nch_ker

        size_data = (self.ny_in, self.nx_in, self.nch_in)
        size_window = self.size_window

        norm = self.norm
        name_data = self.name_data

        num_freq_disp = self.num_freq_disp
        num_freq_save = self.num_freq_save

        ## setup dataset

        dir_data_train = os.path.join(self.dir_data, name_data, 'train')
        dir_data_val = os.path.join(self.dir_data, name_data, 'val')
        if not os.path.exists(os.path.join(dir_data_train)):
            os.makedirs(os.path.join(dir_data_train))
        if not os.path.exists(os.path.join(dir_data_val)):
            os.makedirs(os.path.join(dir_data_val))

        dir_chck = os.path.join(self.dir_checkpoint, self.scope, name_data)
        if not os.path.exists(os.path.join(dir_chck)):
            os.makedirs(os.path.join(dir_chck))
        dir_log_train = os.path.join(self.dir_log, self.scope, name_data, 'train')
        dir_log_val = os.path.join(self.dir_log, self.scope, name_data, 'val')
        if not os.path.exists(os.path.join(dir_log_train)):
            os.makedirs(os.path.join(dir_log_train))
        if not os.path.exists(os.path.join(dir_log_val)):
            os.makedirs(os.path.join(dir_log_val))
        dir_result_train = os.path.join(self.dir_result, self.scope, name_data, 'train')
        dir_result_val = os.path.join(self.dir_result, self.scope, name_data, 'val')
        if not os.path.exists(os.path.join(dir_result_train,'images')):
            os.makedirs(os.path.join(dir_result_train,'images'))
        if not os.path.exists(os.path.join(dir_result_val,'images')):
            os.makedirs(os.path.join(dir_result_val,'images'))

        # transform_train = transforms.Compose([Normalize(mean=0.5, std=0.5), RandomFlip(), RandomCrop((self.ny_out, self.nx_out)), ToTensor()])
        # transform_val = transforms.Compose([Normalize(mean=0.5, std=0.5), RandomFlip(), RandomCrop((self.ny_out, self.nx_out)), ToTensor()])
        dataset_train = MyDataset(data_dir=dir_data_train)
        dataset_val = MyDataset(data_dir=dir_data_val)
        transform_inv = transforms.Compose([ToNumpy(), Denormalize(mean=0.5, std=0.5)])

        loader_train = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=8)
        loader_val = torch.utils.data.DataLoader(dataset_val, batch_size=batch_size, shuffle=True, num_workers=8)

        num_train = len(dataset_train)
        num_val = len(dataset_val)

        num_batch_train = int((num_train / batch_size) + ((num_train % batch_size) != 0))
        num_batch_val = int((num_val / batch_size) + ((num_val % batch_size) != 0))

        if nch_out == 1:
            cmap = 'gray'
        else:
            cmap = None

        ## setup network

        netG1 = Net1(1,1)##模型来源
        netG2 = Net1(1, 1)
        netG3 = Net2(2, 1)
        Dis = Discriminator(size=self.ny_in)
        # netG = UNet_ND()
        #netG = WtUnetPlusPlus(num_classes=1, deep_supervision=False).to(device)
        init_net(netG1, init_type='normal', init_gain=0.02, gpu_ids=gpu_ids)
        init_net(Dis, init_type='normal', init_gain=0.02, gpu_ids=gpu_ids)
        init_net(netG2, init_type='normal', init_gain=0.02, gpu_ids=gpu_ids)
        init_net(netG3, init_type='normal', init_gain=0.02, gpu_ids=gpu_ids)

        ## setup loss & optimization
        criterion = nn.MSELoss().to(device)
        vgg_loss = VGGLoss(device=device, n_layers=5)


        paramsG1 = netG1.parameters()
        paramsDis = Dis.parameters()
        paramsG2 = netG2.parameters()
        paramsG3 = netG3.parameters()
        # optimG1 = torch.optim.RMSprop(paramsG1, lr=5e-5)
        # optimDis = torch.optim.RMSprop(paramsDis, lr=5e-5)
        optimG1 = torch.optim.Adam(paramsG1, lr=1e-4)
        optimDis = torch.optim.Adam(paramsDis, lr=2e-4)
        optimG2 = torch.optim.Adam(paramsG2, lr=lr_G)
        optimG3 = torch.optim.Adam(paramsG3, lr=lr_G)
        ## load from checkpoints
        st_epoch = self.st_epoch

        if train_continue == 'on':
            netG1, optimG1, st_epoch = self.load(dir_chck, netG1,'G1', optimG1, epoch=self.st_epoch, mode=mode)
            netD, optimDis, st_epoch = self.load(dir_chck, Dis, 'Dis',optimDis, epoch=self.st_epoch, mode=mode)
            netG2, optimG2, st_epoch = self.load(dir_chck, netG2, 'G2',optimG2, epoch=self.st_epoch,mode=mode)
            netG3, optimG3, st_epoch = self.load(dir_chck, netG3, 'G3',optimG3, epoch=self.st_epoch,mode=mode)


        ## setup tensorboard
        writer_train = SummaryWriter(log_dir=dir_log_train)
        writer_val = SummaryWriter(log_dir=dir_log_val)
        losstrain_G1 = []
        losstrain_Dis = []
        losstrain_G2 = []
        losstrain_G3 = []
        lossval_G1 = []
        lossval_Dis = []
        lossval_G2 = []
        lossval_G3 = []
        for epoch in range(st_epoch + 1, num_epoch + 1):
            ## training phase
            netG1.train()
            Dis.train()
            netG2.train()
            netG3.train()

            loss_G1_train = []
            loss_Dis_train = []
            loss_G2_train = []
            loss_G3_train = []


            for batch, data in enumerate(loader_train, 1):
                def should(freq):
                    return freq > 0 and (batch % freq == 0 or batch == num_batch_train)

                label = data['label'].to(device)
                input = data['input'].to(device)

                # forward net and get losses
                # for p in Dis.parameters():  # 遍历判别网络D的模型参数
                #      p.data.clamp_(-0.1, 0.1)#0.01

                optimDis.zero_grad()
                out1d = netG1(input).detach()
                gp = cal_gp(Dis, label, out1d, device)
                loss_Dis = -torch.mean(Dis(label))+torch.mean(Dis(out1d))+ 0.5*gp
                # loss_Dis = torch.mean(Dis(label * (1 - mask))-torch.mean(Dis(out1d* (1 - mask)))) \
                #            + torch.mean(Dis(out1d* (1 - mask))-torch.mean(Dis(label* (1 - mask))))
                loss_Dis.backward()
                optimDis.step()

                loss_Dis_train += [loss_Dis.item()]
                writer_train.add_scalar('loss_D_batch', loss_Dis.item(), epoch * len(loader_train) + batch)

                optimG1.zero_grad()
                out1 = netG1(input)
                G1loss = -torch.mean(Dis(out1))
                # G1loss = torch.mean(torch.mean(Dis(out1 * (1 - mask)))-Dis(label * (1 - mask)))+\
                #          torch.mean(torch.mean(Dis(label* (1 - mask)))-Dis(out1* (1 - mask)))
                loss_G1 = G1loss+0.6*vgg_loss(out1 ,label)
                loss_G1.backward()
                optimG1.step()
                # get losses
                loss_G1_train += [loss_G1.item()]
                writer_train.add_scalar('loss_G1_batch', loss_G1.item(), epoch * len(loader_train) + batch)

                optimG2.zero_grad()
                out2 = netG2(input)
                loss_G2 = 0.1*criterion(out2, label)+vgg_loss(out2, label)
                loss_G2.backward()
                optimG2.step()

                loss_G2_train += [loss_G2.item()]
                writer_train.add_scalar('loss_G2_batch', loss_G2.item(), epoch * len(loader_train) + batch)

                optimG3.zero_grad()
                output = netG3(out1.detach(), out2.detach())
                G3loss = 0.1*criterion(output, label)+vgg_loss(output, label)
                loss_G3 = G3loss
                loss_G3.backward()
                optimG3.step()

                loss_G3_train += [loss_G3.item()]
                writer_train.add_scalar('loss_G3_batch', loss_G3.item(), epoch * len(loader_train) + batch)

                print('TRAIN: EPOCH %d: BATCH %04d/%04d: LOSSG1: %.4f LOSSD: %.4f LOSSG2: %.4f LOSSG3: %.4f'
                      % (epoch, batch, num_batch_train, np.mean(loss_G1_train),np.mean(loss_Dis_train),np.mean(loss_G2_train),np.mean(loss_G3_train)))

                if should(num_freq_disp):
                    ## show output
                    input = transform_inv(input)
                    label = transform_inv(label)
                    output = transform_inv(output)
                    out1 = transform_inv(out1)
                    out2 = transform_inv(out2)

                    input = np.clip(input, 0, 1)
                    label = np.clip(label, 0, 1)
                    output = np.clip(output, 0, 1)
                    dif = np.clip(abs(label - input), 0, 1)

                    writer_train.add_images('input', input, num_batch_train * (epoch - 1) + batch, dataformats='NCHW')
                    writer_train.add_images('output', output, num_batch_train * (epoch - 1) + batch, dataformats='NCHW')
                    writer_train.add_images('label', label, num_batch_train * (epoch - 1) + batch, dataformats='NCHW')
                    writer_train.add_images('out1', out1, num_batch_train * (epoch - 1) + batch, dataformats='NCHW')
                    writer_train.add_images('out2', out2, num_batch_train * (epoch - 1) + batch, dataformats='NCHW')
                    for j in range(label.shape[0]):
                        # name = num_train * (epoch - 1) + num_batch_train * (batch - 1) + j
                        name = num_batch_train * (batch - 1) + j
                        fileset = {'name': name,
                                   'input': "%03d-input.png" % name,
                                   'output': "%03d-output.png" % name,
                                   'label': "%03d-label.png" % name,
                                   'dif': "%03d-dif.png" % name}

                        plt.imsave(os.path.join(dir_result_train, 'images', fileset['input']), np.transpose(input, (0, 2, 3, 1))[j, :, :, :].squeeze(), cmap=cmap)
                        plt.imsave(os.path.join(dir_result_train, 'images', fileset['output']), np.transpose(output, (0, 2, 3, 1))[j, :, :, :].squeeze(), cmap=cmap)
                        plt.imsave(os.path.join(dir_result_train, 'images', fileset['label']), np.transpose(label, (0, 2, 3, 1))[j, :, :, :].squeeze(), cmap=cmap)
                        plt.imsave(os.path.join(dir_result_train, 'images', fileset['dif']), np.transpose(dif, (0, 2, 3, 1))[j, :, :, :].squeeze(), cmap=cmap)

                        append_index(dir_result_train, fileset)

            writer_train.add_scalar('loss_G1', np.mean(loss_G1_train), epoch)
            writer_train.add_scalar('loss_D', np.mean(loss_Dis_train), epoch)
            writer_train.add_scalar('loss_G2', np.mean(loss_G2_train), epoch)
            writer_train.add_scalar('loss_G3', np.mean(loss_G3_train), epoch)
            losstrain_Dis += [np.mean(loss_Dis_train)]
            losstrain_G1 += [np.mean(loss_G1_train)]
            losstrain_G2 += [np.mean(loss_G2_train)]
            losstrain_G3 += [np.mean(loss_G3_train)]

            ## validation phase
            with torch.no_grad():
                netG1.eval()
                Dis.eval()
                netG2.eval()
                netG3.eval()

                loss_G1_val = []
                loss_Dis_val= []
                loss_G2_val = []
                loss_G3_val = []

                for batch, data in enumerate(loader_val, 1):
                    def should(freq):
                        return freq > 0 and (batch % freq == 0 or batch == num_batch_val)

                    # input = data['input'].to(device)
                    input = data['input'].to(device)
                    label = data['label'].to(device)

                    # forward net and get losses
                    out1d = netG1(input).detach()
                    loss_Dis = -torch.mean(Dis(label)) + torch.mean(Dis(out1d )) + 0.5 * gp
                    loss_Dis_val += [loss_Dis.item()]
                    writer_val.add_scalar('loss_D_batch', loss_Dis.item(), epoch * len(loader_train) + batch)

                    out1 = netG1(input)
                    G1loss = -torch.mean(Dis(out1))
                    loss_G1 = G1loss+0.6*vgg_loss(out1, label)

                    loss_G1_val += [loss_G1.item()]
                    writer_val.add_scalar('loss_G1_batch', loss_G1.item(), epoch * len(loader_train) + batch)

                    out2 = netG2(input)
                    loss_G2 = criterion(out2, label)+vgg_loss(out2, label)
                    loss_G2_val += [loss_G2.item()]
                    writer_val.add_scalar('loss_G2_batch', loss_G2.item(), epoch * len(loader_train) + batch)

                    output = netG3(out1.detach(), out2.detach())
                    G3loss = criterion(output, label)+vgg_loss(output, label)
                    loss_G3 = G3loss

                    loss_G3_val += [loss_G3.item()]
                    writer_val.add_scalar('loss_G3_batch', loss_G3.item(), epoch * len(loader_train) + batch)


                    print('VALID: EPOCH %d: BATCH %04d/%04d: LOSSG1: %.4f LOSSD: %.4f LOSSG2: %.4f LOSSG3: %.4f'
                          % (epoch, batch, num_batch_val, np.mean(loss_G1_val),np.mean(loss_Dis_val),np.mean(loss_G2_val),np.mean(loss_G3_val)))

                    if should(num_freq_disp):
                        ## show output
                        input = transform_inv(input)
                        label = transform_inv(label)
                        output = transform_inv(output)

                        input = np.clip(input, 0, 1)
                        label = np.clip(label, 0, 1)
                        output = np.clip(output, 0, 1)
                        dif = np.clip(abs(label - input), 0, 1)

                        writer_val.add_images('input', input, num_batch_val * (epoch - 1) + batch, dataformats='NCHW')
                        writer_val.add_images('output', output, num_batch_val * (epoch - 1) + batch, dataformats='NCHW')
                        writer_val.add_images('label', label, num_batch_val * (epoch - 1) + batch, dataformats='NCHW')

                        for j in range(label.shape[0]):
                            # name = num_train * (epoch - 1) + num_batch_train * (batch - 1) + j
                            name = num_batch_train * (batch - 1) + j
                            fileset = {'name': name,
                                       'input': "%03d-input.png" % name,
                                       'output': "%03d-output.png" % name,
                                       'label': "%03d-label.png" % name,
                                       'dif': "%03d-dif.png" % name, }

                            plt.imsave(os.path.join(dir_result_val, 'images', fileset['input']), input[j, :, :, :].squeeze(), cmap=cmap)
                            plt.imsave(os.path.join(dir_result_val, 'images', fileset['output']), output[j, :, :, :].squeeze(), cmap=cmap)
                            plt.imsave(os.path.join(dir_result_val, 'images', fileset['label']), label[j, :, :, :].squeeze(), cmap=cmap)
                            plt.imsave(os.path.join(dir_result_val, 'images', fileset['dif']), dif[j, :, :, :].squeeze(), cmap=cmap)

                            append_index(dir_result_val, fileset)

                writer_val.add_scalar('loss_G1', np.mean(loss_G1_val), epoch)
                writer_train.add_scalar('val_loss_G1', np.mean(loss_G1_val), epoch)
                writer_train.add_scalar('val_loss_D', np.mean(loss_Dis_val), epoch)
                writer_train.add_scalar('val_loss_G2', np.mean(loss_G2_val), epoch)
                writer_train.add_scalar('val_loss_G3', np.mean(loss_G3_val), epoch)
                lossval_Dis += [np.mean(loss_G1_val)]
                lossval_G1 += [np.mean(loss_G1_val)]
                lossval_G2 += [np.mean(loss_G2_val)]
                lossval_G3 += [np.mean(loss_G3_val)]
            # update schduler
            # schedG.step()
            # schedD.step()

            ## save
            if (epoch % num_freq_save) == 0:
                self.save(dir_chck, netG1, optimG1,'G1',epoch)
                self.save(dir_chck, Dis, optimDis,'Dis',epoch)
                self.save(dir_chck, netG2, optimG2,'G2',epoch)
                self.save(dir_chck, netG3, optimG3,'G3',epoch)
        plt.subplot(4, 1, 1)
        plt.plot(losstrain_G1, label='Train_LossG1')
        plt.plot(lossval_G1, label='Val_LossG1')
        plt.xlabel("epoch")
        plt.ylabel("Loss")
        plt.title("Loss for G1")
        plt.legend()

        plt.subplot(4, 1, 2)
        plt.plot(losstrain_Dis, label='Train_LossDis')
        plt.plot(lossval_Dis, label='Val_LossDis')
        plt.xlabel("epoch")
        plt.ylabel("Loss")
        plt.title("Loss for Dis")
        plt.legend()
        # 第三张图
        plt.subplot(4, 1, 3)
        plt.plot(losstrain_G2, label='Train_LossG2')
        plt.plot(lossval_G2, label='Val_LossG2')
        plt.xlabel("epoch")
        plt.ylabel("Loss")
        plt.title("Loss for G2")
        plt.legend()

        # 第三张图
        plt.subplot(4, 1, 4)
        plt.plot(losstrain_G3, label='Train_LossG3')
        plt.plot(lossval_G3, label='Val_LossG3')
        plt.xlabel("epoch")
        plt.ylabel("Loss")
        plt.title("Loss for G3")
        plt.legend()

        plt.tight_layout()  # 调整子图布局，防止重叠
        plt.savefig(os.path.join(dir_result_train, 'trainloss.png'))
        plt.show()
        writer_train.close()
        writer_val.close()

    def test(self):
        mode = self.mode

        batch_size = 1
        device = self.device
        gpu_ids = self.gpu_ids

        nch_in = self.nch_in
        nch_out = self.nch_out
        nch_ker = self.nch_ker

        size_data = (self.ny_in, self.nx_in, self.nch_in)
        size_window = self.size_window


        norm = self.norm

        name_data = self.name_data

        if nch_out == 1:
            cmap = 'gray'
        else:
            cmap = None

        ## setup dataset
        dir_chck = os.path.join(self.dir_checkpoint, self.scope, name_data)

        dir_result_test = os.path.join(self.dir_result, self.scope, name_data, self.dir_result_test)
        if not os.path.exists(os.path.join(dir_result_test, 'images')):
            os.makedirs(os.path.join(dir_result_test, 'images'))

        dir_data_test = os.path.join(self.dir_data, name_data, self.dir_result_test)

        transform_test = transforms.Compose([Normalize(mean=0.5, std=0.5), ToTensor()])
        transform_inv = transforms.Compose([ToNumpy(), Denormalize(mean=0.5, std=0.5)])
        transform_ts2np = ToNumpy()

        # dataset_test = Dataset(dir_data_test, data_type=self.data_type, transform=transform_test, sgm=(0, 25))
        dataset_test = MyTestset(data_dir=dir_data_test)
        loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=1, shuffle=False, num_workers=8)

        num_test = len(dataset_test)

        num_batch_test = int((num_test / batch_size) + ((num_test % batch_size) != 0))

        ## setup network

        netG1 = Net1(1,1)
        netG2 = Net1(1, 1)
        netG3 = Net2(2, 1)
        # Dis = Discriminator(size=self.ny_in)

        init_net(netG1, init_type='normal', init_gain=0.02, gpu_ids=gpu_ids)
        # init_net(Dis, init_type='normal', init_gain=0.02, gpu_ids=gpu_ids)
        init_net(netG2, init_type='normal', init_gain=0.02, gpu_ids=gpu_ids)
        init_net(netG3, init_type='normal', init_gain=0.02, gpu_ids=gpu_ids)

        ## setup loss & optimization
        # fn_REG = nn.L1Loss().to(device)  # L1
        criterion = nn.MSELoss().to(device)
        vgg_loss = VGGLoss(device=device, n_layers=5)
        ## load from checkpoints
        st_epoch = 0

        netG1, st_epoch = self.load(dir_chck, netG1,'G1', epoch=self.test_epoch,mode=mode)
        # Dis, st_epoch = self.load(dir_chck, Dis,'Dis', epoch=self.test_epoch,mode=mode)
        netG2, st_epoch = self.load(dir_chck, netG2,'G2', epoch=self.test_epoch,mode=mode)
        netG3, st_epoch = self.load(dir_chck, netG3,'G3', epoch=self.test_epoch,mode=mode)
        ## test phase
        with torch.no_grad():
            netG1.eval()
            # Dis.eval()
            netG2.eval()
            netG3.eval()
            inputimg=np.zeros((self.nx_in,self.ny_in,len(loader_test)))
            outimg=np.zeros((self.nx_in,self.ny_in,len(loader_test)))
            for i, data in enumerate(loader_test, 1):
                # input = data['input'].to(device)
                input = data['input'].to(device)

                out1 = netG1(input)
                out2 = netG2(input)
                output = netG3(out1.detach(), out2.detach())
                input = transform_inv(input)
                output = transform_inv(output)
                out1 = transform_inv(out1)

                # input = np.clip(input, 0, 1)
                # output = np.clip(output, 0, 1)
                inputimg[:, :, i-1] = input[0, :, :, :].squeeze()
                outimg[:, :, i-1] = output[0, :, :, :].squeeze()
            for j in range(outimg.shape[-1]):
                plt.imsave(os.path.join(dir_result_test, 'images', f'image_{j+1}.png'), outimg[:, :, j], cmap=cmap)
            fig = figshow2(inputimg,outimg)
            fig.savefig(os.path.join(dir_result_test, 'images/denoise.tif'))
            # plt.show()
            print('test finished!')

def set_requires_grad(nets, requires_grad=False):
    """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
    Parameters:
        nets (network list)   -- a list of networks
        requires_grad (bool)  -- whether the networks require gradients or not
    """
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad


def get_scheduler(optimizer, opt):
    """Return a learning rate scheduler

    Parameters:
        optimizer          -- the optimizer of the network
        opt (option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions．　
                              opt.lr_policy is the name of learning rate policy: linear | step | plateau | cosine

    For 'linear', we keep the same learning rate for the first <opt.n_epochs> epochs
    and linearly decay the rate to zero over the next <opt.n_epochs_decay> epochs.
    For other schedulers (step, plateau, and cosine), we use the default PyTorch schedulers.
    See https://pytorch.org/docs/stable/optim.html for more details.
    """
    if opt.lr_policy == 'linear':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.n_epochs) / float(opt.n_epochs_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif opt.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.n_epochs, eta_min=0)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


def append_index(dir_result, fileset, step=False):
    index_path = os.path.join(dir_result, "index.html")
    if os.path.exists(index_path):
        index = open(index_path, "a")
    else:
        index = open(index_path, "w")
        index.write("<html><body><table><tr>")
        if step:
            index.write("<th>step</th>")
        for key, value in fileset.items():
            index.write("<th>%s</th>" % key)
        index.write('</tr>')

    # for fileset in filesets:
    index.write("<tr>")

    if step:
        index.write("<td>%d</td>" % fileset["step"])
    index.write("<td>%s</td>" % fileset["name"])

    del fileset['name']

    for key, value in fileset.items():
        index.write("<td><img src='images/%s'></td>" % value)

    index.write("</tr>")
    return index_path


def add_plot(output, label, writer, epoch=[], ylabel='Density', xlabel='Radius', namescope=[]):
    fig, ax = plt.subplots()

    ax.plot(output.transpose(1, 0).detach().numpy(), '-')
    ax.plot(label.transpose(1, 0).detach().numpy(), '--')

    ax.set_xlim(0, 400)

    ax.grid(True)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)

    writer.add_figure(namescope, fig, epoch)
