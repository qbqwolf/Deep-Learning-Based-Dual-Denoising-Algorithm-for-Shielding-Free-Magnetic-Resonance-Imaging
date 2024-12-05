import random
from torch.autograd import Variable
import numpy as np
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import os
import logging
import torch
from tools.NRSS import NRSS
from matplotlib.patches import Rectangle
import io
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
# Nx=128
# Ny=128
# Nz=16
''''
class Logger:
class Parser:
'''
class Parser:
    def __init__(self, parser):
        self.__parser = parser
        self.__args = parser.parse_args()

        # set gpu ids
        str_ids = self.__args.gpu_ids.split(',')
        self.__args.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                self.__args.gpu_ids.append(id)
        # if len(self.__args.gpu_ids) > 0:
        #     torch.cuda.set_device(self.__args.gpu_ids[0])

    def get_parser(self):
        return self.__parser

    def get_arguments(self):
        return self.__args

    def write_args(self):
        params_dict = vars(self.__args)

        log_dir = os.path.join(params_dict['dir_log'], params_dict['scope'], params_dict['name_data'])
        args_name = os.path.join(log_dir, 'args.txt')

        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        with open(args_name, 'wt') as args_fid:
            args_fid.write('----' * 10 + '\n')
            args_fid.write('{0:^40}'.format('PARAMETER TABLES') + '\n')
            args_fid.write('----' * 10 + '\n')
            for k, v in sorted(params_dict.items()):
                args_fid.write('{}'.format(str(k)) + ' : ' + ('{0:>%d}' % (35 - len(str(k)))).format(str(v)) + '\n')
            args_fid.write('----' * 10 + '\n')

    def print_args(self, name='PARAMETER TABLES'):
        params_dict = vars(self.__args)

        print('----' * 10)
        print('{0:^40}'.format(name))
        print('----' * 10)
        for k, v in sorted(params_dict.items()):
            if '__' not in str(k):
                print('{}'.format(str(k)) + ' : ' + ('{0:>%d}' % (35 - len(str(k)))).format(str(v)))
        print('----' * 10)


class Logger:
    def __init__(self, info=logging.INFO, name=__name__):
        logger = logging.getLogger(name)
        logger.setLevel(info)

        self.__logger = logger

    def get_logger(self, handler_type='stream_handler'):
        if handler_type == 'stream_handler':
            handler = logging.StreamHandler()
            log_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(log_format)
        else:
            handler = logging.FileHandler('utils.log')

        self.__logger.addHandler(handler)

        return self.__logger

def del_file(path):
    ls = os.listdir(path)
    for i in ls:
        c_path = os.path.join(path, i)
        if os.path.isdir(c_path):
            del_file(c_path)
        else:
            os.remove(c_path)

## 先前生成的样本的缓冲区
class ReplayBuffer:
    def __init__(self, max_size=50):
        assert max_size > 0, "Empty buffer or trying to create a black hole. Be careful."
        self.max_size = max_size
        self.data = []

    def push_and_pop(self, data):                       ## 放入一张图像，再从buffer里取一张出来
        to_return = []                                  ## 确保数据的随机性，判断真假图片的鉴别器识别率
        for element in data.data:
            element = torch.unsqueeze(element, 0)
            if len(self.data) < self.max_size:          ## 最多放入50张，没满就一直添加
                self.data.append(element)
                to_return.append(element)
            else:
                if random.uniform(0, 1) > 0.5:          ## 满了就1/2的概率从buffer里取，或者就用当前的输入图片
                    i = random.randint(0, self.max_size - 1)
                    to_return.append(self.data[i].clone())
                    self.data[i] = element
                else:
                    to_return.append(element)
        return Variable(torch.cat(to_return))


## 设置学习率为初始学习率乘以给定lr_lambda函数的值
class LambdaLR:
    def __init__(self, n_epochs, offset, decay_start_epoch):                                                ## (n_epochs = 50, offset = epoch, decay_start_epoch = 30)
        assert (n_epochs - decay_start_epoch) > 0, "Decay must start before the training session ends!"     ## 断言，要让n_epochs > decay_start_epoch 才可以
        self.n_epochs = n_epochs
        self.offset = offset
        self.decay_start_epoch = decay_start_epoch

    def step(self, epoch):                                              ## return    1-max(0, epoch - 30) / (50 - 30)
        return 1.0 - max(0, epoch + self.offset - self.decay_start_epoch) / (self.n_epochs - self.decay_start_epoch)

def normalize_tensor(x, eps=1e-8):
    """
    Normalize the input tensor x to range [-1, 1] along the last dimension
    """
    # Clone the input tensor to avoid changing its values
    x_normalized = x.clone()
    # Compute the norm of x along the last dimension
    norm = torch.norm(x_normalized, p=2, dim=3, keepdim=True)
    # Add a small value to the norm to avoid division by zero
    norm = norm.clamp(min=eps)
    # Divide x by the norm to normalize it
    x_normalized /= norm
    # Compute the maximum and minimum values of x along the last dimension
    max_values, _ = torch.max(x_normalized, dim=-1, keepdim=True)
    min_values, _ = torch.min(x_normalized, dim=-1, keepdim=True)
    # Compute the range of x along the last dimension
    range_values = max_values - min_values
    # Add a small value to the range to avoid division by zero
    range_values = range_values.clamp(min=eps)
    # Scale and shift the normalized tensor to the range [-1, 1]
    x_normalized_scaled = (x_normalized - min_values) / range_values
    x_normalized_scaled = x_normalized_scaled * 2.0 - 1.0
    return x_normalized_scaled

def standardization(tensor):
    mu = torch.mean(tensor.clone(), dim=2, keepdim=True)
    sigma = torch.std(tensor.clone(), dim=2, keepdim=True)
    out = (tensor - mu) / sigma
    return out,mu,sigma
def inv_standardization(tensor, mu, sigma):
    out = tensor * sigma + mu
    return out
def normalization(data):
    data = torch.nan_to_num(data)
    _range = torch.max(torch.abs(data), dim=2, keepdim=True)[0]
    out=data / _range
    return out

def read_mrd_EMI_hym(filename):

    fid = open(filename, 'rb')

    header = {}
    header['Nx'] = np.fromfile(fid, dtype=np.int32, count=1)[0]
    header['Ny'] = np.fromfile(fid, dtype=np.int32, count=1)[0]
    header['Nz'] = np.fromfile(fid, dtype=np.int32, count=1)[0]
    header['Ns'] = np.fromfile(fid, dtype=np.int32, count=1)[0]

    fid.seek(18, 0)
    header['DataTypeCode'] = np.fromfile(fid, dtype=np.int16, count=1)[0]
    fid.seek(152, 0)
    header['Nechoes'] = np.fromfile(fid, dtype=np.int16, count=1)[0]
    fid.seek(156, 0)
    header['Nex'] = np.fromfile(fid, dtype=np.uint8, count=1)[0]

    fid.seek(256, 0)
    text = fid.read(256).decode('utf-8').rstrip()
    header['text'] = text
    fid.seek(512, 0)

    raw_data = np.zeros(( header['Nx'],header['Ny'],header['Nz'], header['Ns']), dtype=np.complex64)
    raw_data_EMI = np.zeros((header['Nx'], header['Ny'],header['Nz'], header['Ns']), dtype=np.complex64)

    for i in range(header['Ns']):
        for j in range(header['Ny']):
            raw_data_tmp = np.fromfile(fid, dtype=np.float32, count=2 * header['Nx'] * header['Nz'])
            raw_data[:, j, :, i] = np.vectorize(complex)(raw_data_tmp[::2], raw_data_tmp[1::2]).reshape(
                (header['Nx'], header['Nz']),order="F").astype(np.complex64)

    for i in range(header['Ns']):
        for j in range(header['Ny']):
            raw_data_tmp = np.fromfile(fid, dtype=np.float32, count=2 * header['Nx'] * header['Nz'])
            raw_data_EMI[:,j, :, i] = np.vectorize(complex)(raw_data_tmp[::2], raw_data_tmp[1::2]).reshape(
                (header['Nx'], header['Nz']),order="F").astype(np.complex64)
    if header['Ns']==1:
        raw_data=raw_data.squeeze(-1)
        raw_data_EMI=raw_data_EMI.squeeze(-1)
    header['file'] = fid.readlines()
    index = np.where(np.array([b'PPL' in line for line in header['file']]))
    str = header['file'][index[0][0]].decode().rstrip()
    header['pplpath'] = str.split()[1]

    indices = np.where([b'views_per_seg' in line for line in header['file']])
    str = header['file'][indices[0][0]].decode().rstrip()
    header['ETL'] = int(str.split()[2])

    fid.close()

    return header, text, raw_data, raw_data_EMI


def read_mrd_fse3d_2echoes(filename):
    header = {}
    text = None
    raw_data_out = None
    emi_data_out = None

    fid = open(filename, 'rb')

    # read header information
    header['Nx'] = np.fromfile(fid, dtype=np.int32, count=1)[0]
    header['Ny'] = np.fromfile(fid, dtype=np.int32, count=1)[0]
    header['Nz'] = np.fromfile(fid, dtype=np.int32, count=1)[0]
    header['Ns'] = np.fromfile(fid, dtype=np.int32, count=1)[0]

    fid.seek(18, 0)
    header['DataTypeCode'] = np.fromfile(fid, dtype=np.int16, count=1)[0]
    fid.seek(152, 0)
    header['Nechoes'] = np.fromfile(fid, dtype=np.int16, count=1)[0]
    fid.seek(156, 0)
    header['Nex'] = np.fromfile(fid, dtype=np.uint8, count=1)[0]

    fid.seek(256, 0)
    text = fid.read(256).decode('utf-8').rstrip('\x00')

    fid.seek(512, 0)

    Nx = header['Nx']
    Ny = header['Ny']
    Nz = header['Nz']
    Ns = header['Ns']

    if header['DataTypeCode'] == 19:
        data_type = 'int16'
    else:
        data_type = 'float32'

    raw_data_tmp = np.fromfile(fid, dtype=data_type, count=2 * Nx * Nz * Ns * Ny)
    raw_data_temp=np.vectorize(complex)(raw_data_tmp[::2], raw_data_tmp[1::2]).reshape(
        (Nx, Nz * Ny, Ns), order="F").astype(np.complex64)
    emi_data_tmp = np.fromfile(fid, dtype=data_type, count=2 * Nx * Nz * Ns * Ny)
    emi_data_temp = np.vectorize(complex)(emi_data_tmp[::2], emi_data_tmp[1::2]).reshape(
        (Nx, Nz * Ny, Ns), order="F").astype(np.complex64)

    lines = fid.readlines()
    header['file'] = [line.strip() for line in lines]

    str_ppl = [line.decode().strip() for line in header['file'] if b'PPL' in line][0]

    header['pplpath'] = ''.join([line.strip() for line in str_ppl if line != '\x00'])
    header['pplpath'] = header['pplpath'][4:]

    str_etl = [line.decode().strip() for line in header['file'] if b'views_per_seg,' in line][0]
    header['ETL'] = int(str_etl.split()[2])
    ETL = header['ETL']

    # seperate RAW data and EMI data
    tmp = emi_data_temp.copy()
    center = int(np.floor(ETL/2))

    if ETL % 2 == 1:
        cond = 1
    else:
        cond = 0

    for i in range(center):
        if cond == 1:
            emi_data_temp[:, i::ETL, :] = raw_data_temp[:, i + center + 1::ETL, :]
            raw_data_temp[:, i + center + 1::ETL, :] = tmp[:, i::ETL, :]
        else:
            emi_data_temp[:, i::ETL, :] = raw_data_temp[:, i + center::ETL, :]
            raw_data_temp[:, i + center::ETL, :] = tmp[:, i::ETL, :]

    tmp_raw = raw_data_temp.copy()
    tmp_emi = emi_data_temp.copy()
    list = np.arange(0, ETL)
    realign_table = np.hstack((list[0::2], list[1::2]))

    for i in range(ETL):
        raw_data_temp[:, realign_table[i]::ETL, :] = tmp_raw[:, i::ETL, :]
        emi_data_temp[:, realign_table[i]::ETL, :] = tmp_emi[:, i::ETL, :]


    raw_data_new = np.zeros((header['Nx'], header['Ny'], header['Nz'], header['Ns']), dtype=complex)
    emi_data_new = np.zeros((header['Nx'], header['Ny'], header['Nz'], header['Ns']), dtype=complex)

    for k in range(Nz):
        ksp = np.zeros((Nx, Ny, Ns), dtype=complex)
        for j in range(Ny // ETL):
            ksp[:, ETL * j:ETL * (j + 1), :] = raw_data_temp[:, (j * Nz + k) * ETL: (j * Nz + (k + 1)) * ETL, :]

        raw_data_new[:, :, k, :] = ksp

    for k in range(Nz):
        ksp = np.zeros((Nx, Ny, Ns), dtype=complex)
        for j in range(Ny // ETL):
            ksp[:, ETL * j:ETL * (j + 1), :] = emi_data_temp[:, (j * Nz  + k) * ETL: (j * Nz+ (k + 1)) * ETL, :]

        emi_data_new[:, :, k, :] = ksp

    fid.close()
    mat_data = np.loadtxt('reorder.txt')
    reorder_idx = mat_data.astype(int)
    raw_data_out = np.zeros_like(raw_data_new.squeeze())
    emi_data_out = np.zeros_like(emi_data_new.squeeze())
    raw_data_out[:, reorder_idx, :] = raw_data_new.squeeze()
    emi_data_out[:, reorder_idx, :] = emi_data_new.squeeze()

    return header, text, raw_data_out, emi_data_out



def read_mrd(filename):
    fid = open(filename, 'rb')

    header = {}
    header['Nx'] = np.fromfile(fid, dtype=np.int32, count=1)[0]
    header['Ny'] = np.fromfile(fid, dtype=np.int32, count=1)[0]
    header['Nz'] = np.fromfile(fid, dtype=np.int32, count=1)[0]
    header['Ns'] = np.fromfile(fid, dtype=np.int32, count=1)[0]

    fid.seek(18, 0)
    header['DataTypeCode'] = np.fromfile(fid, dtype=np.int16, count=1)[0]
    fid.seek(152, 0)
    header['Nechoes'] = np.fromfile(fid, dtype=np.int32, count=1)[0]
    fid.seek(156, 0)
    header['Nex'] = np.fromfile(fid, dtype=np.uint32, count=1)[0]

    fid.seek(256, 0)
    text = fid.read(256).decode('utf-8').rstrip()
    header['text'] = text
    fid.seek(512, 0)

    raw_data = np.zeros(( header['Nx'],header['Ny'],header['Nz'], header['Ns'],header['Nechoes'], header['Nex']), dtype=np.complex64)
    for k in range(header['Nex']):
        for i in range(header['Ns']):
            for j in range(header['Ny']):
                raw_data_tmp = np.fromfile(fid, dtype=np.float32, count=2 * header['Nx'] * header['Nz'])
                raw_data_tmp= np.vectorize(complex)(raw_data_tmp[::2], raw_data_tmp[1::2]).reshape(
                    (header['Nx'], header['Nz']), order="F").astype(np.complex64)
                raw_data_tmp=np.expand_dims(raw_data_tmp,axis=-1)
                raw_data[:, j, :, i, :, k]=raw_data_tmp
    raw_data=raw_data.squeeze(-1).squeeze(-1)
    if header['Ns']==1:
        raw_data=raw_data.squeeze(-1)
    header['file'] = fid.readlines()
    index = np.where(np.array([b'PPL' in line for line in header['file']]))
    str = header['file'][index[0][0]].decode().rstrip()
    header['pplpath'] = str.split()[1]

    indices = np.where([b'views_per_seg' in line for line in header['file']])
    str = header['file'][indices[0][0]].decode().rstrip()
    header['ETL'] = int(str.split()[2])
    if header['Nz'] == 1:
        for i in range(header['Ns']):
            ksp = np.zeros((header['Nx'], header['Ny']), dtype=np.complex64)
            for j in range(header['ETL']):
                # x0=header['Ny'] // header['ETL'] * j
                # x1=header['Ny'] // header['ETL'] * (j+1)
                ksp[:, header['Ny'] // header['ETL'] * j:header['Ny'] // header['ETL'] * (j+1)] = raw_data[:,j::header['ETL'], 0,i]
            raw_data[:, :, 0, i] = ksp
    fid.close()

    return header, text, raw_data

def read_mrd_3D(filename):
    fid = open(filename, 'rb')

    header = {}
    header['Nx'] = np.fromfile(fid, dtype=np.int32, count=1)[0]
    header['Ny'] = np.fromfile(fid, dtype=np.int32, count=1)[0]
    header['Nz'] = np.fromfile(fid, dtype=np.int32, count=1)[0]
    header['Ns'] = np.fromfile(fid, dtype=np.int32, count=1)[0]

    fid.seek(18, 0)
    header['DataTypeCode'] = np.fromfile(fid, dtype=np.int16, count=1)[0]
    fid.seek(152, 0)
    header['Nechoes'] = np.fromfile(fid, dtype=np.int32, count=1)[0]
    fid.seek(156, 0)
    header['Nex'] = np.fromfile(fid, dtype=np.uint32, count=1)[0]

    fid.seek(256, 0)
    text = fid.read(256).decode('utf-8').rstrip()
    header['text'] = text
    fid.seek(512, 0)

    raw_data = np.zeros(( header['Nx'],header['Ny'],header['Nz'], header['Ns'],header['Nechoes']), dtype=np.complex64)
    for k in range(header['Nechoes']):
        for i in range(header['Ns']):
            for j in range(header['Nz']):
                raw_data_tmp = np.fromfile(fid, dtype=np.float32, count=2 * header['Nx'] * header['Ny'])
                raw_data_tmp= np.vectorize(complex)(raw_data_tmp[::2], raw_data_tmp[1::2]).reshape(
                    (header['Nx'], header['Ny']), order="F").astype(np.complex64)
                # raw_data_tmp=np.expand_dims(raw_data_tmp,axis=-1)
                raw_data[:, :, j, i, k]=raw_data_tmp
    # raw_data=raw_data.squeeze(-1).squeeze(-1)
    header['file'] = fid.readlines()
    index = np.where(np.array([b'PPL' in line for line in header['file']]))
    str = header['file'][index[0][0]].decode().rstrip()
    header['pplpath'] = str.split()[1]

    indices = np.where([b'views_per_seg' in line for line in header['file']])
    str = header['file'][indices[0][0]].decode().rstrip()
    header['ETL'] = int(str.split()[2])
    if header['Nz'] == 1:
        for i in range(header['Ns']):
            ksp = np.zeros((header['Nx'], header['Ny']), dtype=np.complex64)
            for j in range(header['ETL']):
                # x0=header['Ny'] // header['ETL'] * j
                # x1=header['Ny'] // header['ETL'] * (j+1)
                ksp[:, header['Ny'] // header['ETL'] * j:header['Ny'] // header['ETL'] * (j+1)] = raw_data[:,j::header['ETL'], 0,i]
            raw_data[:, :, 0, i] = ksp
    fid.close()

    return header, text, raw_data


def fftc(x, dim):
    x1=torch.fft.ifftshift(x, dim=dim)
    x2=torch.fft.fftn(x1, dim=dim)
    x3=torch.fft.fftshift(x2, dim=dim)
    l=torch.tensor(x.size(dim), dtype=x.dtype)
    res = 1/torch.sqrt(l) * x3
    return res
def ifftc(x, dim):
    x1=torch.fft.ifftshift(x, dim=dim)
    x2=torch.fft.ifft(x1, dim=dim)
    x3=torch.fft.fftshift(x2, dim=dim)
    l=torch.tensor(x.size(dim), dtype=x.dtype)
    res = torch.sqrt(l) * x3
    return res

def ifft2c(x):
    S = x.size()
    fctr = S[0]*S[1]
    x = x.reshape(S[0], S[1], torch.prod(torch.tensor(S[2:]), dtype=torch.int))
    res = torch.zeros_like(x)
    for n in range(x.size(2)):
        res[:,:,n] = np.sqrt(fctr)*torch.fft.fftshift(torch.fft.ifft2(torch.fft.ifftshift(x[:,:,n])))
    res = res.reshape(S)
    return res
def fft2c(x):
    S = x.size()
    fctr = S[0]*S[1]
    x = x.reshape(S[0], S[1], torch.prod(torch.tensor(S[2:]), dtype=torch.int))
    res = torch.zeros_like(x)
    for n in range(x.size(2)):
        res[:,:,n] = (1/np.sqrt(fctr))*torch.fft.fftshift(torch.fft.fftn(torch.fft.ifftshift(x[:,:,n])))
    res = res.reshape(S)
    return res
def ifft3c(ksp,n):
    imgs = ifft2c(ifftc(ksp, n))
    return imgs
def fft3c(ksp,n):
    imgs = fft2c(fftc(ksp, n))
    return imgs
def plot_tensor_slices(img3d,row,col):
    # assume tensor has shape (H, W, Z)
    # img3d_normalized = (img3d - torch.min(img3d)) / (torch.max(img3d) - torch.min(img3d))
    num_slices = img3d.shape[-1]
    rows, cols = row,col
    fig, axes = plt.subplots(rows, cols)
    for i in range(num_slices):
        row_idx, col_idx = i // cols, i % cols
        axes[row_idx, col_idx].imshow(img3d[:, :, i], cmap='gray')
    return img3d
def numshow(x,r,c):
    x_copy = np.copy(x)
    x_check = torch.from_numpy(x_copy).squeeze()
    x_checko = ifft3c(x_check.clone(),2)
    check = torch.abs(x_checko)
    img3d=plot_tensor_slices(check,r,c)
    return norm(img3d.numpy())


def tenshow(x,r,c,Nz,Nx,Ny):
    # real_mean = torch.mean(x[:, 0, :, :], dim=2)
    # imag_mean = torch.mean(x[:, 1, :, :], dim=2)
    real_mean = x[:, 0, :, 0]
    imag_mean = x[:, 1, :, 0]
    x_check = real_mean + 1j * imag_mean
    x_check = x_check.permute(1, 0)
    x_check = x_check.numpy()
    x_check = np.reshape(x_check, (Nx,Ny,Nz), order="F")
    x_check = torch.from_numpy(x_check)
    x_checko = ifft3c(x_check.clone(),2)
    check = torch.abs(x_checko)
    _=plot_tensor_slices(check,r,c)
    return norm(check.numpy())

def mulc_tenshow(x,r,c,Nx,Ny,ch):
    real_mean = x[:, 0, :, ch]
    imag_mean = x[:, 1, :, ch]
    x_check = real_mean + 1j * imag_mean
    x_check = x_check.permute(1, 0)
    x_check = x_check.numpy()
    x_check = np.reshape(x_check, (Nx,Ny,r*c), order="F")
    x_check = torch.from_numpy(x_check)
    x_checko = ifft3c(x_check.clone(),2)
    check = torch.abs(x_checko)
    _=plot_tensor_slices(check,r,c)
    return check.numpy()

def kshow(x,r,c):
    check = np.abs(x)
    num_slices = check.shape[-1]
    num=1
    # # 显示
    plt.rcParams['figure.figsize'] = (1.28, 1.28)  # 2.24, 2.24 设置figure_size尺寸
    plt.rcParams['savefig.dpi'] = 100  # 图片像素
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0, 0)
    plt.axis('off')
    for i in range(num_slices):
        plt.imshow(check[...,i], cmap='gray')
        # plt.axis('off')
        # plt.show()
        if i>=4 and i<8:
            plt.savefig(f'D:\文件\研究生\课题\MRI CNN\论文\图片\kspfig\\remnantnoise{num}.tif',bbox_inches='tight',pad_inches =0)
            # cv2.imwrite(f'D:\huyang\kspfig\emi_coil1_emi{num}.png',check[:,:,i],[cv2.IMWRITE_PNG_COMPRESSION, 0])
            num+=1
def compshow(x,r,c):
    x_check = x[:, :, 0 ,:] + x[:, :, 1,:] * 1j
    x_check = torch.from_numpy(x_check)
    x_checko = ifft3c(x_check.clone(),2)
    check = torch.abs(x_checko)
    _=plot_tensor_slices(check,r,c)
    return check.numpy()

def figshow(x,num=None):
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
    num_slices = x.shape[-2]
    cout=x.shape[-1]
    if num is None:
        indices = range(num_slices)
    else:
        num_slices=num
        center = x.shape[-2] // 2
        start_idx = center - num//2
        end_idx = start_idx + num_slices
        indices = list(range(start_idx, end_idx))

    figsize=(num_slices*x.shape[0]/80,cout*x.shape[1]/80)
    plt.rcParams['figure.dpi'] = 80
    plt.rcParams['savefig.dpi'] = 80
    fig, axes = plt.subplots(cout, num_slices,figsize=figsize)
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0, 0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())

    for j in range(cout):
        snum = 0
        for i in indices:
            axes[j, snum].imshow(x[:, :, i,j], cmap='gray', vmin=0, vmax=255,
                      extent=[0, x.shape[0], 0, x.shape[1]])
            axes[j,snum].axis('off')
            snum += 1
    return fig


def calculate_noise_from_corners(image, corner_size=10):
    h, w = image.shape
    # 提取四个角的区域
    top_left = image[:corner_size, :corner_size]
    top_right = image[:corner_size, -corner_size:]
    bottom_left = image[-corner_size:, :corner_size]
    bottom_right = image[-corner_size:, -corner_size:]
    # 将四个角的像素值合并
    corners = np.concatenate([top_left.ravel(), top_right.ravel(),
                              bottom_left.ravel(), bottom_right.ravel()])
    # 计算均值和标准差
    noise_mean = np.mean(corners)
    noise_std = np.std(corners)
    return noise_mean, noise_std

def figshow2(x, num=None, zoom_region=None):
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)

    width, height, slices, counts = x.shape

    if num is None:
        indices = range(slices)
    else:
        num_slices = num
        center = slices // 2
        start_idx = max(center - num // 2, 0)
        end_idx = min(start_idx + num_slices, slices)
        indices = list(range(start_idx, end_idx))
        slices = len(indices)

    figsize = (slices * height / 80, counts * height / 80)
    plt.rcParams['figure.dpi'] = 80
    plt.rcParams['savefig.dpi'] = 80
    fig, axes = plt.subplots(counts, slices, figsize=figsize)

    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0, 0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())

    if counts == 1 and slices == 1:
        axes = np.array([[axes]])
    elif counts == 1:
        axes = axes[np.newaxis, :]
    elif slices == 1:
        axes = axes[:, np.newaxis]
    for j in range(counts):
        for snum, i in enumerate(indices):
            ax = axes[j, snum]
            img = x[:, :, i, j].cpu().numpy() if isinstance(x, torch.Tensor) else x[:, :, i, j]

            ax.imshow(img, cmap='gray', vmin=0, vmax=255, extent=[0, height, 0, height])
            ax.axis('off')

            nrss_value = NRSS(img)

            # 计算基于图像四角的噪声均值和标准差
            noise_mean, noise_std = calculate_noise_from_corners(img,20)

            if zoom_region is None:
                x_center = height // 2
                y_center = height // 2
                zoom_size = height // 4
                x_start = x_center
                y_start = y_center
                zoom_width = zoom_size * 1
                zoom_height = zoom_size * 1
            else:
                x_start = zoom_region.get('x_start', 50)
                y_start = zoom_region.get('y_start', 50)
                zoom_width = zoom_region.get('width', 50)
                zoom_height = zoom_region.get('height', 50)

            ax_inset = inset_axes(ax, width="30%", height="30%", loc='upper right',
                                  bbox_to_anchor=(0, 0, 1, 1),
                                  bbox_transform=ax.transAxes)
            ax_inset.imshow(img[x_start-zoom_width*4//3:x_start, y_start-zoom_height*1//2:y_start+zoom_height*1//2],
                            cmap='gray', vmin=0, vmax=255)#x-上，y+右
            ax_inset.set_xticks([])
            ax_inset.set_yticks([])

            # Add white border around the inset
            for spine in ax_inset.spines.values():
                spine.set_edgecolor('white')
                spine.set_linewidth(1)

            # 在主图像的右下角显示 NRSS 值
            ax.text(0.95, 0.05, f'NRSS: {nrss_value:.2f}',
                    color='white', fontsize=12,
                    ha='right', va='bottom',
                    transform=ax.transAxes,
                    bbox=dict(facecolor='black', alpha=0.5, pad=1))

            # 在图像上显示噪声水平 (均值 + 标准差)
            ax.text(0.05, 0.95, f'std:{noise_std:.2f}\nmean:{noise_mean:.2f}',
                    color='white', fontsize=10,
                    ha='left', va='top',
                    transform=ax.transAxes,
                    bbox=dict(facecolor='black', alpha=0.5, pad=1))

    return fig


def calculate_snr(image):
    # 获取图像大小
    height, width = image.shape

    # 计算感兴趣区域的范围
    roi_height = int(height * 0.6)
    roi_width = int(width * 0.6)
    roi_top = (height - roi_height) // 2
    roi_bottom = roi_top + roi_height
    roi_left = (width - roi_width) // 2
    roi_right = roi_left + roi_width

    # 提取感兴趣区域
    roi = image[roi_top:roi_bottom, roi_left:roi_right]

    # 计算感兴趣区域的信号均值
    signal_mean = np.mean(roi)

    # 计算背景区域的范围
    bg_size = int(height * 0.1)

    # 提取四个角的背景区域
    bg_tl = image[:bg_size, :bg_size]
    bg_tr = image[:bg_size, width - bg_size:]
    bg_bl = image[height - bg_size:, :bg_size]
    bg_br = image[height - bg_size:, width - bg_size:]

    # 计算背景区域的信号标准差
    bg_std = np.std([bg_tl, bg_tr, bg_bl, bg_br])
    std1 = np.std(bg_tl)
    std2 = np.std(bg_tr)
    std3 = np.std(bg_bl)
    std4 = np.std(bg_br)

    # 计算SNR
    snr = 0.66 * signal_mean / bg_std

    return snr


def find_convergence_point(loss, consecutive_batches=10, convergence_threshold=20):
    num_batches = len(loss)
    reference_avg = sum(loss[int(num_batches * 0.7):]) / (num_batches * 0.3)
    moving_average = [sum(loss[i:i + consecutive_batches]) / consecutive_batches for i in
                      range(len(loss) - consecutive_batches + 1)]
    convergence_point = None
    consecutive_count = 0

    for batch_num, avg_loss in enumerate(moving_average):
        if avg_loss <= reference_avg and abs(avg_loss - moving_average[batch_num - 1]) < convergence_threshold:
            consecutive_count += 1
            if consecutive_count == 2:
                convergence_point = batch_num
                break
        else:
            consecutive_count = 0

    return convergence_point
def norm(num):
    min_val = np.min(num)
    max_val = np.max(num)
    result_normalized = (num - min_val) / (max_val - min_val)
    result_normalized = (result_normalized * 255).astype(np.uint8)
    return result_normalized
def normalize_with_mean_std(fig, figin):
    figin_mean = figin.mean()
    figin_std = figin.std()
    normalized_fig = (fig - fig.mean()) / fig.std()
    normalized_fig = normalized_fig * figin_std + figin_mean
    return normalized_fig
