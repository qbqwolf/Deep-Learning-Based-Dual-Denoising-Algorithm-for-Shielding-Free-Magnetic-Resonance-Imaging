import argparse
import os

import torch.backends.cudnn as cudnn
# from train_n2v import *
from train import *
# from trainbw import *
# from traink import *
def del_file(path):
    ls = os.listdir(path)
    for i in ls:
        c_path = os.path.join(path, i)
        if os.path.isdir(c_path):
            del_file(c_path)
        else:
            os.remove(c_path)
def make_and_del(paths,fla):
    for dir in paths:
        if os.path.exists(dir):
            pass
        else:
            os.makedirs(dir)
    for dir in paths:
        if os.path.exists(dir) and fla:
            del_file(dir)
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

cudnn.benchmark = True
cudnn.fastest = True

FLAG_PLATFORM = 'laptop'
#FLAG_PLATFORM = 'colab'

## setup parse
parser = argparse.ArgumentParser(description='Train the unet network',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--gpu_ids', default='0', dest='gpu_ids')
parser.add_argument('--dir_checkpoint', default='./checkpoints', dest='dir_checkpoint')
parser.add_argument('--dir_log', default='./log', dest='dir_log')
parser.add_argument('--dir_result', default='./results', dest='dir_result')
parser.add_argument('--scope', default='secondary_denoising', dest='scope')
parser.add_argument('--dir_data', default='./datasets', dest='dir_data')

parser.add_argument('--name_data', type=str, default='T1phatom', dest='name_data')####choose dataset and result category
parser.add_argument('--dir_result_test', default='test', dest='dir_result_test')###write result file name and input data name of test


parser.add_argument('--mode', default='test', choices=['train', 'test'], dest='mode')###choose mode
parser.add_argument('--train_continue', default='on', choices=['on', 'off'], dest='train_continue')
parser.add_argument('--norm', type=str, default='bnorm', dest='norm')

parser.add_argument('--st_epoch', type=int,  default=160, dest='st_epoch')
parser.add_argument('--num_epoch', type=int,  default=300, dest='num_epoch')
parser.add_argument('--test_epoch', type=int,  default=300, dest='test_epoch')###load epoch number
parser.add_argument('--batch_size', type=int, default=3, dest='batch_size')

parser.add_argument('--lr_G', type=float, default=1e-3, dest='lr_G')

parser.add_argument('--optim', default='adam', choices=['sgd', 'adam', 'rmsprop'], dest='optim')
parser.add_argument('--beta1', default=0.5, dest='beta1')

parser.add_argument('--ny_in', type=int, default=128, dest='ny_in')
parser.add_argument('--nx_in', type=int, default=128, dest='nx_in')
parser.add_argument('--nch_in', type=int, default=1, dest='nch_in')

parser.add_argument('--size_window', type=int, default=(7,7), dest='size_window')#parameter of N2V
parser.add_argument('--ratio', type=int, default=0.9, dest='ratio')


parser.add_argument('--nch_out', type=int, default=1, dest='nch_out')

parser.add_argument('--nch_ker', type=int, default=64, dest='nch_ker')

parser.add_argument('--data_type', default='float32', dest='data_type')

parser.add_argument('--num_freq_disp', type=int,  default=1, dest='num_freq_disp')
parser.add_argument('--num_freq_save', type=int,  default=80, dest='num_freq_save')

PARSER = Parser(parser)

def main():
    ARGS = PARSER.get_arguments()
    PARSER.write_args()
    PARSER.print_args()

    TRAINER = Train(ARGS)

    if ARGS.mode == 'train':
        TRAINER.train()
    elif ARGS.mode == 'test':
        TRAINER.test()

if __name__ == '__main__':
    name='T1phatom'
    fla=0
    resultpath=[f'./results/secondary_denoising/{name}/train',f'./results/secondary_denoising/{name}/val'
        ,f'./results/secondary_denoising/{name}/test']

    make_and_del(resultpath,fla)

    logpath = [f'./results/secondary_denoising/{name}/train/images',f'./results/secondary_denoising/{name}/val/images',
              f'./log/secondary_denoising/{name}/train',f'./log/secondary_denoising/{name}/val']

    make_and_del(logpath, fla)

    main()
    #cd D:\pythonspace\My_ultra_low_field2
    # tensorboard --logdir=log/secondary_denoising/myultra_low/train
    # tensorboard --logdir=log/secondary_denoising/brainweb/train
    # tensorboard --logdir=log/secondary_denoising/kspace/train
    # tensorboard --logdir=log/secondary_denoising/N2V/train
    # tensorboard --logdir=log/secondary_denoising/T1phatom/train