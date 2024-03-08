import numpy as np
import scipy.io as scio
from utils import *
import numpy as np

def normalize_data(data):
    # 计算每个通道的最大值
    max_values = np.abs(data).max()
    # 对每个通道的所有像素进行归一化
    data /= max_values
    return data
row=4
col=5
Nc = 2
filepath = '.\\mrdfile/20230814/Huyang/FFE3D\\'
# filepath = './mrdfile/1214/feiyang/'
header1, text1, raw_data1, raw_data1_EMI = read_mrd_EMI_hym(filepath + 'Scan1.MRD')
header2, text2, raw_data2, raw_data2_EMI = read_mrd_EMI_hym(filepath + 'Scan2.MRD')
header3, text3, raw_data3, raw_data3_EMI = read_mrd_EMI_hym(filepath + 'Scan3.MRD')
header4, text4, raw_data4, raw_data4_EMI = read_mrd_EMI_hym(filepath + 'Scan4.MRD')

# header1, text1, raw_data1, raw_data1_EMI = read_mrd_fse3d_2echoes(filepath + 'Scan1.MRD')
# header2, text2, raw_data2, raw_data2_EMI = read_mrd_fse3d_2echoes(filepath + 'Scan2.MRD')
# header3, text3, raw_data3, raw_data3_EMI = read_mrd_fse3d_2echoes(filepath + 'Scan3.MRD')
# header4, text4, raw_data4, raw_data4_EMI = read_mrd_fse3d_2echoes(filepath + 'Scan4.MRD')
# 顺时针旋转90度
# raw_data1 = np.rot90(raw_data1, k=-1, axes=(0, 1))
# raw_data2 = np.rot90(raw_data2, k=-1, axes=(0, 1))
# raw_data3 = np.rot90(raw_data3, k=-1, axes=(0, 1))
# raw_data4 = np.rot90(raw_data4, k=-1, axes=(0, 1))
# raw_data1_EMI = np.rot90(raw_data1_EMI, k=-1, axes=(0, 1))
# raw_data2_EMI = np.rot90(raw_data2_EMI, k=-1, axes=(0, 1))
# raw_data3_EMI = np.rot90(raw_data3_EMI, k=-1, axes=(0, 1))
# raw_data4_EMI = np.rot90(raw_data4_EMI, k=-1, axes=(0, 1))
numshow(raw_data1,row,col)
plt.show()

raw_data1=np.reshape(raw_data1,(header1["Nx"], header1["Ny"] * header1["Nz"]),order="F");
raw_data2=np.reshape(raw_data2,(header1["Nx"], header1["Ny"] * header1["Nz"]),order="F");
raw_data3=np.reshape(raw_data3,(header1["Nx"], header1["Ny"] * header1["Nz"]),order="F");
raw_data4=np.reshape(raw_data4,(header1["Nx"], header1["Ny"] * header1["Nz"]),order="F");
raw_data1_EMI=np.reshape(raw_data1_EMI,(header1["Nx"], header1["Ny"] * header1["Nz"]),order="F");
raw_data2_EMI=np.reshape(raw_data2_EMI,(header1["Nx"], header1["Ny"] * header1["Nz"]),order="F");
raw_data3_EMI=np.reshape(raw_data3_EMI,(header1["Nx"], header1["Ny"] * header1["Nz"]),order="F");
raw_data4_EMI=np.reshape(raw_data4_EMI,(header1["Nx"], header1["Ny"] * header1["Nz"]),order="F");

#####生成0数组
ksp1 = np.zeros((header1["Nx"], 2, header1["Ny"] * header1["Nz"], Nc));
ksp1[:, 0, :, 0] = np.real(raw_data1)
ksp1[:, 1, :, 0] = np.imag(raw_data1)
ksp1[:, 0, :, 1] = np.real(raw_data2)
ksp1[:, 1, :, 1] = np.imag(raw_data2)

ksp2 = np.zeros((header1["Nx"], 2, header1["Ny"] * header1["Nz"], Nc));
ksp2[:, 0, :, 0] = np.real(raw_data1_EMI)
ksp2[:, 1, :, 0] = np.imag(raw_data1_EMI)
ksp2[:, 0, :, 1] = np.real(raw_data2_EMI)
ksp2[:, 1, :, 1] = np.imag(raw_data2_EMI)

ksp3 = np.zeros((header1["Nx"], 2, header1["Ny"] * header1["Nz"], Nc));
ksp3[:, 0, :, 0] = np.real(raw_data3)
ksp3[:, 1, :, 0] = np.imag(raw_data3)
ksp3[:, 0, :, 1] = np.real(raw_data4)
ksp3[:, 1, :, 1] = np.imag(raw_data4)

ksp4 = np.zeros((header1["Nx"], 2, header1["Ny"] * header1["Nz"], Nc));
ksp4[:, 0, :, 0] = np.real(raw_data3_EMI)
ksp4[:, 1, :, 0] = np.imag(raw_data3_EMI)
ksp4[:, 0, :, 1] = np.real(raw_data4_EMI)
ksp4[:, 1, :, 1] = np.imag(raw_data4_EMI)

scio.savemat('.\\trainmat\\ksp1.mat', {'ksp1':ksp1})
scio.savemat('.\\trainmat\\ksp2.mat', {'ksp2':ksp2})
scio.savemat('.\\trainmat\\ksp3.mat', {'ksp3':ksp3})
scio.savemat('.\\trainmat\\ksp4.mat', {'ksp4':ksp4})
