import numpy as np
import scipy.io as scio
from utils import *
import numpy as np
from seting import Nx,Ny,Nc,Nz,row,col
import cv2
def subtract_mean(arr):
    """
    对输入的数组 arr 沿第 0 和第 2 维度计算均值，并将其从原数组中减去。
    """
    mean_arr = arr.mean(axis=(0, 2), keepdims=True)
    return arr - mean_arr
def normalize_data(data):
    # 计算每个通道的最大值
    max_values = np.abs(data).max()
    # 对每个通道的所有像素进行归一化
    data /= max_values
    return data
# filepath = './rawdatas/mrdfile/20241022_xiaocui/ref_broadband/'
filepath = './rawdatas/mrdfile/20241018_hongjiang/narrowband/'
# filepath = './rawdatas/mrdfile/20241018_huyang/broadband/'
# filepath = './rawdatas/mrdfile/20230814/Huyang/FFE3D/'
# filepath = './rawdatas/mrdfile/20241014/narrowband1/'
header1, text1, raw_data1, raw_data1_EMI = read_mrd_EMI_hym(filepath + 'Scan1.MRD')
header2, text2, raw_data2, raw_data2_EMI = read_mrd_EMI_hym(filepath + 'Scan2.MRD')
header3, text3, raw_data3, raw_data3_EMI = read_mrd_EMI_hym(filepath + 'Scan3.MRD')
header4, text4, raw_data4, raw_data4_EMI = read_mrd_EMI_hym(filepath + 'Scan4.MRD')

# header1, text1, raw_data1, raw_data1_EMI = read_mrd_fse3d_2echoes(filepath + 'Scan1.MRD')
# header2, text2, raw_data2, raw_data2_EMI = read_mrd_fse3d_2echoes(filepath + 'Scan2.MRD')
# header3, text3, raw_data3, raw_data3_EMI = read_mrd_fse3d_2echoes(filepath + 'Scan3.MRD')
# header4, text4, raw_data4, raw_data4_EMI = read_mrd_fse3d_2echoes(filepath + 'Scan4.MRD')
rotation_k = -1  # 旋转 180°
raw_data1 = np.rot90(raw_data1, k=rotation_k, axes=(0, 1))
raw_data2 = np.rot90(raw_data2, k=rotation_k, axes=(0, 1))
raw_data3 = np.rot90(raw_data3, k=rotation_k, axes=(0, 1))
raw_data4 = np.rot90(raw_data4, k=rotation_k, axes=(0, 1))
raw_data1_EMI = np.rot90(raw_data1_EMI, k=rotation_k, axes=(0, 1))
raw_data2_EMI = np.rot90(raw_data2_EMI, k=rotation_k, axes=(0, 1))
raw_data3_EMI = np.rot90(raw_data3_EMI, k=rotation_k, axes=(0, 1))
raw_data4_EMI = np.rot90(raw_data4_EMI, k=rotation_k, axes=(0, 1))

img3d=numshow(raw_data1,row,col)
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
ksp1=subtract_mean(ksp1)
ksp2=subtract_mean(ksp2)
ksp3=subtract_mean(ksp3)
ksp4=subtract_mean(ksp4)
scio.savemat('trainmat/ksp1.mat', {'ksp1':ksp1})
scio.savemat('trainmat/ksp2.mat', {'ksp2':ksp2})
scio.savemat('trainmat/ksp3.mat', {'ksp3':ksp3})
scio.savemat('trainmat/ksp4.mat', {'ksp4':ksp4})

# savepa = './contrast/narrowband/ref/'
# if os.path.exists(savepa):
#     pass
# else:
#     os.makedirs(savepa)
# num = 10
# center = img3d.shape[-1] // 2
# start_idx = center - num // 2
# end_idx = center + num//2
# indices = list(range(start_idx, end_idx))
# id = 0
# for i in indices:
#     cv2.imwrite(savepa + f"image_{id}.png", img3d[:, :, i])
#     id += 1