import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
def transimage():
    inpath = "./datasets/T1/train/"
    opath = "./datasets/T1phatom/train/"
    leng = len(os.listdir(inpath + "input/"))
    oleng = len(os.listdir(opath + "input/"))
    oid = 96
    for i in range(leng):
        input = cv2.imread(inpath + f"input/image_{i + 1}.png", -1)
        label = cv2.imread(inpath + f"label/image_{i + 1}.png", -1)
        cv2.imwrite(opath + f"input/image_{oid + 1}.png", input)
        cv2.imwrite(opath + f"label/image_{oid + 1}.png", label)
        # plt.imsave(opath+f"input/image_{oid+1}.png",input,cmap='gray')
        # plt.imsave(opath+f"label/image_{oid+1}.png",label,cmap='gray')
        oid += 1

    inpath = "./datasets/T1/val/"
    opath = "./datasets/T1phatom/val/"
    leng = len(os.listdir(inpath + "input/"))
    oid = 34
    for i in range(leng):
        input = cv2.imread(inpath + f"input/image_{i + 1}.png", -1)
        label = cv2.imread(inpath + f"label/image_{i + 1}.png", -1)
        cv2.imwrite(opath + f"input/image_{oid + 1}.png", input)
        cv2.imwrite(opath + f"label/image_{oid + 1}.png", label)
        oid += 1
def mergeimgs():
    path="./results/secondary_denoising/T1phatom/test/images/"
    for i in range(5):
        ch1 = cv2.imread(path + f"image_{i + 1}.png", 0)
        ch2 = cv2.imread(path + f"image_{i + 6}.png", 0)
        ch3 = cv2.imread(path + f"image_{i + 11}.png", 0)
        out = (1 / 3) * ch1 + (1 / 3) * ch2 + (1 / 3) * ch3
        cv2.imwrite(path + f"mergeimage_{i + 1}.png", out)
def mergeimsquare():
    path = "./results/secondary_denoising/T1phatom/test/images/"
    for i in range(5):
        ch1 = cv2.imread(path + f"image_{i + 1}.png", -1).astype(np.float64)
        ch2 = cv2.imread(path + f"image_{i + 6}.png", -1).astype(np.float64)
        ch3 = cv2.imread(path + f"image_{i + 11}.png", -1).astype(np.float64)
        # 计算每个像素的均方和
        square_sum = np.sqrt(ch1 ** 2 + ch2 ** 2 + ch3 ** 2)

        out = cv2.normalize(square_sum, None, 0, 255, cv2.NORM_MINMAX)
        cv2.imwrite(path + f"mergesquare_{i + 1}.png", out)
if __name__ == '__main__':
    # transimage()
    # mergeimgs()
    mergeimsquare()