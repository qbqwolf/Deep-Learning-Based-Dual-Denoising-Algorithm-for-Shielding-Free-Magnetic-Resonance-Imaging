import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
def plotbox(labels,list):
    lists=[]
    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(5, 4))
    for i in range(len(list)):
        lists.append(list[i])
    plt.grid(True)  # 显示网格
    plt.boxplot(lists,
                medianprops={'color': 'red', 'linewidth': '1.5'},
                meanline=True,
                autorange=True,
                # showmeans=True,
                # meanprops={'color': 'blue', 'ls': '--', 'linewidth': '1.5'},
                boxprops={'color': 'blue', 'linewidth': '1.5'},
                flierprops={"marker": "o", "markerfacecolor": "red", "markersize": 15},
                labels=labels)

    # plt.show()
def multiplotbox(axes,labels,list1,list2,title):
    lists=[[],[]]
    for i in range(len(list1)):
        lists[0].append(list1[i])
    for i in range(len(list2)):
        lists[1].append(list2[i])
    for i in range(len(lists)):
        axes[i].set_title(title[i],fontsize=16)
        axes[i].grid(True)  # 显示网格
        axes[i].boxplot(lists[i],
                    medianprops={'color': 'red', 'linewidth': '1.5'},
                    meanline=True,
                    autorange=True,
                    # showmeans=True,
                    # meanprops={'color': 'blue', 'ls': '--', 'linewidth': '1.5'},
                    boxprops={'linewidth': '1.5'},
                    flierprops={"marker": "o", "markerfacecolor": "red", "markersize": 10},
                    labels=labels)
        # plt.yticks(np.arange(0, 15, 0.5))

folders = [ "./results/secondary_denoising/Gauss/test/","./results/secondary_denoising/Rayleigh/test/",
            "./results/secondary_denoising/N2V/test/","./results/secondary_denoising/T1/test/"]
BRSf='BRS_results.csv'
# SNRf='snr_results.csv'
# snrflist=[]
brsflist=[]
for folder in folders:
    # snrf = pd.read_csv(folder+SNRf)
    brsf = pd.read_csv(folder + BRSf)
    # snrflist.append(snrf[['secondary_denoising']].values.squeeze(-1).tolist())
    brsflist.append(brsf[['secondary_denoising']].values.squeeze(-1).tolist())
labels = 'Gauss', 'Rayleigh','N2V','Ours'
title=['SNR','BRSQUE']
# fig,axes=plt.subplots(nrows=1,ncols=2,figsize=(10,4))
# plotbox(labels,snrflist)
plotbox(labels,brsflist)
# multiplotbox(axes,labels,snrflist,brsflist,title)
plt.title("BRSQUE scores",fontsize=12)
plt.savefig('./results/secondary_denoising/T1/test/contra.png')
plt.show()