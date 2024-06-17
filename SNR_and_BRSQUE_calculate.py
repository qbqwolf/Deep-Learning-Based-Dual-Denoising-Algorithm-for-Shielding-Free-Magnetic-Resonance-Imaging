from utils import *
import sys
import csv
sys.path.append("./libsvm/python/")
from libsvm.python.brisquequality import *
# img = cv2.imread("D:\\pythonspace\\My_ultra_low_field\\results\\secondary_denoising\\kspace2\\GRET1\\images\\000-input.png",0)
# snr = calculate_snr(img)
# print("SNR of the given image: ",snr)
# img = cv2.imread("D:\\pythonspace\\My_ultra_low_field\\results\\secondary_denoising\\kspace2\\GRET1\\images\\000-output.png",0)
# snr = calculate_snr(img)
# print("SNR of the output image: ",snr)
folders = ["./results/first_denoising\initial", "./results/first_denoising/results_T1", "./results/secondary_denoising/T1_heavy/test/images"]
cpath="./results/secondary_denoising/T1_heavy/test/"
# 保存结果的 CSV 文件路径
csv_filename =cpath+ "snr_results.csv"

# CSV 文件头部
csv_header = ["Image", "raw_data", "first_denoising", "secondary_denoising"]

# 获取每个文件夹内图片的数量
image_count = min([len(os.listdir(folder)) for folder in folders])

# 打开 CSV 文件并写入头部
with open(csv_filename, mode="w", newline="") as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(csv_header)

    # 遍历每张图像
    for image_idx in range(1, image_count + 1):
        snr_values = []

        # 遍历不同的处理阶段
        for folder in folders:
            img_path = os.path.join(folder, f"image_{image_idx}.png")  # 假设图像文件名是 image_1.png、image_2.png 等
            img = cv2.imread(img_path, 0)  # 读取灰度图像
            snr = calculate_snr(img)
            snr_values.append(snr)

        writer.writerow([image_idx] + snr_values)
csv_filename = cpath+"BRS_results.csv"

# CSV 文件头部
csv_header = ["Image", "raw_data", "first_denoising", "secondary_denoising"]

# 获取每个文件夹内图片的数量
image_count = min([len(os.listdir(folder)) for folder in folders])

# 打开 CSV 文件并写入头部
with open(csv_filename, mode="w", newline="") as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(csv_header)

    # 遍历每张图像
    for image_idx in range(1, image_count + 1):
        values = []

        # 遍历不同的处理阶段
        for folder in folders:
            img_path = os.path.join(folder, f"image_{image_idx}.png")  # 假设图像文件名是 image_1.png、image_2.png 等
            qualityscore = test_measure_BRISQUE(img_path)
            values.append(qualityscore)

        writer.writerow([image_idx] + values)
# qualityscore = test_measure_BRISQUE("D:\\pythonspace\\My_ultra_low_field\\results\\secondary_denoising\\kspace2\\GRET1\\images\\000-input.png")
# print("Score of the given image: ", qualityscore)
# qualityscore = test_measure_BRISQUE("D:\\pythonspace\\My_ultra_low_field\\results\\secondary_denoising\\kspace2\\GRET1\\images\\000-output.png")
# print("Score of the output image: ", qualityscore)