import os
import numpy as np
from PIL import Image


def calculate_mean_and_std(root_dir):
    image_list = []

    # 遍历目录下的所有图片文件
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith(".jpg"):
                image_path = os.path.join(root, file)
                print(image_path)
                image = Image.open(image_path)
                image = image.resize((512, 512))  # 调整图片大小为 512x512
                image = np.array(image) / 255.0  # 将像素值缩放到 [0, 1] 范围内
                image_list.append(image)
                # 计算均值和标准差
    images = np.stack(image_list, axis=0)
    mean = np.mean(images, axis=(0, 1, 2))
    std = np.std(images, axis=(0, 1, 2))

    return mean, std





if __name__ == '__main__':

    root_directory = r'D:\Instant-NGP-for-RTX-3000-and-4000\angle_data\real_dataset'
    mean, std = calculate_mean_and_std(root_directory)
    print("Mean:", mean)
    print("Std:", std)