import os

import cv2
from PIL import Image
import numpy as np


# -----------------------
# 该脚本用于生成角度预测的预训练数据
# -----------------------


def transfer(root_dir, video_path, prefix):
    cap = cv2.VideoCapture(video_path)  # 获取视频对象
    # 视频信息获取
    isOpened = cap.isOpened  # 判断是否打开
    fps = cap.get(cv2.CAP_PROP_FPS)
    imageNum = 0
    sum = 0
    timef = 1  # 隔timef帧保存一张图片

    while (isOpened):

        sum += 1

        (frameState, frame) = cap.read()  # 记录每帧及获取状态

        if frameState == True and (sum % timef == 0):

            # 格式转变，BGRtoRGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # 转变成Image
            frame = Image.fromarray(np.uint8(frame))

            frame = np.array(frame)

            # RGBtoBGR满足opencv显示格式
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            imageNum = imageNum + 1
            fileName = os.path.join(save_dir, str(prefix) + '_' + str((imageNum * 10 + 295) % 360) + '.jpg')  # 存储路径
            cv2.imwrite(fileName, frame, [cv2.IMWRITE_JPEG_QUALITY, 100])
            print(fileName + " successfully write in")  # 输出存储状态

        elif frameState == False:
            break

    print('finish!')
    cap.release()


if __name__ == '__main__':
    condition = r'\3'
    video_folder = r'D:\Instant-NGP-for-RTX-3000-and-4000\angle_data\video_dataset' + condition
    save_dir = r'D:\Instant-NGP-for-RTX-3000-and-4000\angle_data\pic_dataset' + condition
    os.makedirs(save_dir, exist_ok=True)
    videos = os.listdir(video_folder)
    for i, name in enumerate(videos):
        transfer(save_dir, os.path.join(video_folder, name), i + 1)
