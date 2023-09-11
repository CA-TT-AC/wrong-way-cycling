import os.path
import random

import cv2
from PIL import Image
import numpy as np
import argparse
# -----------------------
# 该脚本用于生成目标检测的角度验证数据
# -----------------------


# 创建一个ArgumentParser对象
parser = argparse.ArgumentParser()

parser.add_argument("--name", type=str, help="Set the value of the variable")
parser.add_argument("--Eg", type=str, help="Eg")
args = parser.parse_args()
file_name = args.name
Eg = args.Eg
cap = cv2.VideoCapture(r"D:\wise_transportation\data\road_videos\suzhou\\" + file_name + '.MOV')  # 获取视频对象
root_dir = r'D:\wise_transportation\data\2frame_dataset\\' + file_name + '-' + Eg + '/'

import shutil
shutil.rmtree(root_dir)
os.makedirs(root_dir, exist_ok=True)
isOpened = cap.isOpened  # 判断是否打开
# assert isOpened == True
# 视频信息获取
fps = cap.get(cv2.CAP_PROP_FPS)

imageNum = 0
sum = 0

fps = 60

E_skip = float(Eg)*fps  # 抽帧间隔的期望
R_avail = fps // 2  # 抽帧间隔的波动范围

timef = random.randint(E_skip - R_avail // 2, E_skip + R_avail // 2)

while (isOpened):
    sum += 1

    frameState = cap.grab()

    if frameState == True and (sum == timef):
        ret, frame = cap.retrieve()
        # 格式转变，BGRtoRGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # 转变成Image
        frame = Image.fromarray(np.uint8(frame))

        frame = np.array(frame)

        # RGBtoBGR满足opencv显示格式
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        imageNum = imageNum + 1
        os.mkdir(os.path.join(root_dir, str(imageNum)))
        fileName = root_dir + str(imageNum) + '/1.jpg'  # 存储路径
        cv2.imwrite(fileName, frame, [cv2.IMWRITE_JPEG_QUALITY, 100])
        print(fileName + " successfully write in")  # 输出存储状态
        sum += 1
        frameState = cap.grab()
        frameState = cap.grab()
        ret, frame = cap.retrieve()
        # 格式转变，BGRtoRGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # 转变成Image
        frame = Image.fromarray(np.uint8(frame))

        frame = np.array(frame)

        # RGBtoBGR满足opencv显示格式
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        fileName = root_dir + str(imageNum) + '/2.jpg'  # 存储路径
        cv2.imwrite(fileName, frame, [cv2.IMWRITE_JPEG_QUALITY, 100])
        print(fileName + " successfully write in")  # 输出存储状态

        # 更新timef
        timef += random.randint(E_skip - R_avail // 2, E_skip + R_avail // 2)

    elif frameState == False:
        break

print('finish!')
cap.release()
