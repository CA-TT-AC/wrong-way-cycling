import json
import math
import os
import random
import sys

import cv2
import gradio as gr

sys.path.append(r'D:\wise_transportation\wrong-way-cycling')
import numpy as np
import openpyxl
import torch
from matplotlib import pyplot as plt
from scipy.optimize import linear_sum_assignment

from angle_prediction.loss_function import angle2code, code2angle
from infer_image import InferImage
import mmcv
from pipeline_utils import box_iou_rotated
from sklearn.cluster import KMeans
import numpy as np
from detect_pipeline import get_angle
from PIL import Image
from angle_prediction import infer
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

def angle_infer(image, bbox, model, transform):
    image = Image.fromarray(image)
    cropped_patch = image.crop(bbox)
    plt.imshow(cropped_patch)
    angle_pred = infer.test(model, transform, cropped_patch)
    return angle_pred


def average(x, y):
    y, x = max(x, y), min(x, y)
    if y - x > x + 360 - y:
        x = x + 360
    return (x + y) / 2 % 360


def images2angle(paths, inited_model, match_type='Hungary'):
    bboxes = []
    for path in paths:
        img = mmcv.imread(path)
        img = mmcv.imconvert(img, 'bgr', 'rgb')
        bbox = inited_model.ImagePrediction(img).bboxes
        bbox = bbox.tolist()
        # 可视化
        # draw_bbox_on_image(img, bbox, path.split('\\')[-2] + '_' + path.split('\\')[-1])
        for i, one_bbox in enumerate(bbox):
            w = one_bbox[2] - one_bbox[0]
            h = one_bbox[3] - one_bbox[1]
            bbox[i][2] = w
            bbox[i][3] = h
            bbox[i].append(0.0)
        bboxes.append(bbox)

    # print('bbox:\n', bboxes)
    # get ious from one list of bbox to the other
    ious = box_iou_rotated(torch.tensor(bboxes[0]).float(),
                           torch.tensor(bboxes[1]).float()).cpu().numpy()
    # print('iou matrix:\n', ious)
    if len(ious) == 0 or len(ious[0]) == 0:
        return []
    match_list = []
    if match_type == 'Hungary':
        ious[ious > 0.98] = 0
        ious[ious < 0.5] = 0
        # print(ious)
        # 删除全为0的行和列
        nonzero_rows = np.any(ious != 0, axis=1)  # 找到非零行
        nonzero_cols = np.any(ious != 0, axis=0)  # 找到非零列
        ious = ious[nonzero_rows][:, nonzero_cols]  # 使用布尔索引获取非零行和列的子矩阵
        print(ious)
        row_id, col_id = linear_sum_assignment(ious, maximize=True)
        match_list = np.array([row_id, col_id]).transpose()
    else:
        iou_argmax = np.argmax(ious, axis=1)
        iou_max = np.max(ious, axis=1)
        enough_iou_idxs = np.where(iou_max > 0.05)
        for idx in enough_iou_idxs[0]:
            match_list.append((idx, iou_argmax[idx]))
    angles = []
    for i, j in match_list:
        point1 = bboxes[0][i]
        point2 = bboxes[1][j]
        center1 = point1[0] + point1[2] / 2, point1[1] + point1[3] / 2
        center2 = point2[0] + point2[2] / 2, point2[1] + point2[3] / 2
        vec = np.array(center1) - np.array(center2)
        angle = get_angle(vec[0], -vec[1])
        angles.append(angle)
    print(angles)

    return angles


def sort_(elem):
    return int(elem)


def draw_pic(points, scales=None, label_points=None, label_scales=None):
    # 将点列表分解为两个列表：x坐标和y坐标
    x, y = zip(*points)

    # 创建一个图表
    plt.figure(figsize=(10, 6))

    # 绘制折线图
    pred_line = plt.plot(x, y, marker='', linestyle='--', color='blue', label='prediction')  # 'o'表示点的样式
    print(x, y)
    print("scales:", scales)
    scales = np.array(scales) * 400
    # 在相同的坐标上绘制不同大小的点
    plt.scatter(x, y, s=scales, color='blue', marker='o', alpha=0.5)
    if label_points is not None:
        x, y = zip(*label_points)
        # 绘制折线图
        label_line = plt.plot(x, y, marker='', linestyle='dashdot', color='green', label='annotation')  # 'o'表示点的样式
        # 在相同的坐标上绘制不同大小的点
        plt.scatter(x, y, s=label_scales, color='green', marker='o', alpha=0.5)
        plt.legend()
    #     plt.legend((pred_line, label_line), ['prediction', 'annotation'])
    # else:
    #     plt.legend(pred_line, ['prediction'])
    # 设置图表的标题和坐标轴标签
    plt.title("Time per Ratio")
    plt.xlabel("Time")
    plt.ylabel("Ratio")

    # 显示图表
    plt.savefig("output.png", dpi=400)

def ori2minuteResult(times, values, phi):
    print("Phi:", phi)
    useful_data = np.array(values[1:])
    overlapping_data = phi*np.array(values[:-1])
    processed_data = useful_data - overlapping_data
    print("ori:", useful_data)
    print("processed:", processed_data)
    # 设置间隔
    interval = 30

    # 计算可以完整分割的组数
    num_full_groups = processed_data.size // interval

    # 重塑数组，仅包括可以完整分割的部分
    reshaped_data = processed_data[:num_full_groups * interval].reshape(-1, interval)

    # 计算每组的均值
    mean_values = reshaped_data.mean(axis=1)
    print("每分钟均值：", mean_values)
    return mean_values


def video2angle(path, pos_angle, Eg, ui=False):
    # angle_model, transform = infer.init_model_trans()
    detect_model = InferImage()
    cap = cv2.VideoCapture(path)
    isOpened = cap.isOpened  # 判断是否打开
    # assert isOpened == True
    # 视频信息获取
    fps = cap.get(cv2.CAP_PROP_FPS)

    imageNum = 0
    sum = 0

    fps = 30

    E_skip = float(Eg) * fps  # 抽帧间隔的期望
    R_avail = fps // 2  # 抽帧间隔的波动范围

    # mean how many seconds to record information once

    points_wrong = []
    points_right = []
    while isOpened:
        sum += 1
        frameState = cap.grab()
        frames = []
        if (frameState == True) and (sum % (Eg*fps) == 1):
            ret, frame = cap.retrieve()
            # 格式转变，BGRtoRGB
            # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # 转变成Image
            frame = Image.fromarray(np.uint8(frame))

            frame = np.array(frame)

            # RGBtoBGR满足opencv显示格式
            # frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            frames.append(frame)
            imageNum = imageNum + 1

            sum += 1
            if fps == 60:
                cap.grab()
            cap.grab()
            ret, frame = cap.retrieve()
            # 格式转变，BGRtoRGB
            # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # 转变成Image
            frame = Image.fromarray(np.uint8(frame))

            frame = np.array(frame)

            # RGBtoBGR满足opencv显示格式
            # frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            frames.append(frame)

            frames.reverse()
            ret_det = images2angle(frames, detect_model)
            ret_det = np.array(ret_det)
            # ret_angle = np.array(ret_angle)
            # a = np.maximum(ret_det, ret_angle)
            # b = np.minimum(ret_det, ret_angle)
            # divs = np.minimum(a - b, b - a + 360)
            ans = ret_det

            a = np.maximum(ans, pos_angle)
            b = np.minimum(ans, pos_angle)
            divs_with_pos = np.minimum(a - b, b - a + 360)
            divs_with_pos = [i > 90 for i in divs_with_pos]
            print(divs_with_pos)
            print("Time:", sum // fps)
            num_reverse = (sum // fps, int(np.sum(divs_with_pos)))
            num_forward = (sum // fps, len(divs_with_pos) - int(np.sum(divs_with_pos)))
            points_wrong.append(num_reverse)
            points_right.append(num_forward)

            frames.clear()
        elif not frameState:
            break

    # 将数据转换成Pandas Series对象，仅使用数值部分
    times = [pair[0] for pair in points_right]
    values = [pair[1] for pair in points_right]
    series_right = pd.Series(data=values, index=times)

    # 使用ARIMA模型构建，其中order=(1,0,1)表示ARMA(1,1)模型
    model_right = ARIMA(series_right, order=(1, 0, 1))

    # 拟合模型
    model_right_fit = model_right.fit()

    # 打印模型的摘要信息
    # print("right:")
    # print(model_right_fit.summary())
    # print(model_right_fit.params)
    minute_result_right = ori2minuteResult(times, values, model_right_fit.params['ar.L1'])
    # 将数据转换成Pandas Series对象，仅使用数值部分
    times = [pair[0] for pair in points_wrong]
    values = [pair[1] for pair in points_wrong]
    series_wrong = pd.Series(data=values, index=times)

    # 使用ARIMA模型构建，其中order=(1,0,1)表示ARMA(1,1)模型
    model_wrong = ARIMA(series_wrong, order=(1, 0, 0))

    # 拟合模型
    model_wrong_fit = model_wrong.fit()

    # 打印模型的摘要信息
    # print("wrong:")
    # print(model_wrong_fit.summary())
    # print(model_wrong_fit.params)

    minute_result_wrong = ori2minuteResult(times, values, model_wrong_fit.params['ar.L1'])



    # 可选：绘制原始数据和预测数据
    plt.figure(figsize=(10, 6))
    plt.plot(series_right, label='Original Right')
    plt.plot(model_right_fit.predict(), label='Predicted Right')
    plt.plot(series_wrong, label='Original Wrong')
    plt.plot(model_wrong_fit.predict(), label='Predicted Wrong')
    plt.legend()
    plt.show()

    points = []
    scales = []
    print("right:")
    for i in minute_result_right:
        print(i)
    print("wrong:")
    for i in minute_result_wrong:
        print(i)
    exit()
    # for i in range(len(minute_result_wrong)):
    #     points.append((i+1, minute_result_wrong[i]/(minute_result_wrong[i]+minute_result_right[i])))
    #     scales.append(minute_result_wrong[i]+minute_result_right[i])

    # json_file_path = r"D:\wise_transportation\wrong-way-cycling\mmyolo\data\78-2.json"
    #
    # # Read the JSON file
    # with open(json_file_path, 'r') as json_file:
    #     data = json.load(json_file)
    # label_points = []
    # label_scales = []
    # for pack in data:
    #     t, f = pack['right'], pack['wrong']
    #     label_points.append((pack['time'], f / (t + f)))
    #     label_scales.append((t + f) * 20)
    # draw_pic(points, scales, label_points, label_scales)
    #
    # ratio = minute_result_wrong.sum() / (minute_result_wrong.sum() + minute_result_right.sum())
    # print("final ratio:", ratio)
    # return


def twoimages2angle():
    folder = r'D:\PaddleDetection\data'
    paths = os.listdir(folder)
    paths.sort(reverse=True)
    for i in range(len(paths)):
        paths[i] = os.path.join(folder, paths[i])
    ans = images2angle(paths)
    print('angle:', ans)


def restore_i_from_point(point):
    x, y = point[0], point[1]
    radians_i = math.acos(x)
    degrees_i = math.degrees(radians_i)
    return degrees_i


def separate_numbers(data):
    new_data = []
    for i in data:
        i = torch.tensor(i).unsqueeze(0)
        point = angle2code(i)
        new_data.append(np.array(point.squeeze(0).cpu()))
    data = new_data
    data_array = np.array(data)
    kmeans = KMeans(n_clusters=2)
    kmeans.fit(data_array.reshape(-1, 3))
    centers = kmeans.cluster_centers_
    centers_angle = []
    for p in centers:
        centers_angle.append(code2angle(torch.tensor(p).unsqueeze(0).cuda()).float() / torch.pi * 180 + 180)

    a = torch.max(centers_angle[0].float(), centers_angle[1].float())
    b = torch.min(centers_angle[0].float(), centers_angle[1].float())
    div = torch.mean(torch.min(a - b, b - a + 360))
    if div < 90:
        return data_array, []
    labels = kmeans.labels_
    x_group = data_array[labels == 0]
    y_group = data_array[labels == 1]
    if len(x_group) < len(y_group):
        x_group, y_group = y_group, x_group
    x_angle = []
    y_angle = []
    for p in x_group:
        out = code2angle(torch.tensor(p).unsqueeze(0).cuda()).float() / torch.pi * 180 + 180
        x_angle.append(out)
    for p in y_group:
        out = code2angle(torch.tensor(p).unsqueeze(0).cuda()).float() / torch.pi * 180 + 180
        y_angle.append(out)
    return x_angle, y_angle

if __name__ == '__main__':
    # # 创建 Gradio 界面 iface = gr.Interface( video2angle, inputs=[gr.Video(label="upload video"), gr.Number(90,
    # label="Forward orientation"), gr.Number(1.5, label="Expected average gap of Monte Carlo sampling")],
    # outputs=[gr.Image(type="numpy", label="frame processing", ), gr.Textbox(label="Forward Number"), gr.Textbox(
    # label="Reverse Number")], )
    #
    # # 运行界面
    # iface.launch()

    # no ui
    video_path = r'D:\wise_transportation\data\road_videos\videosV2\221-28.MOV'
    eg = 2
    forward = 260
    video2angle(video_path, forward, eg)
    print("finish")
