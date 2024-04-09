import json
import math
import os
import random
import sys

import cv2
import gradio as gr
from matplotlib.backends.backend_agg import FigureCanvasAgg

sys.path.append(r'D:\wise_transportation\wrong-way-cycling')
import numpy as np
import openpyxl
import torch
from matplotlib import pyplot as plt, patches
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


def draw_bboxes(bboxes, image, ret_det, ret_angle, flags, wwc, dpi=100):
    """
    Draw bounding boxes on an image and save it to a file.

    Parameters:
    - bboxes: List of bounding boxes, where each bbox is defined as (x, y, width, height).
    - image: Image on which to draw the bounding boxes.
    - filename: Filename for the saved image.
    """
    # 创建一个图和一个坐标轴，关闭坐标轴
    fig, ax = plt.subplots(1, figsize=(image.shape[1] / dpi, image.shape[0] / dpi), dpi=dpi)
    ax.axis('off')  # 不显示坐标轴
    image = mmcv.imconvert(image, 'bgr', 'rgb')
    # 显示图像
    ax.imshow(image)
    cur = 0
    print('wwc:', wwc)
    # 对每个边界框进行遍历并绘制
    for i, bbox in enumerate(bboxes):
        x, y, width, height, _ = bbox
        if cur >= len(flags) or flags[cur] > 120:
            continue
        print(wwc[cur])
        if not wwc[cur]:
            color = 'green'
        else:
            color = 'red'
        rect = patches.Rectangle((x, y), width, height, linewidth=5, edgecolor=color, facecolor='none')
        # 添加这个patch到坐标轴上
        ax.add_patch(rect)
        txt = "det: " + str(int(ret_det[cur])) + '\n' + "ori:" + str(int(ret_angle[cur]))
        ax.text(x, y, txt, color='blue', fontsize=10, verticalalignment='top',
                bbox=dict(facecolor='white', alpha=0, edgecolor='none'))
        cur += 1

    plt.gca().xaxis.set_major_locator(plt.NullLocator())  # plt.gca()表示获取当前子图"Get Current Axes"。

    plt.gca().yaxis.set_major_locator(plt.NullLocator())

    plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)

    # 将plt转化为numpy数据
    canvas = FigureCanvasAgg(plt.gcf())
    # 绘制图像
    canvas.draw()
    # 获取图像尺寸
    w, h = canvas.get_width_height()
    # 解码string 得到argb图像
    buf = np.fromstring(canvas.tostring_argb(), dtype=np.uint8)
    # 重构成w h 4(argb)图像
    buf.shape = (w, h, 4)
    # 转换为 RGBA
    buf = np.roll(buf, 3, axis=2)
    # 得到 Image RGBA图像对象 (需要Image对象的同学到此为止就可以了)
    image = Image.frombytes("RGBA", (w, h), buf.tostring())
    # 转换为numpy array rgba四通道数组
    image = np.asarray(image)
    # 转换为rgb图像
    rgb_image = image[:, :, :3]
    return rgb_image


def angle_infer(image, bbox, model, transform):
    image = Image.fromarray(image)
    cropped_patch = image.crop(bbox)
    plt.imshow(cropped_patch)
    angle_pred = infer.test(model, transform, cropped_patch)
    return angle_pred


def multi_angle_infer(image, bboxes, model, transform):
    if len(bboxes) == 0:
        return []
    image = Image.fromarray(image)
    inputs = []
    for bbox in bboxes:
        cropped_patch = image.crop(bbox)
        inputs.append(cropped_patch)
    angle_pred = infer.multi_image_test(model, transform, inputs)
    return angle_pred


def average(x, y):
    y, x = max(x, y), min(x, y)
    if y - x > x + 360 - y:
        x = x + 360
    return (x + y) / 2 % 360


# input 2 image paths, as a list or tuple output predicted angle
def images2angle(images, angle_model, transform, detect_model, match_type='Hungary'):
    '''
    修改该函数，使得在匹配之后，每一个bbox有其对应的角度，并在match list中，给出每一个match的角度预测
    '''
    # in bboxes, we save tow list, which contains bboxes of two images
    bboxes = []
    angle_preds = []
    for image in images:
        img = image
        img = mmcv.imconvert(img, 'bgr', 'rgb')
        bbox = detect_model.ImagePrediction(img).bboxes
        bbox = bbox.tolist()
        # angle_pred = []
        angle_pred = multi_angle_infer(img, bbox, angle_model, transform)
        for i, one_bbox in enumerate(bbox):
            # angle_pred.append(angle_infer(img, bbox[i], angle_model, transform))
            w = one_bbox[2] - one_bbox[0]
            h = one_bbox[3] - one_bbox[1]
            bbox[i][2] = w
            bbox[i][3] = h
            bbox[i].append(0.0)
        angle_preds.append(angle_pred)
        bboxes.append(bbox)

    # get ious from one list of bbox to the other
    ious = box_iou_rotated(torch.tensor(bboxes[0]).float(),
                           torch.tensor(bboxes[1]).float()).cpu().numpy()
    # print('iou matrix:\n', ious)
    if len(ious) == 0 or len(ious[0]) == 0:
        return [], [], [[], []]
    match_list = []
    if match_type == 'Hungary':
        ious[ious > 0.98] = 0
        ious[ious < 0.4] = 0
        # print(ious)
        # 删除全为0的行和列
        nonzero_rows = np.any(ious != 0, axis=1)  # 找到非零行
        nonzero_cols = np.any(ious != 0, axis=0)  # 找到非零列
        ious = ious[nonzero_rows][:, nonzero_cols]  # 使用布尔索引获取非零行和列的子矩阵
        # print(ious)
        row_id, col_id = linear_sum_assignment(ious, maximize=True)
        match_list = np.array([row_id, col_id]).transpose()
        # match_list_ = list(match_list_)
    else:
        iou_argmax = np.argmax(ious, axis=1)
        iou_max = np.max(ious, axis=1)
        enough_iou_idxs = np.where(iou_max > 0.05)
        for idx in enough_iou_idxs[0]:
            match_list.append((idx, iou_argmax[idx]))
    angles_det_ans = []
    angles_pred_ans = []
    for i, j in match_list:
        point1 = bboxes[0][i]
        point2 = bboxes[1][j]
        center1 = point1[0] + point1[2] / 2, point1[1] + point1[3] / 2
        center2 = point2[0] + point2[2] / 2, point2[1] + point2[3] / 2
        vec = np.array(center1) - np.array(center2)
        angle = get_angle(vec[0], -vec[1])
        angles_det_ans.append(angle)
        angles_pred_ans.append(average(angle_preds[0][i].cpu().item(), angle_preds[1][j].cpu()).item())
    print("detect pred：", angles_det_ans)
    print("angle, pred: ", angles_pred_ans)
    print('bbox:\n', bboxes)
    return angles_det_ans, angles_pred_ans, bboxes


def sort_(elem):
    return int(elem)


def draw_pic(points, scales=None, label_points=None, label_scales=None):
    # 将点列表分解为两个列表：x坐标和y坐标


    # 创建一个图表
    plt.figure(figsize=(10, 6))
    if len(points) != 0:

        x, y = zip(*points)
        # 绘制折线图
        pred_line = plt.plot(x, y, marker='', linestyle='--', color='blue', label='prediction')  # 'o'表示点的样式
        print(x, y)
        print(scales)
        scales = np.array(scales) * 20
        # 在相同的坐标上绘制不同大小的点
        plt.scatter(x, y, s=scales, color='blue', cmap='coolwarm')
        if label_points is not None:
            x, y = zip(*label_points)
            # 绘制折线图
            label_line = plt.plot(x, y, marker='', linestyle='dashdot', color='green', label='annotation')  # 'o'表示点的样式
            # 在相同的坐标上绘制不同大小的点
            plt.scatter(x, y, s=label_scales, color='green')
            plt.legend()
        #     plt.legend((pred_line, label_line), ['prediction', 'annotation'])
        # else:
        #     plt.legend(pred_line, ['prediction'])
        # 设置图表的标题和坐标轴标签
    plt.title("Time per Ratio")
    plt.xlabel("Time")
    plt.ylabel("Ratio")

    # # 显示图表
    # plt.savefig("output.png", dpi=400)
    # plt.gca().xaxis.set_major_locator(plt.NullLocator())  # plt.gca()表示获取当前子图"Get Current Axes"。
    #
    # plt.gca().yaxis.set_major_locator(plt.NullLocator())

    # plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)

    # 将plt转化为numpy数据
    canvas = FigureCanvasAgg(plt.gcf())
    # 绘制图像
    canvas.draw()
    # 获取图像尺寸
    w, h = canvas.get_width_height()
    # 解码string 得到argb图像
    buf = np.fromstring(canvas.tostring_argb(), dtype=np.uint8)
    # 重构成w h 4(argb)图像
    buf.shape = (w, h, 4)
    # 转换为 RGBA
    buf = np.roll(buf, 3, axis=2)
    # 得到 Image RGBA图像对象 (需要Image对象的同学到此为止就可以了)
    image = Image.frombytes("RGBA", (w, h), buf.tostring())
    # 转换为numpy array rgba四通道数组
    image = np.asarray(image)
    # 转换为rgb图像
    rgb_image = image[:, :, :3]
    return rgb_image


def video2angle(path, pos_angle, Eg, ui=False):
    angle_model, transform = infer.init_model_trans()
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

    timef = random.randint(E_skip - R_avail // 2, E_skip + R_avail // 2)

    num_forwards = 0
    num_reverses = 0
    cur_interval_idx = 1
    # mean how many seconds to record information once
    val_interval = 10

    points = []
    scales = []
    while isOpened:
        sum += 1
        frameState = cap.grab()
        frames = []
        if frameState == True and (sum == timef):
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

            # 更新timef
            timef += random.randint(E_skip - R_avail // 2, E_skip + R_avail // 2)
            frames.reverse()
            ret_det, ret_angle, bboxes = images2angle(frames, angle_model, transform, detect_model)
            ret_det = np.array(ret_det)
            ret_angle = np.array(ret_angle)
            a = np.maximum(ret_det, ret_angle)
            b = np.minimum(ret_det, ret_angle)
            divs = np.minimum(a - b, b - a + 360)
            ans = []
            print("divs:", divs)
            # 使用循环遍历divs列表
            for i in range(len(divs)):
                # 检查当前divs的值是否小于45
                if divs[i] < 120:
                    # 获取当前索引对应的ret_det和ret_angle的值
                    current_ret_det = ret_det[i]
                    current_ret_angle = ret_angle[i]
                    ave = average(current_ret_det, current_ret_angle)
                    ans.append(ave)
            a = np.maximum(ans, pos_angle)
            b = np.minimum(ans, pos_angle)
            divs_with_pos = np.minimum(a - b, b - a + 360)
            divs_with_pos = [i > 90 for i in divs_with_pos]
            num_reverse = int(np.sum(divs_with_pos))
            num_forwards += len(divs_with_pos) - num_reverse
            num_reverses += num_reverse
            img = draw_bboxes(bboxes[0], frames[0], ret_det, ret_angle, divs, divs_with_pos)
            # yield cv2.cvtColor(img, cv2.COLOR_BGR2RGB), len(divs_with_pos) - num_reverse, num_reverse

            if sum > cur_interval_idx * val_interval * fps:
                point = (cur_interval_idx, num_reverses / (num_reverses + num_forwards) if num_reverses + num_forwards else 0)
                scale = num_reverses + num_forwards
                points.append(point)
                scales.append(scale)
                cur_interval_idx += 1
                num_reverses = 0
                num_forwards = 0
            plot = draw_pic(points, scales)
            yield img, plot
                  # len(divs_with_pos) - num_reverse, num_reverse
            frames.clear()
        elif not frameState:
            break

    json_file_path = r"D:\wise_transportation\wrong-way-cycling\mmyolo\data\78-2.json"


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
    # 创建 Gradio 界面
    iface = gr.Interface(video2angle,
                         inputs=[gr.Video(label="upload video"),
                                 gr.Number(90, label="正向角度（例如向上为90，向下为270）"),
                                 gr.Number(3, label="采样间隔（秒）")],
                         outputs=[gr.Image(type="numpy", label="frame processing"),
                                  gr.Image(type="numpy", label="plot"),
                                  ],
                         title="计算机设计大赛：智慧城市——基于集成学习的高效非机动车逆行概率预测",
                         description="拖入视频点击sumbit即可进行概率预测。"
                         )

    # 运行界面
    iface.launch()

    # no ui
    # video_path = r'D:\wise_transportation\data\road_videos\suzhou\upload\78-2.MOV'
    # eg = 2
    # forward = 95
    # video2angle(video_path, forward, eg)
    # print("finish")
