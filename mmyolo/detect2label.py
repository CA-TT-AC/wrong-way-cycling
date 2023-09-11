import math
import os

import numpy as np
import torch
from PIL import Image

from infer_image import InferImage
import mmcv
from pipeline_utils import box_iou_rotated
from sklearn.cluster import KMeans
from scipy.optimize import linear_sum_assignment
import numpy as np



def get_angle(x, y):
    angle = math.atan2(y, x) * 180 / math.pi
    return angle if angle >= 0 else 360 + angle


# input 2 image paths, as a list or tuple output predicted angle
def images2angle(paths, inited_model, match_type='Hungary'):
    bboxes = []
    patches = []
    for ind, path in enumerate(paths):
        img = mmcv.imread(path)
        img = mmcv.imconvert(img, 'bgr', 'rgb')
        bbox = inited_model.ImagePrediction(img).bboxes
        bbox = bbox.tolist()
        for i, one_bbox in enumerate(bbox):

            if ind == 0:
                image = Image.fromarray(img)
                cropped_patch = image.crop(one_bbox)
                patches.append(cropped_patch)
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
    print('iou matrix:\n', ious)
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
    ans = []
    for i, j in match_list:
        point1 = bboxes[0][i]
        point2 = bboxes[1][j]
        center1 = point1[0] + point1[2] / 2, point1[1] + point1[3] / 2
        center2 = point2[0] + point2[2] / 2, point2[1] + point2[3] / 2
        vec = np.array(center1) - np.array(center2)
        angle = get_angle(vec[0], -vec[1])
        ans.append((angle, patches[i]))
    return ans


def sort_(elem):
    return int(elem)


def dataset2angle(path, output_folder_path):
    conditions = os.listdir(path)
    conditions.sort(key=sort_)
    detect_model = InferImage()
    cnt = 0
    for j, condition in enumerate(conditions):
        print('condition:', condition)
        paths = os.listdir(os.path.join(path, condition))
        paths.sort(reverse=True)
        for i in range(len(paths)):
            paths[i] = os.path.join(path, condition, paths[i])
        ret = images2angle(paths, detect_model)
        for angle, img in ret:
            output_filename = f"{cnt}_{int(angle)}.jpg"
            output_path = output_folder_path + '/' + output_filename
            img.save(output_path)
            cnt += 1
    pass


def twoimages2angle():
    folder = r'D:\PaddleDetection\data'
    paths = os.listdir(folder)
    paths.sort(reverse=True)
    for i in range(len(paths)):
        paths[i] = os.path.join(folder, paths[i])
    ans = images2angle(paths)
    print('angle:', ans)



if __name__ == '__main__':
    path = r'D:\wise_transportation\data\2frame_dataset\146-9'
    output_folder_path = r'D:\wise_transportation\data\146-9-auto-label'
    os.makedirs(output_folder_path, exist_ok=True)
    dataset2angle(path, output_folder_path)
