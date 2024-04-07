import os
import json
import numpy as np
import shutil
import argparse


parser = argparse.ArgumentParser(description="Parse dataset and generate output files.")
parser.add_argument('--dataset_root', type=str, required=True, help='Root directory of the dataset')
parser.add_argument('--output_root', type=str, required=True, help='Directory where output files will be saved')

args = parser.parse_args()
 
# 数据集路径
dataset_root = args.dataset_root
images_folder = os.path.join(dataset_root, "Images")
annotations_path = os.path.join(dataset_root, "Annotations/coco_info.json")
 
# 输出路径
output_root = args.output_root
os.makedirs(output_root, exist_ok=True)
 
# 读取annotations.json文件
with open(annotations_path, "r") as f:
    annotations_data = json.load(f)
 
# 提取images, annotations, categories
images = annotations_data["images"]
annotations = annotations_data["annotations"]
categories = annotations_data["categories"]
 
# 随机打乱数据
np.random.shuffle(images)
 
# 训练集，验证集比例
train_ratio, val_ratio = 0.9, 0.1
 
# 计算训练集，验证集的大小
num_images = len(images)
num_train = int(num_images * train_ratio)
num_val = int(num_images * val_ratio)
 
# 划分数据集
train_images = images[:num_train]
val_images = images[num_train:]
 
# 分别为训练集、验证集创建子文件夹
train_folder = os.path.join(output_root, "train")
val_folder = os.path.join(output_root, "val")
os.makedirs(train_folder, exist_ok=True)
os.makedirs(val_folder, exist_ok=True)
 
# 将图片文件复制到相应的子文件夹
for img in train_images:
    shutil.copy(os.path.join(images_folder, img["file_name"]), os.path.join(train_folder, img["file_name"]))
 
for img in val_images:
    shutil.copy(os.path.join(images_folder, img["file_name"]), os.path.join(val_folder, img["file_name"]))
 
# 根据图片id分配annotations
def filter_annotations(annotations, image_ids):
    return [ann for ann in annotations if ann["image_id"] in image_ids]
 
train_ann = filter_annotations(annotations, [img["id"] for img in train_images])
val_ann = filter_annotations(annotations, [img["id"] for img in val_images])
 
# 生成train.json, val.json, test.json
train_json = {"images": train_images, "annotations": train_ann, "categories": categories}
val_json = {"images": val_images, "annotations": val_ann, "categories": categories}

with open(os.path.join(output_root, "train.json"), "w") as f:
    json.dump(train_json, f)
 
with open(os.path.join(output_root, "val.json"), "w") as f:
    json.dump(val_json, f)
 
print("Split Done! ")