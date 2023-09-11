import json
from PIL import Image


def crop_images_from_coco(coco_json_path, image_folder_path, output_folder_path):
    # 读取COCO JSON文件
    with open(coco_json_path, 'r') as json_file:
        coco_data = json.load(json_file)

        # 遍历每个图像
    for i, image_info in enumerate(coco_data['images']):
        image_id = image_info['id']
        image_filename = image_info['file_name']
        image_path = image_folder_path + '/' + image_filename

        # 打开图像
        image = Image.open(image_path)

        # 查找与当前图像相关的边界框
        for j, annotation in enumerate(coco_data['annotations']):
            if annotation['image_id'] == image_id:
                bbox = annotation['bbox']

                # 计算左上角和右下角坐标
                x, y, width, height = bbox
                left = int(x)
                top = int(y)
                right = int(x + width)
                bottom = int(y + height)

                # 裁剪图像
                cropped_image = image.crop((left, top, right, bottom))

                # 保存裁剪后的图像到输出文件夹
                output_filename = f"{j}.jpg"
                output_path = output_folder_path + '/' + output_filename
                cropped_image.save(output_path)

                print(f"Saved cropped image: {output_filename}")

    print("Crop images completed.")


# 示例用法
coco_json_path = r'D:\wise_transportation\dataset\DatasetId_V2.1\Annotations\coco_info.json'
image_folder_path = r'D:\wise_transportation\dataset\DatasetId_V2.1\Images'
output_folder_path = r'D:\wise_transportation\data\cropped_image'

crop_images_from_coco(coco_json_path, image_folder_path, output_folder_path)