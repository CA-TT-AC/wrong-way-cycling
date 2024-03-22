import os.path

import cv2
import mmcv
import numpy as np
from matplotlib import pyplot as plt
from mmdet.apis import inference_detector, init_detector
from mmengine.config import Config
from mmengine.structures import InstanceData
from typing import Sequence, Union

ImagesType = Union[str, np.ndarray, Sequence[str], Sequence[np.ndarray]]

class InferImage():
    """Inference from a image
    Args:
        path_to_config: the path to configuration
        path_to_checkpoint: the path to checkpoint
        threshold: Bbox score threshold
    """
    def __init__(self,path_to_checkpoint: str = r'./ckpt/5m_ver2.pth', path_to_config: str = r'.\configs\custom\5m.py',
                 threshold: float = 0.8) -> None:
        config = Config.fromfile(path_to_config)
        self.model = init_detector(config,
                            path_to_checkpoint,
                            device='cuda:0',
                            cfg_options={})
        self.threshold = threshold
    def ImagePrediction(self, image: ImagesType) -> InstanceData:
        '''
        Args:
            image: the path to the image file or the image object
        Output:
            InstanceData with the image information and predictions
        Example:
        # >>> print(instance_data)
            <InstanceData(
                META INFORMATION
                img_shape: (800, 1196, 3)
                pad_shape: (800, 1216, 3)
                DATA FIELDS
                det_labels: tensor([2, 3])
                det_scores: tensor([0.8000, 0.7000])
                bboxes: tensor([[0.4997, 0.7707, 0.0595, 0.4188],
                            [0.8101, 0.3105, 0.5123, 0.6263]])
                polygons: [[1, 2, 3, 4], [5, 6, 7, 8]]
            ) at 0x7fb492de6280>
        '''
        result = inference_detector(self.model, image)
        pred_instances = result.pred_instances[result.pred_instances.scores > self.threshold]
        return pred_instances
    def __new__(cls, *args, **kwargs):
        if not hasattr(InferImage, "_instance"):
            InferImage._instance = object.__new__(cls)  
        return InferImage._instance
        

def image_infer(
        image: ImagesType,
        path_to_checkpoint: str = r'.\ckpt\5m.pth',
        path_to_config: str = r'.\configs\custom\5m.py',
        threshold: float = 0.3
) -> InstanceData:
    
    config = Config.fromfile(path_to_config)
    model = init_detector(config,
                          path_to_checkpoint,
                          device='cuda:0',
                          cfg_options={})
    result = inference_detector(model, image)
    pred_instances = result.pred_instances[result.pred_instances.scores > threshold]
    return pred_instances


def draw_bbox_on_image(image, bbox, name):
    # 将张量格式的bbox转换为列表
    # bbox = bbox.tolist()

    # 获取图片的宽度和高度
    image_height, image_width = 1, 1

    # 遍历每个边界框
    for box in bbox:
        x_min, y_min, x_max, y_max = box

        # # 将归一化的坐标转换为实际像素坐标
        x_min = int(x_min * image_width)
        y_min = int(y_min * image_height)
        x_max = int(x_max * image_width)
        y_max = int(y_max * image_height)

        # 绘制边界框
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 255), 2)

    # 显示结果
    path = os.path.join(r'D:\wise_transportation\data\visual', name)
    # os.makedirs(path, exist_ok=True)
    cv2.imwrite(path, image)
    return image

if __name__ == '__main__':
    img_path = r"D:\wise_transportation\data\DSC_0062.JPG"
    img = mmcv.imread(img_path)
    img = mmcv.imconvert(img, 'bgr', 'rgb')
    infer = InferImage()
    # infer2 = InferImage()
    print(infer)
    bbox = infer.ImagePrediction(image=img).bboxes
    print(bbox)
    draw_bbox_on_image(img, bbox)
