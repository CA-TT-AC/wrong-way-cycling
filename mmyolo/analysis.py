# from pycocotools.coco import COCO

# dir_path = 'data/Ver3/'
# annotations_path = dir_path + 'Annotations/coco_info.json'

# coco = COCO(annotations_path)


# # get the number of images
# num_images = len(coco.getImgIds())
# print('The number of images: ', num_images)
# # get the number of bboxes
# num_bboxes = len(coco.getAnnIds())
# print('The number of bboxes: ', num_bboxes)
import mmcv
import os
import time
from infer_image import InferImage
from mmdet.apis import inference_detector, init_detector

path='data/Ver3/Images/'
model = init_detector('configs/custom/5s.py','work_dirs/MMYolo/5s_32batch_with_val.pth',device='cuda:0',cfg_options={})
length = len(os.listdir(r"./"+path))
sum=0
for picture in os.listdir(r"./"+path):
    img = mmcv.imread(path+'/'+picture)
    img = mmcv.imconvert(img, 'bgr', 'rgb')
    img=mmcv.imresize(img, (640, 640))
    start = time.time()
    inference_detector(model, img)
    end = time.time()
    sum+=(end-start)
print('length:',length)
print('time:',sum/length)
print('fps:',(length/sum))
