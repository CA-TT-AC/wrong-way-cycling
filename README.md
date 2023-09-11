# wise_transportation

## 介绍
It is a repo to submit all code related to the project.

### blender
目录``blender``下为blender生成脚本.

### Dataset Scripts
Run code in ``dataset_scripts`` to extract frames in videos as you want.  

Run ``mmyolo/detect2label.py`` to generate orientation label by detection model.

### Prediction
目录``angle_prediction``下为角度预测算法部分代码.

### mmyolo
目录``mmyolo``下为目标检测部分代码.
#### Detection Pipeline
这是一个从两张连续帧推导出每个车角度的pipeline，其运行方式是
```
cd mmyolo
python detect_pipeline.py
```
可以在``mmyolo/infer_image.py``的传参中，修改模型，可以选择5m与5s.
训练好的checkpoint默认存放在``mmyolo/ckpt/``目录下，否则您可能需要在``mmyolo/infer_image.py``中修改文件路径.