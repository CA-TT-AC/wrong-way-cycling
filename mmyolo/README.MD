# Model
模型后续放在Hugging Face平台方便更新
[Hugging Face](https://huggingface.co/ascka/MMYolo/tree/main)

- 使用 MIM 安装 MMEngine、 MMCV 和 MMDetection
```shell
pip install -U openmim
mim install -r requirements/mminstall.txt
```
- 安装 MMYOLO
```shell 
mim install -r requirements/albu.txt
mim install "mmyolo"
```
# Train & Test
-  训练
训练集放在data文件夹下，文件夹结构如下：
   - Dataset_one_class
     - Annotations
       - coco_info.json
     - Images
       - ...
```shell
Python tools/train.py configs/custom/5s.py
```
如果训练中途停止，可以在训练命令最后加上 ```--resume``` ,程序会自动从 ```work_dirs``` 中加载最新的权重文件恢复训练。
- 测试
```shell
Python tools/test.py configs/custom/5s.py work_dirs/king/best_coco_bbox_mAP_epoch_40.pth --show-dir show_results
```
运行以上测试命令， 你不仅可以得到模型训练部分所打印的 AP 性能，还可以将推理结果图片自动保存至 ```work_dirs/king/{timestamp}/show_results``` 文件夹中。

# 推理
- 从图片中检测
```python
    from infer_image import image_infer
    data = image_infer(img, path_to_config, path_to_checkpoint, threshold)
    bbox = data.bboxes
    """
    print(data)
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
    print(bbox)
    tensor([[0.4997, 0.7707, 0.0595, 0.4188],
            [0.8101, 0.3105, 0.5123, 0.6263]])
    """

```