# Model
The model checkpoints have been hosted on the Hugging Face platform.
[Hugging Face](https://huggingface.co/ascka/MMYolo/tree/main)

- Install MMEngine, MMCV, and MMDetection using MIM
```shell
pip install -U openmim
mim install -r requirements/mminstall.txt
```
- Install MMYOLO
```shell 
mim install -r requirements/albu.txt
mim install "mmyolo"
```

# Train & Test
- Training
Place the training dataset in the 'data' folder with the following structure:
   - Dataset_one_class
     - Annotations
       - coco_info.json
     - Images
       - ...
```shell
Python tools/train.py configs/custom/5s.py
```
If training is interrupted, you can resume it by adding ```--resume``` to the training command. The program will automatically resume training from the latest saved checkpoint in the 'work_dirs' directory.

- Testing
```shell
Python tools/test.py configs/custom/5s.py work_dirs/king/best_coco_bbox_mAP_epoch_40.pth --show-dir show_results
```
Running the above test command will provide you with AP performance metrics, as well as automatically save inference result images to the 'work_dirs/king/{timestamp}/show_results' folder.

# Inference
- Object Detection from Images
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
